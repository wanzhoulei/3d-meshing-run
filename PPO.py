import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

import copy
import time


# -------------------------
# Utilities
# -------------------------

def flatten_time_batch(tensor: torch.Tensor) -> torch.Tensor:
    """
    Flatten leading (T, B, ...) -> (T*B, ...)
    """
    T, B = tensor.shape[:2]
    return tensor.reshape(T * B, *tensor.shape[2:])


def _pad_lastdim(t: torch.Tensor, target: int, dim: int) -> torch.Tensor:
    """
    Pad tensor t with zeros along dimension `dim` so that size(dim) == target.
    """
    cur = t.size(dim)
    if cur == target:
        return t
    pad_sizes = [0, 0] * t.dim()
    # PyTorch pad uses last dimensions first; we build for general dim by permuting
    # Simpler: allocate new tensor and copy.
    shape = list(t.shape)
    shape[dim] = target
    out = t.new_zeros(shape)
    index = [slice(None)] * t.dim()
    index[dim] = slice(0, cur)
    out[tuple(index)] = t
    return out


def _bg_to_device(bg, device: torch.device):
    """Move a BatchedGraph-like object to a target device."""
    return bg.__class__(
        x=bg.x.to(device),
        edge_feat=bg.edge_feat.to(device),
        face_feat=bg.face_feat.to(device),
        quality_feat=bg.quality_feat.to(device),
        edge_index=bg.edge_index.to(device),
        node_mask=bg.node_mask.to(device),
        edge_mask=bg.edge_mask.to(device),
        edge_node_mask=bg.edge_node_mask.to(device),
        face_node_mask=bg.face_node_mask.to(device),
        edge_action_mask=bg.edge_action_mask.to(device),
        face_action_mask=bg.face_action_mask.to(device),
        aux=bg.aux,
    )


def _device_matches(actual: torch.device, expected: torch.device) -> bool:
    """Robust device match (treat 'cuda' and 'cuda:0' as equivalent)."""
    if actual.type != expected.type:
        return False
    if actual.type != "cuda":
        return True
    # If either side omits explicit index, accept the match.
    if actual.index is None or expected.index is None:
        return True
    return actual.index == expected.index


def compute_gae_from_buffers(buffers, gamma=0.99, gae_lambda=0.95):
    """
    Same GAE logic as your 2D PPO.py.

    buffers must contain:
      reward_buffer: (T,B)
      value_buffer : (T,B)
      done_buffer  : (T,B) float (1.0 means terminated at t)
      last_value_buffer: (B,) optional bootstrap value for state after final rollout step

    Returns:
      advantages_norm: (T,B)
      returns        : (T,B)
      advantages     : (T,B)
    """
    reward_buffer = buffers["reward_buffer"]
    value_buffer  = buffers["value_buffer"]
    done_buffer   = buffers["done_buffer"]

    T, B = reward_buffer.shape
    device = reward_buffer.device
    dtype = reward_buffer.dtype

    advantages = torch.zeros(T, B, dtype=dtype, device=device)

    V = value_buffer
    V_next = torch.zeros_like(V)
    V_next[:-1] = V[1:]
    last_value = buffers.get("last_value_buffer", None)
    if last_value is None:
        V_next[-1] = 0.0
    else:
        V_next[-1] = last_value.to(device=device, dtype=dtype)

    adv_next = torch.zeros(B, dtype=dtype, device=device)

    for t in reversed(range(T)):
        done_t = done_buffer[t]
        not_done = 1.0 - done_t
        delta = reward_buffer[t] + gamma * not_done * V_next[t] - V[t]
        adv = delta + gamma * gae_lambda * not_done * adv_next
        advantages[t] = adv
        adv_next = adv

    returns = advantages + V

    adv_mean = advantages.mean()
    adv_std = advantages.std()
    advantages_norm = (advantages - adv_mean) / (adv_std + 1e-8)

    return advantages_norm, returns, advantages


# -------------------------
# Rollout
# -------------------------

def PolicyRollout(
    envs,
    model,
    T: int,
    batch_from_obs,
    model_action_to_env_with_sizes,
    device: torch.device,
    display_every: int = 20,
    force_cpu_rollout: bool = True, #whether to force rollout to be on cpu entirely
    cpu_embed_for_gpu_rollout: bool = True,
):
    """
    Run a rollout of length T across vectorized envs using current policy.

    This mirrors your 2D PolicyRollout structure, but uses:
      - env obs dicts (3D tet env)
      - batch_from_obs() to build padded graph tensors
      - Mesh3DActorCritic (single module for policy + value)

    Returns:
      buffers: dict of padded rollout tensors (on CPU)
      avg_reward: float
    """
    B = envs.num_envs

    # -------------------------
    # Choose rollout device/model
    # -------------------------
    rollout_device = torch.device("cpu") if force_cpu_rollout else device
    if rollout_device.type == "cuda" and cpu_embed_for_gpu_rollout:
        embed_device = torch.device("cpu")
    else:
        embed_device = rollout_device

    # Detect where the model currently lives (best-effort)
    try:
        model_param_device = next(model.parameters()).device
    except StopIteration:
        model_param_device = torch.device("cpu")

    # Build a rollout_model that matches rollout_device, without per-step transfers
    rollout_model = model
    if rollout_device.type == "cpu" and model_param_device.type != "cpu":
        rollout_model = copy.deepcopy(model).to(rollout_device)
    elif rollout_device.type != model_param_device.type:
        rollout_model = model.to(rollout_device)

    rollout_model.eval()

    obs = envs.reset()

    print("Rolling out policies...")

    # We will store step-wise tensors in lists (since Emax/Fmax/Mmax can vary with time),
    # then pad to global maxima at the end (minimal changes to PPO logic).
    step_graphs = []
    actions_list = []
    old_logp_list = []
    values_list = []
    rewards_list = []
    dones_list = []

    for t in tqdm(range(T)):
        # Build features on embed_device, store rollout buffers on CPU.
        bg_build = batch_from_obs(obs, device=embed_device)
        if bg_build.x.device.type == "cpu":
            bg_store = bg_build
        else:
            bg_store = _bg_to_device(bg_build, torch.device("cpu"))
        step_graphs.append(bg_store)
        if bg_build.x.device == rollout_device:
            bg_model = bg_build
        else:
            bg_model = _bg_to_device(bg_build, rollout_device)

        # Determine per-env true sizes for action index mapping (ENV ordering uses [faces, edges])
        E_sizes = np.array([o["edges"].shape[1] for o in obs], dtype=np.int64)
        F_sizes = np.array([o["faces"].shape[0] for o in obs], dtype=np.int64)

        with torch.no_grad():
            # Guard against accidental mixed-device tensors in rollout.
            if not _device_matches(bg_model.x.device, rollout_device):
                raise RuntimeError(
                    f"Rollout batch is on {bg_model.x.device}, expected {rollout_device}."
                )

            policy_out, value, _, _ = rollout_model(
                x=bg_model.x,
                edge_feat=bg_model.edge_feat,
                face_feat=bg_model.face_feat,
                critic_global_feat=bg_model.quality_feat,
                edge_index=bg_model.edge_index,
                node_mask=bg_model.node_mask,
                edge_mask=bg_model.edge_mask,
                edge_node_mask=bg_model.edge_node_mask,
                face_node_mask=bg_model.face_node_mask,
                edge_action_mask=bg_model.edge_action_mask,
                face_action_mask=bg_model.face_action_mask,
            )

            logits = policy_out.logits  # (B, Emax+Fmax) in MODEL ordering [edges, faces]

            # If an env has no valid actions (all mask false), logits may be all -inf.
            # We handle by forcing action=0, logp=0, value=0 for those envs.
            valid_any = (bg_model.edge_action_mask.any(dim=1) | bg_model.face_action_mask.any(dim=1))  # (B,)

            # sample
            dist = torch.distributions.Categorical(logits=logits)
            a_model = dist.sample()  # (B,)

            # compute logp(a)
            logp_all = torch.log_softmax(logits, dim=1)
            logp_a = logp_all.gather(1, a_model.unsqueeze(1)).squeeze(1)  # (B,)

            # fix invalid-any envs
            a_model = torch.where(valid_any, a_model, torch.zeros_like(a_model))
            logp_a  = torch.where(valid_any, logp_a, torch.zeros_like(logp_a))
            value   = torch.where(valid_any, value, torch.zeros_like(value))

        # Map MODEL action indices -> ENV action indices
        a_model_np = a_model.detach().cpu().numpy().astype(np.int64)
        a_env = model_action_to_env_with_sizes(a_model_np, E_sizes, F_sizes)

        res = envs.step(a_env)
        obs = res.obs

        actions_list.append(a_model.detach().cpu())
        old_logp_list.append(logp_a.detach().cpu())
        values_list.append(value.detach().cpu())
        rewards_list.append(torch.as_tensor(res.reward, dtype=torch.float32))
        dones_list.append(torch.as_tensor(res.done.astype(np.float32), dtype=torch.float32))

        if display_every is not None and (t + 1) % display_every == 0:
            print(f"Step {t+1}/{T}: mean reward {float(res.reward.mean()):.6f} | done {int(res.done.sum())}/{B}")

    # Bootstrap value from the state after the final env.step().
    with torch.no_grad():
        bg_last = batch_from_obs(obs, device=embed_device)
        if bg_last.x.device != rollout_device:
            bg_last = _bg_to_device(bg_last, rollout_device)
        _, last_value, _, _ = rollout_model(
            x=bg_last.x,
            edge_feat=bg_last.edge_feat,
            face_feat=bg_last.face_feat,
            critic_global_feat=bg_last.quality_feat,
            edge_index=bg_last.edge_index,
            node_mask=bg_last.node_mask,
            edge_mask=bg_last.edge_mask,
            edge_node_mask=bg_last.edge_node_mask,
            face_node_mask=bg_last.face_node_mask,
            edge_action_mask=bg_last.edge_action_mask,
            face_action_mask=bg_last.face_action_mask,
        )
    last_value_buffer = last_value.detach().cpu().to(torch.float32)

    # -------------------------
    # Pad all step_graphs to global maxima across time
    # -------------------------
    # Get global maxima
    De = step_graphs[0].edge_feat.shape[-1]
    Df = step_graphs[0].face_feat.shape[-1]
    Dq = step_graphs[0].quality_feat.shape[-1]
    Emax_global = max(bg.edge_feat.shape[1] for bg in step_graphs)
    Fmax_global = max(bg.face_feat.shape[1] for bg in step_graphs)
    Nmax_global = Emax_global + Fmax_global
    Mmax_global = max(bg.edge_index.shape[2] for bg in step_graphs)

    # Allocate buffers on CPU (like your 2D PPO.py does)
    x_buffer = torch.zeros(T, B, Nmax_global, 3, dtype=torch.float32)
    edge_feat_buffer = torch.zeros(T, B, Emax_global, De, dtype=torch.float32)
    face_feat_buffer = torch.zeros(T, B, Fmax_global, Df, dtype=torch.float32)
    quality_feat_buffer = torch.zeros(T, B, Dq, dtype=torch.float32)
    edge_index_buffer = torch.zeros(T, B, 2, Mmax_global, dtype=torch.long)

    node_mask_buffer = torch.zeros(T, B, Nmax_global, dtype=torch.bool)
    edge_mask_buffer = torch.zeros(T, B, Mmax_global, dtype=torch.bool)
    edge_node_mask_buffer = torch.zeros(T, B, Emax_global, dtype=torch.bool)
    face_node_mask_buffer = torch.zeros(T, B, Fmax_global, dtype=torch.bool)
    edge_action_mask_buffer = torch.zeros(T, B, Emax_global, dtype=torch.bool)
    face_action_mask_buffer = torch.zeros(T, B, Fmax_global, dtype=torch.bool)

    action_buffer = torch.zeros(T, B, dtype=torch.long)
    reward_buffer = torch.zeros(T, B, dtype=torch.float32)
    value_buffer = torch.zeros(T, B, dtype=torch.float32)
    logp_buffer = torch.zeros(T, B, dtype=torch.float32)
    done_buffer = torch.zeros(T, B, dtype=torch.float32)

    for t in range(T):
        bg = step_graphs[t]

        # Pad per-step tensors into global buffers
        x_t = _pad_lastdim(bg.x.detach().cpu(), Nmax_global, dim=1)
        node_mask_t = _pad_lastdim(bg.node_mask.detach().cpu(), Nmax_global, dim=1)

        edge_feat_t = _pad_lastdim(bg.edge_feat.detach().cpu(), Emax_global, dim=1)
        face_feat_t = _pad_lastdim(bg.face_feat.detach().cpu(), Fmax_global, dim=1)
        quality_feat_t = bg.quality_feat.detach().cpu()

        edge_node_mask_t = _pad_lastdim(bg.edge_node_mask.detach().cpu(), Emax_global, dim=1)
        face_node_mask_t = _pad_lastdim(bg.face_node_mask.detach().cpu(), Fmax_global, dim=1)

        edge_action_mask_t = _pad_lastdim(bg.edge_action_mask.detach().cpu(), Emax_global, dim=1)
        face_action_mask_t = _pad_lastdim(bg.face_action_mask.detach().cpu(), Fmax_global, dim=1)

        edge_index_t = _pad_lastdim(bg.edge_index.detach().cpu(), Mmax_global, dim=2)
        edge_mask_t = _pad_lastdim(bg.edge_mask.detach().cpu(), Mmax_global, dim=1)

        x_buffer[t] = x_t
        node_mask_buffer[t] = node_mask_t

        edge_feat_buffer[t] = edge_feat_t
        face_feat_buffer[t] = face_feat_t
        quality_feat_buffer[t] = quality_feat_t

        edge_node_mask_buffer[t] = edge_node_mask_t
        face_node_mask_buffer[t] = face_node_mask_t

        edge_action_mask_buffer[t] = edge_action_mask_t
        face_action_mask_buffer[t] = face_action_mask_t

        edge_index_buffer[t] = edge_index_t
        edge_mask_buffer[t] = edge_mask_t

        action_buffer[t] = actions_list[t]
        reward_buffer[t] = rewards_list[t]
        value_buffer[t] = values_list[t].to(torch.float32)
        logp_buffer[t] = old_logp_list[t].to(torch.float32)
        done_buffer[t] = dones_list[t]

    avg_reward = float(reward_buffer.mean().item())
    print(f"\nAverage reward over {T} steps and {B} envs: {avg_reward:.6f}")

    buffers = dict(
        x_buffer=x_buffer,
        edge_feat_buffer=edge_feat_buffer,
        face_feat_buffer=face_feat_buffer,
        quality_feat_buffer=quality_feat_buffer,
        edge_index_buffer=edge_index_buffer,
        node_mask_buffer=node_mask_buffer,
        edge_mask_buffer=edge_mask_buffer,
        edge_node_mask_buffer=edge_node_mask_buffer,
        face_node_mask_buffer=face_node_mask_buffer,
        edge_action_mask_buffer=edge_action_mask_buffer,
        face_action_mask_buffer=face_action_mask_buffer,
        action_buffer=action_buffer,
        reward_buffer=reward_buffer,
        value_buffer=value_buffer,
        last_value_buffer=last_value_buffer,
        logp_buffer=logp_buffer,
        done_buffer=done_buffer,
    )

    return buffers, avg_reward


# -------------------------
# PPO Update
# -------------------------

def PPO_update(
    model,
    optimizer,
    buffers,
    advantages,
    returns,
    *,
    K_epochs: int = 4,
    minibatch_size: int = 2048,
    eps_clip: float = 0.2,
    c_v: float = 0.5,
    c_ent: float = 0.01,
    max_grad_norm: float = 0.5,
    device: torch.device = torch.device("cpu"),
    log_every_minibatch: bool = False,
    use_tqdm: bool = False,
    profile_timing: bool = False,
):
    """
    PPO update that mirrors your 2D PPO_update, but uses 3D batched graph tensors
    and a single Mesh3DActorCritic module.

    Loss:
      L = L_actor + c_v * L_value - c_ent * entropy
    """
    # Flatten rollout buffers
    x_flat = flatten_time_batch(buffers["x_buffer"]).to(device)  # (TB, Nmax, 3)
    edge_feat_flat = flatten_time_batch(buffers["edge_feat_buffer"]).to(device)  # (TB, Emax, De)
    face_feat_flat = flatten_time_batch(buffers["face_feat_buffer"]).to(device)  # (TB, Fmax, Df)
    quality_feat_flat = flatten_time_batch(buffers["quality_feat_buffer"]).to(device)  # (TB, Dq)
    edge_index_flat = flatten_time_batch(buffers["edge_index_buffer"]).to(device).long()  # (TB,2,Mmax)

    node_mask_flat = flatten_time_batch(buffers["node_mask_buffer"]).to(device)  # (TB,Nmax)
    edge_mask_flat = flatten_time_batch(buffers["edge_mask_buffer"]).to(device)  # (TB,Mmax)
    edge_node_mask_flat = flatten_time_batch(buffers["edge_node_mask_buffer"]).to(device)  # (TB,Emax)
    face_node_mask_flat = flatten_time_batch(buffers["face_node_mask_buffer"]).to(device)  # (TB,Fmax)
    edge_action_mask_flat = flatten_time_batch(buffers["edge_action_mask_buffer"]).to(device)  # (TB,Emax)
    face_action_mask_flat = flatten_time_batch(buffers["face_action_mask_buffer"]).to(device)  # (TB,Fmax)

    actions_flat = flatten_time_batch(buffers["action_buffer"]).to(device).long()  # (TB,)
    old_logp_flat = flatten_time_batch(buffers["logp_buffer"]).to(device).float()  # (TB,)
    adv_flat = flatten_time_batch(advantages).to(device).float()  # (TB,)
    returns_flat = flatten_time_batch(returns).to(device).float()  # (TB,)

    # Re-normalize advantages (same as your 2D code)
    adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

    num_samples = x_flat.size(0)
    model.train()

    loss_list = []
    actor_loss_list = []
    value_loss_list = []
    entropy_list = []

    if use_tqdm:
        print("Updating policies...")

    epoch_iter = tqdm(range(K_epochs)) if use_tqdm else range(K_epochs)
    for epoch in epoch_iter:
        indices = torch.randperm(num_samples, device=device)
        if use_tqdm:
            print(f"----------In Epoch ({epoch+1}/{K_epochs})----------")

        loss_avg = 0.0
        actor_avg = 0.0
        value_avg = 0.0
        ent_avg = 0.0
        num_batches = 0
        t_data = 0.0
        t_forward = 0.0
        t_backward = 0.0
        t_opt = 0.0

        mb_iter = tqdm(range(0, num_samples, minibatch_size)) if use_tqdm else range(0, num_samples, minibatch_size)
        for start in mb_iter:
            t0 = time.perf_counter()
            end = min(start + minibatch_size, num_samples)
            mb_idx = indices[start:end]
            num_batches += 1

            x_b = x_flat[mb_idx]
            edge_feat_b = edge_feat_flat[mb_idx]
            face_feat_b = face_feat_flat[mb_idx]
            quality_feat_b = quality_feat_flat[mb_idx]
            edge_index_b = edge_index_flat[mb_idx]

            node_mask_b = node_mask_flat[mb_idx]
            edge_mask_b = edge_mask_flat[mb_idx]
            edge_node_mask_b = edge_node_mask_flat[mb_idx]
            face_node_mask_b = face_node_mask_flat[mb_idx]
            edge_action_mask_b = edge_action_mask_flat[mb_idx]
            face_action_mask_b = face_action_mask_flat[mb_idx]

            actions_b = actions_flat[mb_idx]
            old_logp_b = old_logp_flat[mb_idx]
            adv_b = adv_flat[mb_idx]
            ret_b = returns_flat[mb_idx]
            t_data += time.perf_counter() - t0

            # Forward current policy/value
            t1 = time.perf_counter()
            policy_out, value_new, _, _ = model(
                x=x_b,
                edge_feat=edge_feat_b,
                face_feat=face_feat_b,
                critic_global_feat=quality_feat_b,
                edge_index=edge_index_b,
                node_mask=node_mask_b,
                edge_mask=edge_mask_b,
                edge_node_mask=edge_node_mask_b,
                face_node_mask=face_node_mask_b,
                edge_action_mask=edge_action_mask_b,
                face_action_mask=face_action_mask_b,
            )
            logits_new = policy_out.logits  # (B_mb, Amax)

            log_probs_all = F.log_softmax(logits_new, dim=1)

            logp_new = log_probs_all.gather(1, actions_b.unsqueeze(1)).squeeze(1)
            if c_ent != 0.0:
                probs_all = log_probs_all.exp()
                entropy = -(probs_all * log_probs_all).sum(dim=1).mean()
            else:
                entropy = torch.zeros((), dtype=logp_new.dtype, device=logp_new.device)

            # PPO objective
            ratios = torch.exp(logp_new - old_logp_b)
            surr1 = ratios * adv_b
            surr2 = torch.clamp(ratios, 1.0 - eps_clip, 1.0 + eps_clip) * adv_b
            actor_loss = -torch.min(surr1, surr2).mean()

            value_loss = 0.5 * (value_new - ret_b).pow(2).mean()

            loss = actor_loss + c_v * value_loss - c_ent * entropy
            t_forward += time.perf_counter() - t1

            t2 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            t_backward += time.perf_counter() - t2

            t3 = time.perf_counter()
            if max_grad_norm is not None and max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            t_opt += time.perf_counter() - t3

            if log_every_minibatch:
                print(
                    f"Loss: {loss.item():.3f} | Actor: {actor_loss.item():.3f} | "
                    f"Value: {value_loss.item():.3f} | Entropy: {entropy.item():.3f}"
                )

            loss_avg += loss.item()
            actor_avg += actor_loss.item()
            value_avg += value_loss.item()
            ent_avg += entropy.item()

        loss_list.append(loss_avg / num_batches)
        actor_loss_list.append(actor_avg / num_batches)
        value_loss_list.append(value_avg / num_batches)
        entropy_list.append(ent_avg / num_batches)
        if profile_timing:
            print(
                f"[PPO timing] epoch {epoch+1}: data={t_data:.2f}s "
                f"forward={t_forward:.2f}s backward={t_backward:.2f}s opt={t_opt:.2f}s"
            )

    return loss_list, actor_loss_list, value_loss_list, entropy_list
