"""
Training script for the sacrifice-moves tet mesh refinement agent.

Differences from train.py
--------------------------
1. Uses SacrificeVecEnv (lookback reward) instead of TetMeshRefineVecEnv.
2. Loads tet_dataset_sacrifice_k20_N2000.npz via SacrificeDataset.
3. Output run directories are prefixed "sacrifice-" for easy identification.
4. Adds lookback_window to the saved train_config.json.

Everything else (model architecture, PPO hyperparameters, checkpointing,
plotting) is identical to train.py so that results are directly comparable.

Usage
-----
    python train_sacrifice.py

Generate the dataset first if it does not exist:
    python gen_sacrifice_dataset.py
"""
from __future__ import annotations

import os
import json
from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from sacrifice_dataset import SacrificeDataset
from tet_env_sacrifice import SacrificeVecEnv

from model_face_edge_gpt import Mesh3DActorCritic
from initial_embedding import batch_from_obs, model_action_to_env_with_sizes
from PPO import PolicyRollout, compute_gae_from_buffers, PPO_update


# ---------------------------------------------------------------------------
# Helpers (identical to train.py)
# ---------------------------------------------------------------------------

def save_training_artifacts(
    run_dir,
    avg_reward_list,
    avg_reward_unscaled_round,
    total_loss_round,
    actor_loss_round,
    value_loss_round,
    entropy_round,
):
    metrics_npz_path = os.path.join(run_dir, "training_metrics.npz")
    np.savez(
        metrics_npz_path,
        avg_reward=np.asarray(avg_reward_list, dtype=np.float64),
        avg_reward_unscaled=np.asarray(avg_reward_unscaled_round, dtype=np.float64),
        total_loss=np.asarray(total_loss_round, dtype=np.float64),
        actor_loss=np.asarray(actor_loss_round, dtype=np.float64),
        value_loss=np.asarray(value_loss_round, dtype=np.float64),
        entropy=np.asarray(entropy_round, dtype=np.float64),
    )

    plt.figure()
    plt.plot(avg_reward_unscaled_round, "*-")
    plt.grid(True)
    plt.xlabel("Round")
    plt.ylabel("Average Reward (unscaled rollout mean)")
    plt.title("PPO Training — Sacrifice Task (Unscaled Reward)")
    reward_png = os.path.join(run_dir, "reward_curve.png")
    plt.savefig(reward_png, dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(run_dir, "avg_reward_trace.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(total_loss_round, label="total loss")
    plt.plot(actor_loss_round, label="actor loss")
    plt.plot(value_loss_round, label="value loss")
    plt.grid(True)
    plt.xlabel("Round")
    plt.ylabel("Loss (avg over PPO epochs)")
    plt.title("PPO Loss Curves — Sacrifice Task")
    plt.legend()
    loss_png = os.path.join(run_dir, "loss_curves.png")
    plt.savefig(loss_png, dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(run_dir, "loss_trace.png"), dpi=150, bbox_inches="tight")
    plt.close()

    return metrics_npz_path, reward_png, loss_png


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # -----------------------------------------------------------------------
    # Hyperparameters (same defaults as train.py)
    # -----------------------------------------------------------------------
    num_envs = 32
    T_rollout = 50
    rounds = 180
    gamma = 0.985
    gae_lambda = 0.93

    K_epochs = 24
    minibatch_size = 128
    eps_clip = 0.15
    c_v = 1.5
    c_ent = 0.003
    max_grad_norm = 0.5

    lr = 2e-4
    lr_final = 0.05 * lr
    c_ent_final = c_ent
    lr_decay_start_frac = 0.15
    ent_decay_start_frac = 0.5
    use_tqdm_update = False
    profile_timing_update = True
    enable_torch_compile = False
    force_cpu_rollout = True
    cpu_embed_for_gpu_rollout = True

    d_h = 96
    num_layers = 4
    msg_hidden = 96
    value_hidden = 96
    dropout = 0

    max_steps_per_episode = 150   # slightly longer than base task (harder instances)
    patience = 50
    reward_scale = 10
    tet_quality_mode = "mean_ratio"

    # Sacrifice-task-specific
    lookback_window = 8
    dataset_path = "tet_dataset_sacrifice_k20_N2000.npz"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if device.type == "cuda":
        K_epochs = 10
        minibatch_size = 128
        lr = 2e-4
        lr_final = 0.05 * lr
        c_ent = 0.004
        c_ent_final = 0.0012
        force_cpu_rollout = False
        cpu_embed_for_gpu_rollout = True
    else:
        K_epochs = 4
        minibatch_size = 512
        c_ent = 0.0
        c_ent_final = c_ent
        lr_final = 0.1 * lr
        lr_decay_start_frac = 0.0
        ent_decay_start_frac = 0.0
        max_grad_norm = 0.0
        d_h = 32
        num_layers = 2
        msg_hidden = 32
        value_hidden = 32
        dropout = 0.0

        n_threads = max(1, (os.cpu_count() or 1) - 1)
        torch.set_num_threads(n_threads)
        torch.set_num_interop_threads(1)
        print(f"cpu threads: intra={torch.get_num_threads()} interop=1")
        force_cpu_rollout = True
        cpu_embed_for_gpu_rollout = False

    print(
        f"rollout mode: force_cpu_rollout={force_cpu_rollout} "
        f"cpu_embed_for_gpu_rollout={cpu_embed_for_gpu_rollout}"
    )

    # -----------------------------------------------------------------------
    # Dataset + Env  (sacrifice-specific)
    # -----------------------------------------------------------------------
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Sacrifice dataset not found: {dataset_path}\n"
            "Run:  python gen_sacrifice_dataset.py  to generate it first."
        )

    ds = SacrificeDataset(dataset_path)
    print(f"Loaded sacrifice dataset: {ds.num_samples} samples from {dataset_path}")

    def make_mesh():
        i = np.random.randint(ds.num_samples)
        P, Tbad = ds.get(i)
        return P, Tbad

    envs = SacrificeVecEnv(
        make_mesh_fn=make_mesh,
        num_envs=num_envs,
        max_steps_per_episode=max_steps_per_episode,
        patience=patience,
        reward_scale=reward_scale,
        tet_quality_mode=tet_quality_mode,
        lookback_window=lookback_window,
    )

    # -----------------------------------------------------------------------
    # Model (infer dims from first obs batch — architecture unchanged)
    # -----------------------------------------------------------------------
    obs0 = envs.reset()
    bg0 = batch_from_obs(obs0, device=device)

    d_edge_in = bg0.edge_feat.shape[-1]
    d_face_in = bg0.face_feat.shape[-1]
    critic_extra_dim = bg0.quality_feat.shape[-1]

    model = Mesh3DActorCritic(
        d_edge_in=d_edge_in,
        d_face_in=d_face_in,
        d_h=d_h,
        num_layers=num_layers,
        msg_hidden=msg_hidden,
        value_hidden=value_hidden,
        critic_extra_dim=critic_extra_dim,
        use_coord_update=False,
        dropout=dropout,
    ).to(device)

    if enable_torch_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("torch.compile enabled")
        except Exception as e:
            print(f"torch.compile unavailable: {e}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    # -----------------------------------------------------------------------
    # Output directory — prefixed "sacrifice-" to distinguish from base runs
    # -----------------------------------------------------------------------
    start_dt = datetime.now()
    run_tag = "sacrifice-" + start_dt.strftime("%d-%H-%M-%S")
    out_root = "out"
    run_dir = os.path.join(out_root, run_tag)
    os.makedirs(run_dir, exist_ok=True)
    print("run_dir:", run_dir)

    # -----------------------------------------------------------------------
    # Save config
    # -----------------------------------------------------------------------
    run_config = {
        "start_time": start_dt.isoformat(timespec="seconds"),
        "run_dir": run_dir,
        "task": "sacrifice",
        "dataset": dataset_path,
        "lookback_window": lookback_window,
        "device": str(device),
        "num_envs": num_envs,
        "T_rollout": T_rollout,
        "rounds": rounds,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "K_epochs": K_epochs,
        "minibatch_size": minibatch_size,
        "eps_clip": eps_clip,
        "c_v": c_v,
        "c_ent": c_ent,
        "c_ent_final": c_ent_final,
        "lr_decay": "piecewise-linear",
        "lr_decay_start_frac": lr_decay_start_frac,
        "ent_decay_start_frac": ent_decay_start_frac,
        "max_grad_norm": max_grad_norm,
        "lr": lr,
        "lr_final": lr_final,
        "max_steps_per_episode": max_steps_per_episode,
        "patience": patience,
        "tet_quality_mode": tet_quality_mode,
        "reward_mode": "lookback",
        "use_tqdm_update": use_tqdm_update,
        "profile_timing_update": profile_timing_update,
        "enable_torch_compile": enable_torch_compile,
        "force_cpu_rollout": force_cpu_rollout,
        "cpu_embed_for_gpu_rollout": cpu_embed_for_gpu_rollout,
        "d_h": d_h,
        "num_layers": num_layers,
        "msg_hidden": msg_hidden,
        "value_hidden": value_hidden,
        "critic_extra_dim": critic_extra_dim,
        "reward_scale": reward_scale,
        "critic_quality_features": [
            "q_mean", "q_min", "q_std", "q_worstk_mean_k10",
            "q_softmin_tau0.05", "q_frac_below_0.2",
            "step_frac", "no_improve_frac",
        ],
        "dropout": dropout,
    }
    if device.type == "cpu":
        run_config["torch_num_threads"] = int(torch.get_num_threads())
        run_config["torch_num_interop_threads"] = 1

    with open(os.path.join(run_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    model_summary_path = os.path.join(run_dir, "model_structure.txt")
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    with open(model_summary_path, "w", encoding="utf-8") as f:
        f.write(str(model))
        f.write(f"\n\ntotal_parameters: {num_params}\n")
        f.write(f"trainable_parameters: {num_trainable}\n")

    # -----------------------------------------------------------------------
    # Training loop (identical structure to train.py)
    # -----------------------------------------------------------------------
    avg_reward_list = []
    total_loss_round = []
    actor_loss_round = []
    value_loss_round = []
    entropy_round = []
    avg_reward_unscaled_round = []

    for r in tqdm(range(rounds)):
        print(f"\n============ Round {r} ============")
        if rounds > 1:
            frac = float(r) / float(rounds - 1)
        else:
            frac = 1.0

        if frac <= lr_decay_start_frac:
            lr_now = lr
        else:
            t = (frac - lr_decay_start_frac) / max(1e-8, 1.0 - lr_decay_start_frac)
            lr_now = lr + (lr_final - lr) * t

        if frac <= ent_decay_start_frac:
            c_ent_now = c_ent
        else:
            t_ent = (frac - ent_decay_start_frac) / max(1e-8, 1.0 - ent_decay_start_frac)
            c_ent_now = c_ent + (c_ent_final - c_ent) * t_ent

        for pg in optimizer.param_groups:
            pg["lr"] = lr_now
        print(f"Round lr={lr_now:.6g} c_ent={c_ent_now:.6g}")

        buffers, avg_reward = PolicyRollout(
            envs,
            model,
            T=T_rollout,
            batch_from_obs=batch_from_obs,
            model_action_to_env_with_sizes=model_action_to_env_with_sizes,
            device=device,
            display_every=50,
            force_cpu_rollout=force_cpu_rollout,
            cpu_embed_for_gpu_rollout=cpu_embed_for_gpu_rollout,
        )
        avg_reward_list.append(avg_reward)
        avg_reward_unscaled_round.append(float(avg_reward) / float(reward_scale))

        gae_buffers = {
            "reward_buffer":      buffers["reward_buffer"].to(device),
            "value_buffer":       buffers["value_buffer"].to(device),
            "last_value_buffer":  buffers["last_value_buffer"].to(device),
            "done_buffer":        buffers["done_buffer"].to(device),
        }
        advantages_norm, returns, advantages = compute_gae_from_buffers(
            gae_buffers, gamma=gamma, gae_lambda=gae_lambda
        )

        if device.type == "cuda":
            print("mem allocated (GB):", torch.cuda.memory_allocated() / 1e9,
                  "reserved (GB):", torch.cuda.memory_reserved() / 1e9)

        loss_list, actor_loss_list, value_loss_list, entropy_list = PPO_update(
            model,
            optimizer,
            buffers,
            advantages,
            returns,
            K_epochs=K_epochs,
            minibatch_size=minibatch_size,
            eps_clip=eps_clip,
            c_v=c_v,
            c_ent=c_ent_now,
            max_grad_norm=max_grad_norm,
            device=device,
            use_tqdm=use_tqdm_update,
            profile_timing=profile_timing_update,
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

        total_loss_round.append(float(np.mean(loss_list)))
        actor_loss_round.append(float(np.mean(actor_loss_list)))
        value_loss_round.append(float(np.mean(value_loss_list)))
        entropy_round.append(float(np.mean(entropy_list)))

        print("Current avg reward:", avg_reward)

        should_save_ckpt = ((r + 1) % 10 == 0) or (r == rounds - 1)
        if should_save_ckpt:
            ckpt_path = os.path.join(run_dir, f"model_round{r}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "round": r,
                "avg_reward": avg_reward,
                "run_config": run_config,
            }, ckpt_path)
            print("Saved:", ckpt_path)
            metrics_npz_path, reward_png, loss_png = save_training_artifacts(
                run_dir,
                avg_reward_list,
                avg_reward_unscaled_round,
                total_loss_round,
                actor_loss_round,
                value_loss_round,
                entropy_round,
            )
            print("Saved metrics:", metrics_npz_path)

    # Final artifacts
    metrics_npz_path, reward_png, loss_png = save_training_artifacts(
        run_dir,
        avg_reward_list,
        avg_reward_unscaled_round,
        total_loss_round,
        actor_loss_round,
        value_loss_round,
        entropy_round,
    )
    print("Saved metrics:", metrics_npz_path)
    print("Saved:", reward_png)
    print("Saved:", loss_png)

    final_ckpt = os.path.join(run_dir, "model_final.pt")
    torch.save(model.state_dict(), final_ckpt)
    print(f"Saved final model to {final_ckpt}")


if __name__ == "__main__":
    main()
