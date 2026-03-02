import argparse
import os
import re
import glob
import numpy as np
import torch
from tqdm.auto import tqdm

from tet_mat73_loader import TetMat73Dataset
from tet_mesh_topology_local import TetMeshTopology
from tet_env import softmin_score, worstk_mean_score

from model_face_edge_gpt import Mesh3DActorCritic
from initial_embedding import batch_from_obs, model_action_to_env_with_sizes


def find_latest_checkpoint(out_dir: str = "out") -> str:
    ckpts = glob.glob(os.path.join(out_dir, "model_round*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {out_dir}/model_round*.pt")

    def rid(p: str) -> int:
        m = re.search(r"model_round(\d+)\.pt", os.path.basename(p))
        return int(m.group(1)) if m else -1

    ckpts.sort(key=rid)
    return ckpts[-1]


def obs_from_topo(topo: TetMeshTopology) -> dict:
    return dict(
        points=topo.points,
        tets=topo.tets,
        faces=topo.faces,
        face2tet=topo.face2tet,
        edges=topo.edges,
        candidate_mask=topo.candidate_mask(),
        tet_quality=topo.tet_quality,
    )


def build_model_from_ckpt(ckpt_path: str, device: torch.device, example_obs: dict) -> Mesh3DActorCritic:
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    bg0 = batch_from_obs([example_obs], device=device)
    d_edge_in = bg0.edge_feat.shape[-1]
    d_face_in = bg0.face_feat.shape[-1]

    # Infer architecture from checkpoint to avoid train/eval config mismatch.
    d_h = int(state["edge_embed.net.0.weight"].shape[0])
    msg_hidden = int(state["backbone.layers.0.phi_m.net.0.weight"].shape[0])
    value_hidden = int(state["edge_pool_proj.net.0.weight"].shape[0])

    layer_ids = []
    for k in state.keys():
        m = re.match(r"backbone\.layers\.(\d+)\.", k)
        if m:
            layer_ids.append(int(m.group(1)))
    if not layer_ids:
        raise RuntimeError("Could not infer backbone num_layers from checkpoint.")
    num_layers = max(layer_ids) + 1

    # MLP index layout differs when dropout is enabled.
    dropout = 0.0 if "edge_embed.net.2.weight" in state else 0.2

    model = Mesh3DActorCritic(
        d_edge_in=d_edge_in,
        d_face_in=d_face_in,
        d_h=d_h,
        num_layers=num_layers,
        msg_hidden=msg_hidden,
        value_hidden=value_hidden,
        use_coord_update=False,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.inference_mode()
def run_agent_until_stall(
    P: np.ndarray,
    T_init: np.ndarray,
    model: Mesh3DActorCritic,
    device: torch.device,
    *,
    score_mode: str = "softmin",
    softmin_tau: float = 0.05,
    worstk_k: int = 10,
    eps_improve: float = 1e-6,
    patience_eval: int = 50,
    max_steps: int = 100,
    greedy: bool = False,
):
    topo = TetMeshTopology(P, T_init.copy())

    def score(t):
        q = t.tet_quality
        if score_mode == "softmin":
            return softmin_score(q, tau=softmin_tau)
        if score_mode == "worstk":
            return worstk_mean_score(q, k=worstk_k)
        raise ValueError(score_mode)

    best = score(topo)
    no_improve = 0
    steps = 0
    history = [best]

    while steps < max_steps:
        cand = topo.candidate_mask()
        if not np.any(cand):
            break

        obs = obs_from_topo(topo)
        bg = batch_from_obs([obs], device=device)

        policy_out, _, _, _ = model(
            x=bg.x,
            edge_feat=bg.edge_feat,
            face_feat=bg.face_feat,
            edge_index=bg.edge_index,
            node_mask=bg.node_mask,
            edge_mask=bg.edge_mask,
            edge_node_mask=bg.edge_node_mask,
            face_node_mask=bg.face_node_mask,
            edge_action_mask=bg.edge_action_mask,
            face_action_mask=bg.face_action_mask,
        )
        logits = policy_out.logits[0]  # MODEL ordering: [edges, faces]

        E = obs["edges"].shape[1]
        F = obs["faces"].shape[0]
        cand_face = cand[:F].astype(bool)    # ENV ordering: [faces, edges]
        cand_edge = cand[F:F + E].astype(bool)

        valid_model = np.concatenate([
            np.flatnonzero(cand_edge),      # [0..E-1]
            E + np.flatnonzero(cand_face),  # [E..E+F-1]
        ])
        if valid_model.size == 0:
            break

        valid_model_t = torch.as_tensor(valid_model, device=device, dtype=torch.long)
        if greedy:
            a_model = valid_model[int(torch.argmax(logits[valid_model_t]).item())]
        else:
            sub_logits = logits[valid_model_t]
            dist = torch.distributions.Categorical(logits=sub_logits)
            a_model = valid_model[int(dist.sample().item())]

        a_env = model_action_to_env_with_sizes(
            np.array([a_model], dtype=np.int64),
            np.array([E], dtype=np.int64),
            np.array([F], dtype=np.int64),
        )[0]

        ok = topo.apply_action(int(a_env))
        s_after = score(topo)
        if ok and (s_after - best) >= eps_improve:
            best = s_after
            no_improve = 0
        else:
            no_improve += 1

        history.append(s_after)
        steps += 1
        if no_improve >= patience_eval:
            break

    return topo, np.array(history, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained tet-mesh RL model checkpoint.")
    parser.add_argument("model_path", type=str, nargs="?", default=None, help="Path to model checkpoint (.pt).")
    parser.add_argument("--dataset-path", type=str, default="tet_dataset_grid125_sigma1e-02_N2000.mat")
    parser.add_argument("--out-dir", type=str, default="out")
    parser.add_argument("--num-test", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--score-mode", type=str, default="softmin", choices=["softmin", "worstk"])
    parser.add_argument("--softmin-tau", type=float, default=0.05)
    parser.add_argument("--worstk-k", type=int, default=10)
    parser.add_argument("--patience-eval", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--greedy", action="store_true", default=False, help="Use greedy action selection.")
    args = parser.parse_args()

    ckpt_path = args.model_path if args.model_path is not None else find_latest_checkpoint(args.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("Using checkpoint:", ckpt_path)

    ds = TetMat73Dataset(args.dataset_path, load_all=False)
    print("Number of Samples in this Dataset:", ds.num_samples)
    print("T_good shape:", ds.T_good.shape)

    rng = np.random.default_rng(args.seed)
    test_ids = rng.integers(0, ds.num_samples, size=args.num_test)

    P0, Tbad0 = ds.get(int(test_ids[0]))
    topo0 = TetMeshTopology(P0, Tbad0)
    model = build_model_from_ckpt(ckpt_path, device, obs_from_topo(topo0))

    def metric(q: np.ndarray) -> float:
        if args.score_mode == "softmin":
            return float(softmin_score(q, tau=args.softmin_tau))
        return float(worstk_mean_score(q, k=args.worstk_k))

    results = []
    for i in tqdm(test_ids, desc="Testing"):
        i = int(i)
        P, Tbad = ds.get(i)

        topo_bad = TetMeshTopology(P, Tbad)
        q_bad = topo_bad.tet_quality.copy()

        topo_good = TetMeshTopology(P, ds.T_good)
        q_good = topo_good.tet_quality.copy()

        topo_agent, score_hist = run_agent_until_stall(
            P,
            Tbad,
            model,
            device,
            score_mode=args.score_mode,
            softmin_tau=args.softmin_tau,
            worstk_k=args.worstk_k,
            patience_eval=args.patience_eval,
            max_steps=args.max_steps,
            greedy=args.greedy,
        )
        q_agent = topo_agent.tet_quality.copy()

        results.append(
            dict(
                idx=i,
                min_bad=float(q_bad.min()),
                min_agent=float(q_agent.min()),
                min_good=float(q_good.min()),
                score_bad=metric(q_bad),
                score_agent=metric(q_agent),
                score_good=metric(q_good),
                steps=len(score_hist) - 1,
            )
        )

    ds.close()

    print("\n==== Summary ====")
    print("avg steps:", np.mean([r["steps"] for r in results]))
    print("avg minQ bad   :", np.mean([r["min_bad"] for r in results]))
    print("avg minQ agent :", np.mean([r["min_agent"] for r in results]))
    print("avg minQ good  :", np.mean([r["min_good"] for r in results]))
    print("avg score bad  :", np.mean([r["score_bad"] for r in results]))
    print("avg score agent:", np.mean([r["score_agent"] for r in results]))
    print("avg score good :", np.mean([r["score_good"] for r in results]))


if __name__ == "__main__":
    main()
