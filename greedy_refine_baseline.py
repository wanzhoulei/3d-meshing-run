from __future__ import annotations

import argparse
import copy
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm.auto import tqdm

from tet_env import softmin_score, worstk_mean_score
from tet_mat73_loader import TetMat73Dataset
from tet_mesh_topology_local import TetMeshTopology


def score_from_quality(
    q: np.ndarray,
    score_mode: str = "softmin",
    softmin_tau: float = 0.05,
    worstk_k: int = 10,
) -> float:
    if score_mode == "softmin":
        return float(softmin_score(q, tau=softmin_tau))
    if score_mode == "worstk":
        return float(worstk_mean_score(q, k=worstk_k))
    raise ValueError(f"Unknown score_mode={score_mode}")


def resolve_quality_mode_from_run_dir(run_dir: str, default_mode: str = "mean_ratio") -> str:
    cfg_path = os.path.join(os.path.abspath(run_dir), "train_config.json")
    mode = str(default_mode)
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            mode = str(cfg.get("tet_quality_mode", default_mode))
        except Exception:
            mode = str(default_mode)
    if mode not in ("mean_ratio", "simpqual1", "simpqual2"):
        mode = str(default_mode)
    return mode


def _sorted_face(i: int, j: int, k: int) -> Tuple[int, int, int]:
    a, b, c = int(i), int(j), int(k)
    if a > b:
        a, b = b, a
    if b > c:
        b, c = c, b
    if a > b:
        a, b = b, a
    return (a, b, c)


def _sorted_edge(i: int, j: int) -> Tuple[int, int]:
    a, b = int(i), int(j)
    return (a, b) if a < b else (b, a)


def _tet_faces(t: Sequence[int]) -> List[Tuple[int, int, int]]:
    i, j, k, l = map(int, t)
    return [
        _sorted_face(j, k, l),
        _sorted_face(i, k, l),
        _sorted_face(i, j, l),
        _sorted_face(i, j, k),
    ]


def _tet_edges(t: Sequence[int]) -> List[Tuple[int, int]]:
    i, j, k, l = map(int, t)
    return [
        _sorted_edge(i, j),
        _sorted_edge(i, k),
        _sorted_edge(i, l),
        _sorted_edge(j, k),
        _sorted_edge(j, l),
        _sorted_edge(k, l),
    ]


def _actions_for_tet(topo: TetMeshTopology, tid: int) -> List[int]:
    if tid < 0 or tid >= int(topo.tets.shape[0]):
        return []
    tet = topo.tets[int(tid)]

    fids = topo._active_face_ids()
    eids = topo._active_edge_ids()
    F = int(len(fids))

    fid_to_env: Dict[int, int] = {int(fid): i for i, fid in enumerate(fids.tolist())}
    eid_to_env: Dict[int, int] = {int(eid): F + i for i, eid in enumerate(eids.tolist())}

    actions: List[int] = []

    # Faces -> 2-3 candidate actions.
    for fkey in _tet_faces(tet):
        fid = topo._face_id.get(fkey)
        if fid is None:
            continue
        fid = int(fid)
        if fid >= len(topo.candidate_face_mask):
            continue
        if not bool(topo.candidate_face_mask[fid]):
            continue
        a_env = fid_to_env.get(fid)
        if a_env is not None:
            actions.append(int(a_env))

    # Edges -> 3-2 candidate actions.
    for ekey in _tet_edges(tet):
        eid = topo._edge_id.get(ekey)
        if eid is None:
            continue
        eid = int(eid)
        if eid >= len(topo.candidate_edge_mask):
            continue
        if not bool(topo.candidate_edge_mask[eid]):
            continue
        a_env = eid_to_env.get(eid)
        if a_env is not None:
            actions.append(int(a_env))

    # Deduplicate while preserving order.
    seen = set()
    uniq: List[int] = []
    for a in actions:
        if a in seen:
            continue
        seen.add(a)
        uniq.append(a)

    return uniq


def _pick_target_tet_and_actions(topo: TetMeshTopology) -> Tuple[int, List[int]]:
    """
    Scan all tets from worst quality upward and pick the first tet
    that has at least one valid local action.
    """
    if topo.tets.shape[0] == 0:
        return -1, []

    order = np.argsort(topo.tet_quality)  # ascending (worst first)
    for tid in order.tolist():
        actions = _actions_for_tet(topo, int(tid))
        if len(actions) > 0:
            return int(tid), actions
    return int(order[0]), []


def _decode_action(topo: TetMeshTopology, a_env: int) -> Tuple[str, int]:
    fids = topo._active_face_ids()
    F = int(len(fids))
    a_env = int(a_env)
    if a_env < F:
        return "face (2-3)", a_env
    return "edge (3-2)", int(a_env - F)


def run_greedy_trace_episode(
    P: np.ndarray,
    T_init: np.ndarray,
    *,
    tet_quality_mode: str = "mean_ratio",
    score_mode: str = "softmin",
    softmin_tau: float = 0.05,
    worstk_k: int = 10,
    eps_improve: float = 1e-6,
    patience_eval: int = 80,
    max_steps: int = 800,
    fallback_final_to_best: bool = True,
) -> Dict[str, object]:
    topo = TetMeshTopology(P, T_init.copy(), tet_quality_mode=tet_quality_mode)

    bad_q = topo.tet_quality.copy()
    cur = score_from_quality(topo.tet_quality, score_mode, softmin_tau, worstk_k)
    best = cur
    best_q = topo.tet_quality.copy()
    best_tets = topo.tets.copy()
    best_step = -1

    no_improve = 0
    steps = 0
    action_log: List[Dict[str, object]] = []

    while steps < int(max_steps):
        cand = topo.candidate_mask()
        if not np.any(cand):
            break

        worst_tid, actions = _pick_target_tet_and_actions(topo)
        if len(actions) == 0:
            no_improve += 1
            steps += 1
            if no_improve >= int(patience_eval):
                break
            continue

        best_a: Optional[int] = None
        best_delta = -np.inf

        for a_env in actions:
            # Keep action-index mapping identical to current state.
            # Rebuilding from tets changes internal face/edge IDs, which breaks a_env semantics.
            trial = copy.deepcopy(topo)
            ok = trial.apply_action(int(a_env))
            if not ok:
                continue
            s_new = score_from_quality(trial.tet_quality, score_mode, softmin_tau, worstk_k)
            delta = float(s_new - cur)
            if delta > best_delta:
                best_delta = delta
                best_a = int(a_env)

        if best_a is None:
            no_improve += 1
            steps += 1
            if no_improve >= int(patience_eval):
                break
            continue

        op_type, op_local_index = _decode_action(topo, best_a)
        prev = cur
        ok = topo.apply_action(best_a)
        cur = score_from_quality(topo.tet_quality, score_mode, softmin_tau, worstk_k)
        delta = float(cur - prev)

        action_log.append(
            dict(
                step=int(steps),
                op_type=op_type,
                op_local_index=int(op_local_index),
                env_action=int(best_a),
                valid=bool(ok),
                delta=delta,
                worst_tid=int(worst_tid),
            )
        )

        if ok and (cur - best) >= float(eps_improve):
            best = cur
            best_q = topo.tet_quality.copy()
            best_tets = topo.tets.copy()
            best_step = steps
            no_improve = 0
        else:
            no_improve += 1

        steps += 1
        if no_improve >= int(patience_eval):
            break

    if fallback_final_to_best:
        final_topo = TetMeshTopology(P, best_tets, tet_quality_mode=tet_quality_mode)
        final_q = final_topo.tet_quality.copy()
        final_score = float(best)
    else:
        final_topo = topo
        final_q = topo.tet_quality.copy()
        final_score = float(cur)

    return dict(
        action_log=action_log,
        bad_q=bad_q,
        best_q=best_q,
        final_q=final_q,
        final_tets=final_topo.tets.copy(),
        best_score=float(best),
        final_score=float(final_score),
        steps=int(steps),
        best_step=int(best_step),
        fallback_final_to_best=bool(fallback_final_to_best),
    )


def evaluate_greedy_on_dataset(
    ds: TetMat73Dataset,
    episode_ids: np.ndarray,
    T_good: np.ndarray,
    *,
    tet_quality_mode: str = "mean_ratio",
    score_mode: str = "softmin",
    softmin_tau: float = 0.05,
    worstk_k: int = 10,
    eps_improve: float = 1e-6,
    patience_eval: int = 80,
    max_steps: int = 800,
) -> Dict[str, object]:
    rows: List[Tuple[float, float, float, float, float]] = []
    step_reward_trajs: List[List[float]] = []

    for eid in tqdm(episode_ids, desc="Greedy baseline eval", leave=False):
        P, Tbad = ds.get(int(eid))

        topo_bad = TetMeshTopology(P, Tbad, tet_quality_mode=tet_quality_mode)
        score_bad = score_from_quality(topo_bad.tet_quality, score_mode, softmin_tau, worstk_k)

        topo_good = TetMeshTopology(P, T_good, tet_quality_mode=tet_quality_mode)
        score_good = score_from_quality(topo_good.tet_quality, score_mode, softmin_tau, worstk_k)

        tr = run_greedy_trace_episode(
            P,
            Tbad,
            tet_quality_mode=tet_quality_mode,
            score_mode=score_mode,
            softmin_tau=softmin_tau,
            worstk_k=worstk_k,
            eps_improve=eps_improve,
            patience_eval=patience_eval,
            max_steps=max_steps,
            fallback_final_to_best=True,
        )

        deltas = [float(a["delta"]) for a in tr["action_log"]]
        rows.append((score_bad, float(tr["final_score"]), float(tr["best_score"]), score_good, float(tr["steps"])))
        step_reward_trajs.append(deltas)

    arr = np.asarray(rows, dtype=np.float64)
    max_len = max((len(tr) for tr in step_reward_trajs), default=0)
    step_reward_mean = np.full(max_len, np.nan, dtype=np.float64)
    step_reward_count = np.zeros(max_len, dtype=np.int64)
    for t in range(max_len):
        vals = [tr[t] for tr in step_reward_trajs if t < len(tr)]
        if vals:
            step_reward_mean[t] = float(np.mean(vals))
            step_reward_count[t] = int(len(vals))

    return dict(
        n=len(rows),
        bad_mean=float(arr[:, 0].mean()) if arr.size else np.nan,
        final_mean=float(arr[:, 1].mean()) if arr.size else np.nan,
        best_mean=float(arr[:, 2].mean()) if arr.size else np.nan,
        good_mean=float(arr[:, 3].mean()) if arr.size else np.nan,
        improve_final=float((arr[:, 1] - arr[:, 0]).mean()) if arr.size else np.nan,
        improve_best=float((arr[:, 2] - arr[:, 0]).mean()) if arr.size else np.nan,
        gap_to_good=float((arr[:, 2] - arr[:, 3]).mean()) if arr.size else np.nan,
        steps_mean=float(arr[:, 4].mean()) if arr.size else np.nan,
        step_reward_mean=step_reward_mean,
        step_reward_count=step_reward_count,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Greedy worst-tet baseline for tet refinement.")
    parser.add_argument("--dataset-path", type=str, default="tet_dataset_grid125_sigma1e-02_N2000_test.mat")
    parser.add_argument("--run-dir", type=str, default="out/03-22-16-26")
    parser.add_argument("--tet-quality-mode", type=str, default="", choices=["", "mean_ratio", "simpqual1", "simpqual2"])
    parser.add_argument("--score-mode", type=str, default="softmin", choices=["softmin", "worstk"])
    parser.add_argument("--softmin-tau", type=float, default=0.05)
    parser.add_argument("--worstk-k", type=int, default=10)
    parser.add_argument("--patience-eval", type=int, default=80)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--eps-improve", type=float, default=1e-6)
    parser.add_argument("--num-test", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    mode = args.tet_quality_mode
    if mode == "":
        mode = resolve_quality_mode_from_run_dir(args.run_dir, default_mode="mean_ratio")

    print("dataset:", args.dataset_path)
    print("run_dir:", args.run_dir)
    print("tet_quality_mode:", mode)

    ds = TetMat73Dataset(args.dataset_path, load_all=False)
    rng = np.random.default_rng(args.seed)
    replace = args.num_test > ds.num_samples
    ids = rng.choice(ds.num_samples, size=args.num_test, replace=replace)

    res = evaluate_greedy_on_dataset(
        ds,
        np.asarray(ids, dtype=np.int64),
        ds.T_good,
        tet_quality_mode=mode,
        score_mode=args.score_mode,
        softmin_tau=args.softmin_tau,
        worstk_k=args.worstk_k,
        eps_improve=args.eps_improve,
        patience_eval=args.patience_eval,
        max_steps=args.max_steps,
    )

    print()
    print("bad_mean  final_mean best_mean  good_mean  improve_best  gap_to_good  steps_mean")
    print(
        f"{res['bad_mean']:+.6f}  "
        f"{res['final_mean']:+.6f}  "
        f"{res['best_mean']:+.6f}  "
        f"{res['good_mean']:+.6f}  "
        f"{res['improve_best']:+.6f}  "
        f"{res['gap_to_good']:+.6f}  "
        f"{res['steps_mean']:.2f}"
    )

    ds.close()


if __name__ == "__main__":
    main()
