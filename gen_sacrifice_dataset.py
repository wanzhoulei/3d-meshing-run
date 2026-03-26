"""
Generate the sacrifice-moves tet mesh dataset.

Algorithm per sample
--------------------
1. Load (P, T_good) from the source grid-125 .mat dataset.
2. Apply k random valid flips to T_good -> T_bad (random walk, no quality
   guidance).  This creates topologies that may sit in local quality optima
   from which greedy refinement cannot escape.
3. Run a quick greedy refinement from T_bad.
4. Keep the sample only if:
       greedy_best_score  <  T_good_score - threshold
   i.e. greedy gets stuck strictly below T_good quality.
5. Repeat until `target_n` samples are collected (up to
   `target_n * max_attempts_factor` total tries).

Output: a .npz file compatible with sacrifice_dataset.SacrificeDataset.

Usage
-----
    python gen_sacrifice_dataset.py \
        --source tet_dataset_grid125_sigma1e-02_N2000.mat \
        --output tet_dataset_sacrifice_k20_N2000.npz \
        --k 20 --target-n 2000 --threshold 0.05 --seed 42
"""
from __future__ import annotations

import argparse
import copy

import numpy as np
from tqdm.auto import tqdm

from tet_mat73_loader import TetMat73Dataset
from tet_mesh_topology_local import TetMeshTopology
from tet_env import softmin_score
from greedy_refine_baseline import run_greedy_trace_episode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_walk(
    P: np.ndarray,
    T_start: np.ndarray,
    k: int,
    rng: np.random.Generator,
    tet_quality_mode: str,
) -> tuple[np.ndarray, int]:
    """
    Apply up to k random valid flips starting from T_start (P fixed).

    Returns
    -------
    T_result : (K, 4) int32  — topology after the walk
    flips_done : int          — actual number of successful flips applied
    """
    topo = TetMeshTopology(P, T_start.copy(), tet_quality_mode=tet_quality_mode)
    flips_done = 0
    max_tries = k * 5  # allow retries for invalid samples

    for _ in range(max_tries):
        if flips_done >= k:
            break
        cand = topo.candidate_mask()
        valid_actions = np.where(cand)[0]
        if len(valid_actions) == 0:
            break  # no flips possible
        a = int(rng.choice(valid_actions))
        ok = topo.apply_action(a)
        if ok:
            flips_done += 1

    return topo.tets.copy(), flips_done


def _tgood_softmin_score(
    P: np.ndarray,
    T_good: np.ndarray,
    tet_quality_mode: str,
) -> float:
    topo = TetMeshTopology(P, T_good.copy(), tet_quality_mode=tet_quality_mode)
    return float(softmin_score(topo.tet_quality))


def _greedy_best_score(
    P: np.ndarray,
    T_bad: np.ndarray,
    tet_quality_mode: str,
    patience: int,
    max_steps: int,
) -> float:
    result = run_greedy_trace_episode(
        P,
        T_bad,
        tet_quality_mode=tet_quality_mode,
        score_mode="softmin",
        softmin_tau=0.05,
        patience_eval=patience,
        max_steps=max_steps,
        fallback_final_to_best=True,
    )
    return float(result["best_score"])


def _passes_filter(
    P: np.ndarray,
    T_bad: np.ndarray,
    T_good: np.ndarray,
    tet_quality_mode: str,
    threshold: float,
    greedy_patience: int,
    greedy_max_steps: int,
) -> tuple[bool, float, float]:
    """
    Returns (keep, greedy_score, tgood_score).
    keep is True when greedy_score < tgood_score - threshold.
    """
    tgood_sc = _tgood_softmin_score(P, T_good, tet_quality_mode)
    greedy_sc = _greedy_best_score(P, T_bad, tet_quality_mode, greedy_patience, greedy_max_steps)
    keep = greedy_sc < tgood_sc - threshold
    return keep, greedy_sc, tgood_sc


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_sacrifice_dataset(
    source_path: str,
    output_path: str,
    k: int = 20,
    target_n: int = 2000,
    threshold: float = 0.05,
    max_attempts_factor: int = 10,
    tet_quality_mode: str = "mean_ratio",
    greedy_patience: int = 30,
    greedy_max_steps: int = 150,
    seed: int = 42,
) -> None:
    """
    Generate and save the sacrifice-moves dataset.

    Parameters
    ----------
    source_path : path to the source .mat dataset (TetMat73Dataset format)
    output_path : path for the output .npz file
    k           : number of random flips applied to T_good per sample
    target_n    : desired number of hard samples in the output dataset
    threshold   : greedy must be at least this far below T_good to be kept
    max_attempts_factor : cap total attempts at target_n * this factor
    tet_quality_mode    : per-tet quality metric
    greedy_patience     : patience for the quick greedy filter run
    greedy_max_steps    : max steps for the quick greedy filter run
    seed        : random seed
    """
    rng = np.random.default_rng(seed)
    ds = TetMat73Dataset(source_path, load_all=False)
    T_good = ds.T_good  # (K0, 4) int32, shared across all samples

    # Collectors
    P_list: list[np.ndarray] = []
    T_bad_list: list[np.ndarray] = []
    greedy_scores: list[float] = []
    tgood_scores: list[float] = []

    max_attempts = target_n * max_attempts_factor
    kept = 0
    tried = 0

    pbar = tqdm(total=target_n, desc="Generating sacrifice samples")

    while kept < target_n and tried < max_attempts:
        tried += 1
        i = int(rng.integers(ds.num_samples))
        P, _ = ds.get(i)  # use P from dataset, ignore T_bad (start from T_good)

        # Random walk from T_good
        T_bad, flips_done = _random_walk(P, T_good, k=k, rng=rng,
                                          tet_quality_mode=tet_quality_mode)

        if flips_done == 0:
            # No flips possible; skip
            continue

        # Filter: keep only if greedy gets stuck below T_good quality
        keep, greedy_sc, tgood_sc = _passes_filter(
            P, T_bad, T_good,
            tet_quality_mode=tet_quality_mode,
            threshold=threshold,
            greedy_patience=greedy_patience,
            greedy_max_steps=greedy_max_steps,
        )

        if not keep:
            continue

        P_list.append(P.copy())
        T_bad_list.append(T_bad)
        greedy_scores.append(greedy_sc)
        tgood_scores.append(tgood_sc)
        kept += 1
        pbar.update(1)
        pbar.set_postfix(tried=tried, kept=kept, ratio=f"{kept/tried:.2%}")

    pbar.close()
    ds.close()

    if kept < target_n:
        print(
            f"Warning: only {kept}/{target_n} hard samples found after "
            f"{tried} attempts. Consider lowering --threshold or raising --k."
        )

    # Pack into numpy arrays
    P_arr = np.stack(P_list, axis=0)                   # (kept, V, 3)
    T_bad_flat = np.concatenate(T_bad_list, axis=0)    # (M, 4)
    T_bad_sizes = np.array([t.shape[0] for t in T_bad_list], dtype=np.int64)
    greedy_score_arr = np.array(greedy_scores, dtype=np.float64)
    tgood_score_arr = np.array(tgood_scores, dtype=np.float64)

    np.savez(
        output_path,
        P=P_arr,
        T_good=T_good,
        T_bad_flat=T_bad_flat,
        T_bad_sizes=T_bad_sizes,
        greedy_score=greedy_score_arr,
        tgood_score=tgood_score_arr,
    )
    print(f"\nSaved {kept} samples to {output_path}")
    print(f"  P shape       : {P_arr.shape}")
    print(f"  T_good shape  : {T_good.shape}")
    print(f"  T_bad_flat    : {T_bad_flat.shape}  (avg {T_bad_flat.shape[0]/max(kept,1):.1f} tets/sample)")
    print(f"  greedy score  : mean={greedy_score_arr.mean():.4f}  min={greedy_score_arr.min():.4f}")
    print(f"  T_good score  : mean={tgood_score_arr.mean():.4f}  min={tgood_score_arr.min():.4f}")
    print(f"  avg gap       : {(tgood_score_arr - greedy_score_arr).mean():.4f}")
    print(f"  attempts total: {tried}  (keep rate {kept/max(tried,1):.2%})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate sacrifice-moves tet mesh dataset."
    )
    parser.add_argument(
        "--source", type=str,
        default="tet_dataset_grid125_sigma1e-02_N2000.mat",
        help="Source .mat dataset (TetMat73Dataset format)",
    )
    parser.add_argument(
        "--output", type=str,
        default="tet_dataset_sacrifice_k20_N2000.npz",
        help="Output .npz file path",
    )
    parser.add_argument("--k", type=int, default=20,
                        help="Number of random flips applied to T_good (default: 20)")
    parser.add_argument("--target-n", type=int, default=2000,
                        help="Number of hard samples to collect (default: 2000)")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Min gap (T_good_score - greedy_score) to keep sample (default: 0.05)")
    parser.add_argument("--max-attempts-factor", type=int, default=10,
                        help="Cap total attempts at target_n * this factor (default: 10)")
    parser.add_argument("--tet-quality-mode", type=str, default="mean_ratio",
                        choices=["mean_ratio", "simpqual1", "simpqual2"])
    parser.add_argument("--greedy-patience", type=int, default=30,
                        help="Patience for quick greedy filter run (default: 30)")
    parser.add_argument("--greedy-max-steps", type=int, default=150,
                        help="Max steps for quick greedy filter run (default: 150)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Source dataset  :", args.source)
    print("Output path     :", args.output)
    print(f"k={args.k}  target_n={args.target_n}  threshold={args.threshold}")
    print(f"greedy filter   : patience={args.greedy_patience}  max_steps={args.greedy_max_steps}")
    print(f"tet_quality_mode: {args.tet_quality_mode}")
    print(f"seed            : {args.seed}")
    print()

    generate_sacrifice_dataset(
        source_path=args.source,
        output_path=args.output,
        k=args.k,
        target_n=args.target_n,
        threshold=args.threshold,
        max_attempts_factor=args.max_attempts_factor,
        tet_quality_mode=args.tet_quality_mode,
        greedy_patience=args.greedy_patience,
        greedy_max_steps=args.greedy_max_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
