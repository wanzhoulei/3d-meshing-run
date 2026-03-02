import argparse
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
from tqdm import tqdm

# IMPORTANT: import the LOCAL topology version explicitly
from tet_mesh_topology_local import TetMeshTopology, tet_signed_volume, tet_mean_ratio_quality
from tet_mesh_topology_local import _tet_faces, _sorted_face, _sorted_edge  # ok for testing internals

from tet_mat73_loader import TetMat73Dataset


# -----------------------------
# Validity checks (local-version compliant)
# -----------------------------

@dataclass
class ValidityReport:
    ok: bool
    msg: str = ""
    bad_tid: Optional[int] = None
    stats: Optional[Dict[str, int]] = None


def check_mesh_validity(
    P: np.ndarray,
    T: np.ndarray,
    eps_vol6: float = 1e-12,
) -> ValidityReport:
    """
    Validity checker compatible with LOCAL topology:
      - assumes T are the active tets (shape (K,4)), but still robust.
      - checks:
          * indices in range
          * 4 distinct vertices
          * no duplicate tets
          * no zero/near-zero signed volume
          * faces appear 1 or 2 times only (manifold w boundary)
    eps_vol6 is threshold on |signed_volume_6| (det) not actual volume.
    """
    P = np.asarray(P, dtype=np.float64)
    T = np.asarray(T)

    if T.ndim != 2 or T.shape[1] != 4:
        return ValidityReport(False, f"T must be (K,4), got {T.shape}")

    if not np.issubdtype(T.dtype, np.integer):
        # allow float-but-integer
        if np.all(np.isfinite(T)) and np.all(np.equal(T, np.round(T))):
            T = T.astype(np.int64)
        else:
            return ValidityReport(False, f"T must be integer indices, got dtype {T.dtype}")

    N = P.shape[0]
    K = T.shape[0]
    if K == 0:
        return ValidityReport(False, "No tets (K=0)")

    tmin = int(T.min())
    tmax = int(T.max())
    if tmin < 0 or tmax >= N:
        return ValidityReport(False, f"Index out of range: min={tmin}, max={tmax}, N={N}")

    # distinct vertices + volume check
    for i in range(K):
        a, b, c, d = map(int, T[i])
        if len({a, b, c, d}) != 4:
            return ValidityReport(False, f"Tet {i} has repeated vertices: {T[i].tolist()}", bad_tid=i)
        s6 = tet_signed_volume(P[a], P[b], P[c], P[d])
        if abs(s6) <= eps_vol6:
            return ValidityReport(False, f"Degenerate tet volume at {i}, s6={s6}", bad_tid=i)

    # duplicate tets (up to permutation)
    keys = np.sort(T.astype(np.int64), axis=1)
    uniq = np.unique(keys, axis=0)
    if uniq.shape[0] != K:
        return ValidityReport(False, f"Duplicate tetrahedra: K={K}, unique={uniq.shape[0]}")

    # face incidence counts
    face_count: Dict[Tuple[int, int, int], int] = {}
    for i in range(K):
        tet = T[i]
        for f in _tet_faces(tet):
            fkey = tuple(sorted(map(int, f)))  # robust: f may be tuple/list
            face_count[fkey] = face_count.get(fkey, 0) + 1
            if face_count[fkey] > 2:
                return ValidityReport(False, f"Non-manifold face {fkey}: count>2")

    stats = {
        "K": int(K),
        "F": int(len(face_count)),
        "boundary_faces": int(sum(1 for c in face_count.values() if c == 1)),
        "interior_faces": int(sum(1 for c in face_count.values() if c == 2)),
    }
    return ValidityReport(True, "ok", stats=stats)


def dump_degenerate_tet(P: np.ndarray, T: np.ndarray, tid: int):
    tet = T[tid].astype(int)
    coords = P[tet]
    s6 = tet_signed_volume(coords[0], coords[1], coords[2], coords[3])
    print("---- Degenerate tet dump ----")
    print("tid:", tid)
    print("tet indices:", tet.tolist())
    print("coords:\n", coords)
    print("signed volume6:", s6)
    print("----------------------------")


# -----------------------------
# Random rollout
# -----------------------------

def random_flip_rollout_local(
    P: np.ndarray,
    T0: np.ndarray,
    steps: int,
    rng: random.Random,
    eps_vol6: float = 1e-12,
    verbose_fail: bool = True,
) -> None:
    top = TetMeshTopology(P, T0)

    rep = check_mesh_validity(top.points, top.tets, eps_vol6=eps_vol6)
    assert rep.ok, f"Initial mesh invalid: {rep.msg}"

    # sanity: quality finite
    q = tet_mean_ratio_quality(top.points, top.tets)
    assert np.all(np.isfinite(q))

    last_action = None
    last_kind = None  # "2-3" or "3-2"

    for s in range(steps):
        mask = top.candidate_mask()
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            return

        a = int(rng.choice(valid.tolist()))
        last_action = a

        # determine kind for better debugging
        F = len(top.faces_list) if hasattr(top, "faces_list") else top.faces.shape[0]
        last_kind = "2-3" if a < F else "3-2"

        ok = top.apply_action(a)
        assert ok, f"apply_action returned False for candidate action {a}"

        rep = check_mesh_validity(top.points, top.tets, eps_vol6=eps_vol6)
        if not rep.ok:
            if verbose_fail:
                print(f"\nMesh invalid at step {s+1} after action {last_action} ({last_kind}): {rep.msg}")
                if rep.bad_tid is not None:
                    dump_degenerate_tet(top.points, top.tets, rep.bad_tid)
            raise AssertionError(f"Mesh invalid at step {s+1}: {rep.msg}")


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", type=str, default="tet_dataset_grid125_sigma1e-02_N2000.mat")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--max-samples", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps-vol6", type=float, default=1e-12,
                    help="Degeneracy threshold on |signed_volume_6| (det).")
    ap.add_argument("--no-verbose-fail", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    ds = TetMat73Dataset(args.mat, load_all=True)
    print("Points coordinates shape: ", ds.P.shape)
    print("Number of data points: ", len(ds.T_bad))
    print("Each topology shape: ", ds.T_bad[0].shape)
    print("minQ_bad shape: ", ds.minQ_bad.shape)
    print("minQ_good shape: ", ds.minQ_good.shape)

    total = len(ds.T_bad) if args.max_samples < 0 else min(args.max_samples, len(ds.T_bad))
    print(f"Testing {total} samples with {args.steps} random flips each ...")

    for i in tqdm(range(total)):
        P = ds.P[i]
        T0 = ds.T_bad[i]
        # coerce to int
        if not np.issubdtype(T0.dtype, np.integer):
            T0 = np.asarray(T0, dtype=np.int64)

        try:
            random_flip_rollout_local(
                P, T0,
                steps=args.steps,
                rng=rng,
                eps_vol6=args.eps_vol6,
                verbose_fail=(not args.no_verbose_fail),
            )
        except AssertionError as e:
            print(f"\nFAILED on sample {i}: {e}")
            print("Dumping minimal context:")
            print("  P shape:", P.shape, "T0 shape:", T0.shape, "dtype:", T0.dtype)
            raise

    ds.close()
    print("All tests passed.")


if __name__ == "__main__":
    main()
