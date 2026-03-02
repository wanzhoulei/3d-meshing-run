# test_tet_mesh_topology.py
"""
Extensive randomized tests for tet_mesh_topology.py.

Loads dataset via:
  ds = TetMat73Dataset("tet_dataset_grid125_sigma1e-02_N2000.mat", load_all=True)

Then for each sample:
  - build TetMeshTopology(P, T_bad[i])
  - check mesh validity (combinatorial + geometric)
  - do random valid flips for >=200 steps
  - re-check validity and candidate mask consistency after flips

Run:
  python test_tet_mesh_topology.py --mat tet_dataset_grid125_sigma1e-02_N2000.mat --steps 200
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from tqdm.auto import tqdm

from tet_mesh_topology import (
    TetMeshTopology,
    tet_signed_volume,
    tet_mean_ratio_quality,
    _sorted_edge,
    _sorted_face,
    _tet_faces,
)

from tet_mat73_loader import TetMat73Dataset


# -----------------------------
# Helpers
# -----------------------------

def _as_int_tets(T: np.ndarray) -> np.ndarray:
    """Coerce float-but-integer tets into int64."""
    T = np.asarray(T)
    if np.issubdtype(T.dtype, np.integer):
        return T.astype(np.int64, copy=False)
    if np.all(np.isfinite(T)) and np.all(np.equal(T, np.round(T))):
        return T.astype(np.int64)
    raise TypeError(f"T contains non-integer values; dtype={T.dtype}")


def _sorted_tet(t: np.ndarray) -> Tuple[int, int, int, int]:
    a, b, c, d = map(int, t)
    x = sorted((a, b, c, d))
    return (x[0], x[1], x[2], x[3])


def _face_key(face) -> Tuple[int, int, int]:
    """
    Convert a face returned by _tet_faces into a sorted (a,b,c) int tuple.

    _tet_faces may return tuples, lists, or np arrays. We normalize robustly.
    """
    a, b, c = map(int, face)  # works for tuple/list/np array
    x = sorted((a, b, c))
    return (x[0], x[1], x[2])


# -----------------------------
# Validity checks
# -----------------------------

@dataclass
class ValidityReport:
    ok: bool
    msg: str = ""
    stats: Dict[str, int] | None = None


def check_mesh_validity(
    P: np.ndarray,
    T: np.ndarray,
    eps_vol: float = 1e-14,
) -> ValidityReport:
    """
    Check if (P, T) is a valid tetrahedral simplicial complex (manifold with boundary)
    at the level required by TetMeshTopology local flips.

    Conditions enforced:
      1) T is (K,4), indices in [0, N-1], each tet has 4 distinct vertices.
      2) No duplicate tetrahedra (up to permutation).
      3) Each tetra has non-zero volume (|signed volume| > eps_vol).
      4) Face incidence count is 1 (boundary) or 2 (interior), never >2.
    """
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 3:
        return ValidityReport(False, f"P must be (N,3), got {P.shape}")

    T = np.asarray(T)
    if T.ndim != 2 or T.shape[1] != 4:
        return ValidityReport(False, f"T must be (K,4), got {T.shape}")

    try:
        T = _as_int_tets(T)
    except TypeError as e:
        return ValidityReport(False, str(e))

    N = P.shape[0]
    K = T.shape[0]
    if N <= 0 or K <= 0:
        return ValidityReport(False, f"Empty P or T: N={N}, K={K}")

    tmin = int(T.min())
    tmax = int(T.max())
    if tmin < 0 or tmax >= N:
        return ValidityReport(False, f"Vertex index out of range: min={tmin}, max={tmax}, N={N}")

    # 1) distinct vertices per tet
    for i in range(K):
        if len(set(map(int, T[i]))) != 4:
            return ValidityReport(False, f"Tet {i} has repeated vertices: {T[i].tolist()}")

    # 2) no duplicate tets
    keys = np.array([_sorted_tet(T[i]) for i in range(K)], dtype=np.int64)
    uniq = np.unique(keys, axis=0)
    if uniq.shape[0] != K:
        return ValidityReport(False, f"Duplicate tetrahedra detected: K={K}, unique={uniq.shape[0]}")

    # 3) non-zero volumes
    for i in range(K):
        a, b, c, d = map(int, T[i])
        s6 = tet_signed_volume(P[a], P[b], P[c], P[d])
        if abs(s6) <= 6.0 * eps_vol:
            return ValidityReport(False, f"Degenerate tet volume at {i}, s6={s6}")

    # 4) face incidence counts in {1,2}
    face_count: Dict[Tuple[int, int, int], int] = {}
    for i in range(K):
        tet = T[i]
        for f in _tet_faces(tet):
            fkey = _face_key(f)
            face_count[fkey] = face_count.get(fkey, 0) + 1
            if face_count[fkey] > 2:
                return ValidityReport(False, f"Non-manifold face {fkey}: count>2")

    # sanity: every face counted is 1 or 2 by construction
    stats = {
        "N": int(N),
        "K": int(K),
        "F": int(len(face_count)),
        "boundary_faces": int(sum(1 for c in face_count.values() if c == 1)),
        "interior_faces": int(sum(1 for c in face_count.values() if c == 2)),
    }
    return ValidityReport(True, "ok", stats=stats)


# -----------------------------
# Candidate mask consistency checks
# -----------------------------

def _edge_set_from_tets(T: np.ndarray) -> set[Tuple[int, int]]:
    T = _as_int_tets(T)
    es: set[Tuple[int, int]] = set()
    for tet in T:
        a, b, c, d = map(int, tet)
        for i, j in ((a,b),(a,c),(a,d),(b,c),(b,d),(c,d)):
            es.add(_sorted_edge(i, j))
    return es


def _face_set_from_tets(T: np.ndarray) -> set[Tuple[int, int, int]]:
    T = _as_int_tets(T)
    fs: set[Tuple[int, int, int]] = set()
    for tet in T:
        for f in _tet_faces(tet):
            fs.add(_face_key(f))
    return fs


def check_candidate_masks_consistency(top: TetMeshTopology):
    """
    Assert that any action marked True in the candidate masks satisfies the
    predicates used in tet_mesh_topology.py. (We only enforce True => valid.)
    """
    P = top.points
    T = top.tets
    edge_set = _edge_set_from_tets(T)
    face_set = _face_set_from_tets(T)

    # --- face candidates (2-3) ---
    for fid in np.flatnonzero(top.candidate_face_mask):
        t0, t1 = int(top.face2tet[fid, 0]), int(top.face2tet[fid, 1])
        assert t0 >= 0 and t1 >= 0, f"candidate face {fid} is boundary: face2tet={top.face2tet[fid]}"
        a, b, c = map(int, top.faces[fid])
        d, e = int(top.face_opp[fid, 0]), int(top.face_opp[fid, 1])

        assert _sorted_edge(d, e) not in edge_set, (
            f"candidate 2-3 face {fid} has existing new edge (d,e)=({d},{e})"
        )

        s1 = tet_signed_volume(P[a], P[b], P[c], P[d])
        s2 = tet_signed_volume(P[a], P[b], P[c], P[e])
        assert s1 != 0.0 and s2 != 0.0, f"candidate 2-3 face {fid} degenerate opposite-side test"
        assert s1 * s2 < 0.0, f"candidate 2-3 face {fid} fails sign test: s1={s1}, s2={s2}"

    # --- edge candidates (3-2) ---
    for eid in np.flatnonzero(top.candidate_edge_mask):
        inc = top.edge2tets[eid]
        assert inc.size == 3, f"candidate 3-2 edge {eid} has valence {inc.size}, expected 3"
        u, v = int(top.edges[0, eid]), int(top.edges[1, eid])
        link = top._edge_link_vertices(u, v, inc)
        assert link is not None, f"candidate 3-2 edge {eid} returned None link"
        a, b, c = map(int, link)
        assert len({a, b, c}) == 3, f"candidate 3-2 edge {eid} link vertices not distinct: {link}"

        fkey = _sorted_face(a, b, c)
        assert fkey not in face_set, f"candidate 3-2 edge {eid} would create duplicate face {fkey}"

        s1 = tet_signed_volume(P[a], P[b], P[c], P[u])
        s2 = tet_signed_volume(P[a], P[b], P[c], P[v])
        assert s1 != 0.0 and s2 != 0.0, f"candidate 3-2 edge {eid} degenerate opposite-side test"
        assert s1 * s2 < 0.0, f"candidate 3-2 edge {eid} fails sign test: s1={s1}, s2={s2}"


# -----------------------------
# Random flip rollouts
# -----------------------------

def random_flip_rollout(
    P: np.ndarray,
    T0: np.ndarray,
    steps: int,
    rng: random.Random,
    check_every: int = 1,
) -> None:
    """
    Build TetMeshTopology and do 'steps' random valid flips sampled from candidate_mask().

    Checks:
      - mesh validity after each step (or every check_every steps)
      - candidate masks consistency (True => predicates hold)
      - apply_action returns True for sampled valid action
    """
    top = TetMeshTopology(P, T0)

    rep = check_mesh_validity(top.points, top.tets)
    assert rep.ok, f"Initial mesh invalid: {rep.msg}"

    q = tet_mean_ratio_quality(top.points, top.tets)
    assert np.all(np.isfinite(q)), "Non-finite tet qualities"

    for s in range(steps):
        mask = top.candidate_mask()
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            break

        a = int(rng.choice(valid.tolist()))
        ok = top.apply_action(a)
        assert ok, f"apply_action returned False for supposedly valid action {a}"

        if (s + 1) % check_every == 0:
            rep = check_mesh_validity(top.points, top.tets)
            assert rep.ok, f"Mesh invalid at step {s+1}: {rep.msg}"

            check_candidate_masks_consistency(top)

            q = top.tet_quality
            assert q.shape[0] == top.tets.shape[0]
            assert np.all(np.isfinite(q)), "Non-finite tet qualities after flip"
            assert np.all((q >= -1e-12) & (q <= 1.0 + 1e-12)), "Quality out of [0,1] bounds"


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", type=str, default="tet_dataset_grid125_sigma1e-02_N2000.mat",
                    help="Path to v7.3 .mat file.")
    ap.add_argument("--steps", type=int, default=20, help="Random flips per sample.")
    ap.add_argument("--max-samples", type=int, default=-1,
                    help="How many samples to test (default: all).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check-every", type=int, default=1,
                    help="Run validity/mask checks every N steps (default 1).")
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
        T0 = _as_int_tets(ds.T_bad[i])

        assert T0.min() >= 0, f"Sample {i}: negative index found"
        assert T0.max() < P.shape[0], f"Sample {i}: index out of range"

        try:
            random_flip_rollout(P, T0, steps=args.steps, rng=rng, check_every=args.check_every)
        except AssertionError as e:
            print(f"\nFAILED on sample {i}: {e}")
            print("Dumping minimal context:")
            print("  P shape:", P.shape, "T0 shape:", T0.shape, "dtype:", T0.dtype)
            raise

        if (i + 1) % 10 == 0:
            print(f"  ...passed {i+1}/{total}")

    ds.close()
    print("All tests passed.")


if __name__ == "__main__":
    main()
