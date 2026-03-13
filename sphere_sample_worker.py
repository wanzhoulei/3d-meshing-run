#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import traceback

import gmsh
import numpy as np
from scipy.spatial import Delaunay, QhullError

from tet_quality_metrics import compute_tet_quality


def random_points_in_unit_ball_diverse(
    rng: np.random.Generator,
    n: int,
    boundary_bias: float,
) -> np.ndarray:
    """
    Sample points in unit ball with controllable boundary bias.
    boundary_bias in [0,1]: larger => more points near radius 1.
    """
    n = int(max(0, n))
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)

    # Random directions on sphere.
    dirs = rng.normal(size=(n, 3))
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / np.maximum(norms, 1e-12)

    nb = int(round(float(np.clip(boundary_bias, 0.0, 1.0)) * n))
    ni = n - nb

    # Near-boundary radii and interior radii.
    r_boundary = rng.beta(10.0, 1.8, size=(nb,)) if nb > 0 else np.zeros((0,), dtype=np.float64)
    r_inner = rng.random(size=(ni,)) ** (1.0 / 3.0) if ni > 0 else np.zeros((0,), dtype=np.float64)
    radii = np.concatenate([r_boundary, r_inner], axis=0)
    rng.shuffle(radii)
    return dirs * radii[:, None]


def orient_tets_positive(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    out = np.asarray(tets, dtype=np.int64).copy()
    if out.size == 0:
        return out

    a = points[out[:, 0]]
    b = points[out[:, 1]]
    c = points[out[:, 2]]
    d = points[out[:, 3]]

    det = np.einsum("...i,...i->...", np.cross(b - a, c - a), d - a)
    neg = det < 0.0
    if np.any(neg):
        tmp = out[neg, 0].copy()
        out[neg, 0] = out[neg, 1]
        out[neg, 1] = tmp
    return out


def min_abs_vol6(points: np.ndarray, tets: np.ndarray) -> float:
    if tets.size == 0:
        return 0.0
    a = points[tets[:, 0]]
    b = points[tets[:, 1]]
    c = points[tets[:, 2]]
    d = points[tets[:, 3]]
    vol6 = np.einsum("...i,...i->...", np.cross(b - a, c - a), d - a)
    return float(np.min(np.abs(vol6)))


def apply_random_warp(
    points: np.ndarray,
    rng: np.random.Generator,
    *,
    radius: float,
    affine_strength: float,
    radial_strength: float,
) -> np.ndarray:
    """
    Smooth, near-identity warp to increase node-position diversity while
    keeping element counts fixed and geometry valid.
    """
    P = np.asarray(points, dtype=np.float64)
    c = np.mean(P, axis=0, keepdims=True)
    X = P - c

    # Volume-preserving-ish random affine.
    A = np.eye(3) + rng.normal(scale=float(affine_strength), size=(3, 3))
    try:
        u, s, vt = np.linalg.svd(A, full_matrices=False)
        s = np.clip(s, 0.75, 1.35)
        s = s / np.cbrt(np.prod(s))
        A = (u * s) @ vt
    except np.linalg.LinAlgError:
        A = np.eye(3)
    X = X @ A.T

    r = np.linalg.norm(X, axis=1)
    if np.max(r) > 1e-12:
        n = X / np.maximum(r[:, None], 1e-12)
        k = rng.normal(size=(3,))
        k = k / np.maximum(np.linalg.norm(k), 1e-12)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        freq = rng.integers(2, 6)
        mod = 1.0 + float(radial_strength) * np.sin(freq * (n @ k) + phase) * (r / np.max(r))
        X = X * mod[:, None]

    Pw = X + c
    # Keep within sphere radius envelope for consistency.
    rr = np.linalg.norm(Pw, axis=1)
    max_r = float(np.max(rr))
    if max_r > radius * 0.995:
        Pw = Pw * ((radius * 0.995) / max_r)
    return Pw


def tags_to_local_indices(node_tags: np.ndarray, conn_tags: np.ndarray) -> np.ndarray:
    tag_to_local = {int(t): i for i, t in enumerate(node_tags.tolist())}
    flat = conn_tags.reshape(-1)
    local = np.fromiter((tag_to_local[int(t)] for t in flat), dtype=np.int64, count=flat.size)
    return local.reshape(conn_tags.shape)


def ensure_netgen_in_methods(methods: list[str]) -> list[str]:
    clean = [m.strip() for m in methods if m.strip()]
    has_netgen = any(m.lower() == "netgen" for m in clean)
    if not has_netgen:
        clean.append("Netgen")
    return clean


def generate_sphere_pair_once(
    *,
    radius: float,
    mesh_size: float,
    rng: np.random.Generator,
    n_embedded_points: int,
    embedded_radius_frac: float,
    local_size_jitter_min: float,
    local_size_jitter_max: float,
    boundary_bias: float,
    deformation_strength: float,
    warp_affine_strength: float,
    warp_radial_strength: float,
    optimize_methods: list[str],
    tet_quality_mode: str,
    unsafe_options: bool,
) -> dict:
    gmsh.model.add("sphere_sample_{}".format(int(rng.integers(1, 2_000_000_000))))

    vol_tag = gmsh.model.occ.addSphere(0.0, 0.0, 0.0, float(radius))

    point_tags = []
    if n_embedded_points > 0:
        pts = random_points_in_unit_ball_diverse(
            rng,
            n=n_embedded_points,
            boundary_bias=boundary_bias,
        )

        if deformation_strength > 0.0:
            A = np.eye(3) + rng.normal(scale=float(deformation_strength), size=(3, 3))
            pts = pts @ A.T
            # Project to unit ball after linear deformation.
            norms = np.linalg.norm(pts, axis=1)
            scale = np.maximum(1.0, norms)
            pts = pts / scale[:, None]

        pts = pts * (radius * embedded_radius_frac)

        for x, y, z in pts:
            lc = mesh_size * float(rng.uniform(local_size_jitter_min, local_size_jitter_max))
            pt = gmsh.model.occ.addPoint(float(x), float(y), float(z), float(lc))
            point_tags.append(int(pt))

    gmsh.model.occ.synchronize()

    if point_tags:
        gmsh.model.mesh.embed(0, point_tags, 3, int(vol_tag))

    # Some Gmsh builds crash on specific option writes (e.g. General.RandomSeed).
    # Keep option writes disabled by default for robustness.
    if unsafe_options:
        try:
            gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size * 0.80)
            gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size * 1.20)
            gmsh.option.setNumber("Mesh.Optimize", 1)
            gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        except Exception:
            pass

    gmsh.model.mesh.generate(3)

    used_netgen = False
    for method in optimize_methods:
        if not method:
            continue
        gmsh.model.mesh.optimize(method)
        if method.lower() == "netgen":
            used_netgen = True

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_tags = np.asarray(node_tags, dtype=np.int64)
    points = np.asarray(node_coords, dtype=np.float64).reshape(-1, 3)

    _, elem_node_tags = gmsh.model.mesh.getElementsByType(4)
    t_good_tags = np.asarray(elem_node_tags, dtype=np.int64).reshape(-1, 4)
    t_good = tags_to_local_indices(node_tags, t_good_tags)

    # Main diversity source: smooth random warp after Netgen optimization.
    points = apply_random_warp(
        points,
        rng,
        radius=radius,
        affine_strength=warp_affine_strength,
        radial_strength=warp_radial_strength,
    )
    t_good = orient_tets_positive(points, t_good)
    if min_abs_vol6(points, t_good) < 1e-10:
        raise RuntimeError("degenerate_good_tet_after_warp")

    tri = Delaunay(points)
    t_bad = orient_tets_positive(points, np.asarray(tri.simplices, dtype=np.int64))

    q_good = compute_tet_quality(points, t_good, mode=tet_quality_mode)
    q_bad = compute_tet_quality(points, t_bad, mode=tet_quality_mode)

    return {
        "P": points,
        "T_good": t_good,
        "T_bad": t_bad,
        "minQ_good": float(np.min(q_good)) if q_good.size else np.nan,
        "minQ_bad": float(np.min(q_bad)) if q_bad.size else np.nan,
        "num_nodes": int(points.shape[0]),
        "num_tets_good": int(t_good.shape[0]),
        "num_tets_bad": int(t_bad.shape[0]),
        "netgen_used": bool(used_netgen),
    }


def generate_near_target(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(int(args.seed))
    mesh_size = float(args.mesh_size_start)

    methods = ensure_netgen_in_methods(args.optimize_methods)

    target_nodes = int(args.target_nodes)
    node_tol = int(args.node_tolerance)

    target_tets = int(args.target_tets)
    tet_tol_frac = float(args.tet_tolerance_frac)
    tet_tol_abs = int(math.ceil(abs(target_tets) * max(0.0, tet_tol_frac))) if target_tets > 0 else 0

    best = None
    best_err = float("inf")

    for attempt in range(1, int(args.max_attempts) + 1):
        try:
            sample = generate_sphere_pair_once(
                radius=float(args.radius),
                mesh_size=mesh_size,
                rng=rng,
                n_embedded_points=int(args.n_embedded_points),
                embedded_radius_frac=float(args.embedded_radius_frac),
                local_size_jitter_min=float(args.local_size_jitter_min),
                local_size_jitter_max=float(args.local_size_jitter_max),
                boundary_bias=float(args.boundary_bias),
                deformation_strength=float(args.deformation_strength),
                warp_affine_strength=float(args.warp_affine_strength),
                warp_radial_strength=float(args.warp_radial_strength),
                optimize_methods=methods,
                tet_quality_mode=str(args.tet_quality_mode),
                unsafe_options=bool(args.unsafe_options),
            )
        except (QhullError, RuntimeError, ValueError):
            mesh_size *= 1.015
            continue

        node_err_rel = (abs(sample["num_nodes"] - target_nodes) / max(1.0, float(target_nodes))) if target_nodes > 0 else 0.0
        tet_err_rel = (abs(sample["num_tets_good"] - target_tets) / max(1.0, float(target_tets))) if target_tets > 0 else 0.0
        total_err = (2.5 * tet_err_rel) + node_err_rel

        if total_err < best_err:
            best = sample
            best_err = total_err

        node_ok = True if target_nodes <= 0 else (abs(sample["num_nodes"] - target_nodes) <= node_tol)
        # Padding efficiency is driven by the environment/start mesh (T_bad),
        # so enforce the tet-count band on T_bad.
        tet_bad_ok = True if target_tets <= 0 else (abs(sample["num_tets_bad"] - target_tets) <= tet_tol_abs)
        netgen_ok = bool(sample.get("netgen_used", False))

        if node_ok and tet_bad_ok and netgen_ok:
            sample["matched"] = True
            sample["attempts"] = attempt
            sample["mesh_size_final"] = mesh_size
            return sample

        # Update mesh size primarily for tet count control.
        if target_tets > 0:
            lo = target_tets - tet_tol_abs
            hi = target_tets + tet_tol_abs
            if sample["num_tets_bad"] > hi:
                mesh_size *= 1.03
            elif sample["num_tets_bad"] < lo:
                mesh_size *= 0.97
        else:
            if target_nodes > 0:
                if sample["num_nodes"] > (target_nodes + node_tol):
                    mesh_size *= 1.03
                elif sample["num_nodes"] < (target_nodes - node_tol):
                    mesh_size *= 0.97

        mesh_size *= float(np.exp(rng.normal(0.0, 0.008)))

    if best is None:
        raise RuntimeError("No valid sample generated after all attempts")

    best["matched"] = False
    best["attempts"] = int(args.max_attempts)
    best["mesh_size_final"] = mesh_size
    return best


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate one sphere sample in an isolated process.")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--radius", type=float, default=1.0)
    ap.add_argument("--mesh-size-start", type=float, default=0.28)

    ap.add_argument("--target-nodes", type=int, default=200)
    ap.add_argument("--node-tolerance", type=int, default=5)
    ap.add_argument("--target-tets", type=int, default=600)
    ap.add_argument("--tet-tolerance-frac", type=float, default=0.05)

    ap.add_argument("--max-attempts", type=int, default=24)
    ap.add_argument("--n-embedded-points", type=int, default=72)
    ap.add_argument("--embedded-radius-frac", type=float, default=0.97)
    ap.add_argument("--local-size-jitter-min", type=float, default=0.35)
    ap.add_argument("--local-size-jitter-max", type=float, default=1.95)
    ap.add_argument("--boundary-bias", type=float, default=0.65)
    ap.add_argument("--deformation-strength", type=float, default=0.22)
    ap.add_argument("--warp-affine-strength", type=float, default=0.10)
    ap.add_argument("--warp-radial-strength", type=float, default=0.20)

    ap.add_argument("--tet-quality-mode", type=str, default="mean_ratio")
    ap.add_argument("--optimize-methods", type=str, default="Gmsh,Netgen")
    ap.add_argument("--unsafe-options", action="store_true", default=False)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.optimize_methods = [s.strip() for s in str(args.optimize_methods).split(",") if s.strip()]

    try:
        gmsh.initialize()
        sample = generate_near_target(args)

        np.savez_compressed(
            args.out,
            P=np.asarray(sample["P"], dtype=np.float64),
            T_bad=np.asarray(sample["T_bad"], dtype=np.int64),
            T_good=np.asarray(sample["T_good"], dtype=np.int64),
            minQ_bad=np.asarray([sample["minQ_bad"]], dtype=np.float64),
            minQ_good=np.asarray([sample["minQ_good"]], dtype=np.float64),
            num_nodes=np.asarray([sample["num_nodes"]], dtype=np.int32),
            num_tets_good=np.asarray([sample["num_tets_good"]], dtype=np.int32),
            num_tets_bad=np.asarray([sample["num_tets_bad"]], dtype=np.int32),
            netgen_used=np.asarray([1 if sample.get("netgen_used", False) else 0], dtype=np.int8),
            matched=np.asarray([1 if sample["matched"] else 0], dtype=np.int8),
            attempts=np.asarray([sample["attempts"]], dtype=np.int32),
            mesh_size_final=np.asarray([sample["mesh_size_final"]], dtype=np.float64),
        )
        return 0
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
