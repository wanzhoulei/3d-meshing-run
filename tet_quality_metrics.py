from __future__ import annotations

import numpy as np


def _sqnorm(x: np.ndarray) -> np.ndarray:
    return np.einsum("bi,bi->b", x, x)


def _norm(x: np.ndarray, eps: float = 1e-18) -> np.ndarray:
    return np.sqrt(np.maximum(_sqnorm(x), eps))


def tet_mean_ratio_quality(P: np.ndarray, tets: np.ndarray, eps: float = 1e-18) -> np.ndarray:
    """
    Mean-ratio quality (current project default), clipped to [0, 1].
    """
    tets = np.asarray(tets, dtype=np.int64)
    if tets.size == 0:
        return np.zeros((0,), dtype=np.float64)

    a = P[tets[:, 0]]
    b = P[tets[:, 1]]
    c = P[tets[:, 2]]
    d = P[tets[:, 3]]

    det6 = np.einsum("bi,bi->b", np.cross(b - a, c - a), d - a)
    V = np.abs(det6) / 6.0

    l2 = (
        _sqnorm(b - a) + _sqnorm(c - a) + _sqnorm(d - a) +
        _sqnorm(c - b) + _sqnorm(d - b) + _sqnorm(d - c)
    )

    q = 12.0 * np.power(3.0 * np.maximum(V, 0.0), 2.0 / 3.0) / np.maximum(l2, eps)
    return np.clip(q, 0.0, 1.0).astype(np.float64, copy=False)


def tet_simpqual1_quality(P: np.ndarray, tets: np.ndarray, eps: float = 1e-18) -> np.ndarray:
    """
    SIMPQUAL type 1 (Radius Ratio) in 3D from Persson's MATLAB code.
    """
    tets = np.asarray(tets, dtype=np.int64)
    if tets.size == 0:
        return np.zeros((0,), dtype=np.float64)

    p1 = P[tets[:, 0]]
    p2 = P[tets[:, 1]]
    p3 = P[tets[:, 2]]
    p4 = P[tets[:, 3]]

    d12 = p2 - p1
    d13 = p3 - p1
    d14 = p4 - p1
    d23 = p3 - p2
    d24 = p4 - p2
    d34 = p4 - p3

    v = np.abs(np.einsum("bi,bi->b", np.cross(d12, d13), d14)) / 6.0

    s1 = _norm(np.cross(d12, d13), eps=eps) / 2.0
    s2 = _norm(np.cross(d12, d14), eps=eps) / 2.0
    s3 = _norm(np.cross(d13, d14), eps=eps) / 2.0
    s4 = _norm(np.cross(d23, d24), eps=eps) / 2.0

    p1v = _norm(d12, eps=eps) * _norm(d34, eps=eps)
    p2v = _norm(d23, eps=eps) * _norm(d14, eps=eps)
    p3v = _norm(d13, eps=eps) * _norm(d24, eps=eps)

    term1 = np.maximum(p1v + p2v + p3v, eps)
    term2 = np.maximum(p1v + p2v - p3v, eps)
    term3 = np.maximum(p1v + p3v - p2v, eps)
    term4 = np.maximum(p2v + p3v - p1v, eps)
    denom2 = np.sqrt(np.maximum(term1 * term2 * term3 * term4, eps))
    denom1 = np.maximum(s1 + s2 + s3 + s4, eps)

    q = 216.0 * (v * v) / denom1 / np.maximum(denom2, eps)
    return np.clip(q, 0.0, 1.0).astype(np.float64, copy=False)


def tet_simpqual2_quality(P: np.ndarray, tets: np.ndarray, eps: float = 1e-18) -> np.ndarray:
    """
    SIMPQUAL type 2 (Approximate) in 3D from Persson's MATLAB code.
    """
    tets = np.asarray(tets, dtype=np.int64)
    if tets.size == 0:
        return np.zeros((0,), dtype=np.float64)

    p1 = P[tets[:, 0]]
    p2 = P[tets[:, 1]]
    p3 = P[tets[:, 2]]
    p4 = P[tets[:, 3]]

    d12 = p2 - p1
    d13 = p3 - p1
    d14 = p4 - p1
    d23 = p3 - p2
    d24 = p4 - p2
    d34 = p4 - p3

    vol = np.abs(np.einsum("bi,bi->b", np.cross(d12, d13), d14)) / 6.0
    l2_sum = (
        _sqnorm(d12) + _sqnorm(d13) + _sqnorm(d14) +
        _sqnorm(d23) + _sqnorm(d24) + _sqnorm(d34)
    )
    denom = np.maximum(l2_sum, eps) ** 1.5

    q = 216.0 * vol / np.sqrt(3.0) / np.maximum(denom, eps)
    return np.clip(q, 0.0, 1.0).astype(np.float64, copy=False)


def compute_tet_quality(P: np.ndarray, tets: np.ndarray, mode: str = "mean_ratio") -> np.ndarray:
    """
    Compute per-tet quality with selectable metric.

    Supported modes:
      - "mean_ratio" (default)
      - "simpqual1"  (radius-ratio)
      - "simpqual2"  (approximate)
    """
    mode = str(mode).lower()
    if mode in ("mean_ratio", "meanratio", "mr"):
        return tet_mean_ratio_quality(P, tets)
    if mode in ("simpqual1", "simpqual_type1", "radius_ratio", "rr"):
        return tet_simpqual1_quality(P, tets)
    if mode in ("simpqual2", "simpqual_type2", "approx", "approximate"):
        return tet_simpqual2_quality(P, tets)
    raise ValueError(f"Unknown tet quality mode: {mode}")
