# initial_embedding.py
# ------------------------------------------------------------
# Build initial edge/face features + incidence graph for TetMeshRefineVecEnv obs.
#
# Compatible with:
#   - tet_env.py observation dict:
#       dict(points, tets, faces, face2tet, edges, candidate_mask, tet_quality)
#   - model_face_edge_gpt.Mesh3DActorCritic forward inputs:
#       x, edge_feat, face_feat, edge_index, node_mask, edge_mask,
#       edge_node_mask, face_node_mask, edge_action_mask, face_action_mask
#
# Node ordering (IMPORTANT):
#   nodes [0..E-1]      : edge nodes
#   nodes [E..E+F-1]    : face nodes  (local indexing inside a single mesh)
#
# In padded batch layout used by batch_from_obs:
#   nodes [0..Emax-1]        : edge nodes
#   nodes [Emax..Emax+Fmax-1]: face nodes
#
# Action ordering in ENV (IMPORTANT):
#   env action indices are [faces first then edges] (size F+E)
#   candidate_mask from env is also [faces then edges]
#
# Action ordering in MODEL (IMPORTANT):
#   model logits are [edges first then faces] (size E+F)
#   edge logits occupy [0..E-1], face logits occupy [E..E+F-1]
#
# CHANGE (requested):
#   - Remove boundary-indicator features (duplicates incidence counts).
#     * Edge: remove boundary_edge from edge_feat (keep #faces containing edge).
#     * Face: remove is_face_boundary from face_feat (keep #tets sharing face).
# ------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch


# -------------------------
# Small utility encoders
# -------------------------

def _safe_norm(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.sqrt(torch.clamp((v * v).sum(dim=-1), min=eps))


def _fourier_encode(x: torch.Tensor, num_freqs: int = 6, include_identity: bool = True) -> torch.Tensor:
    """
    Fourier features for continuous scalars.
    Input x: shape (...,) or (...,1)
    Output: (..., D) where D = (include_identity?1:0) + 2*num_freqs
    """
    if x.dim() == 0:
        x = x.view(1)
    if x.size(-1) != 1:
        x = x.unsqueeze(-1)  # (...,1)

    freqs = (2.0 ** torch.arange(num_freqs, device=x.device, dtype=x.dtype))  # (F,)
    angles = x * freqs.view(*([1] * (x.dim() - 1)), -1)  # (...,F)
    sin = torch.sin(angles)
    cos = torch.cos(angles)

    out = [sin, cos]
    if include_identity:
        out = [x] + out
    return torch.cat(out, dim=-1)


def _bucketize_onehot(x: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """
    Bucketize integer tensor x into bins (increasing edges), return one-hot.
    bins: (K,) e.g. [1,2,3,4,6,8,12,20,40]
    returns one-hot (...,K)
    """
    idx = torch.bucketize(x, bins) - 1
    idx = torch.clamp(idx, 0, bins.numel() - 1)
    onehot = torch.zeros(*x.shape, bins.numel(), device=x.device, dtype=torch.float32)
    onehot.scatter_(-1, idx.unsqueeze(-1), 1.0)
    return onehot


def _triangle_angles_from_sides(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Given side lengths a,b,c (each (...,)), return angles (A,B,C) opposite (a,b,c).
    """
    def angle(opposite, s1, s2):
        cosv = (s1*s1 + s2*s2 - opposite*opposite) / torch.clamp(2.0*s1*s2, min=eps)
        cosv = torch.clamp(cosv, -1.0, 1.0)
        return torch.acos(cosv)

    A = angle(a, b, c)
    B = angle(b, a, c)
    C = angle(c, a, b)
    return torch.stack([A, B, C], dim=-1)


def _quality_feat_from_tet_quality(
    tet_quality: np.ndarray,
    *,
    device: torch.device,
    step_frac: float = 0.0,
    no_improve_frac: float = 0.0,
    tau: float = 0.05,
    bad_quality_thresh: float = 0.2,
    worstk_k: int = 10,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Build compact critic features from current tet quality distribution.

    Returns:
      (8,) tensor:
        [mean, min, std, worstk_mean, softmin(tau), frac_below_thresh,
         step_frac, no_improve_frac]
    """
    q_np = np.asarray(tet_quality, dtype=np.float32).reshape(-1)
    if q_np.size == 0:
        base = torch.zeros((6,), dtype=torch.float32, device=device)
    else:
        q = torch.as_tensor(q_np, dtype=torch.float32, device=device)
        q_mean = q.mean()
        q_min = q.min()
        q_std = q.std(unbiased=False)
        k = int(max(1, min(worstk_k, q.numel())))
        q_worstk = torch.topk(q, k=k, largest=False).values.mean()
        q_softmin = -tau * torch.log(torch.exp(-q / max(tau, eps)).mean().clamp_min(eps))
        q_frac_bad = (q < bad_quality_thresh).to(torch.float32).mean()
        base = torch.stack([q_mean, q_min, q_std, q_worstk, q_softmin, q_frac_bad], dim=0)

    step_frac_t = torch.tensor(
        float(np.clip(step_frac, 0.0, 1.0)), dtype=torch.float32, device=device
    )
    no_improve_frac_t = torch.tensor(
        float(np.clip(no_improve_frac, 0.0, 1.0)), dtype=torch.float32, device=device
    )
    return torch.cat([base, torch.stack([step_frac_t, no_improve_frac_t], dim=0)], dim=0)


# -------------------------
# Core: per-mesh embedding
# -------------------------

def initial_embedding_single(
    points: torch.Tensor,      # (N,3)
    tets: torch.Tensor,        # (K,4) long
    faces: torch.Tensor,       # (F,3) long (sorted)
    face2tet: torch.Tensor,    # (F,2) long, -1 indicates boundary
    edges: torch.Tensor,       # (2,E) long (sorted)
    *,
    tet_quality: Optional[torch.Tensor] = None,  # (K,) optional, current per-tet quality
    step_frac: float = 0.0,      # progress feature for actor/critic
    no_improve_frac: float = 0.0, # progress feature for actor/critic
    num_fourier_freqs: int = 6,
    degree_bins: Optional[torch.Tensor] = None,
    inc_bins: Optional[torch.Tensor] = None,
    normalize_by_median_edge: bool = True,
    eps: float = 1e-12
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Build:
      - x_edge (E,3): edge midpoint coords
      - x_face (F,3): face centroid coords
      - edge_feat (E,De)
      - face_feat (F,Df)
      - edge_index (2,M): directed incidence edges between edge-nodes and face-nodes
      - aux: extra scalars for debugging

    The returned edge_index uses *local* node indices:
      edge-node i = i
      face-node f = E + f
    """
    device = points.device
    dtype = points.dtype
    E = edges.shape[1]
    F = faces.shape[0]
    N = points.shape[0]

    # -------------------------
    # Build global scale
    # -------------------------
    vi = points[edges[0]]  # (E,3)
    vj = points[edges[1]]  # (E,3)
    edge_vec = vj - vi
    edge_len = _safe_norm(edge_vec, eps=eps)  # (E,)

    if normalize_by_median_edge:
        scale = torch.median(edge_len).clamp_min(eps)
    else:
        scale = torch.tensor(1.0, device=device, dtype=dtype)

    edge_len_n = edge_len / scale
    edge_loglen = torch.log(edge_len_n.clamp_min(eps))  # (E,)

    # -------------------------
    # Node coordinates for EGNN
    # -------------------------
    x_edge = 0.5 * (vi + vj)  # (E,3)

    f0 = points[faces[:, 0]]
    f1 = points[faces[:, 1]]
    f2 = points[faces[:, 2]]
    x_face = (f0 + f1 + f2) / 3.0  # (F,3)

    # -------------------------
    # Vertex degree
    # -------------------------
    deg = torch.zeros((N,), device=device, dtype=torch.long)
    deg.scatter_add_(0, edges[0], torch.ones((E,), device=device, dtype=torch.long))
    deg.scatter_add_(0, edges[1], torch.ones((E,), device=device, dtype=torch.long))
    deg_u = deg[edges[0]]  # (E,)
    deg_v = deg[edges[1]]  # (E,)

    if degree_bins is None:
        degree_bins = torch.tensor([1, 2, 3, 4, 6, 8, 12, 20, 40], device=device, dtype=torch.long)

    deg_u_oh = _bucketize_onehot(deg_u, degree_bins)
    deg_v_oh = _bucketize_onehot(deg_v, degree_bins)

    # -------------------------
    # Cached numpy views + edge lookup map
    # -------------------------
    e0 = edges[0].detach().cpu().numpy()
    e1 = edges[1].detach().cpu().numpy()
    edge_map = {(int(e0[i]), int(e1[i])): i for i in range(E)}
    faces_np = faces.detach().cpu().numpy() if F > 0 else np.zeros((0, 3), dtype=np.int64)
    tets_np = tets.detach().cpu().numpy() if tets.numel() > 0 else np.zeros((0, 4), dtype=np.int64)

    # -------------------------
    # One pass over faces:
    #   - edge_face_inc
    #   - incidence graph src/dst
    # -------------------------
    edge_face_inc = torch.zeros((E,), device=device, dtype=torch.long)
    src: List[int] = []
    dst: List[int] = []
    for f in range(F):
        a, b, c = map(int, faces_np[f])
        face_node = E + f
        for (u, v) in ((a, b), (b, c), (a, c)):
            if u > v:
                u, v = v, u
            e_idx = edge_map.get((u, v), None)
            if e_idx is None:
                continue
            edge_face_inc[e_idx] += 1
            # edge -> face
            src.append(e_idx)
            dst.append(face_node)
            # face -> edge
            src.append(face_node)
            dst.append(e_idx)

    if inc_bins is None:
        inc_bins = torch.tensor([0, 1, 2, 3, 4, 6, 8, 12, 20], device=device, dtype=torch.long)

    edge_face_inc_oh = _bucketize_onehot(edge_face_inc, inc_bins)

    # -------------------------
    # Local tet-quality features for actor (edge/face neighborhoods)
    # -------------------------
    K = int(tets.shape[0])
    if tet_quality is None:
        q_t = torch.zeros((K,), device=device, dtype=torch.float32)
    else:
        q_t = tet_quality.to(device=device, dtype=torch.float32).reshape(-1)
        if q_t.numel() < K:
            q_pad = torch.zeros((K,), device=device, dtype=torch.float32)
            q_pad[: q_t.numel()] = q_t
            q_t = q_pad
        elif q_t.numel() > K:
            q_t = q_t[:K]

    # Edge-local quality stats over incident tets.
    edge_tet_inc = torch.zeros((E,), device=device, dtype=torch.long)
    edge_q_count = torch.zeros((E,), device=device, dtype=torch.float32)
    edge_q_sum = torch.zeros((E,), device=device, dtype=torch.float32)
    edge_q_sumsq = torch.zeros((E,), device=device, dtype=torch.float32)
    edge_q_min = torch.full((E,), float("inf"), device=device, dtype=torch.float32)
    edge_q_max = torch.full((E,), float("-inf"), device=device, dtype=torch.float32)

    if K > 0:
        q_np = q_t.detach().cpu().numpy()
        for kk in range(K):
            qk = float(q_np[kk])
            i, j, k2, l = map(int, tets_np[kk])
            pairs = [(i, j), (i, k2), (i, l), (j, k2), (j, l), (k2, l)]
            for (u, v) in pairs:
                if u > v:
                    u, v = v, u
                idx = edge_map.get((u, v), None)
                if idx is None:
                    continue
                edge_tet_inc[idx] += 1
                edge_q_count[idx] += 1.0
                edge_q_sum[idx] += qk
                edge_q_sumsq[idx] += qk * qk
                if qk < float(edge_q_min[idx]):
                    edge_q_min[idx] = qk
                if qk > float(edge_q_max[idx]):
                    edge_q_max[idx] = qk

    edge_tet_inc_oh = _bucketize_onehot(edge_tet_inc, inc_bins)

    edge_den = edge_q_count.clamp_min(1.0)
    edge_q_mean = edge_q_sum / edge_den
    edge_q_var = (edge_q_sumsq / edge_den) - edge_q_mean * edge_q_mean
    edge_q_std = torch.sqrt(edge_q_var.clamp_min(0.0))
    edge_has_q = edge_q_count > 0
    edge_q_min = torch.where(edge_has_q, edge_q_min, torch.zeros_like(edge_q_min))
    edge_q_max = torch.where(edge_has_q, edge_q_max, torch.zeros_like(edge_q_max))
    edge_q_span = edge_q_max - edge_q_min
    edge_q_feat = torch.stack([edge_q_mean, edge_q_min, edge_q_max, edge_q_std, edge_q_span], dim=-1)

    # Face-local quality stats over up-to-2 incident tets.
    face_q_mean = torch.zeros((F,), device=device, dtype=torch.float32)
    face_q_min = torch.zeros((F,), device=device, dtype=torch.float32)
    face_q_max = torch.zeros((F,), device=device, dtype=torch.float32)
    face_q_std = torch.zeros((F,), device=device, dtype=torch.float32)
    face_q_span = torch.zeros((F,), device=device, dtype=torch.float32)

    if F > 0 and K > 0:
        f2_np = face2tet.detach().cpu().numpy()
        for f in range(F):
            t0 = int(f2_np[f, 0])
            t1 = int(f2_np[f, 1])
            has0 = 0 <= t0 < K
            has1 = 0 <= t1 < K
            if not has0 and not has1:
                continue
            if has0 and not has1:
                q0 = float(q_np[t0])
                face_q_mean[f] = q0
                face_q_min[f] = q0
                face_q_max[f] = q0
                face_q_std[f] = 0.0
                face_q_span[f] = 0.0
                continue
            if has1 and not has0:
                q1 = float(q_np[t1])
                face_q_mean[f] = q1
                face_q_min[f] = q1
                face_q_max[f] = q1
                face_q_std[f] = 0.0
                face_q_span[f] = 0.0
                continue
            q0 = float(q_np[t0])
            q1 = float(q_np[t1])
            q_mean = 0.5 * (q0 + q1)
            q_min = q0 if q0 < q1 else q1
            q_max = q1 if q1 > q0 else q0
            q_std = 0.5 * abs(q0 - q1)  # population std for two samples
            face_q_mean[f] = q_mean
            face_q_min[f] = q_min
            face_q_max[f] = q_max
            face_q_std[f] = q_std
            face_q_span[f] = q_max - q_min

    face_q_feat = torch.stack([face_q_mean, face_q_min, face_q_max, face_q_std, face_q_span], dim=-1)

    # Global episode progress as actor input (repeated per node).
    step_frac_t = torch.tensor(
        float(np.clip(step_frac, 0.0, 1.0)), dtype=torch.float32, device=device
    )
    no_improve_frac_t = torch.tensor(
        float(np.clip(no_improve_frac, 0.0, 1.0)), dtype=torch.float32, device=device
    )
    edge_prog_feat = torch.stack(
        [
            torch.full((E,), float(step_frac_t.item()), device=device, dtype=torch.float32),
            torch.full((E,), float(no_improve_frac_t.item()), device=device, dtype=torch.float32),
        ],
        dim=-1,
    )
    face_prog_feat = torch.stack(
        [
            torch.full((F,), float(step_frac_t.item()), device=device, dtype=torch.float32),
            torch.full((F,), float(no_improve_frac_t.item()), device=device, dtype=torch.float32),
        ],
        dim=-1,
    )

    # -------------------------
    # Face geometry: area / angles / aspect
    # -------------------------
    e01 = f1 - f0
    e02 = f2 - f0
    n = torch.cross(e01, e02, dim=-1)               # (F,3)
    area = 0.5 * _safe_norm(n, eps=eps)             # (F,)
    area_n = (area / (scale * scale)).clamp_min(eps)
    log_area = torch.log(area_n)                    # (F,)

    l01 = _safe_norm(f1 - f0, eps=eps) / scale
    l12 = _safe_norm(f2 - f1, eps=eps) / scale
    l20 = _safe_norm(f0 - f2, eps=eps) / scale

    ang = _triangle_angles_from_sides(l12, l20, l01, eps=eps)  # (F,3)
    ang_sorted, _ = torch.sort(ang, dim=-1)
    ang_min = ang_sorted[:, 0]
    ang_mid = ang_sorted[:, 1]
    ang_max = ang_sorted[:, 2]

    l_stack = torch.stack([l01, l12, l20], dim=-1)
    l_min = l_stack.min(dim=-1).values.clamp_min(eps)
    l_max = l_stack.max(dim=-1).values
    l_mean = l_stack.mean(dim=-1).clamp_min(eps)

    aspect = (l_max / l_min).clamp_max(1e4)
    log_aspect = torch.log(aspect.clamp_min(eps))
    shape_area = (area_n / (l_mean * l_mean)).clamp_min(eps)   # dimensionless
    log_shape_area = torch.log(shape_area)

    # -------------------------
    # Face incidence counts: how many tets share this face (1 boundary, 2 interior)
    # (kept; boundary flag removed from embedding)
    # -------------------------
    face_tet_inc = (face2tet[:, 0] >= 0).long() + (face2tet[:, 1] >= 0).long()  # (F,) in {1,2}
    face_tet_inc_oh = torch.zeros((F, 3), device=device, dtype=torch.float32)
    face_tet_inc_oh.scatter_(1, torch.clamp(face_tet_inc, 0, 2).unsqueeze(1), 1.0)

    # -------------------------
    # Build edge features (boundary flag REMOVED)
    # -------------------------
    enc_len = _fourier_encode(edge_loglen, num_freqs=num_fourier_freqs, include_identity=True)
    edge_feat = torch.cat(
        [
            enc_len,                     # length (log normalized) Fourier
            deg_u_oh, deg_v_oh,          # endpoint degrees one-hot
            edge_face_inc_oh,            # incident faces bins one-hot
            edge_tet_inc_oh,             # incident tets bins one-hot
            edge_q_feat,                 # local tet-quality stats around this edge
            edge_prog_feat,              # current episode progress (step/no-improve)
            # boundary_edge.unsqueeze(-1)  # REMOVED (duplicate with edge_face_inc)
        ],
        dim=-1
    ).to(torch.float32)

    # -------------------------
    # Build face features (boundary flag REMOVED)
    # -------------------------
    enc_area   = _fourier_encode(log_area, num_freqs=num_fourier_freqs, include_identity=True)
    enc_angmin = _fourier_encode(ang_min, num_freqs=max(2, num_fourier_freqs // 2), include_identity=True)
    enc_angmid = _fourier_encode(ang_mid, num_freqs=max(2, num_fourier_freqs // 2), include_identity=True)
    enc_angmax = _fourier_encode(ang_max, num_freqs=max(2, num_fourier_freqs // 2), include_identity=True)
    enc_lasp   = _fourier_encode(log_aspect, num_freqs=max(2, num_fourier_freqs // 2), include_identity=True)
    enc_lsha   = _fourier_encode(log_shape_area, num_freqs=max(2, num_fourier_freqs // 2), include_identity=True)

    face_feat = torch.cat(
        [
            enc_area,
            enc_angmin, enc_angmid, enc_angmax,
            enc_lasp,
            enc_lsha,
            face_tet_inc_oh,              # kept; implies boundary if ==1
            face_q_feat,                  # local tet-quality stats around this face
            face_prog_feat,               # current episode progress (step/no-improve)
            # is_face_boundary.float().unsqueeze(-1)  # REMOVED (duplicate with face_tet_inc)
        ],
        dim=-1
    ).to(torch.float32)

    if len(src) == 0:
        edge_index = torch.zeros((2, 0), device=device, dtype=torch.long)
    else:
        edge_index = torch.tensor([src, dst], device=device, dtype=torch.long)

    is_face_boundary = (face2tet[:, 1] < 0)  # debug-only aux tensor
    aux = dict(
        scale=scale.detach(),
        edge_len=edge_len.detach(),
        edge_face_inc=edge_face_inc.detach(),
        edge_tet_inc=edge_tet_inc.detach(),
        face_area=area.detach(),
        face_angles=ang.detach(),
        face_aspect=aspect.detach(),
        face_is_boundary=is_face_boundary.detach(),  # still useful for debugging
    )

    return x_edge, x_face, edge_feat, face_feat, edge_index, aux


# -------------------------
# Batch builder (obs -> padded batch tensors)
# -------------------------

@dataclass
class BatchedGraph:
    x: torch.Tensor                 # (B, Nmax, 3)
    edge_feat: torch.Tensor         # (B, Emax, De)
    face_feat: torch.Tensor         # (B, Fmax, Df)
    quality_feat: torch.Tensor      # (B, Dq) critic-only global quality features
    edge_index: torch.Tensor        # (B, 2, Mmax)
    node_mask: torch.Tensor         # (B, Nmax)
    edge_mask: torch.Tensor         # (B, Mmax)
    edge_node_mask: torch.Tensor    # (B, Emax)
    face_node_mask: torch.Tensor    # (B, Fmax)
    edge_action_mask: torch.Tensor  # (B, Emax)
    face_action_mask: torch.Tensor  # (B, Fmax)
    aux: List[Dict[str, torch.Tensor]]


def batch_from_obs(
    obs: List[Dict[str, Any]],
    *,
    device: torch.device,
    num_fourier_freqs: int = 6
) -> BatchedGraph:
    """
    Convert env obs list into padded batch tensors for the model.

    Model expects node ordering: [edges..., faces...]
    Env candidate_mask ordering: [faces..., edges...]

    We split candidate_mask into:
      face_action_mask: first F entries
      edge_action_mask: last E entries
    """
    B = len(obs)

    per = []
    E_list = []
    F_list = []
    M_list = []
    for o in obs:
        P = torch.as_tensor(o["points"], device=device, dtype=torch.float32)
        T = torch.as_tensor(o["tets"], device=device, dtype=torch.long)
        F = torch.as_tensor(o["faces"], device=device, dtype=torch.long)
        F2 = torch.as_tensor(o["face2tet"], device=device, dtype=torch.long)
        E = torch.as_tensor(o["edges"], device=device, dtype=torch.long)  # (2,E)

        x_edge, x_face, edge_feat, face_feat, edge_index, aux = initial_embedding_single(
            P,
            T,
            F,
            F2,
            E,
            tet_quality=torch.as_tensor(o["tet_quality"], device=device, dtype=torch.float32),
            step_frac=float(o.get("step_frac", 0.0)),
            no_improve_frac=float(o.get("no_improve_frac", 0.0)),
            num_fourier_freqs=num_fourier_freqs,
        )
        quality_feat = _quality_feat_from_tet_quality(
            o["tet_quality"],
            device=device,
            step_frac=float(o.get("step_frac", 0.0)),
            no_improve_frac=float(o.get("no_improve_frac", 0.0)),
        )
        per.append((x_edge, x_face, edge_feat, face_feat, edge_index, aux, quality_feat))
        E_list.append(edge_feat.shape[0])
        F_list.append(face_feat.shape[0])
        M_list.append(edge_index.shape[1])

    Emax = int(max(E_list)) if B > 0 else 0
    Fmax = int(max(F_list)) if B > 0 else 0
    Nmax = Emax + Fmax
    Mmax = int(max(M_list)) if B > 0 else 0

    De = per[0][2].shape[1] if B > 0 else 0
    Df = per[0][3].shape[1] if B > 0 else 0
    Dq = per[0][6].shape[0] if B > 0 else 0

    x = torch.zeros((B, Nmax, 3), device=device, dtype=torch.float32)
    edge_feat_b = torch.zeros((B, Emax, De), device=device, dtype=torch.float32)
    face_feat_b = torch.zeros((B, Fmax, Df), device=device, dtype=torch.float32)
    quality_feat_b = torch.zeros((B, Dq), device=device, dtype=torch.float32)

    edge_index_b = torch.zeros((B, 2, Mmax), device=device, dtype=torch.long)
    edge_mask = torch.zeros((B, Mmax), device=device, dtype=torch.bool)

    edge_node_mask = torch.zeros((B, Emax), device=device, dtype=torch.bool)
    face_node_mask = torch.zeros((B, Fmax), device=device, dtype=torch.bool)

    edge_action_mask = torch.zeros((B, Emax), device=device, dtype=torch.bool)
    face_action_mask = torch.zeros((B, Fmax), device=device, dtype=torch.bool)

    aux_list: List[Dict[str, torch.Tensor]] = []

    for b, o in enumerate(obs):
        x_edge, x_face, ef, ff, ei, aux, quality_feat = per[b]
        E = ef.shape[0]
        F = ff.shape[0]
        M = ei.shape[1]

        # coords: edges in [0..E-1], faces in [Emax..Emax+F-1] (padded layout)
        x[b, :E, :] = x_edge
        x[b, Emax:Emax+F, :] = x_face

        # features
        edge_feat_b[b, :E, :] = ef
        face_feat_b[b, :F, :] = ff

        # node masks
        edge_node_mask[b, :E] = True
        face_node_mask[b, :F] = True

        # remap edge_index from local layout (face start at E) to padded layout (face start at Emax)
        if M > 0:
            src = ei[0].clone()
            dst = ei[1].clone()

            src_is_face = (src >= E)
            dst_is_face = (dst >= E)

            src[src_is_face] = Emax + (src[src_is_face] - E)
            dst[dst_is_face] = Emax + (dst[dst_is_face] - E)

            edge_index_b[b, 0, :M] = src
            edge_index_b[b, 1, :M] = dst
            edge_mask[b, :M] = True

        # candidate_mask from env is [faces then edges]
        cand = np.asarray(o["candidate_mask"], dtype=np.bool_)
        cand_face = cand[:F]
        cand_edge = cand[F:F+E]

        face_action_mask[b, :F] = torch.as_tensor(cand_face, device=device)
        edge_action_mask[b, :E] = torch.as_tensor(cand_edge, device=device)
        quality_feat_b[b] = quality_feat

        aux_list.append(aux)

    node_mask = torch.cat([edge_node_mask, face_node_mask], dim=1)  # (B, Nmax)

    return BatchedGraph(
        x=x,
        edge_feat=edge_feat_b,
        face_feat=face_feat_b,
        quality_feat=quality_feat_b,
        edge_index=edge_index_b,
        node_mask=node_mask,
        edge_mask=edge_mask,
        edge_node_mask=edge_node_mask,
        face_node_mask=face_node_mask,
        edge_action_mask=edge_action_mask,
        face_action_mask=face_action_mask,
        aux=aux_list,
    )


# -------------------------
# Action index mapping helpers
# -------------------------

def model_action_to_env_with_sizes(a_model: np.ndarray, E_sizes: np.ndarray, F_sizes: np.ndarray) -> np.ndarray:
    """
    Convert MODEL action indices -> ENV action indices, using true per-env sizes.

    MODEL ordering: [edges(0..E-1), faces(E..E+F-1)]
    ENV ordering:   [faces(0..F-1), edges(F..F+E-1)]
    """
    a_env = np.zeros_like(a_model, dtype=np.int64)
    for b in range(a_model.shape[0]):
        E = int(E_sizes[b])
        F = int(F_sizes[b])
        a = int(a_model[b])
        if a < E:
            a_env[b] = F + a
        else:
            a_env[b] = a - E
    return a_env
