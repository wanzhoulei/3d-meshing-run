# tet_mesh_topology_local.py
from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional, Iterable, Set


# ============================================================
# Geometry / quality
# ============================================================

def tet_signed_volume(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    return float(np.linalg.det(np.stack([b - a, c - a, d - a], axis=1)))

def tet_mean_ratio_quality(P: np.ndarray, tets: np.ndarray, eps: float = 1e-18) -> np.ndarray:
    tets = np.asarray(tets, dtype=np.int64)
    a = P[tets[:, 0]]
    b = P[tets[:, 1]]
    c = P[tets[:, 2]]
    d = P[tets[:, 3]]

    det6 = np.einsum('bi,bi->b', np.cross(b - a, c - a), d - a)
    V = np.abs(det6) / 6.0

    def sq(x): return np.einsum('bi,bi->b', x, x)

    l2 = (
        sq(b - a) + sq(c - a) + sq(d - a) +
        sq(c - b) + sq(d - b) + sq(d - c)
    )

    q = 12.0 * np.power(3.0 * np.maximum(V, 0.0), 2.0 / 3.0) / np.maximum(l2, eps)
    return np.clip(q, 0.0, 1.0)


# ============================================================
# Topology helpers
# ============================================================

def _sorted_face(i: int, j: int, k: int) -> Tuple[int, int, int]:
    a, b, c = int(i), int(j), int(k)
    if a > b: a, b = b, a
    if b > c: b, c = c, b
    if a > b: a, b = b, a
    return (a, b, c)

def _sorted_edge(i: int, j: int) -> Tuple[int, int]:
    a, b = int(i), int(j)
    return (a, b) if a < b else (b, a)

def _tet_faces(t: Iterable[int]) -> List[Tuple[int, int, int]]:
    i, j, k, l = map(int, t)
    return [
        _sorted_face(j, k, l),
        _sorted_face(i, k, l),
        _sorted_face(i, j, l),
        _sorted_face(i, j, k),
    ]

def _tet_edges(t: Iterable[int]) -> List[Tuple[int, int]]:
    i, j, k, l = map(int, t)
    return [
        _sorted_edge(i, j), _sorted_edge(i, k), _sorted_edge(i, l),
        _sorted_edge(j, k), _sorted_edge(j, l), _sorted_edge(k, l),
    ]

def _face_opp_vertex(face: Tuple[int, int, int], tet: Iterable[int]) -> int:
    s = set(map(int, tet))
    s.remove(face[0]); s.remove(face[1]); s.remove(face[2])
    return int(next(iter(s)))


# ============================================================
# Incremental TetMeshTopology (compact active tets)
# ============================================================

class TetMeshTopology:
    """
    Same external API as your original TetMeshTopology, but:
      - no global rebuild after flips
      - tets are ALWAYS compact (active-only)
      - remove uses swap-remove and updates adjacency locally (faces+edges of swapped tet only)
      - flips are GUARANTEED non-degenerate by checking volumes of newly created tets
    """

    def __init__(self, points: np.ndarray, tets: np.ndarray, vol_eps6: float = 1e-12):
        self.points = np.asarray(points, dtype=np.float64)
        self.tets = np.asarray(tets, dtype=np.int64)     # ALWAYS active-only, shape (K,4)
        self.vol_eps6 = float(vol_eps6)                  # threshold on |signed_volume_6|
        self.tet_quality = tet_mean_ratio_quality(self.points, self.tets)

        # Face registry
        self._face_id: Dict[Tuple[int,int,int], int] = {}
        self._face_active: List[bool] = []
        self.faces_list: List[Tuple[int,int,int]] = []
        self.face2tet_list: List[List[int]] = []   # fid -> [t0,t1] (-1 if boundary)
        self.face_opp_list: List[List[int]] = []   # fid -> [opp0,opp1]
        self._active_fids: Set[int] = set()
        self._face_set: Set[Tuple[int,int,int]] = set()
        self._v2faces: Dict[int, Set[int]] = {}

        # Edge registry
        self._edge_id: Dict[Tuple[int,int], int] = {}
        self._edge_active: List[bool] = []
        self.edges_list: List[Tuple[int,int]] = []
        self.edge2tets_list: List[Set[int]] = []   # eid -> set(tids)
        self._active_eids: Set[int] = set()
        self._edge_set: Set[Tuple[int,int]] = set()
        self._v2edges: Dict[int, Set[int]] = {}

        # Build once
        self._build_from_scratch()

        # Candidates
        self._build_candidate_masks()

    # ---------------------------
    # Export arrays (same shapes as original)
    # ---------------------------

    @property
    def faces(self) -> np.ndarray:
        fids = self._active_face_ids()
        if fids.size == 0:
            return np.zeros((0,3), dtype=np.int32)
        return np.asarray([self.faces_list[int(fid)] for fid in fids], dtype=np.int32)

    @property
    def face2tet(self) -> np.ndarray:
        fids = self._active_face_ids()
        if fids.size == 0:
            return np.zeros((0,2), dtype=np.int32)
        return np.asarray([self.face2tet_list[int(fid)] for fid in fids], dtype=np.int32)

    @property
    def face_opp(self) -> np.ndarray:
        fids = self._active_face_ids()
        if fids.size == 0:
            return np.zeros((0,2), dtype=np.int32)
        return np.asarray([self.face_opp_list[int(fid)] for fid in fids], dtype=np.int32)

    @property
    def edges(self) -> np.ndarray:
        eids = self._active_edge_ids()
        if eids.size == 0:
            return np.zeros((2,0), dtype=np.int32)
        return np.asarray([self.edges_list[int(eid)] for eid in eids], dtype=np.int32).T

    @property
    def edge2tets(self) -> List[np.ndarray]:
        eids = self._active_edge_ids()
        out: List[np.ndarray] = []
        for eid in eids:
            s = self.edge2tets_list[int(eid)]
            out.append(np.asarray(sorted(list(s)), dtype=np.int32))
        return out

    # ---------------------------
    # Small geo helper (fast)
    # ---------------------------

    def _abs_vol6(self, i: int, j: int, k: int, l: int) -> float:
        P = self.points
        return abs(tet_signed_volume(P[int(i)], P[int(j)], P[int(k)], P[int(l)]))

    # ---------------------------
    # Registry helpers
    # ---------------------------

    def _get_or_create_face(self, fkey: Tuple[int,int,int]) -> int:
        fid = self._face_id.get(fkey)
        if fid is not None:
            return fid
        fid = len(self.faces_list)
        self._face_id[fkey] = fid
        self.faces_list.append(fkey)
        self.face2tet_list.append([-1, -1])
        self.face_opp_list.append([-1, -1])
        self._face_active.append(True)
        self._active_fids.add(fid)
        self._face_set.add(fkey)
        for v in fkey:
            self._v2faces.setdefault(int(v), set()).add(fid)
        return fid

    def _get_or_create_edge(self, ekey: Tuple[int,int]) -> int:
        eid = self._edge_id.get(ekey)
        if eid is not None:
            return eid
        eid = len(self.edges_list)
        self._edge_id[ekey] = eid
        self.edges_list.append(ekey)
        self.edge2tets_list.append(set())
        self._edge_active.append(True)
        self._active_eids.add(eid)
        self._edge_set.add(ekey)
        self._v2edges.setdefault(int(ekey[0]), set()).add(eid)
        self._v2edges.setdefault(int(ekey[1]), set()).add(eid)
        return eid

    def _deactivate_face_if_unused(self, fid: int, fkey: Tuple[int,int,int]):
        if self.face2tet_list[fid][0] == -1 and self.face2tet_list[fid][1] == -1:
            self._face_active[fid] = False
            self._active_fids.discard(fid)
            self._face_id.pop(fkey, None)
            self._face_set.discard(fkey)
            for v in fkey:
                s = self._v2faces.get(int(v))
                if s is not None:
                    s.discard(fid)
                    if len(s) == 0:
                        self._v2faces.pop(int(v), None)
            if fid < len(self.candidate_face_mask):
                self.candidate_face_mask[fid] = False

    def _deactivate_edge_if_unused(self, eid: int, ekey: Tuple[int,int]):
        if len(self.edge2tets_list[eid]) == 0:
            self._edge_active[eid] = False
            self._active_eids.discard(eid)
            self._edge_id.pop(ekey, None)
            self._edge_set.discard(ekey)
            u, v = int(ekey[0]), int(ekey[1])
            su = self._v2edges.get(u)
            if su is not None:
                su.discard(eid)
                if len(su) == 0:
                    self._v2edges.pop(u, None)
            sv = self._v2edges.get(v)
            if sv is not None:
                sv.discard(eid)
                if len(sv) == 0:
                    self._v2edges.pop(v, None)
            if eid < len(self.candidate_edge_mask):
                self.candidate_edge_mask[eid] = False

    def _ensure_candidate_mask_sizes(self):
        F_total = len(self.faces_list)
        E_total = len(self.edges_list)

        if not hasattr(self, "candidate_face_mask"):
            self.candidate_face_mask = np.zeros((F_total,), dtype=bool)
        elif self.candidate_face_mask.shape[0] < F_total:
            add = F_total - self.candidate_face_mask.shape[0]
            self.candidate_face_mask = np.concatenate([self.candidate_face_mask, np.zeros((add,), dtype=bool)])

        if not hasattr(self, "candidate_edge_mask"):
            self.candidate_edge_mask = np.zeros((E_total,), dtype=bool)
        elif self.candidate_edge_mask.shape[0] < E_total:
            add = E_total - self.candidate_edge_mask.shape[0]
            self.candidate_edge_mask = np.concatenate([self.candidate_edge_mask, np.zeros((add,), dtype=bool)])

    # ---------------------------
    # Build once
    # ---------------------------

    def _build_from_scratch(self):
        K = self.tets.shape[0]
        for tid in range(K):
            tet = self.tets[tid]

            for fkey in _tet_faces(tet):
                fid = self._get_or_create_face(fkey)
                opp = _face_opp_vertex(fkey, tet)
                t0, t1 = self.face2tet_list[fid]
                if t0 == -1:
                    self.face2tet_list[fid][0] = tid
                    self.face_opp_list[fid][0] = opp
                elif t1 == -1:
                    self.face2tet_list[fid][1] = tid
                    self.face_opp_list[fid][1] = opp
                else:
                    raise ValueError(f"Non-manifold input: face {fkey} has >2 incident tets")

            for ekey in _tet_edges(tet):
                eid = self._get_or_create_edge(ekey)
                self.edge2tets_list[eid].add(tid)

    # ---------------------------
    # Candidate masks (now degeneracy-safe)
    # ---------------------------

    def _is_face_candidate(self, fid: int) -> bool:
        if fid < 0 or fid >= len(self.faces_list):
            return False
        if not self._face_active[fid]:
            return False

        P = self.points
        eps6 = self.vol_eps6

        t0, t1 = self.face2tet_list[fid]
        if t0 < 0 or t1 < 0:
            return False
        a, b, c = self.faces_list[fid]
        d, e = self.face_opp_list[fid][0], self.face_opp_list[fid][1]

        # new edge must not exist already
        if _sorted_edge(d, e) in self._edge_set:
            return False

        # opposite-side test wrt face (a,b,c)
        s1 = tet_signed_volume(P[a], P[b], P[c], P[d])
        s2 = tet_signed_volume(P[a], P[b], P[c], P[e])
        if s1 == 0.0 or s2 == 0.0 or s1 * s2 >= 0.0:
            return False

        # ensure all 3 new tets are non-degenerate
        if self._abs_vol6(a, b, d, e) <= eps6:
            return False
        if self._abs_vol6(b, c, d, e) <= eps6:
            return False
        if self._abs_vol6(c, a, d, e) <= eps6:
            return False

        return True

    def _is_edge_candidate(self, eid: int) -> bool:
        if eid < 0 or eid >= len(self.edges_list):
            return False
        if not self._edge_active[eid]:
            return False

        P = self.points
        eps6 = self.vol_eps6
        inc = self.edge2tets_list[eid]
        if len(inc) != 3:
            return False
        u, v = self.edges_list[eid]
        link = self._edge_link_vertices(u, v, np.asarray(sorted(list(inc)), dtype=np.int64))
        if link is None:
            return False
        a, b, c = link

        # keep original safety guard
        if _sorted_face(a, b, c) in self._face_set:
            return False

        # opposite-side test for u and v wrt face (a,b,c)
        s1 = tet_signed_volume(P[a], P[b], P[c], P[u])
        s2 = tet_signed_volume(P[a], P[b], P[c], P[v])
        if s1 == 0.0 or s2 == 0.0 or s1 * s2 >= 0.0:
            return False

        # ensure both new tets are non-degenerate
        if self._abs_vol6(a, b, c, u) <= eps6:
            return False
        if self._abs_vol6(a, b, c, v) <= eps6:
            return False

        return True

    def _build_candidate_masks(self):
        self._ensure_candidate_mask_sizes()
        self.candidate_face_mask[:] = False
        self.candidate_edge_mask[:] = False
        for fid in list(self._active_fids):
            self.candidate_face_mask[fid] = self._is_face_candidate(fid)
        for eid in list(self._active_eids):
            self.candidate_edge_mask[eid] = self._is_edge_candidate(eid)

    def _dirty_entities_from_vertices(self, verts: Set[int]) -> Tuple[Set[int], Set[int]]:
        dirty_fids: Set[int] = set()
        dirty_eids: Set[int] = set()
        for v in verts:
            dirty_fids.update(self._v2faces.get(int(v), set()))
            dirty_eids.update(self._v2edges.get(int(v), set()))
        return dirty_fids, dirty_eids

    def _update_candidate_masks_local(self, dirty_fids: Set[int], dirty_eids: Set[int]) -> None:
        self._ensure_candidate_mask_sizes()
        for fid in dirty_fids:
            self.candidate_face_mask[fid] = self._is_face_candidate(fid)
        for eid in dirty_eids:
            self.candidate_edge_mask[eid] = self._is_edge_candidate(eid)

    def rebuild(self):
        self.tet_quality = tet_mean_ratio_quality(self.points, self.tets)
        self._build_candidate_masks()

    # ---------------------------
    # Link triangle
    # ---------------------------

    def _edge_link_vertices(self, u: int, v: int, incident_tets: np.ndarray) -> Optional[Tuple[int, int, int]]:
        others: List[Tuple[int, int]] = []
        link_set = set()
        for tid in incident_tets.tolist():
            tet = set(map(int, self.tets[int(tid)]))
            if u not in tet or v not in tet:
                return None
            tet.remove(u); tet.remove(v)
            if len(tet) != 2:
                return None
            a, b = tuple(tet)
            others.append((a, b))
            link_set.add(a); link_set.add(b)

        if len(link_set) != 3:
            return None
        link = list(link_set)

        adj = {x: set() for x in link}
        for a, b in others:
            if a not in adj or b not in adj:
                return None
            adj[a].add(b); adj[b].add(a)

        if any(len(adj[x]) != 2 for x in link):
            return None

        a0 = link[0]
        a1 = next(iter(adj[a0]))
        a2 = next(iter(adj[a1] - {a0}))
        return (int(a0), int(a1), int(a2))

    # ---------------------------
    # Action interface
    # ---------------------------

    def action_space_size(self) -> int:
        return int(len(self._active_fids) + len(self._active_eids))

    def candidate_mask(self) -> np.ndarray:
        fids = self._active_face_ids()
        eids = self._active_edge_ids()
        return np.concatenate([self.candidate_face_mask[fids], self.candidate_edge_mask[eids]], axis=0)

    def apply_action(self, a: int) -> bool:
        a = int(a)
        fids = self._active_face_ids()
        eids = self._active_edge_ids()
        F = len(fids)
        if a < 0 or a >= F + len(eids):
            return False

        if a < F:
            fid = int(fids[a])
            if not bool(self.candidate_face_mask[fid]):
                return False
            return self._apply_2_3(fid)
        else:
            eid = int(eids[a - F])
            if not bool(self.candidate_edge_mask[eid]):
                return False
            return self._apply_3_2(eid)

    def _active_face_ids(self) -> np.ndarray:
        if len(self._active_fids) == 0:
            return np.zeros((0,), dtype=np.int64)
        return np.asarray(sorted(self._active_fids), dtype=np.int64)

    def _active_edge_ids(self) -> np.ndarray:
        if len(self._active_eids) == 0:
            return np.zeros((0,), dtype=np.int64)
        return np.asarray(sorted(self._active_eids), dtype=np.int64)

    # ---------------------------
    # Local tet add/remove (COMPACT)
    # ---------------------------

    def _add_tet(self, verts4: Tuple[int,int,int,int]) -> int:
        tid = int(self.tets.shape[0])
        self.tets = np.vstack([self.tets, np.asarray(verts4, dtype=np.int64)[None, :]])
        self.tet_quality = np.hstack([self.tet_quality, tet_mean_ratio_quality(self.points, self.tets[tid:tid+1])])

        tet = self.tets[tid]

        for fkey in _tet_faces(tet):
            fid = self._get_or_create_face(fkey)
            opp = _face_opp_vertex(fkey, tet)
            t0, t1 = self.face2tet_list[fid]
            if t0 == -1:
                self.face2tet_list[fid][0] = tid
                self.face_opp_list[fid][0] = opp
            elif t1 == -1:
                self.face2tet_list[fid][1] = tid
                self.face_opp_list[fid][1] = opp
            else:
                raise ValueError(f"Non-manifold: face {fkey} would have 3 incident tets")

        for ekey in _tet_edges(tet):
            eid = self._get_or_create_edge(ekey)
            self.edge2tets_list[eid].add(tid)

        return tid

    def _replace_tid_in_face(self, fid: int, old: int, new: int, tet_new: np.ndarray):
        t0, t1 = self.face2tet_list[fid]
        if t0 == old:
            self.face2tet_list[fid][0] = new
            self.face_opp_list[fid][0] = _face_opp_vertex(self.faces_list[fid], tet_new)
        elif t1 == old:
            self.face2tet_list[fid][1] = new
            self.face_opp_list[fid][1] = _face_opp_vertex(self.faces_list[fid], tet_new)

    def _remove_tet(self, tid: int) -> None:
        K = int(self.tets.shape[0])
        last = K - 1
        if tid < 0 or tid > last:
            return

        tet_del = self.tets[tid].copy()

        # remove references for tet_del
        for fkey in _tet_faces(tet_del):
            fid = self._face_id.get(fkey)
            if fid is None:
                continue
            t0, t1 = self.face2tet_list[fid]
            if t0 == tid:
                self.face2tet_list[fid][0] = t1
                self.face_opp_list[fid][0] = self.face_opp_list[fid][1]
                self.face2tet_list[fid][1] = -1
                self.face_opp_list[fid][1] = -1
            elif t1 == tid:
                self.face2tet_list[fid][1] = -1
                self.face_opp_list[fid][1] = -1
            self._deactivate_face_if_unused(fid, fkey)

        for ekey in _tet_edges(tet_del):
            eid = self._edge_id.get(ekey)
            if eid is None:
                continue
            s = self.edge2tets_list[eid]
            s.discard(tid)
            self._deactivate_edge_if_unused(eid, ekey)

        # if removing last, pop
        if tid == last:
            self.tets = self.tets[:-1]
            self.tet_quality = self.tet_quality[:-1]
            return

        # swap last into tid
        tet_swap = self.tets[last].copy()
        self.tets[tid] = tet_swap
        self.tet_quality[tid] = self.tet_quality[last]

        # update incidence references: last -> tid (local: 4 faces + 6 edges)
        for fkey in _tet_faces(tet_swap):
            fid = self._face_id[fkey]
            t0, t1 = self.face2tet_list[fid]
            if t0 == last or t1 == last:
                self._replace_tid_in_face(fid, old=last, new=tid, tet_new=tet_swap)

        for ekey in _tet_edges(tet_swap):
            eid = self._edge_id[ekey]
            s = self.edge2tets_list[eid]
            if last in s:
                s.remove(last)
                s.add(tid)

        # pop last
        self.tets = self.tets[:-1]
        self.tet_quality = self.tet_quality[:-1]

    # ---------------------------
    # Apply flips (now guaranteed non-degenerate)
    # ---------------------------

    def _apply_2_3(self, fid: int) -> bool:
        t0, t1 = self.face2tet_list[fid]
        if t0 < 0 or t1 < 0:
            return False
        a, b, c = self.faces_list[fid]
        d, e = self.face_opp_list[fid][0], self.face_opp_list[fid][1]

        # FINAL SAFETY CHECK (in case masks are stale)
        eps6 = self.vol_eps6
        if self._abs_vol6(a, b, d, e) <= eps6: 
            return False
        if self._abs_vol6(b, c, d, e) <= eps6:
            return False
        if self._abs_vol6(c, a, d, e) <= eps6:
            return False

        touched = {int(a), int(b), int(c), int(d), int(e)}
        dirty_fids, dirty_eids = self._dirty_entities_from_vertices(touched)

        # remove higher tid first (swap-remove safety)
        if t0 > t1:
            t0, t1 = t1, t0
        self._remove_tet(int(t1))
        self._remove_tet(int(t0))

        self._add_tet((a, b, d, e))
        self._add_tet((b, c, d, e))
        self._add_tet((c, a, d, e))

        dirty_fids_2, dirty_eids_2 = self._dirty_entities_from_vertices(touched)
        dirty_fids.update(dirty_fids_2)
        dirty_eids.update(dirty_eids_2)
        self._update_candidate_masks_local(dirty_fids, dirty_eids)
        return True

    def _apply_3_2(self, eid: int) -> bool:
        inc = sorted(list(self.edge2tets_list[eid]))
        if len(inc) != 3:
            return False
        u, v = self.edges_list[eid]
        link = self._edge_link_vertices(u, v, np.asarray(inc, dtype=np.int64))
        if link is None:
            return False
        a, b, c = link

        # FINAL SAFETY CHECK
        eps6 = self.vol_eps6
        if self._abs_vol6(a, b, c, u) <= eps6:
            return False
        if self._abs_vol6(a, b, c, v) <= eps6:
            return False

        touched = {int(a), int(b), int(c), int(u), int(v)}
        dirty_fids, dirty_eids = self._dirty_entities_from_vertices(touched)

        # remove in descending order (swap-remove safe)
        for tid in sorted(inc, reverse=True):
            self._remove_tet(int(tid))

        self._add_tet((a, b, c, u))
        self._add_tet((a, b, c, v))

        dirty_fids_2, dirty_eids_2 = self._dirty_entities_from_vertices(touched)
        dirty_fids.update(dirty_fids_2)
        dirty_eids.update(dirty_eids_2)
        self._update_candidate_masks_local(dirty_fids, dirty_eids)
        return True
