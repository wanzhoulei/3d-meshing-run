from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable


# ============================================================
# Geometry / quality
# ============================================================

def tet_signed_volume(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    """
    Signed volume *6 of tetra (a,b,c,d): det([b-a, c-a, d-a]).

    Arguments:
        a: np.ndarray, coordinate of 1st node 
        b: np.ndarray, coordinate of 2nd node
        c: np.ndarray, coordinate of 3rd node
        d: np.ndarray, coordinate of 4th node

    Returns:
        float, -6 * volumn of this tet
    """
    return float(np.linalg.det(np.stack([b - a, c - a, d - a], axis=1)))

def tet_volume(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    """The volumn of the tet"""
    return abs(tet_signed_volume(a, b, c, d)) / 6.0

def tet_mean_ratio_quality(P: np.ndarray, tets: np.ndarray, eps: float = 1e-18) -> np.ndarray:
    """
    Mean-ratio quality in [0,1] (1 for equilateral tetra), computed as:
        q = 12 * (3*V)^(2/3) / sum_{edges} ||e||^2
    where V is tetra volume. For equilateral with edge length 1, q==1.
    It computes the mean ratio quality of all tets in this graph.

    Arguments:
        P:    (N,3) float
        tets: (K,4) int
    Returns:
        q: (K,) float
    """
    tets = np.asarray(tets, dtype=np.int64)
    a = P[tets[:, 0]]
    b = P[tets[:, 1]]
    c = P[tets[:, 2]]
    d = P[tets[:, 3]]

    # volumes
    # det([b-a, c-a, d-a]) gives signed 6V
    det6 = np.einsum('bi,bi->b', np.cross(b - a, c - a), d - a)  # scalar triple
    V = np.abs(det6) / 6.0

    # squared edge lengths (6 edges)
    def sq(x): return np.einsum('bi,bi->b', x, x)

    l2 = (
        sq(b - a) + sq(c - a) + sq(d - a) +
        sq(c - b) + sq(d - b) + sq(d - c)
    )

    q = 12.0 * np.power(3.0 * np.maximum(V, 0.0), 2.0 / 3.0) / np.maximum(l2, eps)
    # numerically clamp
    return np.clip(q, 0.0, 1.0)

def segment_intersects_triangle_interior(pd, pe, pa, pb, pc, eps=1e-12):
    """
    Return True iff segment pd->pe intersects triangle (pa,pb,pc)
    at a point strictly inside the triangle (not on edges/vertices)
    AND strictly inside the segment (not at endpoints).

    Uses Möller-Trumbore ray-triangle intersection adapted to a segment,
    with strict barycentric constraints.
    """
    d = pe - pd  # segment direction
    e1 = pb - pa
    e2 = pc - pa

    pvec = np.cross(d, e2)
    det = float(np.dot(e1, pvec))

    # Parallel or nearly parallel => no proper single-point intersection
    if abs(det) < eps:
        return False

    inv_det = 1.0 / det
    tvec = pd - pa
    u = float(np.dot(tvec, pvec)) * inv_det

    # strict interior: u in (0,1)
    if u <= eps or u >= 1.0 - eps:
        return False

    qvec = np.cross(tvec, e1)
    v = float(np.dot(d, qvec)) * inv_det

    # strict interior: v in (0,1) and u+v < 1
    if v <= eps or v >= 1.0 - eps:
        return False
    if u + v >= 1.0 - eps:
        return False

    # Solve for segment parameter t along pd + t*d. For segment, require t in (0,1).
    t = float(np.dot(e2, qvec)) * inv_det
    if t <= eps or t >= 1.0 - eps:
        return False

    return True


# ============================================================
# Topology helpers
# ============================================================

def _sorted_face(i: int, j: int, k: int) -> Tuple[int, int, int]:
    """
    Sort the face indices in increasing order
    Returns a tuple of face indices in increasing order
    """
    a, b, c = int(i), int(j), int(k)
    if a > b: a, b = b, a
    if b > c: b, c = c, b
    if a > b: a, b = b, a
    return (a, b, c)

def _sorted_edge(i: int, j: int) -> Tuple[int, int]:
    """
    Sort the edge indices in increasing order
    Returns a tuple of edge indices in increasing order
    """
    a, b = int(i), int(j)
    return (a, b) if a < b else (b, a)

def _tet_faces(t: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    Return the 4 faces (as sorted triples) of a tet vertex array length 4.

    Arguments:
        t: np.ndarray, indices array of length 4 for a tet 
    
    Returns:
        a list of 4 triples in increasing order of all four faces of the tet
    """
    i, j, k, l = map(int, t)
    return [
        _sorted_face(j, k, l),  # opposite i
        _sorted_face(i, k, l),  # opposite j
        _sorted_face(i, j, l),  # opposite k
        _sorted_face(i, j, k),  # opposite l
    ]

def _tet_edges(t: np.ndarray) -> List[Tuple[int, int]]:
    """
    Returns a list of tuples of edge nodes 

    Arguments: 
        t: np.ndarray, indices array of length 4 for a tet 
    Returns:
        a list of 6 tuples of indices in increasing oder of all 6 edges of the tet
    """
    i, j, k, l = map(int, t)
    return [
        _sorted_edge(i, j), _sorted_edge(i, k), _sorted_edge(i, l),
        _sorted_edge(j, k), _sorted_edge(j, l), _sorted_edge(k, l),
    ]


# ============================================================
# Tet mesh topology (2-3 and 3-2 flips)
# ============================================================

from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class BuildResult:
    """
    Adjacency + incidence structures derived from a tetrahedral mesh.

    Given:
      - points P: (N,3) float
      - tets   T: (K,4) int (vertex indices)

    This struct stores *canonicalized* faces/edges and their incident tetrahedra,
    which are the basic primitives needed to:
      - identify boundary vs interior elements,
      - query local neighborhoods,
      - validate and execute local bistellar flips (2-3 and 3-2).

    Fields
    ------
    faces : np.ndarray, shape (F, 3), dtype int32
        The list of unique triangular faces in the mesh.
        Each row is a sorted vertex triple (a,b,c) with a<b<c.
        Faces are canonicalized by sorting so the same geometric face has
        a unique key independent of tetra orientation.

        Interpretation:
          faces[f] = (a,b,c) is a triangle that appears as a face of one or
          two tetrahedra.

    face2tet : np.ndarray, shape (F, 2), dtype int32, stores tet indices
        For each face f, the indices of up to two incident tetrahedra.

          face2tet[f,0] = t0  (always >= 0)
          face2tet[f,1] = t1  (>=0 if interior face, -1 if boundary face)

        Boundary test:
          face2tet[f,1] == -1  ⇔  face is on the boundary.

        This is used for 2-3 flips:
          a valid 2-3 flip requires an interior face, i.e. two incident tets.

    face_opp : np.ndarray, shape (F, 2), dtype int32
        For each face f and each incident tet in face2tet, store the “opposite”
        vertex of that tetrahedron across the face.

        If faces[f] = (a,b,c) and face2tet[f,0] = t0, then:
          face_opp[f,0] = d
        where d is the 4th vertex in tet t0 such that tet vertices are {a,b,c,d}.

        Similarly, if face2tet[f,1] = t1 (interior face), then:
          face_opp[f,1] = e
        where e is the 4th vertex in tet t1 such that tet vertices are {a,b,c,e}.

        If the face is boundary (face2tet[f,1] == -1), then:
          face_opp[f,1] == -1.

    edges : np.ndarray, shape (2, E), dtype int32
        The list of unique edges in the mesh.
        Each column is a sorted vertex pair (u,v) with u < v.

        Interpretation:
          edges[:,e] = (u,v) is an edge that appears in at least one tetrahedron.

    edge2tets : List[np.ndarray], length E, stores indicies of tets 
        For each edge e, a 1D int32 array listing the indices of all tetrahedra
        incident to that edge.

          edge2tets[e] = [t0, t1, ..., t_(deg-1)]

        The length of edge2tets[e] is the edge valence/degree (deg).

        This is used for 3-2 flips:
          a candidate 3-2 flip requires an interior edge with deg == 3
          (exactly 3 tetrahedra around the edge).
    """

    faces: np.ndarray
    face2tet: np.ndarray
    face_opp: np.ndarray
    edges: np.ndarray
    edge2tets: List[np.ndarray]

class TetMeshTopology:
    """
    A simple (correct-first) tetra mesh topology manager for RL flips.
    Compared to your 2D implementation, this version rebuilds adjacency after each flip.
    That keeps the implementation short and robust for a toy dataset.

    Supported local moves (fixed vertices):
      - 2-3 flip: internal face shared by 2 tets  -> replace with 3 tets around new edge
      - 3-2 flip: internal edge with valence 3    -> replace with 2 tets around new face

    Observation-facing public fields (NumPy):
      points: (N,3) float64, node coordinates 
      tets  : (K,4) int32, tet node idices list 

      faces : (F,3) int32  sorted vertex ids of every face
      face2tet: (F,2) int32 (t0,t1 or (t,-1) on boundary), tet indices 
      face_opp: (F,2) int32 opposite vertices (d,e) corresponding to face2tet

      edges : (2,E) int32 (sorted endpoints)
      edge2tets: list of length E with int32 arrays of incident tet ids

      candidate_face_mask: (F,) bool  valid 2-3
      candidate_edge_mask: (E,) bool  valid 3-2 (valence==3)

      tet_quality: (K,) float64
    """

    def __init__(self, points: np.ndarray, tets: np.ndarray):
        self.points = np.asarray(points, dtype=np.float64) #geometry
        self.tets = np.asarray(tets, dtype=np.int32) #topology

        self.faces = np.zeros((0, 3), dtype=np.int32)
        self.face2tet = np.zeros((0, 2), dtype=np.int32)
        self.face_opp = np.zeros((0, 2), dtype=np.int32)

        self.edges = np.zeros((2, 0), dtype=np.int32)
        self.edge2tets: List[np.ndarray] = []

        self.candidate_face_mask = np.zeros((0,), dtype=bool)
        self.candidate_edge_mask = np.zeros((0,), dtype=bool)

        self.tet_quality = np.zeros((self.tets.shape[0],), dtype=np.float64)

        self.rebuild()

    # ---------------------------
    # Build adjacency + masks
    # ---------------------------

    def rebuild(self):
        """Rebuild everything"""
        self._build_faces_edges()
        self.tet_quality = tet_mean_ratio_quality(self.points, self.tets)
        self._build_candidate_masks()

    def _build_faces_edges(self):
        """
        builds the following: 
        self.faces, self.face2tet, self.face_opp
        self.edges, self.edge2tets
        """

        K = self.tets.shape[0] #number of tets in this graph

        #initialize a map that goes from Triple of face nodes -> tuple of incident tet indices
        face_map: Dict[Tuple[int, int, int], List[Tuple[int, int]]] = {}

        # store (tet_id, opposite_vertex)
        for tid in range(K):
            tet = self.tets[tid]
            # faces with opposites: face idx 0 is opposite tet[0], etc (as in _tet_faces)
            faces = _tet_faces(tet)
            opps = [int(tet[0]), int(tet[1]), int(tet[2]), int(tet[3])]
            # careful: our _tet_faces order is opposite [i,j,k,l] respectively, matching above
            for fkey, opp in zip(faces, opps):
                face_map.setdefault(fkey, []).append((tid, opp))

        F = len(face_map)
        faces = np.zeros((F, 3), dtype=np.int32)
        face2tet = np.full((F, 2), -1, dtype=np.int32)
        face_opp = np.full((F, 2), -1, dtype=np.int32)

        for idx, (fkey, inc) in enumerate(face_map.items()):
            faces[idx, :] = np.array(fkey, dtype=np.int32)
            # inc is list of (tid, opp)
            if len(inc) >= 1:
                face2tet[idx, 0] = int(inc[0][0])
                face_opp[idx, 0] = int(inc[0][1])
            if len(inc) >= 2:
                face2tet[idx, 1] = int(inc[1][0])
                face_opp[idx, 1] = int(inc[1][1])
            # if >2, non-manifold; keep first two

        # edges
        edge_map: Dict[Tuple[int, int], List[int]] = {}
        for tid in range(K):
            tet = self.tets[tid]
            for ekey in _tet_edges(tet):
                edge_map.setdefault(ekey, []).append(tid)

        E = len(edge_map)
        edges = np.zeros((2, E), dtype=np.int32)
        edge2tets: List[np.ndarray] = [None] * E  # type: ignore

        for idx, (ekey, inc) in enumerate(edge_map.items()):
            edges[0, idx], edges[1, idx] = int(ekey[0]), int(ekey[1])
            edge2tets[idx] = np.asarray(inc, dtype=np.int32)

        self.faces, self.face2tet, self.face_opp = faces, face2tet, face_opp
        self.edges, self.edge2tets = edges, edge2tets

        # Build fast lookups for validity checks
        self._edge_set = set((_sorted_edge(int(edges[0, i]), int(edges[1, i]))) for i in range(E))
        self._face_set = set((_sorted_face(int(faces[i, 0]), int(faces[i, 1]), int(faces[i, 2]))) for i in range(F))

    def _build_candidate_masks(self):
        """
        build two validity masks: candidate_face_mask AND candidate_edge_mask
        """
        # 2-3: internal face with exactly two incident tets,
        # and new edge between opposites doesn't already exist,
        # and opposites are on opposite sides of face plane.
        F = self.faces.shape[0] #number of faces 
        cand_f = np.zeros((F,), dtype=bool) #initial all zero candidate list
        P = self.points

        """
        For a face to be valid for 2-3 flip:
        1. It cannot be a boundary face 
        2. If you connect d e, the opposite two nodes of the face, it will intersect with the face 
            in the interior (not on the face boundary)
        """
        for fid in range(F):  # for every face
            t0, t1 = int(self.face2tet[fid, 0]), int(self.face2tet[fid, 1])
            if t0 < 0 or t1 < 0:
                continue  # (1) boundary face => not valid

            a, b, c = map(int, self.faces[fid])
            d, e = int(self.face_opp[fid, 0]), int(self.face_opp[fid, 1])

            pa, pb, pc = P[a], P[b], P[c]
            pd, pe = P[d], P[e]

            # (2) require segment de intersects the face triangle strictly in its interior
            if not segment_intersects_triangle_interior(pd, pe, pa, pb, pc, eps=1e-12):
                continue

            cand_f[fid] = True

        """
        An edge is valid for 3-2 flip iif:
            1. It is an internal edge 
            2. Has exactly 3 valent tets
            3. Its link must have 3 vertices (the two be face is valid)
            4. opposite-side test for u,v wrt face (a,b,c)
            * 2, 3 ensures 1
        """
        E = self.edges.shape[1]  # number of edges
        cand_e = np.zeros((E,), dtype=bool)
        for eid in range(E):
            inc = self.edge2tets[eid]          # incident tet id list/array
            if inc.size != 3:
                continue                       # (1) valence must be exactly 3
            u = int(self.edges[0, eid])
            v = int(self.edges[1, eid])
            link = self._edge_link_vertices(u, v, inc)  # should represent the link face vertices
            if link is None:
                continue
            a, b, c = map(int, link)

            #opposite-side test for u and v wrt face (a,b,c)
            va = self.points[a]; vb = self.points[b]; vc = self.points[c]
            vu = self.points[u]; vv = self.points[v]

            s1 = tet_signed_volume(va, vb, vc, vu)
            s2 = tet_signed_volume(va, vb, vc, vv)

            if s1 == 0.0 or s2 == 0.0:
                continue
            if s1 * s2 >= 0.0:
                continue

            cand_e[eid] = True


        self.candidate_face_mask = cand_f
        self.candidate_edge_mask = cand_e

    def _edge_link_vertices(self, u: int, v: int, incident_tets: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """
        For an edge (u,v) with 3 incident tets, recover the 3 link vertices (a,b,c)
        such that the incident tets are (u,v,a,b), (u,v,b,c), (u,v,c,a) (up to order).
        Returns (a,b,c) in some cyclic order, or None if structure is not consistent.
        """
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
        # Build adjacency among link vertices: each tet contributes an edge between its two "other" vertices
        adj = {x: set() for x in link}
        for a, b in others:
            if a not in adj or b not in adj:
                return None
            adj[a].add(b); adj[b].add(a)

        # For a triangle, each vertex should have degree 2
        if any(len(adj[x]) != 2 for x in link):
            return None

        # Return a cycle order: pick arbitrary start and walk
        a0 = link[0]
        a1 = next(iter(adj[a0]))
        a2 = next(iter(adj[a1] - {a0}))
        return (int(a0), int(a1), int(a2))

    # ---------------------------
    # Action interface
    # ---------------------------

    def action_space_size(self) -> int:
        return int(self.faces.shape[0] + self.edges.shape[1])

    def candidate_mask(self) -> np.ndarray:
        """Concatenated (F+E,) mask for [faces then edges]."""
        return np.concatenate([self.candidate_face_mask, self.candidate_edge_mask], axis=0)

    # ---------------------------
    # Apply flips
    # ---------------------------

    def apply_action(self, a: int) -> bool:
        """
        Apply an action in [0, F+E).
        Returns:
            True if a valid move was applied, False otherwise (no-op).
        """
        a = int(a)
        F = self.faces.shape[0]
        if a < 0 or a >= F + self.edges.shape[1]:
            return False

        if a < F: #2-3 flip
            fid = a
            if not bool(self.candidate_face_mask[fid]):
                return False
            self._apply_2_3(fid)
            return True
        else:
            eid = a - F
            if not bool(self.candidate_edge_mask[eid]):
                return False
            self._apply_3_2(eid)
            return True

    def _apply_2_3(self, fid: int):
        """
        2-3 flip on internal face (a,b,c) shared by tets (a,b,c,d) and (a,b,c,e).
        Replace with three tets:
            (a,b,d,e), (b,c,d,e), (c,a,d,e)
        """
        t0, t1 = int(self.face2tet[fid, 0]), int(self.face2tet[fid, 1]) #two incident tets
        a, b, c = map(int, self.faces[fid]) 
        d, e = int(self.face_opp[fid, 0]), int(self.face_opp[fid, 1])

        # Remove t0 and t1, add 3 new
        keep = np.ones((self.tets.shape[0],), dtype=bool)
        keep[t0] = False
        keep[t1] = False
        kept = self.tets[keep]

        new_tets = np.asarray([
            [a, b, d, e],
            [b, c, d, e],
            [c, a, d, e],
        ], dtype=np.int32)

        self.tets = np.concatenate([kept, new_tets], axis=0)
        #is it too inefficient?
        #this is actually local change but the code is doing global rebuild
        self.rebuild()

    def _apply_3_2(self, eid: int):
        """
        3-2 flip on internal edge (u,v) with exactly 3 incident tets.
        Let link vertices be (a,b,c). Replace the 3 tets with 2:
            (a,b,c,u) and (a,b,c,v)
        """
        u, v = int(self.edges[0, eid]), int(self.edges[1, eid])
        inc = self.edge2tets[eid]
        link = self._edge_link_vertices(u, v, inc)
        if link is None:
            # should not happen if candidate masks were correct; do nothing
            return
        a, b, c = link

        # Remove the 3 incident tets
        keep = np.ones((self.tets.shape[0],), dtype=bool)
        keep[inc] = False
        kept = self.tets[keep]

        new_tets = np.asarray([
            [a, b, c, u],
            [a, b, c, v],
        ], dtype=np.int32)

        self.tets = np.concatenate([kept, new_tets], axis=0)
        self.rebuild()
