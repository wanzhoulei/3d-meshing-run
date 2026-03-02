from __future__ import annotations
import h5py
import numpy as np
from typing import List, Tuple, Dict, Any, Optional


class TetMat73Dataset:
    """
    Loader for the MATLAB v7.3 .mat produced by gen_tet_dataset.m.

    Exposes:
      - P: (N,125,3) float64 (stacked)
      - T_good: (K0,4) int32 (0-based)
      - T_bad: list length N, each (Ki,4) int32 (0-based)
      - minQ_bad, minQ_good: (N,) float64
      - params: dict (best-effort)
    """

    def __init__(self, filename: str, load_all: bool = True):
        self.filename = filename
        self.f = h5py.File(filename, "r")

        # fixed good topology (note: h5py reads MATLAB arrays transposed)
        self.T_good = (np.array(self.f["T_good"]).T - 1).astype(np.int32)

        # quality arrays
        self.minQ_bad = np.array(self.f["minQ_bad"]).reshape(-1, order="F").astype(np.float64)
        self.minQ_good = np.array(self.f["minQ_good"]).reshape(-1, order="F").astype(np.float64)

        # cell arrays stored as object refs
        P_cell = self.f["P"]        # likely (1,N)
        T_cell = self.f["T_bad"]    # likely (1,N)
        if P_cell.shape != T_cell.shape:
            raise ValueError(f"Shape mismatch: P{P_cell.shape} vs T_bad{T_cell.shape}")

        # Determine N robustly
        self.num_samples = int(max(P_cell.shape))

        self._P_cell = P_cell
        self._T_cell = T_cell

        self.P: Optional[np.ndarray] = None
        self.T_bad: Optional[List[np.ndarray]] = None

        self.params: Dict[str, Any] = self._read_params()

        if load_all:
            self._load_all()

    def _read_params(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if "params" not in self.f:
            return out
        g = self.f["params"]
        # MATLAB structs in v7.3 are groups; each field is a dataset (often scalar or vector)
        for key in g.keys():
            try:
                out[key] = np.array(g[key]).reshape(-1, order="F")
            except Exception:
                pass
        return out

    def _cell_ref(self, cell_ds, i: int):
        # Works for (1,N) or (N,1)
        if cell_ds.shape[0] == 1:
            return cell_ds[0, i]
        if cell_ds.shape[1] == 1:
            return cell_ds[i, 0]
        # generic fallback (column-major linearization)
        r, c = cell_ds.shape
        rr = i % r
        cc = i // r
        return cell_ds[rr, cc]

    def get(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (P_i, T_bad_i) for sample i. P is (125,3), T is (Ki,4), 0-based."""
        i = int(i)
        if not (0 <= i < self.num_samples):
            raise IndexError(i)

        p_ref = self._cell_ref(self._P_cell, i)
        t_ref = self._cell_ref(self._T_cell, i)

        P = np.array(self.f[p_ref]).T.astype(np.float64)     # (125,3)
        T = (np.array(self.f[t_ref]).T - 1).astype(np.int32) # (Ki,4), 0-based

        return P, T

    def _load_all(self):
        P_list: List[np.ndarray] = []
        T_list: List[np.ndarray] = []
        for i in range(self.num_samples):
            Pi, Ti = self.get(i)
            P_list.append(Pi)
            T_list.append(Ti)
        self.P = np.stack(P_list, axis=0)   # (N,125,3)
        self.T_bad = T_list

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass
