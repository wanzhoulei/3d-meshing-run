import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Assuming utility functions like scatter_add are available in util.py
from util import scatter_add, make_batch_masks

class MLP(nn.Module):
    """
    Standard multi-layer perceptron utility.
    """
    def __init__(self, in_dim, hid_dim, out_dim, num_layers=2, act=nn.SiLU, dropout=0.0):
        super().__init__()
        layers = []
        dims = [in_dim] + [hid_dim] * (num_layers - 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(act())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class HeteroEdgeFaceLayer(nn.Module):
    """
    An E(n) Equivariant Layer for 3D Mesh Graphs with Face and Edge nodes.
    Flow: 
    1. Edge + Face features -> Updated Edge features
    2. Edge + Face features -> Updated Face features
    3. Geometric updates (Coordinates)
    """
    def __init__(self, d_f, d_e, hid_dim, dropout=0.0):
        super().__init__()
        # Message from Face to Edge: Phi_f2e(h_f, h_e, dist)
        self.phi_f2e = MLP(d_f + d_e + 1, hid_dim, d_e, num_layers=2, dropout=dropout)
        
        # Message from Edge to Face: Phi_e2f(h_e, h_f, dist)
        self.phi_e2f = MLP(d_e + d_f + 1, hid_dim, d_f, num_layers=2, dropout=dropout)
        
        # Coordinate update (3D)
        self.phi_x = MLP(d_e, hid_dim, 1, num_layers=2, dropout=dropout)
        
        # Final Node Updates
        self.phi_h_f = MLP(d_f * 2, hid_dim, d_f, num_layers=2, dropout=dropout)
        self.phi_h_e = MLP(d_e * 2, hid_dim, d_e, num_layers=2, dropout=dropout)

    def forward(self, pos_f, pos_e, h_f, h_e, f2e_index, e2f_index, f_mask=None, e_mask=None):
        """
        pos_f/e: (B, N_f/e, 3)
        h_f/e:   (B, N_f/e, d_f/e)
        f2e_index: (B, 2, num_edges_in_faces) - maps which faces connect to which edges
        """
        B, Nf, _ = h_f.shape
        _, Ne, _ = h_e.shape
        
        # 1. Edge Update (Receive info from incident faces)
        idx_f, idx_e = f2e_index[:, 0, :], f2e_index[:, 1, :]
        batch_idx = torch.arange(B, device=h_f.device).view(B, 1).expand_as(idx_f)
        
        dist_fe = torch.norm(pos_f[batch_idx, idx_f] - pos_e[batch_idx, idx_e], dim=-1, keepdim=True)
        m_f2e_input = torch.cat([h_f[batch_idx, idx_f], h_e[batch_idx, idx_e], dist_fe], dim=-1)
        m_f2e = self.phi_f2e(m_f2e_input)
        
        aggr_f2e = scatter_add(m_f2e, idx_e, Ne)
        h_e_next = h_e + self.phi_h_e(torch.cat([h_e, aggr_f2e], dim=-1))

        # 2. Face Update (Receive info from boundary edges)
        idx_e_rev, idx_f_rev = e2f_index[:, 0, :], e2f_index[:, 1, :]
        dist_ef = torch.norm(pos_e[batch_idx, idx_e_rev] - pos_f[batch_idx, idx_f_rev], dim=-1, keepdim=True)
        m_e2f_input = torch.cat([h_e[batch_idx, idx_e_rev], h_f[batch_idx, idx_f_rev], dist_ef], dim=-1)
        m_e2f = self.phi_e2f(m_e2f_input)
        
        aggr_e2f = scatter_add(m_e2f, idx_f_rev, Nf)
        h_f_next = h_f + self.phi_h_f(torch.cat([h_f, aggr_e2f], dim=-1))

        # 3. Coordinate update (based on edge states)
        # We move edge positions relative to face centers
        rel_pos = pos_e[batch_idx, idx_e] - pos_f[batch_idx, idx_f]
        trans = rel_pos * self.phi_x(m_f2e)
        pos_e_next = pos_e + scatter_add(trans, idx_e, Ne)

        if f_mask is not None: h_f_next *= f_mask.unsqueeze(-1)
        if e_mask is not None: h_e_next *= e_mask.unsqueeze(-1)

        return pos_f, pos_e_next, h_f_next, h_e_next

class MeshActor3D(nn.Module):
    """
    3D Actor: Processes Face and Edge nodes to output a unified action distribution.
    """
    def __init__(self, d_f, d_e, L, hid_dim=64):
        super().__init__()
        self.layers = nn.ModuleList([
            HeteroEdgeFaceLayer(d_f, d_e, hid_dim) for _ in range(L)
        ])
        
        self.face_head = MLP(d_f, hid_dim, 1, num_layers=3)
        self.edge_head = MLP(d_e, hid_dim, 1, num_layers=3)

    def forward(self, pos_f, pos_e, h_f, h_e, f2e_index, e2f_index, cand_mask_f, cand_mask_e):
        # Message Passing
        for layer in self.layers:
            pos_f, pos_e, h_f, h_e = layer(pos_f, pos_e, h_f, h_e, f2e_index, e2f_index)
        
        # Compute Logits
        f_logits = self.face_head(h_f).squeeze(-1) # (B, Nf)
        e_logits = self.edge_head(h_e).squeeze(-1) # (B, Ne)
        
        # Combined logits for total action space
        logits = torch.cat([f_logits, e_logits], dim=1)
        full_mask = torch.cat([cand_mask_f, cand_mask_e], dim=1)
        
        neg_inf = torch.finfo(logits.dtype).min
        logits = torch.where(full_mask, logits, torch.full_like(logits, neg_inf))
        
        return logits, h_f, h_e

class MeshCritic3D(nn.Module):
    """
    3D Critic: Global pooling of both Face and Edge features to estimate state value V.
    """
    def __init__(self, d_f, d_e, hid_dim):
        super().__init__()
        self.phi_f = MLP(d_f, hid_dim, hid_dim)
        self.phi_e = MLP(d_e, hid_dim, hid_dim)
        self.v_head = MLP(hid_dim, hid_dim, 1, num_layers=3)

    def forward(self, h_f, h_e, f_mask, e_mask):
        # Map features to common space
        zf = self.phi_f(h_f) * f_mask.unsqueeze(-1)
        ze = self.phi_e(h_e) * e_mask.unsqueeze(-1)
        
        # Global Aggregate (Sum pooling)
        g = zf.sum(dim=1) + ze.sum(dim=1)
        return self.v_head(g).squeeze(-1)