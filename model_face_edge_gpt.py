
"""
model_3d.py

Actor-Critic networks for 3D tetrahedral mesh *topology* refinement with PPO.

You described a **bipartite incidence graph** whose nodes are:
  - EDGE nodes: one per mesh edge (topological edge in the tetra-complex)
  - FACE nodes: one per triangular face

Connectivity constraints:
  - A FACE node connects only to its 3 incident EDGE nodes.
  - An EDGE node connects only to its 1-2 incident FACE nodes (1 if boundary).

We implement an E(n)-equivariant backbone (EGNN-style message passing) on this
bipartite graph. Each node carries:
  - coordinates x in R^3 (used for equivariance)
  - a feature vector h (learned embeddings)

Typical coordinate choices (you supply these from the environment):
  - Edge node position: midpoint of its two vertices
  - Face node position: centroid of its three vertices

Policy (actor):
  - After L equivariant layers, we obtain final embeddings for all nodes.
  - Two separate scalar heads:
        edge_head(h_edge^L) -> logit per edge action
        face_head(h_face^L) -> logit per face action
  - Apply validity masks (candidate masks) to set invalid logits to a large negative value.
  - Concatenate edge+face logits into one flat action-logit vector.

Value (critic):
  - Type-aware global pooling:
        g_edge = masked_mean( proj_edge(h_edge^L) )
        g_face = masked_mean( proj_face(h_face^L) )
        g = concat(g_edge, g_face)
  - value = value_mlp(g) -> scalar per batch element

Batching:
  - This implementation supports padding:
        x:         (B, N_padded, 3)
        h:         (B, N_padded, d_h)
        node_mask: (B, N_padded)          True for real nodes, False for padded
        edge_index:(B, 2, M_padded)       incidence edges, padded
        edge_mask: (B, M_padded)          True for real incidence edges, False for padded
    where N_padded = E_max + F_max is padded total nodes per batch element and
    nodes are ordered as:
        [edge nodes..., face nodes...]

  - You also pass:
        edge_node_mask: (B, E_max)  True for real edge nodes
        face_node_mask: (B, F_max)  True for real face nodes
    These are used to slice/pool/type heads.

  - Candidate/action masks:
        edge_action_mask: (B, E_max)  True if the edge action is valid
        face_action_mask: (B, F_max)  True if the face action is valid

Notes:
  - This file is self-contained (no dependency on your 2D util.py).
  - It is intentionally conservative and well-documented so you can edit it
    as your action semantics solidify (e.g., “flip an edge”, “2-3 flip”, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor], dim: int) -> torch.Tensor:
    """
    Compute mean along `dim` with a boolean mask.

    Args:
        x:    Tensor (..., D, ...)
        mask: Boolean mask broadcastable to x with singleton in feature dims.
              Common shapes:
                - (B, N) for node-wise, use mask.unsqueeze(-1)
        dim:  Reduction dimension

    Returns:
        Tensor with `dim` reduced.
    """
    if mask is None:
        return x.mean(dim=dim)
    # Ensure mask is float and broadcastable
    m = mask.to(dtype=x.dtype, device=x.device)
    while m.dim() < x.dim():
        m = m.unsqueeze(-1)
    x = x * m
    denom = m.sum(dim=dim).clamp_min(1.0)
    return x.sum(dim=dim) / denom


class MLP(nn.Module):
    """
    Simple MLP helper.

    Uses [Linear -> Act -> Dropout] blocks for hidden layers, and a final Linear.
    """

    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        num_layers: int = 2,
        act: type[nn.Module] = nn.SiLU,
        dropout: float = 0.0,
        layernorm: bool = False,
    ) -> None:
        """
        MLP constructor

        Arguments:
            in_dim: int, the input layer dim
            hid_dim: int, the hidden layer dim
            out_dim: int, the output layer dim
            num_layers: int, number of hidden layers
            act: activation function,default silu
            dropout: dropout rate added to dropout after each non-final layer 
            layernorm: whether to add layernorm after each non-final layer, default false
        """
        super().__init__()
        dims = [in_dim] + [hid_dim] * num_layers + [out_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i + 1 != len(dims) - 1:
                if layernorm:
                    layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(act())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------------------------------------------------------
# E(n)-equivariant backbone (EGNN-style)
# -----------------------------------------------------------------------------

class EGNNLayer(nn.Module):
    """
    A single EGNN-style layer (Satorras et al.) adapted to *padded batch* graphs.

    For each directed edge (i -> j), compute:
        d_ij = ||x_i - x_j||^2     Distance between each node
        m_ij = phi_m([h_i, h_j, d_ij]) <- phi_m is a NN, message sending from j to i
    Aggregate to nodes:
        m_i = sum_{j in N(i)} m_ij       total mesage received from all neighboring edges

    Update node features:
        h_i <- phi_h([h_i, m_i]) <- phi_h is another NN

    Optional coordinate update:
        x_i <- x_i + (1/|N(i)|) * sum_j (x_i - x_j) * phi_x(m_ij) <- update coordinate by forces acting on it
            through all neighboring edges
    which preserves E(n)-equivariance (translation/rotation) when phi_x outputs scalars.

    Shapes (padded batch):
        x:         (B, N, 3), padded batched node positions for both faces and edges N = Emax + Fmax
        h:         (B, N, d_h), padded batched node features for both faces and edges 
        edge_index:(B, 2, M)   directed edges (src, dst), it is also padded to make M the largest edge num in batch
        node_mask: (B, N),     True for real nodes, False for padded nodes 
        edge_mask: (B, M),     True for real edges, False for padded edges

    We assume `edge_index[b, :, m]` is valid only when edge_mask[b, m] is True
    """
    def __init__(
        self,
        d_h: int, #embedding dimension for both edge and face nodes 
        msg_hidden: int = 128, #hidden message dimension 
        use_coord_update: bool = True, #whether to use coordinate update in each message passing 
        dropout: float = 0.0,
        msgMLP_depth = 2, #the number of hidden layers in the message MLP
        fMLP_depth = 2, #number of hidden layers in feature updating MLP
        crdMLP_depth = 2 #number of hidden layers in coordinate updating MLP
    ) -> None:
        super().__init__()
        self.use_coord_update = use_coord_update

        # message: (h_i, h_j, d_ij) -> m_ij in R^{d_h}
        self.phi_m = MLP(2 * d_h + 1, msg_hidden, d_h, num_layers=msgMLP_depth, dropout=dropout, act=nn.SiLU)

        # feature update: (h_i, m_i) -> delta_h
        self.phi_h = MLP(2 * d_h, msg_hidden, d_h, num_layers=fMLP_depth, dropout=dropout, act=nn.SiLU)

        # coordinate scalar: m_ij -> s_ij in R (per-edge scalar)
        self.phi_x = MLP(d_h, msg_hidden, 1, num_layers=crdMLP_depth, dropout=dropout, act=nn.SiLU)

        self.ln = nn.LayerNorm(d_h)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
        edge_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward Method
        
        Arguments:
            x: tensor, shape (B, N_padded, 3), padded batched node coordinates 
            h: tensor, shape (B, N_padded, d_h) padded batched node initial embedding
            edge_index: shape(B, 2, M_padded)   incidence edges, padded, both directions 
            node_mask: shape (B, N_padded), whether the node is real or padded node
            edge_mask: (B, M),     True for real edges, False for padded edges

        Returns:
            Tuple(tensor, tensor): tuple of output x coordinates and final feature embedding 
                ((B, N_padded, 3), (B, N_padded, d_h))
        """

        B, N, _ = x.shape
        _, _, M = edge_index.shape

        if node_mask is None:
            node_mask = torch.ones(B, N, dtype=torch.bool, device=x.device)
        if edge_mask is None:
            edge_mask = torch.ones(B, M, dtype=torch.bool, device=x.device)

        src = edge_index[:, 0, :]  # (B, M), source node indices for all edges, padded
        dst = edge_index[:, 1, :]  # (B, M), dst node indices for all edges, padded

        # Gather per-edge (x_i, x_j, h_i, h_j)
        # Use torch.gather with expanded indices
        def batch_gather_nodes(t: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
            # t: (B, N, D), idx: (B, M) -> (B, M, D)
            B_, N_, D_ = t.shape
            idx_exp = idx.unsqueeze(-1).expand(B_, idx.shape[1], D_)
            return t.gather(dim=1, index=idx_exp)

        x_i = batch_gather_nodes(x, src) # (B, M, 3) source node coords
        x_j = batch_gather_nodes(x, dst) # (B, M, 3) dst node coords
        h_i = batch_gather_nodes(h, src) # (B, M, d_h) source node initial feature
        h_j = batch_gather_nodes(h, dst) # (B, M, d_h) dst node initial feature 

        # Squared distances, shape (B, M, 1)
        d_ij = ((x_i - x_j) ** 2).sum(dim=-1, keepdim=True) 

        # Message (B, M, d_h)
        m_ij = self.phi_m(torch.cat([h_i, h_j, d_ij], dim=-1))

        # Zero out masked edges
        m_edge = edge_mask.to(dtype=m_ij.dtype, device=m_ij.device).unsqueeze(-1)
        m_ij = m_ij * m_edge

        # Aggregate messages to destination nodes: m_i = sum_{j} m_{j->i}  (i.e., dst aggregation)
        m_i = torch.zeros(B, N, m_ij.shape[-1], device=m_ij.device, dtype=m_ij.dtype)
        # scatter-add along node dimension
        dst_exp = dst.unsqueeze(-1).expand_as(m_ij)  # (B, M, d_h)
        m_i.scatter_add_(dim=1, index=dst_exp, src=m_ij)

        # Feature update
        h_next = self.phi_h(torch.cat([h, m_i], dim=-1))
        h_next = self.ln(h + h_next)  # residual + norm

        # Mask padded nodes
        h_next = h_next * node_mask.to(dtype=h_next.dtype, device=h_next.device).unsqueeze(-1)

        # Optional coordinate update
        x_next = x
        if self.use_coord_update:
            s_ij = self.phi_x(m_ij)  # (B, M, 1)
            delta = (x_i - x_j) * s_ij  # (B, M, 3)
            delta = delta * m_edge  # keep masking consistent

            # Aggregate to destination nodes
            dx = torch.zeros(B, N, 3, device=x.device, dtype=x.dtype)
            dst_exp3 = dst.unsqueeze(-1).expand_as(delta)
            dx.scatter_add_(dim=1, index=dst_exp3, src=delta)

            # Normalize by degree (masked)
            deg = torch.zeros(B, N, 1, device=x.device, dtype=x.dtype)
            one = torch.ones(B, M, 1, device=x.device, dtype=x.dtype) * m_edge
            deg.scatter_add_(dim=1, index=dst.unsqueeze(-1), src=one)
            dx = dx / deg.clamp_min(1.0)

            x_next = x + dx
            x_next = x_next * node_mask.to(dtype=x_next.dtype, device=x_next.device).unsqueeze(-1)

        return x_next, h_next


class EGNNBackbone(nn.Module):
    """
    Stack of EGNN layers.

    Inputs are padded batch graphs. See EGNNLayer for shapes.
    """
    def __init__(
        self,
        d_h: int, #feature dimension 
        num_layers: int = 4, #number of EGNN layers 
        msg_hidden: int = 128, #hidden message dim in each layer 
        use_coord_update: bool = True, #whether to update coordinate in each layer
        dropout: float = 0.0, 
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            EGNNLayer(d_h, msg_hidden=msg_hidden, use_coord_update=use_coord_update, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
        edge_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            x, h = layer(x, h, edge_index, node_mask=node_mask, edge_mask=edge_mask)
        return x, h


# -----------------------------------------------------------------------------
# Actor–Critic: bipartite edge/face nodes
# -----------------------------------------------------------------------------

@dataclass
class PolicyOutput:
    """
    Returned by the actor head.
    """
    logits: torch.Tensor          # (B, A_max) where A_max = E_max + F_max
    edge_logits: torch.Tensor     # (B, E_max)
    face_logits: torch.Tensor     # (B, F_max)


class Mesh3DActorCritic(nn.Module):
    """
    Combined actor-critic module (common backbone, separate heads).

    Expected node ordering:
        nodes 0..E_max-1          -> edge nodes
        nodes E_max..E_max+F_max-1-> face nodes

    You *must* provide edge_node_mask and face_node_mask to indicate
    how many of those padded nodes are real per batch element.

    Candidate action masks are used to invalidate impossible actions.
    """
    def __init__(
        self,
        d_edge_in: int,  #initial edge embedding dimension
        d_face_in: int,  #initial face embedding dimension
        d_h: int = 128,  #shared embedding dimension later for both face and edge
        num_layers: int = 4, #number of GNN layers conducted
        msg_hidden: int = 128, #hidden message dimension
        use_coord_update: bool = False, #whether to use coordinate update after each MP
        dropout: float = 0.0, #dropout rate
        head_hidden: int = 128, #head MLP hidden dimension 
        value_hidden: int = 128, #value MLP hidden dimension
        critic_extra_dim: int = 0, #extra global critic-only input dim
    ) -> None:
        super().__init__()
        self.critic_extra_dim = int(critic_extra_dim)

        # Project raw node features into a shared hidden space d_h
        #this is done before the GNN so that the edge and face embeddings are in the same space
        self.edge_embed = MLP(d_edge_in, d_h, d_h, num_layers=1, dropout=dropout, act=nn.SiLU)
        self.face_embed = MLP(d_face_in, d_h, d_h, num_layers=1, dropout=dropout, act=nn.SiLU)

        self.backbone = EGNNBackbone(
            d_h=d_h,
            num_layers=num_layers,
            msg_hidden=msg_hidden,
            use_coord_update=use_coord_update,
            dropout=dropout,
        )

        # Actor heads (produce scalar logits)
        #it maps final embedding to scalar logits 
        #do it separately in two MLPs so that the logits can be in the same scale
        self.edge_head = MLP(d_h, head_hidden, 1, num_layers=2, dropout=dropout, act=nn.SiLU)
        self.face_head = MLP(d_h, head_hidden, 1, num_layers=2, dropout=dropout, act=nn.SiLU)

        # Critic: type-aware pooling then value MLP
        #transform both edge and face final embedding before pooling 
        self.edge_pool_proj = MLP(d_h, value_hidden, value_hidden, num_layers=1, dropout=dropout, act=nn.SiLU)
        self.face_pool_proj = MLP(d_h, value_hidden, value_hidden, num_layers=1, dropout=dropout, act=nn.SiLU)
        #after pooling, take the concatenated pooled global embeddings for edges and faces 
        #map it to a scalar, which is the predicted global value of this graph
        value_in_dim = 2 * value_hidden + self.critic_extra_dim
        self.value_mlp = MLP(value_in_dim, value_hidden, 1, num_layers=2, dropout=dropout, act=nn.SiLU)

        # Constant used for masking invalid actions
        self.register_buffer("_neg_inf", torch.tensor(-1e9))

    @staticmethod
    def _split_nodes(
        h: torch.Tensor,
        E_max: int,
        F_max: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split (B, N, d) hidden features into edge and face blocks.

        Returns:
            h_edge: (B, E_max, d)
            h_face: (B, F_max, d)
        """
        h_edge = h[:, :E_max, :]
        h_face = h[:, E_max:E_max + F_max, :]
        return h_edge, h_face

    def forward(
        self,
        # geometry & features
        x: torch.Tensor,                 # (B, N, 3), initial node coordinates
        edge_feat: torch.Tensor,         # (B, E_max, d_edge_in), initial edge node features 
        face_feat: torch.Tensor,         # (B, F_max, d_face_in), initial face node features 

        # graph structure
        edge_index: torch.Tensor,        # (B, 2, M) directed incidence edges between edge- and face-nodes
        node_mask: Optional[torch.Tensor] = None,   # (B, N)
        edge_mask: Optional[torch.Tensor] = None,   # (B, M)

        # type masks
        edge_node_mask: Optional[torch.Tensor] = None,  # (B, E_max)
        face_node_mask: Optional[torch.Tensor] = None,  # (B, F_max)

        # action validity masks (candidate mask)
        edge_action_mask: Optional[torch.Tensor] = None,  # (B, E_max)
        face_action_mask: Optional[torch.Tensor] = None,  # (B, F_max)

        # optional critic-only global features (e.g. tet_quality summary)
        critic_global_feat: Optional[torch.Tensor] = None,  # (B, critic_extra_dim)
    ) -> Tuple[PolicyOutput, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            policy_out: PolicyOutput (logits and per-type logits)
            value:      (B,) state-value estimate
            x_L:        (B, N, 3) final coordinates (if coord updates enabled; else unchanged)
            h_L:        (B, N, d_h) final node embeddings
        """
        B, E_max, _ = edge_feat.shape
        _, F_max, _ = face_feat.shape
        N_expected = E_max + F_max
        assert x.shape[1] == N_expected, f"x has N={x.shape[1]} but expected E_max+F_max={N_expected}"

        if edge_node_mask is None:
            edge_node_mask = torch.ones(B, E_max, dtype=torch.bool, device=x.device)
        if face_node_mask is None:
            face_node_mask = torch.ones(B, F_max, dtype=torch.bool, device=x.device)

        # Build the combined node feature tensor h0 in shared dimension d_h
        h_edge0 = self.edge_embed(edge_feat)  # (B, E_max, d_h)
        h_face0 = self.face_embed(face_feat)  # (B, F_max, d_h)
        h0 = torch.cat([h_edge0, h_face0], dim=1)  # (B, N, d_h)
    
        # node_mask: (B, N) bool
        if node_mask is None:
            node_mask = torch.cat([edge_node_mask, face_node_mask], dim=1)  # (B, N)

        #mask out false nodes
        node_mask_f = node_mask.unsqueeze(-1).type_as(h0)   # (B, N, 1)
        h0 = h0 * node_mask_f
        x  = x  * node_mask_f

        # Backbone, final coordinates and node embeddings
        x_L, h_L = self.backbone(x, h0, edge_index, node_mask=node_mask, edge_mask=edge_mask)

        # Split node embeddings by type
        h_edgeL, h_faceL = self._split_nodes(h_L, E_max=E_max, F_max=F_max)

        # Actor logits per type
        edge_logits = self.edge_head(h_edgeL).squeeze(-1)  # (B, E_max)
        face_logits = self.face_head(h_faceL).squeeze(-1)  # (B, F_max)

        # Invalidate padded nodes (node masks) first
        edge_logits = edge_logits.masked_fill(~edge_node_mask, self._neg_inf)
        face_logits = face_logits.masked_fill(~face_node_mask, self._neg_inf)

        # Then invalidate invalid actions (candidate masks)
        if edge_action_mask is not None:
            edge_logits = edge_logits.masked_fill(~edge_action_mask, self._neg_inf)
        if face_action_mask is not None:
            face_logits = face_logits.masked_fill(~face_action_mask, self._neg_inf)

        # Concatenate into one action space
        logits = torch.cat([edge_logits, face_logits], dim=1)  # (B, E_max + F_max)
        policy_out = PolicyOutput(logits=logits, edge_logits=edge_logits, face_logits=face_logits)

        # Critic value: pool separately per type (masked mean) then concat
        edge_z = self.edge_pool_proj(h_edgeL)  # (B, E_max, value_hidden)
        face_z = self.face_pool_proj(h_faceL)  # (B, F_max, value_hidden)

        g_edge = masked_mean(edge_z, edge_node_mask, dim=1)  # (B, value_hidden)
        g_face = masked_mean(face_z, face_node_mask, dim=1)  # (B, value_hidden)

        g = torch.cat([g_edge, g_face], dim=-1)  # (B, 2*value_hidden)
        if self.critic_extra_dim > 0:
            if critic_global_feat is None:
                critic_global_feat = torch.zeros(
                    (B, self.critic_extra_dim), dtype=g.dtype, device=g.device
                )
            else:
                if critic_global_feat.shape != (B, self.critic_extra_dim):
                    raise ValueError(
                        f"critic_global_feat shape {tuple(critic_global_feat.shape)} "
                        f"!= ({B}, {self.critic_extra_dim})"
                    )
                critic_global_feat = critic_global_feat.to(device=g.device, dtype=g.dtype)
            g = torch.cat([g, critic_global_feat], dim=-1)
        value = self.value_mlp(g).squeeze(-1)    # (B,)

        return policy_out, value, x_L, h_L


# -----------------------------------------------------------------------------
# Minimal smoke test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    B = 2
    E_max = 5
    F_max = 4
    N = E_max + F_max

    d_edge_in = 8
    d_face_in = 10

    # Random positions: in practice edge midpoints + face centroids
    x = torch.randn(B, N, 3)

    edge_feat = torch.randn(B, E_max, d_edge_in)
    face_feat = torch.randn(B, F_max, d_face_in)

    # Build a random bipartite incidence edge list (directed)
    # For each face, connect to 3 edges (here random for smoke test)
    M = F_max * 3 * 2  # we will include both directions
    edge_index = torch.zeros(B, 2, M, dtype=torch.long)

    for b in range(B):
        cols = []
        for f in range(F_max):
            face_node_id = E_max + f
            # choose 3 edges
            e_ids = torch.randperm(E_max)[:3].tolist()
            for e in e_ids:
                # undirected -> add both directions
                cols.append((e, face_node_id))
                cols.append((face_node_id, e))
        cols = cols[:M]
        edge_index[b, 0, :] = torch.tensor([c[0] for c in cols], dtype=torch.long)
        edge_index[b, 1, :] = torch.tensor([c[1] for c in cols], dtype=torch.long)

    node_mask = torch.ones(B, N, dtype=torch.bool)
    edge_mask = torch.ones(B, M, dtype=torch.bool)

    edge_node_mask = torch.ones(B, E_max, dtype=torch.bool)
    face_node_mask = torch.ones(B, F_max, dtype=torch.bool)

    # Make some actions invalid
    edge_action_mask = torch.ones(B, E_max, dtype=torch.bool)
    face_action_mask = torch.ones(B, F_max, dtype=torch.bool)
    edge_action_mask[:, 0] = False
    face_action_mask[:, -1] = False

    model = Mesh3DActorCritic(
        d_edge_in=d_edge_in,
        d_face_in=d_face_in,
        d_h=64,
        num_layers=3,
        msg_hidden=128,
        use_coord_update=False,
        dropout=0.1,
        head_hidden=64,
        value_hidden=64,
    )

    policy_out, value, x_L, h_L = model(
        x=x,
        edge_feat=edge_feat,
        face_feat=face_feat,
        edge_index=edge_index,
        node_mask=node_mask,
        edge_mask=edge_mask,
        edge_node_mask=edge_node_mask,
        face_node_mask=face_node_mask,
        edge_action_mask=edge_action_mask,
        face_action_mask=face_action_mask,
    )

    print("logits:", policy_out.logits.shape)
    print("value:", value.shape)
