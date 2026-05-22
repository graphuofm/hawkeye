"""A standard TGN (memory + temporal-attention embedding) — the base interaction
model that GraphEagleVision plugs into. Adapted from the PyG / TGB TGN example.

Lifecycle per epoch:
    tgn.set_edges(edge_t, edge_msg)   # full arrays, indexed by global edge id (once)
    tgn.reset()                       # start of each epoch
    for batch:
        z = tgn.embed(node_ids)       # current embeddings for a set of nodes
        ... use z ...
        tgn.update_state(src, dst, t, msg)
        tgn.detach()                  # end of each batch
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.nn import Linear

from torch_geometric.nn import TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
    TGNMemory,
)


class _GraphAttentionEmbedding(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, msg_dim: int, time_enc, dropout: float = 0.1):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2, dropout=dropout, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class TGNModel(nn.Module):
    def __init__(self, num_nodes: int, raw_msg_dim: int, memory_dim: int = 100,
                 time_dim: int = 100, embedding_dim: int = 100, neighbor_size: int = 10, dropout: float = 0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.raw_msg_dim = max(1, int(raw_msg_dim))
        self.embedding_dim = embedding_dim
        self.neighbor_size = neighbor_size
        self.memory = TGNMemory(
            num_nodes, self.raw_msg_dim, memory_dim, time_dim,
            message_module=IdentityMessage(self.raw_msg_dim, memory_dim, time_dim),
            aggregator_module=LastAggregator(),
        )
        self.gnn = _GraphAttentionEmbedding(memory_dim, embedding_dim, self.raw_msg_dim,
                                            self.memory.time_enc, dropout=dropout)
        self.assoc = None              # int tensor [num_nodes], allocated lazily on device
        self.neighbor_loader = None    # allocated lazily on device
        self.edge_t = None             # [E] float, indexed by global edge id
        self.edge_msg = None           # [E, raw_msg_dim] float

    # ------------------------------------------------------------------ #
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _lazy_init(self) -> None:
        dev = self.device
        if self.neighbor_loader is None or self.neighbor_loader.neighbors.device != dev:
            self.neighbor_loader = LastNeighborLoader(self.num_nodes, size=self.neighbor_size, device=dev)
        if self.assoc is None or self.assoc.device != dev:
            self.assoc = torch.empty(self.num_nodes, dtype=torch.long, device=dev)

    def set_edges(self, edge_t: torch.Tensor, edge_msg: torch.Tensor) -> None:
        dev = self.device
        # TGNMemory keeps `last_update` as Long, so timestamps must stay integer-typed.
        self.edge_t = edge_t.to(dev).long()
        if edge_msg.dim() == 1:
            edge_msg = edge_msg.unsqueeze(-1)
        self.edge_msg = edge_msg.to(dev).float()

    def reset(self) -> None:
        self._lazy_init()
        self.memory.reset_state()
        self.neighbor_loader.reset_state()

    def detach(self) -> None:
        self.memory.detach()

    # ------------------------------------------------------------------ #
    def embed(self, node_ids: torch.Tensor) -> torch.Tensor:
        """TGN embeddings for ``node_ids`` (1-D long tensor) using the current memory
        + temporal-neighbour cache. Call after reset()/update_state() of prior batches."""
        node_ids = node_ids.to(self.device)
        n_id, edge_index, e_id = self.neighbor_loader(node_ids)
        self.assoc[n_id] = torch.arange(n_id.size(0), device=self.device)
        memory, last_update = self.memory(n_id)
        z = self.gnn(memory, last_update, edge_index, self.edge_t[e_id], self.edge_msg[e_id])
        return z[self.assoc[node_ids]]

    def update_state(self, src: torch.Tensor, dst: torch.Tensor, t: torch.Tensor, msg: torch.Tensor) -> None:
        dev = self.device
        src, dst = src.to(dev).long(), dst.to(dev).long()
        t = t.to(dev).long()
        if msg.dim() == 1:
            msg = msg.unsqueeze(-1)
        msg = msg.to(dev).float()
        self.memory.update_state(src, dst, t, msg)
        self.neighbor_loader.insert(src, dst)


class TGNLinkPredictor(nn.Module):
    """The vanilla TGN link head (for the pure-TGN baseline)."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        h = (self.lin_src(z_src) + self.lin_dst(z_dst)).relu()
        return self.lin_final(h).squeeze(-1)
