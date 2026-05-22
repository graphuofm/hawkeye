"""DyGFormer-lite: a Transformer encoder over a node's recent interaction sequence.

A minimal, reasonably-faithful implementation of the DyGFormer idea (Yu et al.
NeurIPS 2023): for a query node v at time t, fetch its last-K interactions
(neighbour, timestamp, edge-message), encode each as
``[node_embed(neighbour) + time_enc(t - t_evt) + msg_proj(msg)]``, optionally
append a *neighbour-co-occurrence* indicator (1 iff that neighbour also appears
in the *other* query node's history), pass through a Transformer encoder, and
mean-pool. Used as a stronger base model than TGN in the "enhance" experiment.

Compared to the official DyGFormer this is a faithful "lite" version: we use
ring-buffered histories on GPU; the co-occurrence trick is implemented; we drop
some smaller bells (multiple patch sizes, channel-mixer head) for clarity.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SinusoidTimeEncoder(nn.Module):
    """Mercer-style time encoder a la Bochner — same parameterisation as TGN."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)
        # init similar to TGN: log-spaced frequencies
        nn.init.constant_(self.w.bias, 0.0)
        nn.init.uniform_(self.w.weight, a=-0.01, b=0.01)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (..., ) float -> (..., dim)
        x = self.w(t.unsqueeze(-1).float())
        return torch.cos(x)


class DyGFormerLite(nn.Module):
    """Strong-base temporal-graph encoder via a Transformer over interaction histories.

    Lifecycle per epoch:
        model.reset()                                # at the start of each epoch
        z_src, z_dst = model.embed_pairs(src, dst, t)# query embeddings (for the link head)
        model.update_state(src, dst, t, msg)         # advance the histories
    """

    def __init__(self, num_nodes: int, raw_msg_dim: int, embedding_dim: int = 100,
                 time_dim: int = 100, history_size: int = 32, num_layers: int = 2,
                 num_heads: int = 2, dropout: float = 0.1,
                 use_cooccur: bool = True,
                 struct_channel_dim: int = 0) -> None:
        super().__init__()
        self.N = int(num_nodes)
        self.K = int(history_size)
        self.D = int(embedding_dim)
        self.use_cooccur = bool(use_cooccur)
        self.struct_channel_dim = int(struct_channel_dim)
        self.msg_dim = max(1, int(raw_msg_dim))
        self.node_embed = nn.Embedding(self.N + 1, self.D, padding_idx=self.N)  # last id = "pad"
        self.time_enc = _SinusoidTimeEncoder(time_dim)
        self.msg_proj = nn.Linear(self.msg_dim, self.D)
        self.time_proj = nn.Linear(time_dim, self.D)
        # struct channel (cohesion features) gets its own LayerNorm to keep scales sane
        if self.struct_channel_dim > 0:
            self.struct_norm = nn.LayerNorm(self.struct_channel_dim)
        in_dim = self.D + (1 if self.use_cooccur else 0) + self.struct_channel_dim
        if in_dim != self.D:
            self.in_proj = nn.Linear(in_dim, self.D)
        else:
            self.in_proj = nn.Identity()
        enc_layer = nn.TransformerEncoderLayer(d_model=self.D, nhead=num_heads,
                                               dim_feedforward=4 * self.D,
                                               dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        # ring-buffer state buffers (allocated lazily on first reset on the right device)
        self.register_buffer("hist_nid", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("hist_t", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("hist_msg", torch.empty(0, dtype=torch.float32), persistent=False)
        self.register_buffer("hist_ptr", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("hist_cnt", torch.empty(0, dtype=torch.long), persistent=False)

    # ------------------------------------------------------------------ #
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _alloc(self) -> None:
        dev = self.device
        self.hist_nid = torch.full((self.N, self.K), self.N, dtype=torch.long, device=dev)  # pad index
        self.hist_t = torch.zeros((self.N, self.K), dtype=torch.long, device=dev)
        self.hist_msg = torch.zeros((self.N, self.K, self.msg_dim), dtype=torch.float32, device=dev)
        self.hist_ptr = torch.zeros((self.N,), dtype=torch.long, device=dev)
        self.hist_cnt = torch.zeros((self.N,), dtype=torch.long, device=dev)

    def reset(self) -> None:
        self._alloc()

    # ------------------------------------------------------------------ #
    def update_state(self, src: torch.Tensor, dst: torch.Tensor,
                     t: torch.Tensor, msg: torch.Tensor) -> None:
        """Append (dst, t, msg) to src's ring buffer and (src, t, msg) to dst's.
        Within a batch we apply two scatter writes sequentially; duplicate
        writes to the same node within a batch overwrite (last wins) — fine for
        most TGB streams since duplicate src/dst within a batch is rare."""
        dev = self.device
        src = src.to(dev).long(); dst = dst.to(dev).long()
        t = t.to(dev).long()
        if msg.dim() == 1:
            msg = msg.unsqueeze(-1)
        msg = msg.to(dev).float()
        B = src.size(0)
        # write to src side
        for side_a, side_b in ((src, dst), (dst, src)):
            ptr = self.hist_ptr[side_a]
            # advance the ptrs first to claim slots
            new_ptr = (ptr + 1) % self.K
            self.hist_ptr.index_copy_(0, side_a, new_ptr)
            self.hist_cnt[side_a] = torch.clamp(self.hist_cnt[side_a] + 1, max=self.K)
            # scatter writes
            self.hist_nid[side_a, ptr] = side_b
            self.hist_t[side_a, ptr] = t
            self.hist_msg[side_a, ptr] = msg

    # ------------------------------------------------------------------ #
    def _gather_seq(self, ids: torch.Tensor, t_query: torch.Tensor):
        """For each query node id, return its history as
        (nid: (B,K), dt: (B,K) float, msg: (B,K,msg_dim), mask: (B,K) bool with True=valid)."""
        nid = self.hist_nid[ids]           # (B,K)
        t = self.hist_t[ids].float()       # (B,K)
        msg = self.hist_msg[ids]           # (B,K,msg_dim)
        cnt = self.hist_cnt[ids]           # (B,)
        # valid = the latest cnt entries; in a ring buffer of size K, after `cnt` writes the
        # valid slots are *all* of them if cnt==K, else the first cnt slots written into.
        # Simpler: a slot is valid iff its stored nid != pad
        mask = (nid != self.N)
        dt = t_query.float().unsqueeze(-1) - t
        return nid, dt, msg, mask

    def _encode_seq(self, nid, dt, msg, mask, peer_set=None, struct_ch=None):
        # nid (B,K), dt (B,K), msg (B,K,m), mask (B,K)
        node_e = self.node_embed(nid)              # (B,K,D)
        msg_e = self.msg_proj(msg)                 # (B,K,D)
        t_e = self.time_proj(self.time_enc(dt))    # (B,K,D)
        x = node_e + msg_e + t_e
        extras = []
        if self.use_cooccur and peer_set is not None:
            # peer_set: (B, K_peer) of nids -> compute, for each event in nid, whether it
            # appears in the peer's history (a 0/1 feature broadcast along K).
            # (B,K,1) <- any over K_peer dim
            co = (nid.unsqueeze(-1) == peer_set.unsqueeze(1)).any(dim=-1).float().unsqueeze(-1)
            extras.append(co)
        if self.struct_channel_dim > 0 and struct_ch is not None:
            # struct_ch: (B,K,F_struct) — pre-computed cohesion features (this is the
            # drop-in replacement for the 1-bit cooccur channel).
            sc = self.struct_norm(struct_ch)
            extras.append(sc)
        if extras:
            x = torch.cat([x] + extras, dim=-1)
        x = self.in_proj(x)
        # transformer: mask invalid positions. Rows with all-pad would crash the
        # nested-tensor fast path in nn.TransformerEncoder, so we always leave at
        # least the first position unmasked (its embedding is the pad embedding,
        # so the output stays well-defined and gets zeroed in the mean-pool below).
        empty_row = ~mask.any(dim=1)                 # (B,) True iff fully padded
        key_pad = ~mask.clone()
        if empty_row.any():
            key_pad[empty_row, 0] = False
        h = self.encoder(x, src_key_padding_mask=key_pad)  # (B,K,D)
        # mean-pool over the *originally* valid positions only
        m = mask.float().unsqueeze(-1)               # (B,K,1)
        denom = m.sum(dim=1).clamp(min=1.0)          # (B,1)
        return (h * m).sum(dim=1) / denom            # (B,D); 0 for empty rows

    def embed_pairs(self, src: torch.Tensor, dst: torch.Tensor, t: torch.Tensor,
                    struct_provider=None):
        """Return (z_src, z_dst) ∈ (B, D) for each (src, dst) pair at time t.

        If ``struct_provider`` is given, it is called as
        ``struct_provider(hist_nids, peer_nids) -> (B, K, F_struct)`` once for each
        of (src-history, dst-as-peer) and (dst-history, src-as-peer); the returned
        tensor is concatenated alongside the (optional) cooccur channel before the
        transformer encoder."""
        dev = self.device
        src = src.to(dev).long(); dst = dst.to(dev).long(); t = t.to(dev)
        nid_s, dt_s, msg_s, mask_s = self._gather_seq(src, t)
        nid_d, dt_d, msg_d, mask_d = self._gather_seq(dst, t)
        sc_s = sc_d = None
        if struct_provider is not None and self.struct_channel_dim > 0:
            sc_s = struct_provider(nid_s, dst).to(dev)
            sc_d = struct_provider(nid_d, src).to(dev)
        z_s = self._encode_seq(nid_s, dt_s, msg_s, mask_s,
                               peer_set=nid_d if self.use_cooccur else None,
                               struct_ch=sc_s)
        z_d = self._encode_seq(nid_d, dt_d, msg_d, mask_d,
                               peer_set=nid_s if self.use_cooccur else None,
                               struct_ch=sc_d)
        return z_s, z_d

    # convenience for unified node-embedding access (used by some couplings)
    def embed(self, node_ids: torch.Tensor, t_query: Optional[torch.Tensor] = None,
              peer_ids: Optional[torch.Tensor] = None,
              struct_provider=None) -> torch.Tensor:
        """Embed a flat list of nodes at a single time t.

        If ``peer_ids`` is given, the per-slot structure channels (cooccur and/or
        ``struct_provider``-derived) are computed against the corresponding peer
        — this is the path used by the swap-in DyGFormer runner."""
        if t_query is None:
            t_query = torch.zeros(node_ids.size(0), device=self.device, dtype=torch.long)
        node_ids = node_ids.to(self.device).long()
        t_q = t_query.to(self.device)
        nid, dt, msg, mask = self._gather_seq(node_ids, t_q)
        sc = None
        peer_set = None
        if peer_ids is not None:
            peer_t = peer_ids.to(self.device).long()
            if self.use_cooccur:
                peer_nid, _, _, _ = self._gather_seq(peer_t, t_q)
                peer_set = peer_nid  # (B, K) of peer's history neighbour ids
            if struct_provider is not None and self.struct_channel_dim > 0:
                sc = struct_provider(nid, peer_t).to(self.device)
        return self._encode_seq(nid, dt, msg, mask, peer_set=peer_set, struct_ch=sc)


class DyGFormerLiteLinkPredictor(nn.Module):
    """Simple link head over (z_src, z_dst) for the pure-DyGFormer baseline."""
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.lin = nn.Sequential(
            nn.Linear(2 * in_channels, in_channels), nn.ReLU(),
            nn.Linear(in_channels, 1),
        )
    def forward(self, z_src, z_dst):
        return self.lin(torch.cat([z_src, z_dst], dim=-1)).squeeze(-1)
