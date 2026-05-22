"""Fusion of interaction embedding (from a base TG model) and structural embedding."""
from __future__ import annotations

import torch
import torch.nn as nn


class StructOnlyFusion(nn.Module):
    """Pure-structural mode: no interaction signal used."""

    def __init__(self, struct_dim: int, output_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(struct_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.output_dim = output_dim

    def forward(self, h_struct: torch.Tensor, h_inter: torch.Tensor | None = None) -> torch.Tensor:
        return self.norm(self.proj(h_struct))


class ConcatFusion(nn.Module):
    def __init__(self, inter_dim: int, struct_dim: int, output_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(inter_dim + struct_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.output_dim = output_dim

    def forward(self, h_struct: torch.Tensor, h_inter: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(torch.cat([h_inter, h_struct], dim=-1)))


class AdditiveFusion(nn.Module):
    def __init__(self, inter_dim: int, struct_dim: int, output_dim: int) -> None:
        super().__init__()
        self.inter_proj = nn.Linear(inter_dim, output_dim)
        self.struct_proj = nn.Linear(struct_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.output_dim = output_dim

    def forward(self, h_struct: torch.Tensor, h_inter: torch.Tensor) -> torch.Tensor:
        return self.norm(self.inter_proj(h_inter) + self.struct_proj(h_struct))


class GatedFusion(nn.Module):
    """h = g * h_inter' + (1-g) * h_struct';  g = sigmoid(W[h_inter; h_struct]).

    The gate value is logged for interpretability (high -> relies on interaction).
    """

    def __init__(self, inter_dim: int, struct_dim: int, output_dim: int) -> None:
        super().__init__()
        total = inter_dim + struct_dim
        self.gate = nn.Sequential(nn.Linear(total, output_dim), nn.Sigmoid())
        self.inter_proj = nn.Linear(inter_dim, output_dim)
        self.struct_proj = nn.Linear(struct_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.output_dim = output_dim
        self.last_gate: torch.Tensor | None = None

    def forward(self, h_struct: torch.Tensor, h_inter: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([h_inter, h_struct], dim=-1)
        g = self.gate(combined)
        self.last_gate = g.detach()
        h = g * self.inter_proj(h_inter) + (1.0 - g) * self.struct_proj(h_struct)
        return self.norm(h)


class AttentionFusion(nn.Module):
    """Cross-attention late fusion: project both to ``output_dim``, let each attend
    to the other (a tiny 2-token MHA), then pool the two updated tokens."""

    def __init__(self, inter_dim: int, struct_dim: int, output_dim: int, heads: int = 4) -> None:
        super().__init__()
        h = max(heads, 1)
        while output_dim % h != 0:
            h -= 1
        self.inter_proj = nn.Linear(inter_dim, output_dim)
        self.struct_proj = nn.Linear(struct_dim, output_dim)
        self.attn = nn.MultiheadAttention(output_dim, num_heads=h, batch_first=True)
        self.norm = nn.LayerNorm(output_dim)
        self.output_dim = output_dim

    def forward(self, h_struct: torch.Tensor, h_inter: torch.Tensor) -> torch.Tensor:
        a = self.inter_proj(h_inter)         # (N, d)
        b = self.struct_proj(h_struct)       # (N, d)
        tok = torch.stack([a, b], dim=1)     # (N, 2, d)
        out, _ = self.attn(tok, tok, tok)    # (N, 2, d)
        return self.norm((out + tok).mean(dim=1))


class FiLMFusion(nn.Module):
    """FiLM modulation: h_struct produces (gamma, beta); h := norm(gamma * inter' + beta)."""

    def __init__(self, inter_dim: int, struct_dim: int, output_dim: int) -> None:
        super().__init__()
        self.inter_proj = nn.Linear(inter_dim, output_dim)
        self.film = nn.Linear(struct_dim, 2 * output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.output_dim = output_dim

    def forward(self, h_struct: torch.Tensor, h_inter: torch.Tensor) -> torch.Tensor:
        gb = self.film(h_struct)
        gamma, beta = gb.chunk(2, dim=-1)
        return self.norm((1.0 + gamma) * self.inter_proj(h_inter) + beta)


def build_fusion(mode: str, struct_dim: int, output_dim: int, inter_dim: int = 0) -> nn.Module:
    m = mode.lower()
    if m == "struct_only":
        return StructOnlyFusion(struct_dim, output_dim)
    if inter_dim <= 0:
        raise ValueError(f"fusion mode {mode!r} requires inter_dim > 0")
    if m == "concat":
        return ConcatFusion(inter_dim, struct_dim, output_dim)
    if m == "additive":
        return AdditiveFusion(inter_dim, struct_dim, output_dim)
    if m == "gated":
        return GatedFusion(inter_dim, struct_dim, output_dim)
    if m in ("attn", "attention"):
        return AttentionFusion(inter_dim, struct_dim, output_dim)
    if m == "film":
        return FiLMFusion(inter_dim, struct_dim, output_dim)
    raise KeyError(f"unknown fusion mode {mode!r}")
