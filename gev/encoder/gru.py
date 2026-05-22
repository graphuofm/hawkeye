from __future__ import annotations

import torch
import torch.nn as nn


class GRUEncoder(nn.Module):
    """GRU over a window of raw K-dim indicator vectors (ablation alternative to MLP).

    Expects input of shape (batch, window, K). Heavier (needs stored sequences).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.output_dim = output_dim

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(sequences)
        return self.norm(self.proj(h.squeeze(0)))
