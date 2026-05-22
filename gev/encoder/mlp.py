from __future__ import annotations

import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    """Lightweight structural encoder: stacked Linear/ReLU/Dropout + final LayerNorm."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        num_layers = max(1, num_layers)
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        layers.append(nn.LayerNorm(output_dim))
        self.net = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IdentityEncoder(nn.Module):
    """No encoder — raw rolling-stats features pass through (for ablation)."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.output_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
