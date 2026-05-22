"""Misc utilities: seeding, device, MRR."""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def get_device(prefer: Optional[str] = None):
    import torch

    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reciprocal_rank(pos_score: float, neg_scores: np.ndarray) -> float:
    """Rank of the positive among [pos] + negatives (ties broken pessimistically)."""
    neg_scores = np.asarray(neg_scores)
    # number of negatives strictly greater + ties counted as half/worst
    greater = int((neg_scores > pos_score).sum())
    equal = int((neg_scores == pos_score).sum())
    rank = greater + equal // 2 + 1
    return 1.0 / rank


def count_parameters(module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
