"""
evaluate.py – evaluation helpers & implementation verification
"""
from __future__ import annotations

import json
from typing import List

import torch

# local imports – *only* runtime symbols necessary for verification  ----------
from .train import (
    EdgeScoreMemory,
    EXP3Bandit,
    FlashSparseAttention,
    FlashGATLayer,
    FlashGAT,
    VanillaGAT,
)

__all__ = ["evaluate", "verify_implementation"]


# -----------------------------------------------------------------------------

def evaluate(model, data, mask, device):
    """Return accuracy on the provided boolean mask."""
    model.eval()
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
        pred = out.argmax(dim=-1)
        correct = int((pred[mask] == data.y[mask].to(device)).sum())
    return correct / int(mask.sum())


# -----------------------------------------------------------------------------
REQUIRED_COMPONENTS: List[str] = [
    "EdgeScoreMemory",
    "EXP3Bandit",
    "FlashSparseAttention",
    "FlashGATLayer",
    "FlashGAT",
    "VanillaGAT",
]


def verify_implementation() -> bool:
    """Light-weight sanity check that core symbols are importable."""
    missing = [c for c in REQUIRED_COMPONENTS if c not in globals()]
    if missing:
        print("[Verify] Missing components:", json.dumps(missing))
        return False
    print("[Verify] All core components present – OK")
    return True
