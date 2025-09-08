"""
preprocess.py – dataset loading & feature pre-processing
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F


def load_dataset(name: str = "ogbn-products"):
    """Download (if needed) & return PyG graph with train/val/test masks."""

    try:
        from ogb.nodeproppred import PygNodePropPredDataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "ogb package not installed – `pip install ogb` is required"
        ) from exc

    dataset = PygNodePropPredDataset(name=name, root=str(Path("data").resolve()))
    data = dataset[0]

    # normalise features & cast to half precision (saves memory)
    data.x = F.layer_norm(data.x.float(), (data.x.size(-1),)).half()
    return data, dataset.num_classes
