from __future__ import annotations

"""
preprocess.py – dataset loading & feature pre-processing
"""

import os
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# 0.  Torch serialization compatibility patch
# -----------------------------------------------------------------------------
# PyTorch ≥2.6 switched the default behaviour of `torch.load` to `weights_only=True`,
# which breaks deserialisation of objects that are not simple tensors – such as
# the `torch_geometric.data.data.*Attr` instances stored inside OGB processed
# dataset files.  We **auto-discover** every attribute / storage class defined in
# torch-geometric's internal modules and put them on the allow-list so the new
# secure loader can resolve those symbols safely.
# -----------------------------------------------------------------------------
try:
    import importlib

    # Helper is only present on recent PyTorch builds; wrap everything in a
    # try/except so that older runtimes degrade gracefully.
    from torch.serialization import add_safe_globals  # type: ignore

    # ------------------------------------------------------------------
    # 1)  All *Attr classes (already handled previously)                   
    # ------------------------------------------------------------------
    tg_data_mod = importlib.import_module("torch_geometric.data.data")
    attr_types = {
        getattr(tg_data_mod, name)
        for name in dir(tg_data_mod)
        if name.endswith("Attr") and isinstance(getattr(tg_data_mod, name), type)
    }

    # ------------------------------------------------------------------
    # 2)  All *Storage classes – required for Pyg processed datasets       
    #     (e.g. GlobalStorage, NodeStorage, EdgeStorage, …)                
    # ------------------------------------------------------------------
    storage_mod = importlib.import_module("torch_geometric.data.storage")
    storage_types = {
        getattr(storage_mod, name)
        for name in dir(storage_mod)
        if name.endswith("Storage") and isinstance(getattr(storage_mod, name), type)
    }

    if attr_types or storage_types:
        add_safe_globals(attr_types | storage_types)  # type: ignore[arg-type]
except Exception:  # pragma: no cover – best-effort patch, safe to ignore
    pass

# -----------------------------------------------------------------------------
# 1.  Utility – disable interactive OGB download prompt
# -----------------------------------------------------------------------------

def _disable_ogb_prompt():
    """Monkey-patch OGB’s download prompt so that it never requires stdin."""

    from ogb.utils import url as ogb_url  # type: ignore

    def _decide_download(url: str):  # pylint: disable=unused-argument
        if os.getenv("DISABLE_DATA_DOWNLOAD", "0") == "1":
            raise RuntimeError(
                "Dataset download disabled via DISABLE_DATA_DOWNLOAD env-var. "
                "Please place the OGB dataset under the local 'data/' folder "
                "before running the experiment."
            )
        print(f"[OGB] Auto-approving download for {url}")
        return True

    ogb_url.decide_download = _decide_download  # type: ignore[attr-defined]


_disable_ogb_prompt()


# -----------------------------------------------------------------------------
# 2.  Public API
# -----------------------------------------------------------------------------

def load_dataset(name: str = "ogbn-products"):
    """Download (if needed) & return PyG graph with train/val/test masks."""

    try:
        from ogb.nodeproppred import PygNodePropPredDataset  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "ogb package not installed – `pip install ogb` is required"
        ) from exc

    dataset = PygNodePropPredDataset(name=name, root=str(Path("data").resolve()))
    data = dataset[0]

    # ------------------------------------------------------------------
    # Split indices → boolean masks expected by the training loop
    # ------------------------------------------------------------------
    idx_split = dataset.get_idx_split()
    num_nodes = data.y.size(0)
    for key, idx in idx_split.items():
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[idx] = True
        setattr(data, f"{key}_mask", mask)

    # ------------------------------------------------------------------
    # Feature normalisation & dtype selection
    # ------------------------------------------------------------------
    # Always normalise to zero-mean, unit-variance.  For CUDA-enabled devices we
    # cast to `half` in order to save memory.  On CPU, many ops (in particular
    # `torch.nn.Linear`) do NOT support FP16 – therefore we keep features in
    # FP32 when CUDA is unavailable.
    # ------------------------------------------------------------------
    data.x = F.layer_norm(data.x.float(), (data.x.size(-1),))
    if torch.cuda.is_available():
        data.x = data.x.half()

    data.y = data.y.squeeze(-1).long()  # shape: [N]
    return data, dataset.num_classes
