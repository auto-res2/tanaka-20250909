"""
preprocess.py – dataset loading & feature pre-processing
"""
from __future__ import annotations

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
# the `torch_geometric.data.data.DataEdgeAttr` instances stored inside OGB
# processed dataset files.  We *explicitly* allow‐list the missing type so that
# the default secure loader succeeds without having to fall back to the less
# secure `weights_only=False` option.
# -----------------------------------------------------------------------------
try:
    # The class lives exactly under the dotted path mentioned in PyTorch’s error
    # message, therefore importing it guarantees the correct reference.
    from torch_geometric.data.data import DataEdgeAttr  # type: ignore

    # The helper is only available on recent PyTorch versions – we guard the
    # import so that older runtimes degrade gracefully.
    from torch.serialization import add_safe_globals  # type: ignore

    # Register the type globally for the entire lifetime of the interpreter.
    add_safe_globals({DataEdgeAttr})  # pylint: disable=no-member
except Exception:  # pragma: no cover – the patch is best-effort
    # If either PyG or the new `torch.serialization` API is unavailable we fall
    # back silently – older PyTorch (<2.6) never required this workaround.
    pass

# -----------------------------------------------------------------------------
# 1.  Utility – disable interactive OGB download prompt
# -----------------------------------------------------------------------------

def _disable_ogb_prompt():
    """Monkey-patch OGB’s download prompt so that it never requires stdin.

    In CI / non-interactive environments an unexpected `input()` call raises an
    EOFError which terminates the run.  We instead auto-approve the download –
    this mirrors a user typing “y”.  If the runtime wishes to block downloads
    it can set the environment variable `DISABLE_DATA_DOWNLOAD=1`, in which
    case we raise a *clear* error right away.
    """

    from ogb.utils import url as ogb_url  # type: ignore

    def _decide_download(url: str):  # pylint: disable=unused-argument
        if os.getenv("DISABLE_DATA_DOWNLOAD", "0") == "1":
            raise RuntimeError(
                "Dataset download disabled via DISABLE_DATA_DOWNLOAD env-var. "
                "Please place the OGB dataset under the local 'data/' folder "
                "before running the experiment."
            )
        # Auto-approve download (acts like the user typed “y”).
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

    # Normalise features & cast to half precision (saves memory)
    data.x = F.layer_norm(data.x.float(), (data.x.size(-1),)).half()
    data.y = data.y.squeeze(-1).long()  # shape: [N]
    return data, dataset.num_classes