"""
evaluate.py – evaluation, basic metrics & plotting stubs
"""
from __future__ import annotations
import json
import pathlib
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .preprocess import DATASET_REGISTRY
from .train import MODEL_REGISTRY

# -----------------------------------------------------------------------------
#  Metrics
# -----------------------------------------------------------------------------

def mae(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:  # noqa: D401
    """Mean-absolute-error."""
    return (pred - tgt).abs().mean()


def mse(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:  # noqa: D401
    """Mean-squared-error."""
    return ((pred - tgt) ** 2).mean()

# -----------------------------------------------------------------------------
#  Evaluator
# -----------------------------------------------------------------------------
class Evaluator:
    """Simple evaluator producing MAE & MSE."""

    def __init__(self, cfg: Dict[str, Any]):
        dataset_cls = DATASET_REGISTRY[cfg["dataset"]]
        self.dataset = dataset_cls(cfg, split="val")
        self.loader = DataLoader(
            self.dataset,
            batch_size=cfg.get("batch_size", 32),
            shuffle=False,
            num_workers=0,
        )
        model_cls = MODEL_REGISTRY[cfg["model"]]
        self.model = model_cls(horizon=cfg["horizon"], input_dim=self.dataset.num_vars)
        self.model.load_state_dict(torch.load(cfg["ckpt_path"]))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    def evaluate(self) -> Dict[str, float]:
        maes, mses = [], []
        with torch.no_grad():
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                maes.append(mae(pred, y).item())
                mses.append(mse(pred, y).item())
        return {"MAE": float(np.mean(maes)), "MSE": float(np.mean(mses))}

# -----------------------------------------------------------------------------
#  Plotting helper – stores placeholder figure files for compliance
# -----------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt  # heavy import only if available

    def save_forecast_plot(pred: np.ndarray, tgt: np.ndarray, out_path: pathlib.Path) -> None:
        plt.figure(figsize=(10, 3))
        plt.plot(tgt, label="target")
        plt.plot(pred, label="pred")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
except Exception:  # pragma: no cover – headless environment

    def save_forecast_plot(*_args: Any, **_kw: Any) -> None:  # noqa: D401
        pass
