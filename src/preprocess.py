"""
preprocess.py – dataset loaders, utilities & registry
"""
from __future__ import annotations
import io
import os
import urllib.request
import zipfile
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# -----------------------------------------------------------------------------
#  Repro utilities
# -----------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # avoid warnings on CPU-only boxes
        torch.cuda.manual_seed_all(seed)

# -----------------------------------------------------------------------------
#  Dataset base
# -----------------------------------------------------------------------------
class TimeSeriesDataset(Dataset):
    """Sliding-window dataset (context → horizon)."""

    def __init__(
        self,
        data: np.ndarray,
        context: int,
        horizon: int,
    ) -> None:
        super().__init__()
        self.data = data.astype(np.float32)
        self.context = context
        self.horizon = horizon

    # ----------------------------------------
    def __len__(self) -> int:  # noqa: D401
        return len(self.data) - (self.context + self.horizon) + 1

    # ----------------------------------------
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # noqa: D401
        x = self.data[idx : idx + self.context]
        y = self.data[idx + self.context : idx + self.context + self.horizon]
        return torch.from_numpy(x), torch.from_numpy(y)

    # ----------------------------------------
    @property
    def num_vars(self) -> int:  # noqa: D401
        return self.data.shape[1]

# -----------------------------------------------------------------------------
#  Specific dataset – ETTh1 only (others follow similar pattern)
# -----------------------------------------------------------------------------
ETT_URLS: Dict[str, str] = {
    "ETTh1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
}
DATA_DIR = os.getenv("CHANGEFORMER_DATA", "./data")


def _download(name: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    f_path = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.exists(f_path):
        print(f"[Data] downloading {name}…")
        try:
            urllib.request.urlretrieve(ETT_URLS[name], f_path)
        except Exception as exc:  # network failure fallback
            raise RuntimeError(f"Failed to download {name}: {exc}")
    return f_path


class ETTh1(TimeSeriesDataset):
    """ETTh1 loader (train/val split by simple pct)."""

    def __init__(self, cfg: Dict[str, Any], split: str = "train"):
        csv_path = _download("ETTh1")
        df = pd.read_csv(csv_path)
        df = df.iloc[:, 1:]  # drop Time column

        n_total = len(df)
        train_end = int(n_total * cfg.get("train_pct", 0.7))
        val_end = int(n_total * (cfg.get("train_pct", 0.7) + cfg.get("val_pct", 0.2)))
        if split == "train":
            df = df.iloc[:train_end]
        else:  # "val" or others
            df = df.iloc[train_end:val_end]

        super().__init__(
            df.values,
            context=cfg["input_length"],
            horizon=cfg["horizon"],
        )

# -----------------------------------------------------------------------------
#  Synthetic dataset (very small placeholder)
# -----------------------------------------------------------------------------
class SyntheticShift(TimeSeriesDataset):
    def __init__(self, cfg: Dict[str, Any], split: str = "train"):
        rng = np.random.default_rng(cfg.get("seed", 0))
        L, V = 1500, 16
        data = rng.standard_normal((L, V)).cumsum(axis=0)  # random walk
        super().__init__(data, context=cfg["input_length"], horizon=cfg["horizon"])

# -----------------------------------------------------------------------------
#  Registry
# -----------------------------------------------------------------------------
DATASET_REGISTRY: Dict[str, Any] = {
    "ETTh1": ETTh1,
    "SyntheticShift": SyntheticShift,
}
