"""
train.py – model definitions and training utilities
"""
from __future__ import annotations
import os
import math
import time
import pathlib
from typing import Any, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from .preprocess import set_global_seed, DATASET_REGISTRY

# -----------------------------------------------------------------------------
#  Model components (minimal but functional ChangeFormer subset)
# -----------------------------------------------------------------------------
class OCPDLayer(nn.Module):
    """Online change-point detector (1-D CNN → probability)."""

    def __init__(self, d_model: int, ksize: int = 7):
        super().__init__()
        self.conv = nn.Conv1d(d_model, 1, ksize, padding=ksize // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, d, T)
        return torch.sigmoid(self.conv(x)).squeeze(1)  # (B, T)


class ChangeFormerCore(nn.Module):
    """Very compact CHANGEFORMER core – enough for integration tests."""

    def __init__(self, horizon: int, input_dim: int = 1, d_model: int = 256):
        super().__init__()
        self.horizon = horizon
        self.input_dim = input_dim  # number of variables (channels)
        self.embed = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, batch_first=True, dim_feedforward=d_model * 4
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=4)
        # Output a forecast for each variable at each horizon step
        self.head = nn.Linear(d_model, horizon * input_dim)
        self.ocpd = OCPDLayer(d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, L, C)
        x = self.embed(x)
        ocp = self.ocpd(x.transpose(1, 2))
        h = self.encoder(x)
        out_flat = self.head(h[:, -1])  # (B, horizon * input_dim)
        out = out_flat.view(x.size(0), self.horizon, self.input_dim)  # (B, H, C)
        return out, ocp


class ChangeFormer(nn.Module):
    """Wrapper returned by REGISTRY – exposes only forecast for Trainer."""

    def __init__(self, horizon: int, input_dim: int = 1):
        super().__init__()
        self.core = ChangeFormerCore(horizon=horizon, input_dim=input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core(x)[0]


MODEL_REGISTRY: Dict[str, Any] = {"ChangeFormer": ChangeFormer}

# -----------------------------------------------------------------------------
#  Trainer
# -----------------------------------------------------------------------------
class Trainer:
    """Single-GPU trainer stub – sufficient for unit / functional testing."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        set_global_seed(cfg.get("seed", 42))

        # -------- dataset / dataloader -------------------------------------------------
        ds_name = cfg["dataset"]
        dataset_cls = DATASET_REGISTRY[ds_name]
        self.dataset = dataset_cls(cfg)
        self.loader = DataLoader(
            self.dataset,
            batch_size=cfg.get("batch_size", 32),
            shuffle=True,
            num_workers=min(4, os.cpu_count() or 1),
            pin_memory=True,
        )

        # -------- model / optimisation ------------------------------------------------
        model_cls = MODEL_REGISTRY[cfg["model"]]
        self.model = model_cls(horizon=cfg["horizon"], input_dim=self.dataset.num_vars)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.get("lr", 3e-4),
            weight_decay=cfg.get("weight_decay", 1e-2),
        )
        self.grad_clip = cfg.get("grad_clip", 1.0)
        self.epochs = cfg.get("epochs", 5)

    # ---------------------------------------------------------------------
    def _step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        self.optim.zero_grad()
        pred = self.model(x)
        loss = (pred - y).abs().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optim.step()
        return loss.item()

    # ---------------------------------------------------------------------
    def train(self) -> None:
        print("[Trainer] Start training…")
        for ep in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)
                epoch_loss += self._step(x, y)
            epoch_loss /= max(1, len(self.loader))
            print(f"  • epoch {ep:03d}/{self.epochs} | loss={epoch_loss:.5f}")

    # ---------------------------------------------------------------------
    @property
    def trained_model(self) -> nn.Module:
        return self.model
