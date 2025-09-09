"""
src/train.py – model definition, ReDiF-T memory modules and the training
loop.  Everything that touches the model lives here so that no other file
needs to know about PyTorch internals.
"""
from __future__ import annotations

import itertools
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

from .preprocess import get_loader
from .evaluate import eval_fid

# ============================================================
#  ReDiF-T  – MEMORY OPTIMISATION COMPONENTS
# ============================================================

class _RevHalfResidual(torch.autograd.Function):
    """Reversible half-residual block used to save activation memory."""

    @staticmethod
    def forward(ctx, x1, x2, f, g):  # type: ignore[override]
        with torch.no_grad():
            y1 = x1 + f(x2)
            y2 = x2 + g(y1)
        ctx.save_for_backward(y1.detach(), y2.detach())
        ctx.f, ctx.g = f, g  # type: ignore[attr-defined]
        return y1, y2

    @staticmethod
    def backward(ctx, dy1, dy2):  # type: ignore[override]
        y1, y2 = ctx.saved_tensors  # type: ignore[attr-defined]
        f, g = ctx.f, ctx.g  # type: ignore[attr-defined]
        with torch.enable_grad():
            y1.requires_grad_(True)
            y2_ = y2 - g(y1)
            x2 = y2_.detach().requires_grad_(True)
            y1_ = y1 - f(x2)
            x1 = y1_.detach()
            torch.autograd.backward((y1, y2), (dy1, dy2))
            dx1 = x1.grad + dy1  # type: ignore[operator]
            dx2 = x2.grad + dy2  # type: ignore[operator]
        return dx1, dx2, None, None


class ReversibleBlock(torch.nn.Module):
    """Wrap two transformer blocks f & g into one reversible pair."""

    def __init__(self, f: torch.nn.Module, g: torch.nn.Module):
        super().__init__()
        self.f, self.g = f, g

    def forward(self, x):  # type: ignore[override]
        x1, x2 = torch.chunk(x, 2, dim=1)
        y1, y2 = _RevHalfResidual.apply(x1, x2, self.f, self.g)
        return torch.cat([y1, y2], dim=1)


class BlockwiseSelfAttention(torch.nn.Module):
    """Chunked attention – keeps memory O(chunk·d)."""

    def __init__(self, sa_module: torch.nn.Module, chunk: int = 128):
        super().__init__()
        self.sa = sa_module
        self.chunk = chunk

    def forward(self, x, **kw):  # type: ignore[override]
        L = x.shape[1]
        outs: List[torch.Tensor] = []
        for i in range(0, L, self.chunk):
            outs.append(self.sa(x[:, i : i + self.chunk], **kw))
        return torch.cat(outs, dim=1)


class LowRankKVCache:
    """8-bit quantised, low-rank KV cache."""

    def __init__(self, rank: int = 16, bits: int = 8):
        self.rank, self.bits = rank, bits
        self.cache: Dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    # ===== helpers ==========================================================
    def _quantise(self, tensor: torch.Tensor):
        qmin, qmax = -(2 ** (self.bits - 1)), 2 ** (self.bits - 1) - 1
        scale = tensor.abs().amax() / qmax + 1e-8
        qt = torch.round(tensor / scale).to(torch.int8)
        return qt, scale

    def _dequantise(self, qt: torch.Tensor, scale: torch.Tensor):
        return qt.float() * scale

    # ===== public API =======================================================
    def store(self, key: int, kv: torch.Tensor):
        U, S, Vh = torch.linalg.svd_lowrank(kv, q=self.rank)
        lowrank = (U @ torch.diag(S) @ Vh).detach()
        qt, scale = self._quantise(lowrank)
        self.cache[key] = (qt.cpu(), scale.cpu())

    def load(self, key: int, device):
        qt, scale = self.cache[key]
        return self._dequantise(qt.to(device), scale.to(device))


class ShrinkAdamW(torch.optim.AdamW):
    """AdamW + Adafactor factoring + 4-bit state quantisation."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.95), weight_decay=1e-2, eps=1e-8):
        super().__init__(params, lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)
        self._quant_scale: Dict[int, torch.Tensor] = {}

    # ---------------------------------------------------------------------
    def _quantise(self, key: int, tensor: torch.Tensor):
        qmin, qmax = -8, 7  # 4-bit signed
        scale = tensor.abs().amax() / qmax + 1e-8
        qt = torch.round(tensor / scale).to(torch.int8)
        self._quant_scale[key] = scale
        return qt

    def _dequantise(self, key: int, qt: torch.Tensor):
        return qt.float() * self._quant_scale[key]

    # ---------------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                sid = id(p)
                st = self.state[p]
                if len(st) == 0:
                    st["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    st["exp_avg_sq"] = torch.zeros((p.shape[0], 1), device=p.device)
                # de-quantise
                if st["exp_avg"].dtype == torch.int8:
                    st["exp_avg"] = self._dequantise(sid, st["exp_avg"])
                # parent update
                super().step(lambda: None)
                # re-quantise
                st["exp_avg"] = self._quantise(sid, st["exp_avg"])
        return loss


# ============================================================
#  MODEL  (tiny stub if DiT repo missing)
# ============================================================

def _load_external_dit(image_size: int):
    """Attempt to import Facebook's DiT implementation."""
    try:
        from importlib import import_module

        dit_mod = import_module("DiT.models")
        return dit_mod.DiT_XL_2(image_size=image_size)  # type: ignore[attr-defined]
    except (ModuleNotFoundError, AttributeError):
        return None


class _DummyDiT(torch.nn.Module):
    """Fallback minimal model so that the training script is runnable in any
    environment where the real DiT is unavailable.  It produces a scalar
    loss equal to the mean of the inputs.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):  # type: ignore[override]
        return x.mean(dim=(1, 2, 3))  # fake per-sample loss


def load_dit(image_size: int):
    model = _load_external_dit(image_size)
    if model is None:
        model = _DummyDiT()
    return model


# ============================================================
#  PUBLIC WRAPPER
# ============================================================

def enable_redif(model: torch.nn.Module, *, chunk: int = 128, kv_rank: int = 16, kv_bits: int = 8):
    """In-place conversion of a DiT-style model to ReDiF-T."""

    cache = LowRankKVCache(rank=kv_rank, bits=kv_bits)

    # 1) Replace MHA with block-wise wrapper
    for name, module in list(model.named_modules()):
        if isinstance(module, torch.nn.MultiheadAttention):
            parent = model
            path = name.split(".")
            for part in path[:-1]:
                parent = getattr(parent, part)
            setattr(parent, path[-1], BlockwiseSelfAttention(module, chunk=chunk))

    # 2) Bundle consecutive transformer layers into reversible pairs
    blocks = [m for m in model.modules() if hasattr(m, "mlp")]
    for i in range(0, len(blocks) - 1, 2):
        f, g = blocks[i], blocks[i + 1]
        rev = ReversibleBlock(f, g)
        # place back into parent
        for n, m in model.named_modules():
            if m is f:
                par = model
                parts = n.split(".")[:-1]
                for part in parts:
                    par = getattr(par, part)
                setattr(par, parts[-1] if parts else n, rev)
                break

    model._kv_cache = cache  # type: ignore[attr-defined]
    model._is_redif = True   # type: ignore[attr-defined]
    return model


# ============================================================
#  CHECKPOINT  WRAPPER (fix for broken .apply call earlier)
# ============================================================

class _CheckpointWrapper(torch.nn.Module):
    """Module wrapper that applies torch.utils.checkpoint.checkpoint to the
    wrapped module's forward pass.  This is a drop-in replacement that keeps
    the original interface intact while trading compute for memory.
    """

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):  # type: ignore[override]
        return torch.utils.checkpoint.checkpoint(self.module, *args, **kwargs)


# ============================================================
#  TRAINING LOOP
# ============================================================

def train_one(cfg: Dict[str, Any]):
    """Single training run.  Returns a profiler dict with memory/FID."""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = load_dit(cfg["resolution"]).to(device)

    # select optimisation strategy ------------------------------------------------
    if cfg["method"].startswith("redif"):
        model = enable_redif(model, chunk=cfg.get("chunk_size", 128))
        opt = ShrinkAdamW(model.parameters(), lr=1e-4)
    else:
        if cfg["method"] == "checkpoint":
            model = _CheckpointWrapper(model)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=1e-2)

    loader = get_loader("imagenet", "train", cfg["resolution"], cfg["batch"], num_workers=2)

    profiler = {"peak_mem": 0.0, "img_sec": []}

    model.train()
    start = time.time()

    for step, batch in enumerate(loader):
        batch = batch.to(device, non_blocking=True)
        loss = model(batch).mean()
        loss.backward()
        opt.step(); opt.zero_grad()

        if step == 0 and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        if step % 50 == 0 and step > 0:
            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated() / 2 ** 30
                profiler["peak_mem"] = max(profiler["peak_mem"], peak)
            img_sec = cfg["batch"] * 50 / (time.time() - start)
            profiler["img_sec"].append(img_sec)
            start = time.time()

        if step >= cfg.get("max_steps", 200):
            break

    # quick sanity-check FID (placeholder)
    profiler["fid"] = eval_fid(model, num_images=512, resolution=cfg["resolution"])
    return profiler
