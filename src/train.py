"""
train.py
Contains model–related classes and helper utilities for the Compressed-State Reuse (CSR) method.
"""
from __future__ import annotations

import types
from typing import Dict, Any

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
#  Core CSR Components
# ---------------------------------------------------------------------------

class CSREncoder(nn.Module):
    """1×1 depth-wise convolution followed by vector-quantisation.

    Parameters
    ----------
    in_ch: int
        Number of input channels.
    compression_ratio: int, default = 8
        Channel reduction factor.
    codebook_size: int, default = 512
        Size of the VQ code-book.
    """

    def __init__(self, in_ch: int, compression_ratio: int = 8, codebook_size: int = 512):
        super().__init__()
        if in_ch % compression_ratio != 0:
            raise ValueError("in_ch must be divisible by compression_ratio")
        self.conv = nn.Conv2d(in_ch, in_ch // compression_ratio, kernel_size=1, groups=in_ch)
        self.codebook_size = codebook_size
        self.codebook_dim = in_ch // compression_ratio
        self.codebook = nn.Embedding(codebook_size, self.codebook_dim)
        nn.init.uniform_(self.codebook.weight, -1 / self.codebook_dim ** 0.5, 1 / self.codebook_dim ** 0.5)

    @torch.no_grad()
    def forward(self, x: torch.Tensor):  # noqa: D401, N802 – signature kept identical to paper
        z_e = self.conv(x)  # (B, C', H, W)
        # Flatten spatial dimensions, then compute L2-distance to code-book entries.
        flat = z_e.permute(0, 2, 3, 1).reshape(-1, self.codebook_dim)
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(1)
        )
        idx = dist.argmin(dim=-1)
        z_q = (
            self.codebook(idx)
            .view(z_e.permute(0, 2, 3, 1).shape)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        return z_q, idx.view(z_q.shape[0], 1, *z_q.shape[2:])


class CSRDecoder(nn.Module):
    """Symmetric lightweight decoder which reconstructs the hidden feature."""

    def __init__(self, out_ch: int, compression_ratio: int = 8):
        super().__init__()
        if out_ch % compression_ratio != 0:
            raise ValueError("out_ch must be divisible by compression_ratio")
        self.deconv = nn.Conv2d(out_ch // compression_ratio, out_ch, kernel_size=1, groups=out_ch // compression_ratio)

    def forward(self, z_q: torch.Tensor):  # noqa: D401 – signature mirrored for clarity
        return self.deconv(z_q)


class CSRRouter(nn.Module):
    """Predicts categorical reuse length *K* ∈ {0 … K_max}."""

    def __init__(self, feature_dim: int, K_max: int = 8, age_lambda: float = 0.1):
        super().__init__()
        self.K_max = K_max
        self.age_lambda = age_lambda
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.SiLU(),
            nn.Linear(feature_dim // 4, K_max + 1),
        )

    def forward(self, pooled: torch.Tensor, age_onehot: torch.Tensor):  # noqa: D401 – kept simple
        logits = self.mlp(pooled) - self.age_lambda * age_onehot  # ageing penalty encourages longer reuse
        return torch.softmax(logits, dim=-1)


class CSRCacheManager:
    """Maintains a per-layer cache of compressed tensors with ageing."""

    def __init__(self, K_max: int):
        self.K_max = K_max
        # layer_name → {timestep → tensor}
        self._cache: Dict[str, Dict[int, torch.Tensor]] = {}

    # ---------------------------------------------------------------------
    #  CRUD
    # ---------------------------------------------------------------------
    def get(self, layer: str, t: int):
        layer_cache = self._cache.get(layer, {})
        return layer_cache.get(t)

    def put(self, layer: str, t: int, tensor: torch.Tensor):
        if layer not in self._cache:
            self._cache[layer] = {}
        self._cache[layer][t] = tensor

        # Drop cache entries older than K_max steps to bound VRAM.
        for old_t in list(self._cache[layer].keys()):
            if t - old_t > self.K_max:
                del self._cache[layer][old_t]

    # ---------------------------------------------------------------------
    #  Diagnostics helpers
    # ---------------------------------------------------------------------
    def vram_bytes(self) -> int:
        total = 0
        for layer_cache in self._cache.values():
            for tensor in layer_cache.values():
                total += tensor.element_size() * tensor.nelement()
        return total


# ---------------------------------------------------------------------------
#  Utility to wrap a Diffusers UNet with CSR modules (no retraining required)
# ---------------------------------------------------------------------------

def csr_wrap(pipe, *, compression_ratio: int = 8, K_max: int = 8):
    """Inserts CSR encoder/decoder/router after every SpatialTransformer block.

    Parameters
    ----------
    pipe : diffusers.DiffusionPipeline
        The diffusion pipeline to be wrapped.
    compression_ratio : int
        Channel reduction factor for the encoder and decoder.
    K_max : int
        Maximum number of timesteps a cached state can be reused.
    """

    # Local import to avoid a hard dependency for users who only preprocess.
    try:
        from diffusers.models.attention_processor import SpatialTransformer
    except ImportError as exc:  # pragma: no cover – informative error
        raise ImportError("diffusers >= 0.20 is required for csr_wrap") from exc

    cache_mgr = CSRCacheManager(K_max)
    device = pipe.device if hasattr(pipe, "device") else torch.device("cpu")

    for name, module in pipe.unet.named_modules():
        if isinstance(module, SpatialTransformer):
            in_ch = module.norm.out_channels  # type: ignore[attr-defined]
            enc = CSREncoder(in_ch, compression_ratio).to(device)
            dec = CSRDecoder(in_ch, compression_ratio).to(device)
            router = CSRRouter(in_ch, K_max).to(device)

            module.__dict__["_csr_components"] = {"enc": enc, "dec": dec, "router": router}
            original_forward = module.forward  # noqa: B008 – we preserve the bound method

            def _forward(self, hidden_states: torch.Tensor, *args, **kwargs):  # noqa: ANN001
                timestep = int(kwargs.get("timestep", 0))
                key = f"{name}"

                # 1) Attempt reuse if present in cache.
                cached = cache_mgr.get(key, timestep)
                if cached is not None:
                    return cached

                # 2) Otherwise compress → compute original → store reconstructed state.
                with torch.no_grad():
                    z_q, _ = enc(hidden_states.detach())
                    rec = dec(z_q)
                out = original_forward(hidden_states, *args, **kwargs)
                cache_mgr.put(key, timestep, rec)
                return out

            # Bind the patched method.
            module.forward = types.MethodType(_forward, module)

    return pipe
