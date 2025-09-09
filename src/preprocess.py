"""
src/preprocess.py – data loading / augmentation utilities
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

import yaml

import torch
import torchvision.transforms as T
from torch import Tensor

# WebDataset & diffusers are optional – the code falls back to stubs when
# unavailable so that the project remains runnable in minimal environments.
try:
    import webdataset as wds
except ModuleNotFoundError:  # pragma: no cover – stub fallback

    class _Stub:
        def __getattr__(self, item):  # noqa: D401
            raise ModuleNotFoundError("WebDataset is required at runtime for real training.")

    wds = _Stub()  # type: ignore[assignment]

# ---------------------------------------------------------------------------
#  CONFIG HANDLING
# ---------------------------------------------------------------------------
_DEF_CFG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"


def _load_cfg() -> Dict:
    with _DEF_CFG_PATH.open() as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
#  OPTIONAL  VAE ENCODER  (only loaded when actually requested)
# ---------------------------------------------------------------------------
_VAE = None


def _load_vae(device="cpu"):
    global _VAE
    if _VAE is None:
        try:
            from diffusers import AutoencoderKL

            _VAE = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
            _VAE.eval().requires_grad_(False).to(device)
        except ModuleNotFoundError:
            raise RuntimeError("diffusers not installed – set 'vae_encode: false' in the YAML or install the package.")
    return _VAE


@torch.no_grad()
def _vae_encode(img: Tensor):
    vae = _load_vae(device=img.device)
    latents = vae.encode(2 * img - 1).latent_dist.sample() * 0.18215
    return latents


# ---------------------------------------------------------------------------
#  TRANSFORM PIPELINE BUILDER
# ---------------------------------------------------------------------------

def _build_transform(ops: List[Dict], resolution: int):
    tfms = []
    for op in ops:
        if "random_resized_crop" in op:
            tfms.append(T.RandomResizedCrop(resolution, interpolation=T.InterpolationMode.BICUBIC))
        if "center_crop" in op:
            size = op["center_crop"]["size"]
            tfms.append(T.CenterCrop(size))
        if "rotation" in op:
            tfms.append(T.RandomRotation([0, 270], interpolation=T.InterpolationMode.BICUBIC))
        if "random_hflip" in op:
            p = op["random_hflip"].get("p", 0.5)
            tfms.append(T.RandomHorizontalFlip(p))
    tfms.append(T.ToTensor())  # uint8 → float32 [0,1]
    return T.Compose(tfms)


# ---------------------------------------------------------------------------
#  DATA LOADER FACTORY
# ---------------------------------------------------------------------------

def get_loader(name: str, split: str, resolution: int, batch_size: int, *, num_workers: int = 4):
    cfg = _load_cfg()["datasets"][name]

    # url may be str or {train, val}
    url = cfg["url"][split] if isinstance(cfg["url"], dict) else cfg["url"]
    shards = url if url.endswith(".tar") else url + "/*"

    ds = (
        wds.WebDataset(shards, handler=wds.warn_and_continue)  # type: ignore[attr-defined]
        .shuffle(10_000)
        .decode("pil")
        .to_tuple("jpg")
        .map(lambda img: _build_transform(cfg["preprocessing"], resolution)(img))
    )

    if any("vae_encode" in p for p in cfg["preprocessing"]):
        ds = ds.map(_vae_encode)

    loader = wds.WebLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)  # type: ignore[attr-defined]
    return loader
