"""
preprocess.py
All dataset loading and preprocessing utilities (ImageNet-V2 & COCO-2017 val).
"""
from __future__ import annotations

from typing import Tuple

import pathlib
import tarfile
import zipfile

import requests
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

__all__ = [
    "get_imagenet_v2_loader",
]

_DATA_ROOT = pathlib.Path("data")
_DATA_ROOT.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
#  Download helpers
# ----------------------------------------------------------------------------


def _download(url: str, dest: pathlib.Path) -> pathlib.Path:
    """Stream-download *url* to *dest* with a progress-bar.

    The file is skipped if *dest* already exists.
    """

    if dest.exists():
        return dest

    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

    return dest


# ----------------------------------------------------------------------------
#  ImageNet-V2
# ----------------------------------------------------------------------------

_IMAGENETV2_URL = "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2.tar.gz"


def _prepare_imagenet_v2() -> pathlib.Path:
    tar_path = _DATA_ROOT / "imagenetv2.tar.gz"
    _download(_IMAGENETV2_URL, tar_path)

    extract_root = _DATA_ROOT / "imagenetv2"
    if extract_root.exists():
        return extract_root

    with tarfile.open(tar_path) as tar:
        tar.extractall(_DATA_ROOT)
    return extract_root


class _ImageFolder(Dataset):
    """Tiny wrapper adding deterministic transforms used for FID evaluation."""

    def __init__(self, root: pathlib.Path, size: int = 256):
        self._data = torchvision.datasets.ImageFolder(root)
        self._tfm = T.Compose([T.Resize(size + 32), T.CenterCrop(size), T.ToTensor()])

    # PyTorch Dataset protocol ------------------------------------------------
    def __len__(self):  # noqa: D401 – quick wrapper
        return len(self._data)

    def __getitem__(self, idx):  # noqa: D401 – quick wrapper
        img, _ = self._data[idx]
        return self._tfm(img)


# ----------------------------------------------------------------------------
#  Public loader helpers
# ----------------------------------------------------------------------------

def get_imagenet_v2_loader(*, batch_size: int, size: int = 256) -> DataLoader:
    """Returns a pinned-memory, non-shuffled DataLoader over ImageNet-V2."""

    root = _prepare_imagenet_v2()
    ds = _ImageFolder(root, size)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)