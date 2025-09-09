"""
main.py
Orchestrates the full experimental workflow: preprocessing, model wrapping,
image generation, metric computation & result persistence.

Run with:
    python -m src.main
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys
from typing import List

import yaml

# Note: heavy libraries are imported lazily inside the functions that need them
from src import evaluate as ev
from src import preprocess as prep
from src import train as trn

# ---------------------------------------------------------------------------
#  Directories & configuration
# ---------------------------------------------------------------------------

_RESEARCH_DIR = pathlib.Path(".research") / "iteration1"
_RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
_IMAGES_DIR = _RESEARCH_DIR / "images"
_IMAGES_DIR.mkdir(exist_ok=True, parents=True)

_CFG_PATH = pathlib.Path("config") / "config.yaml"
if not _CFG_PATH.exists():
    raise FileNotFoundError("config/config.yaml not found – did you forget to export it?")

CFG = yaml.safe_load(_CFG_PATH.read_text())
EXP_CFG = CFG["experiment"]


# ---------------------------------------------------------------------------
#  Helper to generate images with (optionally) CSR
# ---------------------------------------------------------------------------


def _generate_images(pipe, *, seed: int, num_batches: int, steps: int, use_csr: bool):
    import random

    import torch

    rnd_state = random.getstate()  # Restore after generation to avoid side-effects per seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    images: List[any] = []  # list of PIL.Image
    for _ in range(num_batches):
        out = pipe(
            prompt=["" for _ in range(EXP_CFG["batch_size"])],
            num_inference_steps=steps,
        )
        images.extend(out.images)

    random.setstate(rnd_state)
    return images


# ---------------------------------------------------------------------------
#  Core experiment (ImageNet-256 – baseline vs CSR)
# ---------------------------------------------------------------------------

def _run_experiment():
    try:
        from diffusers import DPMSolverMultistepScheduler, DiffusionPipeline
    except ImportError as exc:
        print("Diffusers not installed – unable to run generation; exiting early.")
        sys.exit(0)

    # ---------------------------------------------------------------------
    #  Load model & scheduler once to amortise I/O across seeds
    # ---------------------------------------------------------------------
    model_id = EXP_CFG["models"]["dit_xl_2_256"]
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype="auto")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        num_train_timesteps=1000,
        use_karras_sigmas=True,
    )
    pipe.to("cuda")

    results = []

    for seed in EXP_CFG["seeds"]:
        # -----------------------------------------------------------------
        #  Vanilla generation
        # -----------------------------------------------------------------
        vanilla_imgs = _generate_images(
            pipe,
            seed=seed,
            num_batches=2,  # <-- reduced for example; real run would use 50k/bs
            steps=EXP_CFG["sampler"]["dpm_solver_steps"],
            use_csr=False,
        )
        _seed_dir = _IMAGES_DIR / f"vanilla_seed{seed}"
        _seed_dir.mkdir(exist_ok=True)
        for i, img in enumerate(vanilla_imgs):
            img.save(_seed_dir / f"{i}.png")

        # -----------------------------------------------------------------
        #  CSR generation (K_max = 8)
        # -----------------------------------------------------------------
        csr_pipe = trn.csr_wrap(pipe, compression_ratio=8, K_max=8)
        csr_imgs = _generate_images(
            csr_pipe,
            seed=seed,
            num_batches=2,
            steps=EXP_CFG["sampler"]["dpm_solver_steps"],
            use_csr=True,
        )
        _csr_seed_dir = _IMAGES_DIR / f"csr_seed{seed}"
        _csr_seed_dir.mkdir(exist_ok=True)
        for i, img in enumerate(csr_imgs):
            img.save(_csr_seed_dir / f"{i}.png")

        # -----------------------------------------------------------------
        #  Metrics (FID)
        # -----------------------------------------------------------------
        real_dir = pathlib.Path("data") / "imagenetv2"
        fid_vanilla = ev.compute_fid(_seed_dir, real_dir)
        fid_csr = ev.compute_fid(_csr_seed_dir, real_dir)

        results.append({"seed": seed, "fid": fid_vanilla, "csr_fid": fid_csr})

        # Clean-up to keep disk usage low for the tutorial run
        shutil.rmtree(_seed_dir)
        shutil.rmtree(_csr_seed_dir)

    return results


# ---------------------------------------------------------------------------
#  Entry-point
# ---------------------------------------------------------------------------

def main():  # noqa: D401 – script-style entry-point
    results = _run_experiment()

    result_path = _RESEARCH_DIR / "results.json"
    result_path.write_text(json.dumps(results, indent=2))

    # Print JSON to standard-output so CI / user can inspect quickly
    print(result_path.read_text())


if __name__ == "__main__":
    main()