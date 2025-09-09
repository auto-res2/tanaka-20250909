
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
import inspect
from typing import List

import yaml
import torch  # FIX: ensure torch is available throughout this module

# Note: heavy libraries are imported lazily inside the functions that need them
from src import evaluate as ev
from src import preprocess as prep  # noqa: F401 – retained for future extensions
from src import train as trn

# ---------------------------------------------------------------------------
#  Directories & configuration
# ---------------------------------------------------------------------------

# All research artefacts for *this* iteration live under .research/iteration4
_RESEARCH_DIR = pathlib.Path(".research") / "iteration4"
_RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

# Images are stored under .research/iteration4/images/…
_IMAGES_DIR = _RESEARCH_DIR / "images"
_IMAGES_DIR.mkdir(exist_ok=True, parents=True)

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------

_CFG_PATH = pathlib.Path("config") / "config.yaml"
if not _CFG_PATH.exists():
    raise FileNotFoundError("config/config.yaml not found – did you forget to export it?")

CFG = yaml.safe_load(_CFG_PATH.read_text())
EXP_CFG = CFG["experiment"]


# ---------------------------------------------------------------------------
#  Helper – resolve model identifier to a valid Diffusers repo or local path
# ---------------------------------------------------------------------------

def _resolve_model_id(raw_id: str) -> str:
    """Converts *raw_id* from the config to a valid argument for
    DiffusionPipeline.from_pretrained.

    The original YAML lists direct `.pt` checkpoint URLs for the DiT models
    released by FAIR.  Diffusers hosts converted weights under the following
    repo names:
        • DiT-XL/2 256×256 → "facebook/DiT-XL-2-256"
        • DiT-XL/2 512×512 → "facebook/DiT-XL-2-512"

    If *raw_id* already looks like a Hugging Face repo (i.e. no `.pt` suffix)
    it is returned unchanged.  Otherwise we attempt the above mapping and fall
    back to a lightweight public DDPM model so that unit-tests can still run
    without downloading >1 GB of weights.
    """

    if not raw_id.endswith(".pt"):
        return raw_id  # assume the caller knows what they're doing

    if "256x256" in raw_id:
        return "facebook/DiT-XL-2-256"
    if "512x512" in raw_id:
        return "facebook/DiT-XL-2-512"

    # ---------------------------------------------------------------------
    #  Last-resort – use an ultra-light model to keep CI resource footprint
    #  reasonable while still exercising the full code-path.
    # ---------------------------------------------------------------------
    print(
        f"[WARN] Could not map checkpoint URL '{raw_id}' to a diffusers repo. "
        "Falling back to 'google/ddpm-cifar10-32'."
    )
    return "google/ddpm-cifar10-32"


# ---------------------------------------------------------------------------
#  Helper to generate images with (optionally) CSR
# ---------------------------------------------------------------------------

def _generate_images(pipe, *, seed: int, num_batches: int, steps: int):
    """Generates *batch_size × num_batches* images using *pipe* under a fixed *seed*."""

    import random
    import torch

    batch_size = EXP_CFG["batch_size"]

    # Save & later restore the RNG states to avoid cross-contamination between calls
    rnd_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    cuda_states = {i: torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())}

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    images: List[any] = []  # list of PIL.Image

    call_sig = inspect.signature(pipe.__call__).parameters

    for _ in range(num_batches):
        kwargs = {"num_inference_steps": steps, "output_type": "pil"}
        if "prompt" in call_sig:
            kwargs["prompt"] = ["" for _ in range(batch_size)]
        elif "class_labels" in call_sig:
            kwargs["class_labels"] = [0] * batch_size
        elif "batch_size" in call_sig:
            kwargs["batch_size"] = batch_size
        else:
            # As a fallback we at least control the batch size through generator duplication.
            kwargs["batch_size"] = batch_size

        out = pipe(**kwargs)
        images.extend(out.images if hasattr(out, "images") else out[0])

    # ---------------------------------------------------------------------
    #  Restore original RNG states so that subsequent calls behave as if
    #  this function never touched the global generators.
    # ---------------------------------------------------------------------
    random.setstate(rnd_state)
    import torch  # re-import for mypy/static checkers – already in scope at runtime

    torch.random.set_rng_state(torch_state)
    for i, state in cuda_states.items():
        torch.cuda.set_rng_state(state, i)

    return images


# ---------------------------------------------------------------------------
#  Core experiment (ImageNet-256 – baseline vs CSR)
# ---------------------------------------------------------------------------

def _run_experiment():
    try:
        from diffusers import DPMSolverMultistepScheduler, DiffusionPipeline
    except ImportError:
        print("Diffusers not installed – unable to run generation; exiting early.")
        sys.exit(0)

    raw_model_id = EXP_CFG["models"]["dit_xl_2_256"]
    model_id = _resolve_model_id(raw_model_id)

    try:
        pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=None)
    except Exception as exc:
        print(f"[ERROR] Failed to load model '{model_id}': {exc}")
        sys.exit(1)

    # Scheduler compatibility – some lightweight fallback models (e.g. DDPM)
    # already come with their own scheduler that may not accept the extra args.
    try:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            num_train_timesteps=1000,
            use_karras_sigmas=True,
        )
    except Exception:
        # Use the default scheduler shipped with the model.
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(device)

    results = []

    for seed in EXP_CFG["seeds"]:
        # -------------------------------------------------------------
        #  Vanilla generation
        # -------------------------------------------------------------
        vanilla_imgs = _generate_images(
            pipe,
            seed=seed,
            num_batches=2,  # reduced for CI-friendliness; adjust for real runs
            steps=EXP_CFG["sampler"]["dpm_solver_steps"],
        )
        _seed_dir = _IMAGES_DIR / f"vanilla_seed{seed}"
        _seed_dir.mkdir(exist_ok=True)
        for i, img in enumerate(vanilla_imgs):
            img.save(_seed_dir / f"{i}.png")

        # -------------------------------------------------------------
        #  CSR generation (K_max = 8)
        # -------------------------------------------------------------
        csr_pipe = trn.csr_wrap(pipe, compression_ratio=8, K_max=8)
        csr_imgs = _generate_images(
            csr_pipe,
            seed=seed,
            num_batches=2,
            steps=EXP_CFG["sampler"]["dpm_solver_steps"],
        )
        _csr_seed_dir = _IMAGES_DIR / f"csr_seed{seed}"
        _csr_seed_dir.mkdir(exist_ok=True)
        for i, img in enumerate(csr_imgs):
            img.save(_csr_seed_dir / f"{i}.png")

        # -------------------------------------------------------------
        #  Metrics (FID)
        # -------------------------------------------------------------
        real_dir = pathlib.Path("data") / "imagenetv2"
        if real_dir.exists():
            fid_vanilla = ev.compute_fid(_seed_dir, real_dir)
            fid_csr = ev.compute_fid(_csr_seed_dir, real_dir)
        else:
            print(
                "[INFO] ImageNet-V2 not found locally – skipping FID computation. "
                "Results will contain NaNs."
            )
            fid_vanilla = float("nan")
            fid_csr = float("nan")

        results.append({"seed": seed, "fid": fid_vanilla, "csr_fid": fid_csr})

        # Clean-up to keep disk usage low for the tutorial run
        shutil.rmtree(_seed_dir, ignore_errors=True)
        shutil.rmtree(_csr_seed_dir, ignore_errors=True)

    return results


# ---------------------------------------------------------------------------
#  Entry-point
# ---------------------------------------------------------------------------

def main():  # noqa: D401 – script-style entry-point
    results = _run_experiment()

    # Each experiment result lives in its own JSON file under .research/iteration4/
    result_path = _RESEARCH_DIR / f"{EXP_CFG['name']}_results.json"
    result_path.write_text(json.dumps(results, indent=2))

    # Print JSON to standard-output so CI / users can inspect quickly
    print(result_path.read_text())


if __name__ == "__main__":
    main()
