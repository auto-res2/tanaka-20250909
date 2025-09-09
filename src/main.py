from __future__ import annotations

"""
src/main.py – orchestrator.  Launch with
    python -m src.main
"""
import itertools
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

from .evaluate import aggregate, lineplot, save_json
from .train import train_one

# ---------------------------------------------------------------------------
#  CONFIG  (load YAML produced from the original experiment code)
# ---------------------------------------------------------------------------
_CFG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with _CFG_PATH.open() as f:
    CFG_ROOT: Dict = yaml.safe_load(f)


# ---------------------------------------------------------------------------
#  EXPERIMENT 1  – SCALING CURVE
# ---------------------------------------------------------------------------

def _run_exp1():
    exp_cfg = CFG_ROOT["experiments"]["exp1_scaling"]
    resolutions: List[int] = exp_cfg["grid"]["resolution"]
    methods: List[str] = exp_cfg["grid"]["method"]
    seeds: List[int] = exp_cfg["seeds"]

    peak_dict = {m: [] for m in methods}

    for R, m in itertools.product(resolutions, methods):
        metrics: List[float] = []
        for sd in seeds:
            np.random.seed(sd)
            cfg_run = {"resolution": R, "method": m, "batch": 4, "max_steps": 100}
            prof = train_one(cfg_run)
            metrics.append(prof["peak_mem"])  # type: ignore[arg-type]
        mu, sigma = aggregate(metrics)
        peak_dict[m].append(mu)

    # ------------------------------------------------------------------
    #  FIGURE + JSON OUTPUT (as mandated by instructions)
    # ------------------------------------------------------------------
    fig_dir = Path(".research/iteration2/images")
    fig_dir.mkdir(parents=True, exist_ok=True)
    figure_path = fig_dir / "exp1_peak_memory.pdf"
    lineplot(resolutions, [peak_dict[m] for m in methods], methods, "Peak GPU Memory", str(figure_path))

    save_json({"experiment": "exp1_scaling", "figure": str(figure_path)}, "exp1_results.json")


# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------

def main():
    print("=============== ReDiF-T  EXPERIMENT SUITE =================")
    _run_exp1()


if __name__ == "__main__":
    main()