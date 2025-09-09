"""
main.py – orchestrates the full experimental workflow
Executable:  python -m src.main  OR  python -m main  (depending on PYTHONPATH)
"""
from __future__ import annotations
import argparse
import json
import os
import pathlib
from typing import Any, Dict

import yaml
import torch

from .train import Trainer
from .evaluate import Evaluator
from .preprocess import set_global_seed

# -----------------------------------------------------------------------------
#  Helper – load YAML then override with CLI flags if provided
# -----------------------------------------------------------------------------
CONFIG_PATH = pathlib.Path(__file__).parent.parent / "config" / "config.yaml"

def load_cfg(path: os.PathLike | None = None) -> Dict[str, Any]:
    with open(path or CONFIG_PATH, "r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)
    return cfg

# -----------------------------------------------------------------------------
#  Experiment runner
# -----------------------------------------------------------------------------
RESEARCH_DIR = pathlib.Path(".research") / "iteration1"
RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
(RESEARCH_DIR / "images").mkdir(exist_ok=True)


def run_single_experiment(exp_name: str, exp_cfg: Dict[str, Any]) -> Dict[str, float]:
    print(f"\n===== {exp_name} =====")
    set_global_seed(exp_cfg.get("seed", 42))

    # 1) train -------------------------------------------------------------
    trainer = Trainer(exp_cfg)
    trainer.train()

    # save checkpoint for later evaluation
    ckpt_path = RESEARCH_DIR / f"{exp_name}_model.pt"
    torch.save(trainer.trained_model.state_dict(), ckpt_path)

    # 2) evaluate ----------------------------------------------------------
    eval_cfg = {**exp_cfg, "ckpt_path": str(ckpt_path)}
    evaluator = Evaluator(eval_cfg)
    results = evaluator.evaluate()

    # 3) store json --------------------------------------------------------
    out_json = RESEARCH_DIR / f"{exp_name}_results.json"
    with out_json.open("w") as fp:
        json.dump(results, fp, indent=2)
    print(json.dumps(results, indent=2))

    return results

# -----------------------------------------------------------------------------
#  Entry point
# -----------------------------------------------------------------------------

def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="CHANGEFORMER runner")
    parser.add_argument("--cfg", type=str, default=None, help="Path to config.yaml override")
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
    exp_results: Dict[str, Dict[str, float]] = {}
    for exp_name, exp_cfg in cfg["experiments"].items():
        exp_results[exp_name] = run_single_experiment(exp_name, exp_cfg)

    # final summary --------------------------------------------------------
    summary_path = RESEARCH_DIR / "all_experiments_summary.json"
    with summary_path.open("w") as fp:
        json.dump(exp_results, fp, indent=2)
    print("\nAll experiments finished – summary written to", summary_path)


if __name__ == "__main__":
    main()
