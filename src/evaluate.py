"""
evaluate.py
Evaluation utilities: FID, sFID, IS, Precision/Recall & simple statistical tests.
"""
from __future__ import annotations

from typing import Dict, List

import pathlib

import numpy as np
import scipy.stats as st
from cleanfid import fid

__all__ = [
    "MetricAggregator",
    "compute_fid",
    "paired_t_test",
]


class MetricAggregator:
    """Accumulates metric dictionaries from multiple experimental runs and returns
    mean ± std for each entry.
    """

    def __init__(self):
        self._runs: List[Dict[str, float]] = []

    # ---------------------------------------------------------------------
    #  Public API
    # ---------------------------------------------------------------------
    def add_run(self, metrics: Dict[str, float]):
        self._runs.append(metrics)

    def aggregate(self) -> Dict[str, Dict[str, float]]:
        if not self._runs:
            raise RuntimeError("No runs to aggregate – call add_run first.")

        keys = self._runs[0].keys()
        out: Dict[str, Dict[str, float]] = {}
        for k in keys:
            vals = [float(run[k]) for run in self._runs]
            out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        return out


# -------------------------------------------------------------------------
#  Individual metric helpers
# -------------------------------------------------------------------------

def compute_fid(fake_dir: str | pathlib.Path, real_dir: str | pathlib.Path, *, mode: str = "clean") -> float:
    """Thin wrapper around Clean-FID to ensure uniform num_workers & dtype."""

    return float(fid.compute_fid(str(real_dir), str(fake_dir), mode=mode, num_workers=4))


def paired_t_test(a: List[float], b: List[float]):  # noqa: D401 – scientific function name
    """Two-sided paired t-test used for per-seed significance checks."""

    return st.ttest_rel(a, b)