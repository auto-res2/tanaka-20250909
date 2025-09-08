from __future__ import annotations

"""
main.py – experiment orchestration
Run via:  python -m src.main
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml

from .preprocess import load_dataset
from .train import FlashGAT, train_one_epoch
from .evaluate import evaluate, verify_implementation

# -----------------------------------------------------------------------------
# 1.  Load configuration (env override is supported)
# -----------------------------------------------------------------------------

CFG_PATH = os.getenv(
    "FLASHGAT_CONFIG", str(Path(__file__).parent.parent / "config" / "config.yaml")
)
with open(CFG_PATH, "r", encoding="utf-8") as fp:
    CONFIG = yaml.safe_load(fp)


# -----------------------------------------------------------------------------
# 2.  Utility helpers
# -----------------------------------------------------------------------------


def _prepare_dirs():
    """Ensure the mandatory research directory structure exists."""

    # All artefacts for *this* iteration must live under `.research/iteration7`.
    img_dir = Path(".research/iteration7/images")
    img_dir.mkdir(parents=True, exist_ok=True)

    json_dir = Path(".research/iteration7")
    json_dir.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 3.  Main routine
# -----------------------------------------------------------------------------


def run():
    _prepare_dirs()

    seed = CONFIG["seeds"][0]
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mp = CONFIG["hardware"].get("mixed_precision", "bf16") == "bf16"

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset_name = "ogbn-products"  # single-dataset demo
    data, num_classes = load_dataset(dataset_name)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = FlashGAT(
        in_dim=data.x.size(-1),
        hid_dim=CONFIG["model"]["hidden_dim"],
        num_layers=CONFIG["model"]["num_layers"],
        heads=CONFIG["model"]["heads"],
        B=CONFIG["flashgat"]["B"],
        causal=CONFIG["flashgat"]["causal"],
        num_classes=num_classes,
    ).to(device)
    model.set_edge_memory(num_edges=data.edge_index.size(1), device=device)

    # optimiser + AMP scaler
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["optim"]["lr"],
        weight_decay=CONFIG["optim"]["weight_decay"],
    )
    scaler = torch.cuda.amp.GradScaler()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    t0 = time.time()
    best_val = 0.0
    for epoch in range(CONFIG["optim"]["epochs"]):
        loss = train_one_epoch(model, data, optim, scaler, device, mp)
        val_acc = evaluate(model, data, data.valid_mask, device)
        best_val = max(best_val, val_acc)
        print(f"Epoch {epoch:02d} | loss={loss:.4f} | val={val_acc*100:.2f}%")
        if val_acc >= CONFIG["datasets"][dataset_name]["val_acc_target"]:
            print("Target validation accuracy reached – early stop")
            break

    test_acc = evaluate(model, data, data.test_mask, device)
    elapsed = time.time() - t0

    # ------------------------------------------------------------------
    # Output JSON & persistence
    # ------------------------------------------------------------------
    result = {
        "seed": seed,
        "dataset": dataset_name,
        "wall_clock_s": elapsed,
        "val_acc": best_val,
        "test_acc": test_acc,
    }

    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out_path = Path(".research/iteration7") / f"flashgat_{dataset_name}_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2)

    # mandatory STDOUT for verification
    print("\n=== Implementation Verification ===")
    print("Passed" if verify_implementation() else "Failed")
    print("\n=== Results (also saved to .research/iteration7) ===")
    print(json.dumps(result, indent=2))


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    run()
