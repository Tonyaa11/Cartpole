"""
Test-set evaluation of the trained deep model (Workshop 2).

Machine Learning - Workshop 2 (Advanced Learning Models, Part A)

What it does
------------
1. Loads ``checkpoints/model_best.pt`` (model weights + scaler + threshold
   saved by ``src/train.py``) WITHOUT retraining anything.
2. Rebuilds the same past-only windows of ``data/splits.json`` (same split,
   same rows as Workshop 1) and runs inference on the held-out TEST episodes.
3. Reports the SAME metrics as Workshop 1 (accuracy, precision, recall, F1,
   ROC-AUC, PR-AUC, confusion matrix) at two thresholds:
   * 0.5, and
   * the threshold selected on the VALIDATION set during training.
4. Loads ``runs/baseline_results_base.csv`` (Workshop 1) and prints a
   comparison table: baselines vs. the deep model.
5. Persists everything to ``runs/eval_<run_name>.csv|json``.

Usage
-----
    python src/evaluate.py                       # best checkpoint
    python src/evaluate.py --checkpoint checkpoints/model_best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from dataset import ROOT_DIR, load_raw, load_splits
from model import build_model
from sequences import build_all_partitions
from train_lstm import best_f1_threshold, evaluate_loader, make_loaders, set_global_seed

RUNS_DIR = ROOT_DIR / "runs" / "workshop2"
CKPT_PATH = ROOT_DIR / "checkpoints" / "model_best.pt"
W1_RESULTS = ROOT_DIR / "runs" / "workshop1" / "baseline_results_base.csv"


def load_checkpoint(path: Path = CKPT_PATH) -> dict:
    """Load a checkpoint dictionary (weights are loaded separately).

    Args:
        path (Path): Path to the ``.pt`` file.

    Returns:
        dict: Checkpoint contents (``model_cfg``, ``window``, ``features``,
        ``threshold``, ``seed``, ...).

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Train first: python src/train.py"
        )
    return torch.load(path, map_location="cpu", weights_only=False)


def evaluate(checkpoint_path: Path = CKPT_PATH) -> dict:
    """Evaluate the saved model on the TEST split and compare with Workshop 1.

    Args:
        checkpoint_path (Path): Checkpoint to evaluate.

    Returns:
        dict: ``{"deep": {...metrics...}, "comparison": DataFrame}``.
    """
    set_global_seed(42)
    ckpt = load_checkpoint(checkpoint_path)

    # Rebuild data with the SAME configuration stored in the checkpoint
    df = load_raw()
    splits = load_splits()
    parts, _ = build_all_partitions(
        df, splits, window=int(ckpt["window"]), mode=str(ckpt["features"])
    )
    loaders = make_loaders(parts, batch_size=256, seed=int(ckpt.get("seed", 42)))

    model = build_model(ckpt["model_cfg"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    criterion = torch.nn.BCEWithLogitsLoss()  # only for a comparable val loss
    y_te, s_te, _ = evaluate_loader(model, loaders["test"], criterion, torch.device("cpu"))
    y_va, s_va, _ = evaluate_loader(model, loaders["val"], criterion, torch.device("cpu"))

    # Threshold selected on VALIDATION (stored in the checkpoint; recomputed
    # here as a consistency check - both must agree)
    thr_stored = float(ckpt["threshold"])
    thr_recomputed = best_f1_threshold(y_va, s_va)

    def metrics(y, scores, thr) -> dict:
        pred = scores >= thr
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        return {
            "threshold": float(thr),
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, scores),
            "pr_auc": average_precision_score(y, scores),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        }

    deep = {
        "model": "Deep LSTM (Workshop 2)" if ckpt["model_cfg"]["cell"] == "lstm"
        else f"Deep {ckpt['model_cfg']['cell'].upper()} (Workshop 2)",
        "run_name": ckpt.get("run_name", "model_best"),
        "best_epoch": ckpt.get("best_epoch"),
        "window": ckpt["window"],
        "features": ckpt["features"],
        "threshold_stored": thr_stored,
        "threshold_recomputed": thr_recomputed,
        "val_f1": metrics(y_va, s_va, thr_stored),
        "test_at_0.5": metrics(y_te, s_te, 0.5),
        "test_at_val_threshold": metrics(y_te, s_te, thr_stored),
    }

    # ---- Comparison with Workshop 1 -------------------------------------- #
    rows = []
    if W1_RESULTS.exists():
        w1 = pd.read_csv(W1_RESULTS)
        w1 = w1[w1["setting"] == "test @ val-threshold"]
        for _, r in w1.iterrows():
            rows.append(
                {
                    "model": r["model"], "workshop": "W1",
                    "precision": r["precision"], "recall": r["recall"],
                    "f1": r["f1"], "roc_auc": r["roc_auc"], "pr_auc": r["pr_auc"],
                    "accuracy": r["accuracy"],
                    "tn": int(r["tn"]), "fp": int(r["fp"]),
                    "fn": int(r["fn"]), "tp": int(r["tp"]),
                }
            )
    t = deep["test_at_val_threshold"]
    rows.append(
        {
            "model": deep["model"], "workshop": "W2",
            "precision": t["precision"], "recall": t["recall"], "f1": t["f1"],
            "roc_auc": t["roc_auc"], "pr_auc": t["pr_auc"],
            "accuracy": t["accuracy"],
            "tn": t["tn"], "fp": t["fp"], "fn": t["fn"], "tp": t["tp"],
        }
    )
    comparison = pd.DataFrame(rows)

    # ---- Console report --------------------------------------------------- #
    print("=" * 78)
    print(f"EVALUATION | checkpoint: {checkpoint_path}")
    print(f"window={ckpt['window']} | features={ckpt['features']} | "
          f"cell={ckpt['model_cfg']['cell']} | best epoch={ckpt.get('best_epoch')}")
    print(f"threshold (val): stored {thr_stored:.4f} | recomputed {thr_recomputed:.4f}")
    print("=" * 78)
    for name, m in [("VAL ", deep["val_f1"]),
                    ("TEST@0.5", deep["test_at_0.5"]),
                    ("TEST@thr", deep["test_at_val_threshold"])]:
        print(f"{name:>9}: acc {m['accuracy']:.3f} | P {m['precision']:.3f} | "
              f"R {m['recall']:.3f} | F1 {m['f1']:.3f} | AUC {m['roc_auc']:.3f} | "
              f"PR-AUC {m['pr_auc']:.3f} | [[{m['tn']} {m['fp']}] [{m['fn']} {m['tp']}]]")

    print("\n--- Comparison (test @ validation threshold) ---")
    print(
        comparison[["model", "workshop", "precision", "recall", "f1",
                    "roc_auc", "pr_auc"]]
        .to_string(index=False, float_format=lambda v: f"{v:.3f}")
    )

    # ---- Persist ---------------------------------------------------------- #
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    tag = deep["run_name"]
    out = {"checkpoint": str(checkpoint_path), **deep}
    with open(RUNS_DIR / f"eval_{tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    comparison.to_csv(RUNS_DIR / f"comparison_{tag}.csv", index=False)

    print(f"\nresults -> {RUNS_DIR / f'eval_{tag}.json'}")
    print(f"table   -> {RUNS_DIR / f'comparison_{tag}.csv'}")
    return {"deep": deep, "comparison": comparison}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    p = argparse.ArgumentParser(description="Evaluate the CartPole deep model")
    p.add_argument("--checkpoint", type=Path, default=CKPT_PATH,
                   help="path to the .pt checkpoint")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args().checkpoint)
