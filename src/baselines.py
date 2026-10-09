"""
Supervised baselines for the CartPole failure-prediction task.

Machine Learning - Workshop 1

Three traditional classifiers are trained on the engineered features:
    Baseline 1: Logistic Regression   (linear model)
    Baseline 2: Random Forest         (tree-based bagging ensemble)
    Baseline 3: Gradient Boosting     (tree-based boosting ensemble)

Evaluation protocol
-------------------
* Models are fitted on TRAIN only.
* The decision threshold is chosen on VALIDATION (maximising F1), so the test
  set is never used to tune anything.
* Metrics are reported on TEST at the validation threshold and at 0.5.
* Accuracy is reported but NOT used for decisions: with ~5.5% positives, a model
  that always predicts "safe" already reaches ~94.5% accuracy.

Usage
-----
    python src/dataset.py                 # create data/splits.json (once)
    python src/baselines.py               # train + evaluate
    python src/baselines.py --augment     # add noisy copies of the train set
    python src/baselines.py --raw-only    # ablation: no engineered features
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight

from dataset import (
    RAW_FEATURE_COLS,
    ROOT_DIR,
    SEED,
    TARGET_COL,
    apply_scaler,
    augment_with_noise,
    fit_scaler,
    get_or_create_splits,
    load_raw,
    set_global_seed,
    subset,
)
from features import build_features

RUNS_DIR = ROOT_DIR / "runs"


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def prepare_data(
    augment: bool = False, raw_only: bool = False
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """Build scaled (X, y) for train / val / test following the saved split.

    Pipeline order (important to avoid leakage):
        1. Load the saved episode split.
        2. (Optional) augment the TRAIN raw rows only.
        3. Build features per partition (temporal features stay inside episodes).
        4. Fit the scaler on TRAIN and apply it to val/test.

    Args:
        augment (bool): If True, append noisy copies of the training episodes.
        raw_only (bool): If True, use only the 5 raw columns (ablation of the
            feature-engineering step).

    Returns:
        Dict[str, Tuple[pd.DataFrame, pd.Series]]: ``{"train": (X, y), ...}``.
    """
    df = load_raw()
    splits = get_or_create_splits(df)

    parts = {name: subset(df, eps) for name, eps in splits.items()}

    if augment:
        parts["train"] = augment_with_noise(parts["train"])

    data = {}
    for name, part in parts.items():
        X = build_features(part)
        if raw_only:
            X = X[RAW_FEATURE_COLS]
        data[name] = (X, part[TARGET_COL].astype(int))

    # Normalisation parameters come from TRAIN only
    scaler = fit_scaler(data["train"][0])
    return {
        name: (apply_scaler(scaler, X), y) for name, (X, y) in data.items()
    }


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def get_models(seed: int = SEED) -> Dict[str, object]:
    """Instantiate the three baseline classifiers.

    Class imbalance is handled with class weights (linear model and forest) or
    with sample weights (boosting), never by touching val/test.

    Args:
        seed (int): Random seed.

    Returns:
        Dict[str, object]: Model name -> unfitted estimator.
    """
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=seed
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, random_state=seed
        ),
    }


def fit_model(name: str, model: object, X: pd.DataFrame, y: pd.Series) -> object:
    """Fit one model (boosting receives balanced sample weights).

    Args:
        name (str): Model name.
        model (object): Unfitted estimator.
        X (pd.DataFrame): Training features.
        y (pd.Series): Training labels.

    Returns:
        object: Fitted estimator.
    """
    if name == "Gradient Boosting":
        weights = compute_sample_weight("balanced", y)
        model.fit(X, y, sample_weight=weights)
    else:
        model.fit(X, y)
    return model


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def best_f1_threshold(y_true: pd.Series, scores: np.ndarray) -> float:
    """Find the probability threshold that maximises F1 (use on VALIDATION).

    Args:
        y_true (pd.Series): True labels.
        scores (np.ndarray): Predicted probabilities of the positive class.

    Returns:
        float: Threshold with the highest F1 on the given data.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])


def compute_metrics(y_true: pd.Series, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    """Compute the classification metrics requested by the workshop.

    Args:
        y_true (pd.Series): True labels.
        scores (np.ndarray): Predicted probabilities of the positive class.
        threshold (float): Decision threshold.

    Returns:
        Dict[str, float]: accuracy, precision, recall, f1, roc_auc, pr_auc and
        the confusion-matrix counts (tn, fp, fn, tp).
    """
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# --------------------------------------------------------------------------- #
# Main experiment
# --------------------------------------------------------------------------- #
def run_baselines(augment: bool = False, raw_only: bool = False) -> pd.DataFrame:
    """Train the three baselines and evaluate them on val and test.

    Args:
        augment (bool): Use noise augmentation on the training set.
        raw_only (bool): Use only the raw features (feature-engineering ablation).

    Returns:
        pd.DataFrame: One row per (model, evaluation setting) with all metrics.
    """
    set_global_seed()
    data = prepare_data(augment=augment, raw_only=raw_only)
    X_tr, y_tr = data["train"]
    X_va, y_va = data["val"]
    X_te, y_te = data["test"]

    print("=" * 70)
    print(f"BASELINES | augmentation = {augment} | raw features only = {raw_only}")
    print("=" * 70)
    print(f"train: {X_tr.shape} | val: {X_va.shape} | test: {X_te.shape}")
    print(f"positives -> train {y_tr.mean():.3%} | val {y_va.mean():.3%} | test {y_te.mean():.3%}\n")

    rows = []
    for name, model in get_models().items():
        t0 = time.time()
        model = fit_model(name, model, X_tr, y_tr)
        fit_time = time.time() - t0

        # Threshold selected on VALIDATION only
        val_scores = model.predict_proba(X_va)[:, 1]
        thr = best_f1_threshold(y_va, val_scores)

        test_scores = model.predict_proba(X_te)[:, 1]
        for setting, t in [("test @ 0.5", 0.5), ("test @ val-threshold", thr)]:
            m = compute_metrics(y_te, test_scores, t)
            rows.append({"model": name, "setting": setting, "fit_time_s": fit_time, **m})

        # Console report for the validation-threshold setting
        m = rows[-1]
        print(f"--- {name}  (fit {fit_time:.1f}s, val-threshold = {thr:.3f})")
        print(
            f"    precision {m['precision']:.3f} | recall {m['recall']:.3f} | "
            f"F1 {m['f1']:.3f} | ROC-AUC {m['roc_auc']:.3f} | PR-AUC {m['pr_auc']:.3f} | "
            f"acc {m['accuracy']:.3f}"
        )
        print(f"    confusion matrix [[TN FP] [FN TP]] = [[{m['tn']} {m['fp']}] [{m['fn']} {m['tp']}]]\n")

    results = pd.DataFrame(rows)
    tag = "raw" if raw_only else ("augmented" if augment else "base")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RUNS_DIR / f"baseline_results_{tag}.csv", index=False)
    with open(RUNS_DIR / f"baseline_results_{tag}.json", "w") as f:
        json.dump(results.to_dict(orient="records"), f, indent=2)
    print(f"Results saved in {RUNS_DIR}/baseline_results_{tag}.(csv|json)")
    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train CartPole baselines")
    parser.add_argument("--augment", action="store_true", help="noise augmentation (train only)")
    parser.add_argument("--raw-only", action="store_true", help="use only raw features (ablation)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_baselines(augment=args.augment, raw_only=args.raw_only)
