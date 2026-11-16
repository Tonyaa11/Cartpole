"""
make_figures_w2.py - Generate the Workshop 2 report figures as PNG files.

Machine Learning - Workshop 2 (Advanced Learning Models, Part A)

Every figure is computed from the real training logs (``runs/train_log_*.csv``),
the real checkpoint (``checkpoints/model_best.pt``) and the real Workshop 1
baselines (``runs/baseline_results_base.csv``); nothing is hard-coded.

Files written to ``report/figures/`` (names expected by ``report/main_w2.tex``):

    fig2-architecture.png  schematic of the model (input -> LSTM -> head)
    fig2-curves.png      train/val loss + validation metrics per epoch
    fig2-confusion.png   test confusion matrix of the deep model
    fig2-roc-pr.png      ROC and PR curves: deep model vs. the W1 baselines
    fig2-comparison.png  bar chart: W1 baselines vs. the W2 deep model

Usage
-----
    python src/make_figures_w2.py
    python src/make_figures_w2.py --run w2_gru_w16_raw   # other run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from dataset import ROOT_DIR, load_raw, load_splits
from model import build_model
from sequences import build_all_partitions
from train_lstm import evaluate_loader, make_loaders, set_global_seed

FIG_DIR = ROOT_DIR / "report" / "figures"
RUNS_DIR = ROOT_DIR / "runs" / "workshop2"
CKPT_PATH = ROOT_DIR / "checkpoints" / "workshop2" / "model_best.pt"
W1_RESULTS = ROOT_DIR / "runs" / "workshop1" / "baseline_results_base.csv"

PALETTE = ["#4C72B0", "#C44E52"]
W1_COLORS = {"Logistic Regression": "#8c8c8c",
             "Random Forest": "#dd8452",
             "Gradient Boosting": "#55a868"}
DPI = 200


def save(fig: plt.Figure, name: str) -> None:
    """Save a figure in ``report/figures`` and close it.

    Args:
        fig (plt.Figure): Figure to save.
        name (str): File name (with extension).

    Returns:
        None
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT_DIR)}")


def load_run(run_name: str | None) -> tuple[dict, pd.DataFrame]:
    """Load the best-checkpoint summary and the per-epoch training log.

    Args:
        run_name (str | None): Run name; if None, taken from the checkpoint.

    Returns:
        tuple[dict, pd.DataFrame]: Summary JSON and epoch log.
    """
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    name = run_name or ckpt.get("run_name", "w2_lstm_w16_raw")
    summary_path = RUNS_DIR / f"train_summary_{name}.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    else:
        summary = {"run_name": name}
    log_path = RUNS_DIR / f"train_log_{name}.csv"
    log = pd.read_csv(log_path)
    return {**summary, "ckpt": ckpt}, log


def fig_architecture() -> None:
    """Schematic diagram of the deep model (input window -> LSTM -> MLP head).

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(7.6, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    def box(x, w, title, subtitle, fc, ec):
        ax.add_patch(plt.Rectangle((x, 0.9), w, 1.2, facecolor=fc,
                                   edgecolor=ec, lw=1.4))
        ax.text(x + w / 2, 1.75, title, ha="center", va="center",
                fontsize=8.5, fontweight="bold")
        ax.text(x + w / 2, 1.3, subtitle, ha="center", va="center",
                fontsize=7)

    def arrow(x0, x1):
        ax.annotate("", xy=(x1, 1.5), xytext=(x0, 1.5),
                    arrowprops=dict(arrowstyle="->", lw=1.4, color="k"))

    box(0.2, 2.0, "Input window", "(B, 16, 5)\nz-score scaled", "#e8eef7", "#4C72B0")
    arrow(2.2, 2.6)
    box(2.6, 2.2, "pack_padded", "masks padding", "#f2f2f2", "#8c8c8c")
    arrow(4.8, 5.2)
    box(5.2, 2.2, "LSTM  x 2", "64 units, drop 0.2", "#dbe8f5", "#4C72B0")
    arrow(7.4, 7.8)
    box(7.8, 2.0, "MLP head", "64-32-1, ReLU", "#f5e8e8", "#C44E52")

    ax.text(6.3, 2.5, "h$_T$ (last real step)", ha="center", fontsize=7,
            style="italic")
    ax.text(8.8, 0.55, "logit $\\hat{y}$  $\\rightarrow$  sigmoid $\\rightarrow$ P(failure imminent)",
            ha="center", fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig2-architecture.png")


def fig_learning_curves(log: pd.DataFrame, name: str) -> None:
    """Loss and validation metrics per epoch.

    Args:
        log (pd.DataFrame): Per-epoch training log.
        name (str): Run name (annotated in the figure).

    Returns:
        None
    """
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.9))

    axes[0].plot(log["epoch"], log["train_loss"], label="train", color=PALETTE[0])
    axes[0].plot(log["epoch"], log["val_loss"], label="val", color=PALETTE[1])
    axes[0].set_title("(a) Loss (BCE, pos-weighted)", fontsize=9)
    axes[0].set_xlabel("epoch")

    axes[1].plot(log["epoch"], log["val_f1"], "o-", color=PALETTE[0], ms=3, label="F1")
    axes[1].plot(log["epoch"], log["val_pr_auc"], "s-", color=PALETTE[1], ms=3, label="PR-AUC")
    axes[1].plot(log["epoch"], log["val_roc_auc"], "^-", color="#55a868", ms=3, label="ROC-AUC")
    best = log["val_f1"].idxmax()
    axes[1].axvline(log.loc[best, "epoch"], color="k", ls="--", lw=0.8,
                    label=f"best (ep {int(log.loc[best, 'epoch'])})")
    axes[1].set_title("(b) Validation metrics", fontsize=9)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0.6, 1.01)

    axes[2].plot(log["epoch"], log["threshold"], "o-", color="#dd8452", ms=3,
                 label="threshold (val)")
    axes[2].set_title("(c) Selected decision threshold", fontsize=9)
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("threshold")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(f"Learning curves — {name}", fontsize=9, y=1.02)
    fig.tight_layout()
    save(fig, "fig2-curves.png")


def get_test_scores() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Re-run the best checkpoint on the TEST split.

    Returns:
        tuple: ``y_true``, ``y_score``, ``y_val_score`` and the summary dict
        (metrics at the validation threshold).
    """
    set_global_seed(42)
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    df = load_raw()
    splits = load_splits()
    parts, _ = build_all_partitions(
        df, splits, window=int(ckpt["window"]), mode=str(ckpt["features"])
    )
    loaders = make_loaders(parts, batch_size=256, seed=int(ckpt.get("seed", 42)))
    model = build_model(ckpt["model_cfg"])
    model.load_state_dict(ckpt["model_state"])

    crit = torch.nn.BCEWithLogitsLoss()
    y_te, s_te, _ = evaluate_loader(model, loaders["test"], crit, torch.device("cpu"))
    y_va, s_va, _ = evaluate_loader(model, loaders["val"], crit, torch.device("cpu"))
    return y_te, s_te, y_va, ckpt


def fig_confusion(y_true: np.ndarray, scores: np.ndarray, thr: float) -> None:
    """Test confusion matrix of the deep model at the validation threshold.

    Args:
        y_true (np.ndarray): Test labels.
        scores (np.ndarray): Test probabilities.
        thr (float): Decision threshold.

    Returns:
        None
    """
    pred = scores >= thr
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(3.6, 3.1))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["safe", "danger"], yticklabels=["safe", "danger"], ax=ax)
    ax.set_title(f"Deep model — test (threshold {thr:.2f})", fontsize=9)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.tight_layout()
    save(fig, "fig2-confusion.png")


def fig_roc_pr(y_true: np.ndarray, deep_scores: np.ndarray) -> None:
    """ROC and PR curves of the deep model and the three W1 baselines.

    The W1 curves are recomputed with ``baselines.py`` (same split, same seed),
    so the curves match the reported table.

    Args:
        y_true (np.ndarray): Test labels.
        deep_scores (np.ndarray): Deep-model test probabilities.

    Returns:
        None
    """
    from baselines_ml import fit_model, get_models, prepare_data

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))

    fpr, tpr, _ = roc_curve(y_true, deep_scores)
    prec, rec, _ = precision_recall_curve(y_true, deep_scores)
    axes[0].plot(fpr, tpr, color="#4C72B0", lw=2,
                 label=f"Deep LSTM (AUC={auc(fpr, tpr):.3f})")
    axes[1].plot(rec, prec, color="#4C72B0", lw=2,
                 label=f"Deep LSTM (AP={auc(rec, prec):.3f})")

    # Workshop 1 baselines (refit with the identical pipeline)
    data = prepare_data()
    X_tr, y_tr = data["train"]
    X_va, y_va = data["val"]
    X_te, y_te = data["test"]
    assert (np.asarray(y_true) == np.asarray(y_te)).all(), "test rows differ!"

    from baselines_ml import best_f1_threshold

    for name, model in get_models().items():
        model = fit_model(name, model, X_tr, y_tr)
        scores = model.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, scores)
        prec, rec, _ = precision_recall_curve(y_te, scores)
        axes[0].plot(fpr, tpr, color=W1_COLORS[name], lw=1.2, ls="--",
                     label=f"{name} (AUC={auc(fpr, tpr):.3f})")
        axes[1].plot(rec, prec, color=W1_COLORS[name], lw=1.2, ls="--",
                     label=f"{name} (AP={auc(rec, prec):.3f})")

    axes[0].plot([0, 1], [0, 1], "k:", lw=0.7)
    axes[0].set_title("(a) ROC curve (test)", fontsize=9)
    axes[0].set_xlabel("false positive rate")
    axes[0].set_ylabel("true positive rate")

    base = y_te.mean()
    axes[1].axhline(base, color="k", ls=":", lw=0.7, label=f"baseline ({base:.3f})")
    axes[1].set_title("(b) Precision–Recall curve (test)", fontsize=9)
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    save(fig, "fig2-roc-pr.png")


def fig_comparison(comparison_csv: Path) -> None:
    """Bar chart comparing the W1 baselines with the W2 deep model.

    Args:
        comparison_csv (Path): CSV written by ``src/evaluate.py``.

    Returns:
        None
    """
    df = pd.read_csv(comparison_csv)
    metrics = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
    labels = [f"{r.model}\n({r.workshop})" for r in df.itertuples()]

    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    x = np.arange(len(metrics))
    width = 0.8 / len(df)
    colors = ["#8c8c8c", "#dd8452", "#55a868", "#4C72B0"]
    for i, row in df.iterrows():
        vals = [row[m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=row["model"],
                      color=colors[i % len(colors)],
                      edgecolor="white", lw=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=6.5, rotation=90)

    ax.set_xticks(x + width * (len(df) - 1) / 2)
    ax.set_xticklabels(metrics, fontsize=8)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("score")
    ax.set_title("Workshop 1 baselines vs. Workshop 2 deep model (test)", fontsize=9)
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    save(fig, "fig2-comparison.png")


def main() -> None:
    """Generate all Workshop 2 figures."""
    set_global_seed(42)
    sns.set_theme(style="whitegrid")

    p = argparse.ArgumentParser(description="Workshop 2 report figures")
    p.add_argument("--run", default=None, help="run name (default: from checkpoint)")
    p.add_argument("--comparison", type=Path,
                   default=RUNS_DIR / "comparison_w2_lstm_w16_raw.csv",
                   help="comparison CSV from src/evaluate.py")
    args = p.parse_args()

    print("=" * 60)
    print("GENERATING WORKSHOP 2 FIGURES ->", FIG_DIR)
    print("=" * 60)

    summary, log = load_run(args.run)
    fig_architecture()
    fig_learning_curves(log, summary.get("run_name", "run"))

    y_te, s_te, _, ckpt = get_test_scores()
    fig_confusion(y_te, s_te, float(ckpt["threshold"]))
    fig_roc_pr(y_te, s_te)

    if args.comparison.exists():
        fig_comparison(args.comparison)
    else:
        print(f"  WARNING: {args.comparison} not found "
              "(run: python src/evaluate.py) - skipping comparison figure")


if __name__ == "__main__":
    main()
