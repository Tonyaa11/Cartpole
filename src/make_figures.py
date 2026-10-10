"""
make_figures.py - Generate the report figures as PNG files.

Machine Learning - Workshop 1

Every figure is computed from the real dataset and the real baseline models
(same code, split and seed as ``baselines.py``); nothing is hard-coded. The files
are written to ``report/figures/`` with the exact names that ``report/main.tex``
expects, so after running this script the report shows the figures automatically:

    fig-target-episodes.png   class counts | episode lengths | label vs. distance to end
    fig-distributions.png     class-conditional histograms of the 4 state variables
    fig-corr.png              correlation matrix of the raw features and the label
    fig-confusion.png         test confusion matrices of the three baselines

Usage
-----
    python src/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed (works in Colab, servers and scripts)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from baselines import best_f1_threshold, fit_model, get_models, prepare_data
from dataset import (
    EPISODE_COL,
    RAW_FEATURE_COLS,
    ROOT_DIR,
    STATE_COLS,
    TARGET_COL,
    load_raw,
    set_global_seed,
)

FIG_DIR = ROOT_DIR / "report" / "figures"
PALETTE = ["#4C72B0", "#C44E52"]  # safe (blue) / danger (red)
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


def fig_target_and_episodes(df) -> None:
    """Class counts, episode-length histogram and label vs. steps to episode end.

    Args:
        df (pd.DataFrame): Raw dataframe.

    Returns:
        None
    """
    counts = df[TARGET_COL].value_counts().sort_index()
    lengths = df.groupby(EPISODE_COL).size()
    steps_to_end = df.groupby(EPISODE_COL)["t"].transform("max") - df["t"]
    rate = df.groupby(steps_to_end)[TARGET_COL].mean().head(15)

    fig, axes = plt.subplots(1, 3, figsize=(10, 2.9))

    # (a) class counts
    axes[0].bar(["safe (0)", "danger (1)"], counts.values, color=PALETTE)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    axes[0].set_title(f"(a) Class distribution ({counts.max() / counts.min():.1f}:1)", fontsize=9)
    axes[0].set_ylabel("rows")

    # (b) episode lengths (log y-axis because of the heavy skew)
    axes[1].hist(lengths, bins=40, color=PALETTE[0])
    axes[1].set_yscale("log")
    axes[1].set_title("(b) Episode length", fontsize=9)
    axes[1].set_xlabel("steps")
    axes[1].set_ylabel("episodes (log)")

    # (c) label probability by distance to the end of the episode
    axes[2].bar(rate.index, rate.values, color=PALETTE[1])
    axes[2].set_title("(c) P(label = 1) vs. steps before end", fontsize=9)
    axes[2].set_xlabel("steps before episode end")
    axes[2].set_ylabel("P(label = 1)")

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "fig-target-episodes.png")


def fig_distributions(df) -> None:
    """Class-conditional histograms of the four state variables.

    Args:
        df (pd.DataFrame): Raw dataframe.

    Returns:
        None
    """
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.6))
    for ax, col in zip(axes, STATE_COLS):
        sns.histplot(
            data=df, x=col, hue=TARGET_COL, stat="density", common_norm=False,
            bins=40, element="step", ax=ax, palette=PALETTE, legend=(col == STATE_COLS[-1]),
        )
        ax.set_title(col, fontsize=9)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "fig-distributions.png")


def fig_correlation(df) -> None:
    """Correlation matrix of the raw features and the label.

    Args:
        df (pd.DataFrame): Raw dataframe.

    Returns:
        None
    """
    corr = df[RAW_FEATURE_COLS + [TARGET_COL]].corr()
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                annot_kws={"size": 7}, ax=ax)
    ax.set_title("Correlation (raw features)", fontsize=9)
    fig.tight_layout()
    save(fig, "fig-corr.png")


def fig_confusion() -> None:
    """Test confusion matrices of the three baselines (validation threshold).

    Uses exactly the pipeline of ``baselines.py`` (same split, features, seed).

    Returns:
        None
    """
    data = prepare_data()
    X_tr, y_tr = data["train"]
    X_va, y_va = data["val"]
    X_te, y_te = data["test"]

    fig, axes = plt.subplots(1, 3, figsize=(10, 2.9))
    for ax, (name, model) in zip(axes, get_models().items()):
        model = fit_model(name, model, X_tr, y_tr)
        thr = best_f1_threshold(y_va, model.predict_proba(X_va)[:, 1])  # threshold from validation
        pred = (model.predict_proba(X_te)[:, 1] >= thr).astype(int)
        sns.heatmap(
            confusion_matrix(y_te, pred, labels=[0, 1]), annot=True, fmt="d", cmap="Blues",
            cbar=False, xticklabels=["safe", "danger"], yticklabels=["safe", "danger"], ax=ax,
        )
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
    fig.tight_layout()
    save(fig, "fig-confusion.png")


def main() -> None:
    """Generate all figures of the report."""
    set_global_seed()
    sns.set_theme(style="whitegrid")
    df = load_raw()

    print("=" * 60)
    print("GENERATING REPORT FIGURES ->", FIG_DIR)
    print("=" * 60)
    fig_target_and_episodes(df)
    fig_distributions(df)
    fig_correlation(df)
    fig_confusion()


if __name__ == "__main__":
    main()
