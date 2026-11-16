"""
Shared, framework-independent utilities for Workshop 4 (Autoencoders & VAEs).

Machine Learning - Workshop 4

This module contains everything that does NOT depend on PyTorch:
    * paths and the experiment configuration,
    * data preparation (re-uses the Workshop 1 split, features and scaler),
    * anomaly-detection and reconstruction metrics,
    * model-selection rules,
    * small I/O and plotting helpers.

Keeping these pieces free of PyTorch makes them easy to unit-test and lets the
three scripts (``train_ae.py``, ``evaluate_ae.py``, ``analyze_latent.py``) share
exactly the same data pipeline and metric definitions.

Design decisions (cited in the report)
--------------------------------------
1. **Same data as Workshop 1.** Inputs are the 16 engineered features of
   ``features.build_features`` computed per partition, standardised with a
   scaler fitted on the TRAIN episodes only. The episode-level split in
   ``data/splits.json`` is re-used unchanged, so every number is directly
   comparable with the Workshop 1 baselines.
2. **Anomaly definition.** The dataset has a natural normal/abnormal split:
   ``label = 0`` (safe state) is *normal* and ``label = 1`` (failure within the
   next five steps) is *anomalous*. The "normal-only" models never see a
   pre-failure state during training.
3. **Thresholds are tuned on VALIDATION only** (maximum F1, as in Workshop 1).
   The test set is used once, for the final report.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from dataset import (
    EPISODE_COL,
    ROOT_DIR,
    SEED,
    TARGET_COL,
    TIME_COL,
    fit_scaler,
    get_or_create_splits,
    load_raw,
    subset,
)
from features import build_features

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
RUNS_DIR = ROOT_DIR / "runs" / "workshop4"
CKPT_DIR = ROOT_DIR / "checkpoints"
CKPT_RUNS_DIR = CKPT_DIR / "workshop4"
REPORT_DIR = ROOT_DIR / "report" / "workshop4"
FIG_DIR = REPORT_DIR / "figures"
GEN_DIR = REPORT_DIR / "generated"
W1_RESULTS_CSV = ROOT_DIR / "runs" / "baseline_results_base.csv"
W2_CKPT = CKPT_DIR / "w2_lstm_best.pt"  # Workshop 2 model (2-layer LSTM, window 16)

PARTITIONS = ("train", "val", "test")

# Regimes: which training rows a model sees
REGIME_ALL = "all"        # every training row (reconstruction + latent analysis)
REGIME_NORMAL = "normal"  # only label == 0 rows (anomaly detection)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    """Hyper-parameters of one training run.

    Attributes:
        model (str): ``"ae"``, ``"vae"`` or ``"sup"`` (supervised reference MLP).
        regime (str): ``"all"`` or ``"normal"`` (rows used for training).
        z_dim (int): Width of the bottleneck / latent space.
        hidden (List[int]): Hidden widths of the encoder (decoder mirrors them).
        beta (float): Weight of the KL term (VAE only).
        kl_warmup_epochs (int): Linear KL annealing from 0 to ``beta`` (VAE only).
        noise_std (float): Std of the Gaussian input corruption (denoising AE);
            0 disables it.
        epochs (int): Training epochs.
        batch_size (int): Mini-batch size.
        lr (float): Adam learning rate.
        weight_decay (float): Adam weight decay.
        seed (int): Random seed.
        name (str): Run name (filled automatically when empty).
    """

    model: str = "ae"
    regime: str = REGIME_ALL
    z_dim: int = 4
    hidden: List[int] = field(default_factory=lambda: [64, 32])
    beta: float = 1.0
    kl_warmup_epochs: int = 10
    noise_std: float = 0.0
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    seed: int = SEED
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = run_name(self)

    def to_dict(self) -> Dict:
        """Return the configuration as a JSON-serialisable dictionary."""
        return asdict(self)


def run_name(cfg: TrainConfig) -> str:
    """Build a readable, unique run name from a configuration.

    Args:
        cfg (TrainConfig): Run configuration.

    Returns:
        str: e.g. ``ae_z4_all``, ``vae_b1.0_z4_normal``, ``dae_z4_normal``.
    """
    if cfg.model == "sup":
        return f"sup_mlp_z{cfg.z_dim}"
    prefix = "dae" if (cfg.model == "ae" and cfg.noise_std > 0) else cfg.model
    beta = f"_b{cfg.beta:g}" if cfg.model == "vae" else ""
    return f"{prefix}{beta}_z{cfg.z_dim}_{cfg.regime}"


def set_seed(seed: int = SEED) -> None:
    """Fix the Python and NumPy random seeds (PyTorch is seeded in autoencoder.py).

    Args:
        seed (int): Seed value.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class Partition:
    """One partition of the Workshop 4 data (all arrays share the same row order).

    Attributes:
        X (np.ndarray): Standardised features, shape (n, 16), float32.
        y (np.ndarray): Labels (0 = safe / normal, 1 = failure imminent / anomaly).
        episode (np.ndarray): Episode id of each row.
        t (np.ndarray): Time index of each row.
        steps_to_end (np.ndarray): Steps until the episode ends. ANALYSIS ONLY:
            it looks into the future, so it is never used as a model input.
        raw (pd.DataFrame): Unscaled engineered features (for plots in physical units).
    """

    X: np.ndarray
    y: np.ndarray
    episode: np.ndarray
    t: np.ndarray
    steps_to_end: np.ndarray
    raw: pd.DataFrame

    def normal(self) -> np.ndarray:
        """Return the standardised features of the normal rows (label 0)."""
        return self.X[self.y == 0]


@dataclass
class W4Data:
    """Container with the three partitions and the fitted scaler."""

    train: Partition
    val: Partition
    test: Partition
    feature_names: List[str]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray

    def __getitem__(self, name: str) -> Partition:
        return getattr(self, name)

    def training_matrix(self, regime: str) -> np.ndarray:
        """Training rows for a regime (``all`` or ``normal``)."""
        return self.train.X if regime == REGIME_ALL else self.train.normal()

    def validation_matrix(self, regime: str) -> np.ndarray:
        """Validation rows used for early stopping in a regime."""
        return self.val.X if regime == REGIME_ALL else self.val.normal()

    def inverse_scale(self, X: np.ndarray) -> np.ndarray:
        """Map standardised features back to physical units."""
        return X * self.scaler_scale + self.scaler_mean


def load_w4_data() -> W4Data:
    """Build the Workshop 4 dataset from the Workshop 1 pipeline.

    Pipeline (identical to Workshop 1, so there is no leakage):
        1. Load the raw CSV and the saved episode split.
        2. Build the 16 engineered features per partition.
        3. Fit the z-score scaler on ALL training rows and apply it to every
           partition (the same input space is used by every model).

    Returns:
        W4Data: Standardised partitions plus scaler statistics.
    """
    df = load_raw()
    splits = get_or_create_splits(df)

    feats, parts = {}, {}
    for name in PARTITIONS:
        part = subset(df, splits[name])
        parts[name] = part
        feats[name] = build_features(part)

    scaler = fit_scaler(feats["train"])
    names = list(feats["train"].columns)

    out = {}
    for name in PARTITIONS:
        part = parts[name]
        X = scaler.transform(feats[name]).astype(np.float32)
        steps_to_end = (
            part.groupby(EPISODE_COL)[TIME_COL].transform("max") - part[TIME_COL]
        ).to_numpy()
        out[name] = Partition(
            X=X,
            y=part[TARGET_COL].to_numpy().astype(int),
            episode=part[EPISODE_COL].to_numpy(),
            t=part[TIME_COL].to_numpy(),
            steps_to_end=steps_to_end,
            raw=feats[name].reset_index(drop=True),
        )

    return W4Data(
        train=out["train"],
        val=out["val"],
        test=out["test"],
        feature_names=names,
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
    )


def w2_windows(meta: Dict, partition: str) -> tuple:
    """Rebuild the Workshop 2 LSTM inputs for one partition, in Workshop 4 row order.

    The Workshop 2 model reads, for every row, a past-only window of the last
    ``meta["window"]`` steps of the same episode (``sequences.window_episodes``),
    standardised with the scaler stored in its checkpoint. Rows are sorted by
    (episode, t), exactly like :func:`load_w4_data`, so the outputs align row
    by row with the autoencoder inputs.

    Args:
        meta (Dict): Workshop 2 checkpoint metadata (``window``, ``features``,
            ``scaler_mean``, ``scaler_scale``).
        partition (str): ``"train"``, ``"val"`` or ``"test"``.

    Returns:
        tuple: ``(X, lengths, episode, t)`` with ``X`` of shape (n, window, F).
    """
    from sequences import compute_features, window_episodes

    df = load_raw()
    part = subset(df, get_or_create_splits(df)[partition])
    feats = compute_features(part, str(meta["features"])).to_numpy(dtype=np.float64)
    mean = np.asarray(meta["scaler_mean"], dtype=np.float64)
    scale = np.asarray(meta["scaler_scale"], dtype=np.float64)
    scaled = ((feats - mean) / scale).astype(np.float32)
    X, lengths = window_episodes(scaled, part[EPISODE_COL].to_numpy(), int(meta["window"]))
    return X, lengths, part[EPISODE_COL].to_numpy(), part[TIME_COL].to_numpy()


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def per_sample_mse(X: np.ndarray, X_hat: np.ndarray) -> np.ndarray:
    """Mean squared reconstruction error of every row (the anomaly score).

    Args:
        X (np.ndarray): Original standardised inputs, shape (n, d).
        X_hat (np.ndarray): Reconstructions, shape (n, d).

    Returns:
        np.ndarray: Shape (n,).
    """
    return np.mean((X - X_hat) ** 2, axis=1)


def reconstruction_metrics(X: np.ndarray, X_hat: np.ndarray) -> Dict[str, float]:
    """Reconstruction-quality metrics for continuous (tabular) data.

    * ``mse``: mean squared error in standardised units (lower is better).
    * ``r2``: explained variance, 1 - SSE / SST computed over all entries.
    * ``mean_corr``: mean Pearson correlation between each feature and its
      reconstruction (the "correlation" metric suggested for continuous data).

    Args:
        X (np.ndarray): Original inputs.
        X_hat (np.ndarray): Reconstructions.

    Returns:
        Dict[str, float]: Metrics plus the per-feature MSE and correlation lists.
    """
    err = (X - X_hat) ** 2
    sst = np.sum((X - X.mean(axis=0)) ** 2)
    corrs = []
    for j in range(X.shape[1]):
        a, b = X[:, j], X_hat[:, j]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            corrs.append(0.0)
        else:
            corrs.append(float(np.corrcoef(a, b)[0, 1]))
    return {
        "mse": float(err.mean()),
        "r2": float(1.0 - err.sum() / sst),
        "mean_corr": float(np.mean(corrs)),
        "per_feature_mse": [float(v) for v in err.mean(axis=0)],
        "per_feature_corr": corrs,
    }


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Threshold that maximises F1 (always called on VALIDATION data).

    Args:
        y_true (np.ndarray): Binary labels.
        scores (np.ndarray): Anomaly scores / probabilities (higher = anomalous).

    Returns:
        float: Selected threshold.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])


def detection_metrics(
    y_true: np.ndarray, scores: np.ndarray, threshold: Optional[float] = None
) -> Dict[str, float]:
    """Threshold-free and threshold-based anomaly-detection metrics.

    Args:
        y_true (np.ndarray): Binary labels (1 = anomaly).
        scores (np.ndarray): Anomaly scores (higher = more anomalous).
        threshold (Optional[float]): Decision threshold. If None, only the
            threshold-free metrics are returned.

    Returns:
        Dict[str, float]: roc_auc, pr_auc and, with a threshold, precision,
        recall, f1 and the confusion-matrix counts.
    """
    out = {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
    }
    if threshold is not None:
        y_pred = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        out.update(
            threshold=float(threshold),
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            f1=float(f1_score(y_true, y_pred, zero_division=0)),
            tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        )
    return out


def evaluate_scores(
    val_y: np.ndarray, val_scores: np.ndarray, test_y: np.ndarray, test_scores: np.ndarray,
    val_normal_quantile: float = 0.99,
) -> Dict[str, Dict[str, float]]:
    """Full anomaly-detection protocol for one score.

    Two thresholds are reported, both chosen WITHOUT looking at the test set:
        * ``val_f1``: maximises F1 on validation (same rule as Workshop 1);
        * ``val_q99``: 99th percentile of the validation NORMAL scores
          (a purely unsupervised rule that needs no anomaly labels).

    Args:
        val_y (np.ndarray): Validation labels.
        val_scores (np.ndarray): Validation scores.
        test_y (np.ndarray): Test labels.
        test_scores (np.ndarray): Test scores.
        val_normal_quantile (float): Quantile for the unsupervised threshold.

    Returns:
        Dict[str, Dict[str, float]]: ``{"val": ..., "test_f1": ..., "test_q99": ...}``.
    """
    thr_f1 = best_f1_threshold(val_y, val_scores)
    thr_q = float(np.quantile(val_scores[val_y == 0], val_normal_quantile))
    return {
        "val": detection_metrics(val_y, val_scores, thr_f1),
        "test_f1": detection_metrics(test_y, test_scores, thr_f1),
        "test_q99": detection_metrics(test_y, test_scores, thr_q),
    }


# --------------------------------------------------------------------------- #
# Model selection
# --------------------------------------------------------------------------- #
def select_by_val_auc(summaries: Sequence[Dict], key: str = "val_auc") -> Dict:
    """Return the run summary with the highest validation ROC-AUC.

    Ties (within 1e-4) are broken in favour of the run listed first, so the
    candidate lists are ordered from the simplest model (smallest latent space,
    smallest beta) to the most complex.

    Args:
        summaries (Sequence[Dict]): Run summaries containing ``key``.
        key (str): Metric to maximise.

    Returns:
        Dict: The selected summary.
    """
    best = summaries[0]
    for s in summaries[1:]:
        if s[key] > best[key] + 1e-4:
            best = s
    return best


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def save_json(obj: Dict, path: Path) -> None:
    """Write a dictionary as indented JSON (creates parent folders)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path) -> Dict:
    """Read a JSON file."""
    with open(path) as f:
        return json.load(f)


def load_w1_baselines() -> pd.DataFrame:
    """Load the Workshop 1 test results (threshold chosen on validation).

    Returns:
        pd.DataFrame: One row per baseline model.
    """
    df = pd.read_csv(W1_RESULTS_CSV)
    return df[df["setting"] == "test @ val-threshold"].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Plot style
# --------------------------------------------------------------------------- #
COLORS = {
    "normal": "#4C72B0",   # same blue as Workshop 1 ("safe")
    "anomaly": "#C44E52",  # same red as Workshop 1 ("danger")
    "ae": "#4C72B0",
    "vae": "#DD8452",
    "sup": "#55A868",
    "dae": "#8172B3",
    "raw": "#7F7F7F",
}
DPI = 200


def setup_matplotlib() -> None:
    """Common matplotlib configuration for every Workshop 4 figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 100,
    })


def save_fig(fig, name: str) -> None:
    """Save a figure as PNG in ``report/workshop4/figures`` and close it."""
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT_DIR)}")
