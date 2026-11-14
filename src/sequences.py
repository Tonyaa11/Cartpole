"""
Sequence (sliding-window) dataset for the CartPole failure-prediction task.

Machine Learning - Workshop 2 (Advanced Learning Models, Part A)

Why sequences?
--------------
Workshop 1 classified EACH ROW independently with hand-crafted features. The
deep model instead reads the temporal context directly: every row becomes one
training sample described by the ``window`` most recent steps of the SAME
episode (including the current step). The label of the sample is the label of
the last step of the window.

Design decisions
----------------
1. **Past-only windows.** The window ends AT the predicted row; nothing from
   the future is used (no leakage, same rule as Workshop 1).
2. **One sample per raw row.** Test metrics are therefore computed on exactly
   the same rows as the Workshop 1 baselines (direct comparison).
3. **Short episodes are kept.** Rows near the beginning of an episode use a
   shorter window (``lengths``); the packed representation masks the padding.
4. **Padding with zeros.** Features are z-score standardised first, so zero =
   the training mean (an "uninformative" value); padding is masked anyway.
5. **Scaler fitted on TRAIN only** (same leakage rule as Workshop 1).
6. **``t`` is not a feature** (same rule as Workshop 1).

Usage
-----
    from sequences import build_sequences
    X, lengths, y, scaler = build_sequences(df, episodes, window=16)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from dataset import (
    EPISODE_COL,
    RAW_FEATURE_COLS,
    TARGET_COL,
)

# Feature modes
MODE_RAW = "raw"  # 5 columns: x, x_dot, theta, theta_dot, action
MODE_ENGINEERED = "engineered"  # 16 columns from features.build_features


def compute_features(df: pd.DataFrame, mode: str = MODE_RAW) -> pd.DataFrame:
    """Return the per-row feature matrix for the requested mode.

    Args:
        df (pd.DataFrame): Raw dataframe (a partition or the full data).
        mode (str): ``"raw"`` (5 columns) or ``"engineered"`` (16 columns).

    Returns:
        pd.DataFrame: Feature matrix (same index as ``df``).

    Raises:
        ValueError: If ``mode`` is unknown.
    """
    if mode == MODE_RAW:
        return df[RAW_FEATURE_COLS].astype(float)
    if mode == MODE_ENGINEERED:
        from features import build_features  # local import (features imports dataset)

        return build_features(df)
    raise ValueError(f"unknown feature mode: {mode!r}")


def feature_count(mode: str = MODE_RAW) -> int:
    """Number of features produced by ``compute_features``.

    Args:
        mode (str): ``"raw"`` or ``"engineered"``.

    Returns:
        int: Feature dimension per time step.
    """
    return len(RAW_FEATURE_COLS) if mode == MODE_RAW else 16


def fit_scaler(train_features: pd.DataFrame) -> StandardScaler:
    """Fit a z-score scaler on the TRAINING features only.

    Args:
        train_features (pd.DataFrame): Training feature matrix.

    Returns:
        StandardScaler: Fitted scaler (mean/std from TRAIN only).
    """
    scaler = StandardScaler()
    scaler.fit(train_features)
    return scaler


def window_episodes(
    features: np.ndarray, episodes: np.ndarray, window: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Build past-only sliding windows, one window per row.

    Args:
        features (np.ndarray): ``(N, F)`` scaled features, sorted by
            (episode, t).
        episodes (np.ndarray): ``(N,)`` episode id of each row (sorted).
        window (int): Maximum window length (number of past steps, including
            the current row).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - ``X`` of shape ``(N, window, F)`` (zero-padded on the LEFT side
              of short windows is not used: real steps are left-aligned and
              the rest is zero padding at the END, which the packed RNN masks);
            - ``lengths`` of shape ``(N,)`` with the true number of steps.
    """
    n_rows, n_feats = features.shape
    X = np.zeros((n_rows, window, n_feats), dtype=np.float32)
    lengths = np.zeros(n_rows, dtype=np.int64)

    # Segment boundaries of each episode (rows are sorted by (episode, t))
    edges = np.flatnonzero(np.diff(episodes)) + 1
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [n_rows]])

    for s, e in zip(starts, ends):
        length = e - s
        for t in range(length):
            row = s + t
            lo = max(0, t + 1 - window)  # past-only slice [lo, t]
            n_steps = t + 1 - lo
            X[row, :n_steps] = features[s + lo : s + t + 1]
            lengths[row] = n_steps

    return X, lengths


def _build_from_frame(
    part: pd.DataFrame,
    window: int,
    mode: str,
    scaler: StandardScaler | None = None,
    fit: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Build (X, lengths, y) from an already-filtered, sorted dataframe.

    Args:
        part (pd.DataFrame): Rows of ONE partition, sorted by (episode, t).
        window (int): Maximum window length.
        mode (str): Feature mode (``"raw"`` / ``"engineered"``).
        scaler (StandardScaler | None): Scaler to apply (if not fitting).
        fit (bool): If True, fit the scaler on THIS frame (train only).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
            ``X (N, window, F)``, ``lengths (N,)``, ``y (N,)`` and the scaler.

    Raises:
        ValueError: If no scaler is provided and ``fit`` is False.
    """
    part = part.sort_values([EPISODE_COL, "t"] if "t" in part.columns else EPISODE_COL)

    feats = compute_features(part, mode)
    if fit:
        scaler = fit_scaler(feats)
    if scaler is None:
        raise ValueError("provide a scaler or set fit=True")

    scaled = pd.DataFrame(
        scaler.transform(feats), columns=feats.columns, index=feats.index
    ).astype(np.float32)

    X, lengths = window_episodes(
        scaled.to_numpy(dtype=np.float32),
        part[EPISODE_COL].to_numpy(),
        window,
    )
    y = part[TARGET_COL].to_numpy(dtype=np.float32)
    return X, lengths, y, scaler


def build_sequences(
    df: pd.DataFrame,
    episodes: List[int],
    window: int = 16,
    mode: str = MODE_RAW,
    scaler: StandardScaler | None = None,
    fit: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Build (X, lengths, y) for one partition from its raw rows.

    Args:
        df (pd.DataFrame): Full raw dataframe (sorted by (episode, t)).
        episodes (List[int]): Episode ids that belong to this partition.
        window (int): Maximum window length.
        mode (str): Feature mode (``"raw"`` / ``"engineered"``).
        scaler (StandardScaler | None): Scaler to apply. Required unless
            ``fit=True``.
        fit (bool): If True, fit the scaler on THIS partition (use only for
            the training partition) and ignore ``scaler``.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
            ``X (N, window, F) float32``, ``lengths (N,) int64``,
            ``y (N,) float32`` and the scaler used.
    """
    part = df[df[EPISODE_COL].isin(episodes)]
    return _build_from_frame(part, window, mode, scaler=scaler, fit=fit)


def augment_rows(
    part: pd.DataFrame,
    noise_scale: float = 0.02,
    n_copies: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """Add noisy copies of a partition's rows (TRAIN ONLY, same rule as W1).

    A real CartPole sensor is never perfectly precise: adding a small Gaussian
    noise (``noise_scale`` x std of each state variable) makes the model less
    sensitive to tiny numerical differences. Labels are kept; copies receive
    new episode ids so their windows are built independently.

    Args:
        part (pd.DataFrame): Raw TRAINING rows only (never val/test).
        noise_scale (float): Noise std as a fraction of each variable's std.
        n_copies (int): Number of noisy copies to append.
        seed (int): Random seed.

    Returns:
        pd.DataFrame: Original rows plus the noisy copies.
    """
    from dataset import augment_with_noise

    return augment_with_noise(
        part, n_copies=n_copies, noise_scale=noise_scale, seed=seed
    )


def build_all_partitions(
    df: pd.DataFrame,
    splits: Dict[str, List[int]],
    window: int = 16,
    mode: str = MODE_RAW,
    augment: bool = False,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], StandardScaler]:
    """Build sequences for train/val/test with ONE scaler fitted on TRAIN.

    Args:
        df (pd.DataFrame): Full raw dataframe.
        splits (Dict[str, List[int]]): Episode ids per partition
            (from ``data/splits.json``).
        window (int): Maximum window length.
        mode (str): Feature mode.
        augment (bool): If True, append noise-augmented copies of the TRAIN
            rows only (never val/test).

    Returns:
        Tuple[dict, StandardScaler]:
            ``{"train": (X, lengths, y), "val": ..., "test": ...}`` and the
            scaler fitted on the training partition only.
    """
    train_df = df[df[EPISODE_COL].isin(splits["train"])]
    if augment:
        train_df = augment_rows(train_df)

    X_tr, L_tr, y_tr, scaler = _build_from_frame(
        train_df, window, mode, fit=True
    )
    parts = {"train": (X_tr, L_tr, y_tr)}
    for name in ("val", "test"):
        val_df = df[df[EPISODE_COL].isin(splits[name])]
        X, L, y, _ = _build_from_frame(val_df, window, mode, scaler=scaler)
        parts[name] = (X, L, y)
    return parts, scaler


def describe_partition(
    name: str, X: np.ndarray, lengths: np.ndarray, y: np.ndarray
) -> str:
    """One-line summary of a partition (rows, windows, positives).

    Args:
        name (str): Partition name.
        X (np.ndarray): Window tensor.
        lengths (np.ndarray): True window lengths.
        y (np.ndarray): Labels.

    Returns:
        str: Human-readable summary line.
    """
    return (
        f"{name:>5}: {len(y):6d} windows | shape {X.shape} | "
        f"mean length {lengths.mean():.1f} | positives {int(y.sum()):5d} "
        f"({100 * y.mean():.2f}%)"
    )
