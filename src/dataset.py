"""
CartPole Failure-Prediction Dataset: loading, splitting, scaling and augmentation.

Machine Learning - Workshop 1 (Data Preparation & Supervised Learning Baselines)

Task
----
Binary classification: given the state of a CartPole at time ``t``
(x, x_dot, theta, theta_dot) and the action taken, predict whether the episode
is about to fail (``label = 1`` in the last 5 steps of a failed episode).

Design decisions (documented here so the report can cite them)
--------------------------------------------------------------
1. **Split by episode, never by row.**
   Rows of the same episode are strongly autocorrelated. Splitting by row would
   put almost-identical consecutive states in train and test (data leakage).
   We therefore split the 400 episodes 70/15/15.
2. **Stratification by episode length.**
   Episode lengths are very skewed (some last 8 steps, some 500). We stratify
   the episode split by length quartile so that long episodes (which contribute
   many rows) are spread across train/val/test.
3. **Scaling statistics computed ONLY on the training set** (no leakage).
4. **Augmentation applied ONLY to the training set** (Gaussian sensor noise).
5. **The split is saved to ``data/splits.json``** and re-loaded afterwards;
   we never re-split randomly.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Global constants
# --------------------------------------------------------------------------- #
SEED = 42

# Repository paths (resolved relative to this file so scripts work from anywhere)
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT_DIR / "data" / "raw" / "cartpole_dataset.csv"
SPLITS_JSON = ROOT_DIR / "data" / "splits.json"

# Columns of the raw CSV
STATE_COLS = ["x", "x_dot", "theta", "theta_dot"]
RAW_FEATURE_COLS = STATE_COLS + ["action"]
TARGET_COL = "label"
EPISODE_COL = "episode"
TIME_COL = "t"  # NOT used as a feature (see features.py)


def set_global_seed(seed: int = SEED) -> None:
    """Fix the random seeds of Python and NumPy (and scikit-learn through NumPy).

    Args:
        seed (int): Seed value used for all random number generators.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)


def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute the SHA256 checksum of a file (integrity verification).

    Args:
        path (Path): File whose checksum is computed.
        chunk_size (int): Read block size in bytes (avoids loading the whole file).

    Returns:
        str: Hexadecimal SHA256 digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def load_raw(csv_path: Path = RAW_CSV) -> pd.DataFrame:
    """Load the raw CSV and run basic sanity checks.

    Args:
        csv_path (Path): Location of ``cartpole_dataset.csv``.

    Returns:
        pd.DataFrame: Raw dataframe sorted by (episode, t) so that temporal
        operations (lags, rolling windows) are well defined.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If an expected column is missing.
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)

    expected = [EPISODE_COL, TIME_COL] + RAW_FEATURE_COLS + [TARGET_COL]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Sorting guarantees that "previous step" means the previous time step
    df = df.sort_values([EPISODE_COL, TIME_COL]).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Episode-level split
# --------------------------------------------------------------------------- #
def make_episode_split(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = SEED,
) -> Dict[str, List[int]]:
    """Split EPISODES (not rows) into train/val/test, stratified by length.

    Args:
        df (pd.DataFrame): Raw dataframe with ``episode`` column.
        val_size (float): Fraction of episodes for validation.
        test_size (float): Fraction of episodes for test.
        seed (int): Random seed for reproducibility.

    Returns:
        Dict[str, List[int]]: ``{"train": [...], "val": [...], "test": [...]}``
        with lists of episode ids.
    """
    # One row per episode with its length
    ep = df.groupby(EPISODE_COL).size().rename("length").reset_index()

    # Length quartiles -> stratification key (rank first to break ties)
    ep["bin"] = pd.qcut(ep["length"].rank(method="first"), q=4, labels=False)

    # Step 1: separate the test episodes
    trainval, test = train_test_split(
        ep, test_size=test_size, stratify=ep["bin"], random_state=seed
    )

    # Step 2: split the remainder into train / val
    # (val_size is relative to the TOTAL, so rescale it)
    rel_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        trainval, test_size=rel_val, stratify=trainval["bin"], random_state=seed
    )

    return {
        "train": sorted(int(e) for e in train[EPISODE_COL]),
        "val": sorted(int(e) for e in val[EPISODE_COL]),
        "test": sorted(int(e) for e in test[EPISODE_COL]),
    }


def save_splits(
    splits: Dict[str, List[int]],
    df: pd.DataFrame,
    path: Path = SPLITS_JSON,
    csv_path: Path = RAW_CSV,
) -> None:
    """Persist the split (plus metadata) to ``data/splits.json``.

    Args:
        splits (Dict[str, List[int]]): Episode ids per partition.
        df (pd.DataFrame): Raw dataframe (used to report row counts).
        path (Path): Output JSON path.
        csv_path (Path): Raw CSV (its SHA256 is stored for traceability).

    Returns:
        None
    """
    meta = {
        "seed": SEED,
        "split_unit": "episode",
        "stratified_by": "episode length quartile",
        "csv_sha256": sha256_of_file(csv_path),
        "n_episodes": {k: len(v) for k, v in splits.items()},
        "n_rows": {
            k: int(df[EPISODE_COL].isin(v).sum()) for k, v in splits.items()
        },
        "n_positive_rows": {
            k: int(df.loc[df[EPISODE_COL].isin(v), TARGET_COL].sum())
            for k, v in splits.items()
        },
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"meta": meta, "episodes": splits}, f, indent=2)


def load_splits(path: Path = SPLITS_JSON) -> Dict[str, List[int]]:
    """Load the saved episode split (the ONLY source of truth for partitions).

    Args:
        path (Path): Path to ``splits.json``.

    Returns:
        Dict[str, List[int]]: Episode ids per partition.
    """
    with open(path) as f:
        return json.load(f)["episodes"]


def get_or_create_splits(df: pd.DataFrame, path: Path = SPLITS_JSON) -> Dict[str, List[int]]:
    """Load ``splits.json`` if it exists, otherwise create and save it once.

    Args:
        df (pd.DataFrame): Raw dataframe.
        path (Path): Path of the splits file.

    Returns:
        Dict[str, List[int]]: Episode ids per partition.
    """
    if Path(path).exists():
        return load_splits(path)
    splits = make_episode_split(df)
    save_splits(splits, df, path)
    return splits


def subset(df: pd.DataFrame, episodes: List[int]) -> pd.DataFrame:
    """Return the rows that belong to the given episodes.

    Args:
        df (pd.DataFrame): Raw dataframe.
        episodes (List[int]): Episode ids to keep.

    Returns:
        pd.DataFrame: Filtered copy (index reset).
    """
    return df[df[EPISODE_COL].isin(episodes)].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Augmentation (TRAIN ONLY)
# --------------------------------------------------------------------------- #
def augment_with_noise(
    df_train: pd.DataFrame,
    n_copies: int = 1,
    noise_scale: float = 0.02,
    seed: int = SEED,
) -> pd.DataFrame:
    """Create noisy copies of the TRAINING episodes (sensor-noise augmentation).

    Justification: a real CartPole sensor is never perfectly precise. Adding a
    small Gaussian noise (``noise_scale`` x std of each state variable) makes the
    classifier less sensitive to tiny numerical differences. The label is kept.
    Copies receive new episode ids so temporal features (lags, rolling windows)
    are computed independently per copy.

    Args:
        df_train (pd.DataFrame): Raw TRAINING rows only (never val/test).
        n_copies (int): Number of noisy copies to append.
        noise_scale (float): Noise std as a fraction of each variable's std.
        seed (int): Random seed.

    Returns:
        pd.DataFrame: Original rows plus the noisy copies.
    """
    rng = np.random.RandomState(seed)
    stds = df_train[STATE_COLS].std().values  # computed on train only
    id_offset = int(df_train[EPISODE_COL].max()) + 1

    frames = [df_train]
    for c in range(1, n_copies + 1):
        noisy = df_train.copy()
        noise = rng.normal(0.0, 1.0, size=(len(noisy), len(STATE_COLS))) * stds * noise_scale
        noisy[STATE_COLS] = noisy[STATE_COLS].values + noise
        noisy[EPISODE_COL] = noisy[EPISODE_COL] + c * id_offset  # new unique ids
        frames.append(noisy)

    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Scaling (fit on TRAIN only)
# --------------------------------------------------------------------------- #
def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit a z-score scaler on the training features only.

    Z-score is used because the features have very different ranges
    (e.g. x in meters vs. theta in radians) and the linear baseline is
    scale-sensitive.

    Args:
        X_train (pd.DataFrame): Training feature matrix.

    Returns:
        StandardScaler: Fitted scaler (mean/std come from TRAIN only).
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def apply_scaler(scaler: StandardScaler, X: pd.DataFrame) -> pd.DataFrame:
    """Apply an already-fitted scaler (keeps column names and index).

    Args:
        scaler (StandardScaler): Scaler fitted on train.
        X (pd.DataFrame): Features to transform.

    Returns:
        pd.DataFrame: Scaled features.
    """
    return pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)


# --------------------------------------------------------------------------- #
# CLI: create splits.json
# --------------------------------------------------------------------------- #
def main() -> None:
    """Create ``data/splits.json`` from the raw CSV (run once)."""
    set_global_seed()
    df = load_raw()
    splits = make_episode_split(df)
    save_splits(splits, df)

    print("=" * 60)
    print("EPISODE-LEVEL SPLIT SAVED ->", SPLITS_JSON)
    print("=" * 60)
    for name, eps in splits.items():
        part = subset(df, eps)
        print(
            f"{name:>5}: {len(eps):3d} episodes | {len(part):6d} rows | "
            f"positives = {int(part[TARGET_COL].sum()):5d} "
            f"({100 * part[TARGET_COL].mean():.2f}%)"
        )


if __name__ == "__main__":
    main()
