"""
dataset_info.py - Structured summary of the CartPole dataset.

Machine Learning - Workshop 1

Scans the dataset directory and prints:
    * file integrity (size on disk and SHA256, optionally compared with an
      expected checksum),
    * shape, column types and missing values,
    * target distribution and imbalance ratio,
    * feature ranges (min / max / mean / std),
    * episode statistics (length distribution, how episodes end).

Usage
-----
    python dataset_info.py
    python dataset_info.py --csv data/raw/cartpole_dataset.csv \
        --sha256 ec067d5dcc498f2b5c5486d24f290283431ef3a0db0a4952cfde84f07e4f53f8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow "python dataset_info.py" from the repository root
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from dataset import (  # noqa: E402
    EPISODE_COL,
    RAW_CSV,
    RAW_FEATURE_COLS,
    TARGET_COL,
    TIME_COL,
    sha256_of_file,
)

# SHA256 reported when the dataset was generated (used by default)
EXPECTED_SHA256 = "ec067d5dcc498f2b5c5486d24f290283431ef3a0db0a4952cfde84f07e4f53f8"


def section(title: str) -> None:
    """Print a section header.

    Args:
        title (str): Header text.

    Returns:
        None
    """
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main(csv_path: Path, expected_sha: str) -> None:
    """Print the structured dataset summary.

    Args:
        csv_path (Path): Path to the dataset CSV.
        expected_sha (str): Expected SHA256 checksum ("" to skip the check).

    Returns:
        None
    """
    # ---- 1. Integrity ------------------------------------------------------ #
    section("1. FILE INTEGRITY")
    size_mb = csv_path.stat().st_size / (1024 ** 2)
    digest = sha256_of_file(csv_path)
    print(f"File       : {csv_path}")
    print(f"Size       : {size_mb:.2f} MB")
    print(f"SHA256     : {digest}")
    if expected_sha:
        print(f"Checksum   : {'OK (matches expected)' if digest == expected_sha else 'MISMATCH!'}")

    df = pd.read_csv(csv_path)

    # ---- 2. Structure ------------------------------------------------------ #
    section("2. STRUCTURE")
    print(f"Rows x columns : {df.shape[0]} x {df.shape[1]}")
    print(f"Episodes       : {df[EPISODE_COL].nunique()}")
    print("\nColumn dtypes:")
    print(df.dtypes.to_string())
    print(f"\nMissing values : {int(df.isna().sum().sum())}")
    print(f"Duplicate rows : {int(df.duplicated().sum())}")

    # ---- 3. Target distribution ------------------------------------------- #
    section("3. TARGET DISTRIBUTION (label)")
    counts = df[TARGET_COL].value_counts().sort_index()
    for cls, n in counts.items():
        print(f"class {cls}: {n:6d} samples ({100 * n / len(df):5.2f}%)")
    print(f"Imbalance ratio (majority/minority): {counts.max() / counts.min():.1f} : 1")

    # ---- 4. Feature ranges ------------------------------------------------- #
    section("4. FEATURE RANGES")
    stats = df[RAW_FEATURE_COLS].agg(["min", "max", "mean", "std"]).T
    print(stats.round(4).to_string())

    # ---- 5. Episodes ------------------------------------------------------- #
    section("5. EPISODE STATISTICS")
    lengths = df.groupby(EPISODE_COL).size()
    print(lengths.describe().round(2).to_string())
    n_no_pos = int((df.groupby(EPISODE_COL)[TARGET_COL].sum() == 0).sum())
    print(f"\nEpisodes without any positive step (reached the step limit): {n_no_pos}")
    print(f"Time index column '{TIME_COL}' range: {df[TIME_COL].min()} - {df[TIME_COL].max()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CartPole dataset summary")
    parser.add_argument("--csv", type=Path, default=RAW_CSV)
    parser.add_argument("--sha256", default=EXPECTED_SHA256)
    args = parser.parse_args()
    main(args.csv, args.sha256)
