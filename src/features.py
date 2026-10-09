"""
Feature engineering for the CartPole failure-prediction task.

Machine Learning - Workshop 1

Golden rules
------------
* **No future information.** Every feature at time ``t`` uses only rows with
  time <= ``t`` of the SAME episode. The label is defined by the future
  (the last 5 steps before a failure), so any feature that looks ahead
  (or counts steps to the end of the episode) would leak the target.
* **``t`` is not a feature.** The time index is a bookkeeping column; using it
  would let the model learn episode-length artefacts instead of dynamics.
* Features are derived from physics knowledge of CartPole: an episode fails when
  |theta| > ~12 degrees (0.2095 rad) or |x| > 2.4 m, so the *distance to those
  limits* and the *speed of approach* are the natural predictors.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from dataset import EPISODE_COL, RAW_FEATURE_COLS

# Standard CartPole termination thresholds (domain constants)
THETA_LIMIT = 0.2095  # radians (~12 degrees)
X_LIMIT = 2.4         # meters

# Cap for the "time to limit" estimate (seconds) when the system is not approaching
TTL_CAP = 10.0


def _time_to_limit(margin: pd.Series, approach_speed: pd.Series) -> pd.Series:
    """Estimate the time to reach a limit assuming constant velocity.

    Args:
        margin (pd.Series): Remaining distance to the limit (>= 0 means inside).
        approach_speed (pd.Series): Speed towards the limit (> 0 means approaching).

    Returns:
        pd.Series: ``margin / approach_speed`` if approaching, otherwise TTL_CAP.
        Values are clipped to ``[0, TTL_CAP]``.
    """
    safe_speed = approach_speed.where(approach_speed > 1e-6, np.nan)
    ttl = (margin / safe_speed).fillna(TTL_CAP)
    return ttl.clip(lower=0.0, upper=TTL_CAP)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the feature matrix from the raw dataframe.

    The dataframe MUST be sorted by (episode, t) (``dataset.load_raw`` does it).

    Feature groups:
        1. Raw state + action (5): x, x_dot, theta, theta_dot, action.
        2. Distance to failure limits (4): |x|, |theta|, x_margin, theta_margin.
        3. Short-term dynamics (3): first differences of x, theta, theta_dot
           with respect to the previous step (0 at the first step).
        4. Rolling windows over the last 3 steps (2): mean |theta|, mean |theta_dot|.
        5. Time-to-limit estimates (2): seconds to reach the theta / x limit
           if the current velocity were kept constant.

    Args:
        df (pd.DataFrame): Raw dataframe (a single partition or the full data).

    Returns:
        pd.DataFrame: Feature matrix (same index as ``df``), 16 columns,
        float64, with no NaN values.
    """
    g = df.groupby(EPISODE_COL, sort=False)
    feats = df[RAW_FEATURE_COLS].copy()

    # --- Group 2: distance to the failure limits --------------------------- #
    feats["abs_x"] = df["x"].abs()
    feats["abs_theta"] = df["theta"].abs()
    feats["x_margin"] = X_LIMIT - feats["abs_x"]
    feats["theta_margin"] = THETA_LIMIT - feats["abs_theta"]

    # --- Group 3: first differences (previous step of the SAME episode) ----- #
    for col in ["x", "theta", "theta_dot"]:
        feats[f"d_{col}"] = g[col].diff().fillna(0.0)

    # --- Group 4: rolling windows (past + current only) --------------------- #
    # min_periods=1 -> the first steps of an episode use the available history
    feats["roll3_abs_theta"] = (
        feats["abs_theta"].groupby(df[EPISODE_COL]).transform(
            lambda s: s.rolling(3, min_periods=1).mean()
        )
    )
    feats["roll3_abs_theta_dot"] = (
        df["theta_dot"].abs().groupby(df[EPISODE_COL]).transform(
            lambda s: s.rolling(3, min_periods=1).mean()
        )
    )

    # --- Group 5: time to reach each limit at constant velocity ------------- #
    # approach speed = velocity projected towards the side where the variable lies
    theta_approach = np.sign(df["theta"]) * df["theta_dot"]
    x_approach = np.sign(df["x"]) * df["x_dot"]
    feats["ttl_theta"] = _time_to_limit(feats["theta_margin"], theta_approach)
    feats["ttl_x"] = _time_to_limit(feats["x_margin"], x_approach)

    return feats.astype(float)


def feature_names() -> List[str]:
    """Return the ordered list of feature names produced by ``build_features``.

    Returns:
        List[str]: Column names.
    """
    return [
        "x", "x_dot", "theta", "theta_dot", "action",
        "abs_x", "abs_theta", "x_margin", "theta_margin",
        "d_x", "d_theta", "d_theta_dot",
        "roll3_abs_theta", "roll3_abs_theta_dot",
        "ttl_theta", "ttl_x",
    ]
