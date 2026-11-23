"""
w1_bridge.py - Reuse the Workshop 1 failure classifier inside the RL workshop.

Machine Learning - Workshop 3 (Reinforcement Learning)

Workshop 1 trained a Gradient Boosting classifier that predicts whether a CartPole
state is within five steps of a failure ("danger"). This module rebuilds that model
with EXACTLY the Workshop 1 pipeline (same split, 16 features, scaler fitted on the
training set, decision threshold chosen on validation) and exposes two uses:

1. ``danger_rate(trajectory)``: share of the steps of a trajectory that the W1
   classifier flags as dangerous. It lets us measure every policy of Workshop 3
   with the Workshop 1 model (a metric shared by both workshops).
2. ``danger_proba_candidates(...)``: probability of danger for the states reached
   by each action, used by the "W1-guided" baseline policy (one-step look-ahead).

The Workshop 1 modules live in ``src/w1`` (``dataset.py``, ``features.py`` and
``baselines.py``) and are imported as a package, so they never clash with the
Workshop 3 module of the same name (``src/w3/baselines.py``).

Usage
-----
    python src/w3/w1_bridge.py      # rebuilds the model and checks the online features
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import w1.baselines as w1_baselines  # noqa: E402  (Workshop 1 models and threshold rule)
import w1.dataset as w1_dataset    # noqa: E402  (Workshop 1)
import w1.features as w1_features  # noqa: E402  (Workshop 1)


class W1DangerModel:
    """Workshop 1 Gradient Boosting failure classifier, rebuilt and ready to use."""

    def __init__(self) -> None:
        """Fit the W1 pipeline: features -> scaler (train) -> GB -> threshold (val)."""
        w1 = w1_baselines
        w1_dataset.set_global_seed()
        df = w1_dataset.load_raw()
        splits = w1_dataset.get_or_create_splits(df)
        parts = {k: w1_dataset.subset(df, v) for k, v in splits.items()}

        X_tr = w1_features.build_features(parts["train"])
        X_va = w1_features.build_features(parts["val"])
        self.columns = list(X_tr.columns)
        self.scaler = w1_dataset.fit_scaler(X_tr)            # statistics from W1 train only

        model = w1.get_models()["Gradient Boosting"]
        self.model = w1.fit_model("Gradient Boosting", model,
                                  w1_dataset.apply_scaler(self.scaler, X_tr),
                                  parts["train"]["label"].astype(int))
        val_scores = self.model.predict_proba(w1_dataset.apply_scaler(self.scaler, X_va))[:, 1]
        self.threshold = w1.best_f1_threshold(parts["val"]["label"].astype(int), val_scores)
        self.n_train_samples = len(X_tr)

    # ------------------------------------------------------------------ #
    # Batch use: full trajectories
    # ------------------------------------------------------------------ #
    def trajectory_proba(self, states: np.ndarray, actions: Sequence[int]) -> np.ndarray:
        """Danger probability of every step of one trajectory.

        Args:
            states (np.ndarray): States s_0..s_{T-1}, shape (T, 4).
            actions (Sequence[int]): Action taken in each of those states.

        Returns:
            np.ndarray: Probability of the "danger" class per step, shape (T,).
        """
        df = pd.DataFrame(states, columns=["x", "x_dot", "theta", "theta_dot"])
        df.insert(0, "t", np.arange(len(df)))
        df.insert(0, "episode", 0)
        df["action"] = np.asarray(actions, dtype=int)
        X = w1_features.build_features(df)[self.columns]
        return self.model.predict_proba(w1_dataset.apply_scaler(self.scaler, X))[:, 1]

    def danger_rate(self, states: np.ndarray, actions: Sequence[int]) -> float:
        """Share of the steps flagged as dangerous by the W1 classifier.

        Args:
            states (np.ndarray): States of one trajectory, shape (T, 4).
            actions (Sequence[int]): Actions taken in those states.

        Returns:
            float: Fraction of steps with probability >= the W1 threshold.
        """
        return float(np.mean(self.trajectory_proba(states, actions) >= self.threshold))

    # ------------------------------------------------------------------ #
    # Online use: one step at a time (W1-guided policy)
    # ------------------------------------------------------------------ #
    def online_features(self, history: List[np.ndarray], state: np.ndarray, action: int) -> np.ndarray:
        """Compute the 16 W1 features of ``state`` from the recent history.

        This is a fast NumPy version of ``features.build_features`` for one row;
        ``self_test`` checks that both give the same numbers.

        Args:
            history (List[np.ndarray]): Previous states of the episode (only the
                last two are used), oldest first.
            state (np.ndarray): The state to describe.
            action (int): Action associated with that state.

        Returns:
            np.ndarray: Feature vector in the W1 column order, shape (16,).
        """
        TL, XL, CAP = w1_features.THETA_LIMIT, w1_features.X_LIMIT, w1_features.TTL_CAP
        x, x_dot, th, th_dot = state
        prev = history[-1] if history else None
        d_x = x - prev[0] if prev is not None else 0.0
        d_th = th - prev[2] if prev is not None else 0.0
        d_thd = th_dot - prev[3] if prev is not None else 0.0
        window = list(history[-2:]) + [state]                  # rolling mean of 3 steps
        roll_th = float(np.mean([abs(s[2]) for s in window]))
        roll_thd = float(np.mean([abs(s[3]) for s in window]))

        def ttl(margin: float, speed: float) -> float:
            return float(np.clip(margin / speed, 0.0, CAP)) if speed > 1e-6 else CAP

        th_margin, x_margin = TL - abs(th), XL - abs(x)
        return np.array([
            x, x_dot, th, th_dot, float(action),
            abs(x), abs(th), x_margin, th_margin,
            d_x, d_th, d_thd, roll_th, roll_thd,
            ttl(th_margin, np.sign(th) * th_dot), ttl(x_margin, np.sign(x) * x_dot),
        ])

    def danger_proba_candidates(self, history: List[np.ndarray], candidates: List[np.ndarray],
                                actions: List[int]) -> np.ndarray:
        """Danger probability of several candidate next states (one per action).

        Args:
            history (List[np.ndarray]): States of the episode so far, ending with
                the current state.
            candidates (List[np.ndarray]): Predicted next state for each action.
            actions (List[int]): The action that leads to each candidate.

        Returns:
            np.ndarray: One probability per candidate.
        """
        X = np.stack([self.online_features(history, c, a) for c, a in zip(candidates, actions)])
        # Same z-score as the W1 scaler, applied with NumPy (much faster than a DataFrame
        # inside the control loop); the model receives the column names it was fitted with
        X = (X - self.scaler.mean_) / self.scaler.scale_
        return self.model.predict_proba(pd.DataFrame(X, columns=self.columns))[:, 1]

    def self_test(self, n_episodes: int = 20) -> float:
        """Check that ``online_features`` reproduces ``build_features`` exactly.

        Args:
            n_episodes (int): Number of dataset episodes to compare.

        Returns:
            float: Maximum absolute difference between both implementations.
        """
        df = w1_dataset.load_raw()
        df = df[df["episode"].isin(df["episode"].unique()[:n_episodes])].reset_index(drop=True)
        ref = w1_features.build_features(df)[self.columns].values
        rows = []
        for _, ep in df.groupby("episode", sort=False):
            states = ep[["x", "x_dot", "theta", "theta_dot"]].values
            for i in range(len(states)):
                rows.append(self.online_features(list(states[:i]), states[i], int(ep["action"].iloc[i])))
        return float(np.max(np.abs(np.stack(rows) - ref)))


def summary(model: W1DangerModel) -> Dict[str, float]:
    """Short description of the rebuilt W1 model (for logs).

    Args:
        model (W1DangerModel): The fitted model.

    Returns:
        Dict[str, float]: Threshold and number of training samples.
    """
    return {"threshold": float(model.threshold), "n_train_samples": int(model.n_train_samples)}


if __name__ == "__main__":
    m = W1DangerModel()
    print("W1 Gradient Boosting rebuilt:", summary(m))
    print(f"online_features vs build_features, max |diff| = {m.self_test():.2e}")
