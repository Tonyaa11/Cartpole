"""
w2_bridge.py - Reuse the Workshop 2 deep sequence model inside the RL workshop.

Machine Learning - Workshop 3 (Reinforcement Learning)

Workshop 2 trained a 2-layer LSTM that reads the last ``window`` (= 16) steps of an
episode (x, x_dot, theta, theta_dot, action) and predicts whether the episode is
within five steps of a failure. Its checkpoint
``checkpoints/workshop2/model_best.pt`` is self-contained: weights, architecture,
the z-score statistics fitted on the W2 training set and the decision threshold
chosen on the W2 validation set. Nothing is retrained here.

This module mirrors ``w1_bridge.py`` so both supervised models are used in the same
two ways:

1. ``danger_rate`` / ``trajectory_proba``: share of the steps of a trajectory that
   the W2 model flags as dangerous (a metric shared by Workshops 2, 3 and 4).
2. ``danger_proba_candidates``: probability of danger of the state reached by each
   action, used by the "W2-guided" baseline policy (one-step look-ahead).

It is also used by ``src/w4/monitor_w3.py`` (Workshop 4 monitoring of the agents).

Usage
-----
    python src/w3/w2_bridge.py      # loads the model and checks it against the W2 pipeline
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from w2.model import build_model  # noqa: E402  (Workshop 2 architecture)
from w3.common import ROOT_DIR  # noqa: E402

W2_CHECKPOINT = ROOT_DIR / "checkpoints" / "workshop2" / "model_best.pt"


class W2DangerModel:
    """Workshop 2 LSTM failure predictor, loaded from its checkpoint."""

    def __init__(self, checkpoint: Path = W2_CHECKPOINT) -> None:
        """Load the checkpoint (weights, scaler statistics, window and threshold).

        Args:
            checkpoint (Path): Path of the Workshop 2 checkpoint.

        Raises:
            ValueError: If the checkpoint does not use the raw 5-column features
                this bridge builds.
        """
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if ckpt.get("features") != "raw":
            raise ValueError("w2_bridge expects a Workshop 2 model trained on raw features")
        self.model = build_model(ckpt["model_cfg"])
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.window = int(ckpt["window"])
        self.threshold = float(ckpt["threshold"])            # chosen on W2 validation
        self.mean = np.asarray(ckpt["scaler_mean"], dtype=np.float32)    # W2 train stats
        self.scale = np.asarray(ckpt["scaler_scale"], dtype=np.float32)
        self.run_name = ckpt.get("run_name", "w2")
        self.n_parameters = int(sum(p.numel() for p in self.model.parameters()))

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _rows(self, states: np.ndarray, actions: Sequence[int]) -> np.ndarray:
        """Scale (state, action) rows exactly as Workshop 2 did.

        Args:
            states (np.ndarray): States, shape (T, 4).
            actions (Sequence[int]): Action of each row, length T.

        Returns:
            np.ndarray: Scaled rows (x, x_dot, theta, theta_dot, action), (T, 5).
        """
        rows = np.column_stack([np.asarray(states, dtype=np.float32),
                                np.asarray(actions, dtype=np.float32)])
        return (rows - self.mean) / self.scale

    @torch.no_grad()
    def _predict_windows(self, X: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """Run the LSTM on right-padded windows.

        Args:
            X (np.ndarray): Windows, shape (B, window, 5), right zero-padded.
            lengths (np.ndarray): True number of steps of each window, (B,).

        Returns:
            np.ndarray: Probability of the "danger" class, shape (B,).
        """
        logits = self.model(torch.as_tensor(X, dtype=torch.float32),
                            torch.as_tensor(lengths, dtype=torch.long))
        return torch.sigmoid(logits).numpy()

    # ------------------------------------------------------------------ #
    # Batch use: full trajectories
    # ------------------------------------------------------------------ #
    def trajectory_proba(self, states: np.ndarray, actions: Sequence[int],
                         batch_size: int = 1024) -> np.ndarray:
        """Danger probability of every step of one trajectory.

        Step i is described by the past-only window ending at step i (at most
        ``window`` steps), the same construction as ``src/w2/sequences.py``.

        Args:
            states (np.ndarray): States s_0..s_{T-1}, shape (T, 4).
            actions (Sequence[int]): Action taken in each of those states.
            batch_size (int): Windows per forward pass.

        Returns:
            np.ndarray: Probability per step, shape (T,).
        """
        rows = self._rows(states, actions)
        T, F = rows.shape
        X = np.zeros((T, self.window, F), dtype=np.float32)
        lengths = np.zeros(T, dtype=np.int64)
        for t in range(T):
            lo = max(0, t + 1 - self.window)
            X[t, : t + 1 - lo] = rows[lo : t + 1]   # real steps left-aligned (right padding)
            lengths[t] = t + 1 - lo
        out = [self._predict_windows(X[i:i + batch_size], lengths[i:i + batch_size])
               for i in range(0, T, batch_size)]
        return np.concatenate(out)

    def danger_rate(self, states: np.ndarray, actions: Sequence[int]) -> float:
        """Share of the steps flagged as dangerous by the W2 model.

        Args:
            states (np.ndarray): States of one trajectory, shape (T, 4).
            actions (Sequence[int]): Actions taken in those states.

        Returns:
            float: Fraction of steps with probability >= the W2 threshold.
        """
        return float(np.mean(self.trajectory_proba(states, actions) >= self.threshold))

    # ------------------------------------------------------------------ #
    # Online use: one step at a time (W2-guided policy)
    # ------------------------------------------------------------------ #
    def danger_proba_candidates(self, history_states: List[np.ndarray], history_actions: List[int],
                                candidates: List[np.ndarray], actions: List[int]) -> np.ndarray:
        """Danger probability of several candidate next states (one per action).

        Each candidate window is the recent history of the episode (states with
        the actions actually taken in them) followed by the candidate state. As
        in the W1-guided policy, the candidate row carries the action that leads
        to it, because the action that will be taken there is not known yet.

        Args:
            history_states (List[np.ndarray]): States of the episode so far, ending
                with the current state.
            history_actions (List[int]): Action taken in each history state; the
                current state has no action yet, so this list is one shorter.
            candidates (List[np.ndarray]): Predicted next state for each action.
            actions (List[int]): The action that leads to each candidate.

        Returns:
            np.ndarray: One probability per candidate.
        """
        # The current state gets, for each candidate, the action being evaluated
        keep = self.window - 1
        past_s = history_states[:-1][-(keep - 1):] if keep > 1 else []
        past_a = history_actions[-(keep - 1):] if keep > 1 else []
        windows, lengths = [], []
        for cand, a in zip(candidates, actions):
            states = list(past_s) + [history_states[-1], cand]
            acts = list(past_a) + [a, a]
            rows = self._rows(np.stack(states), acts)
            pad = np.zeros((self.window, rows.shape[1]), dtype=np.float32)
            pad[: len(rows)] = rows
            windows.append(pad)
            lengths.append(len(rows))
        return self._predict_windows(np.stack(windows), np.array(lengths))


def summary(model: W2DangerModel) -> Dict[str, float]:
    """Short description of the loaded W2 model (for logs).

    Args:
        model (W2DangerModel): The loaded model.

    Returns:
        Dict[str, float]: Run name, window, threshold and parameter count.
    """
    return {"run_name": model.run_name, "window": model.window,
            "threshold": round(model.threshold, 4), "n_parameters": model.n_parameters}


def self_test(model: W2DangerModel) -> Dict[str, float]:
    """Check the bridge against the Workshop 2 pipeline on the W2 test episodes.

    Rebuilds the test windows with ``src/w2/sequences.py`` (the W2 code) and
    compares its probabilities with ``trajectory_proba`` (this module). It also
    recomputes the W2 test F1 at the stored threshold, which must match the W2
    report (0.865).

    Args:
        model (W2DangerModel): The loaded model.

    Returns:
        Dict[str, float]: Max probability difference and the recomputed test F1.
    """
    from sklearn.metrics import f1_score

    from w1.dataset import load_raw, load_splits      # Workshop 1 data pipeline
    from w2.sequences import compute_features, window_episodes  # Workshop 2 windows

    df = load_raw()
    test_eps = load_splits()["test"]
    probs, labels = [], []
    for ep in test_eps:
        part = df[df["episode"] == ep]
        probs.append(model.trajectory_proba(part[["x", "x_dot", "theta", "theta_dot"]].values,
                                            part["action"].values))
        labels.append(part["label"].values)
    p, y = np.concatenate(probs), np.concatenate(labels)

    # Reference: the W2 sequence builder with the W2 training scaler
    part = df[df["episode"].isin(test_eps)].sort_values(["episode", "t"])
    feats = ((compute_features(part, "raw").values - model.mean) / model.scale).astype(np.float32)
    X, lengths = window_episodes(feats, part["episode"].values, model.window)
    ref = model._predict_windows(X, lengths)
    return {"max_abs_diff": float(np.max(np.abs(ref - p))),
            "test_f1": float(f1_score(y, (p >= model.threshold).astype(int)))}


if __name__ == "__main__":
    torch.set_num_threads(1)
    m = W2DangerModel()
    print("W2 model loaded:", summary(m))
    print("bridge vs W2 pipeline on the W2 test set:", self_test(m))
