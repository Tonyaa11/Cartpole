"""
w2_bridge.py - Reuse the Workshop 2 LSTM failure classifier inside Workshop 4.

Machine Learning - Workshop 4 (Autoencoders & VAE)

Wraps the Workshop 2 LSTM (FailurePredictor) so that Workshop 4 scripts can
evaluate it as a supervised-learning baseline for anomaly / failure detection.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from w4.w4_common import PARTITIONS, W2_CKPT, W4Data, w2_windows
from w2.model import build_model


class W2DangerModel:
    """Workshop 2 LSTM failure classifier, loaded from checkpoint."""

    def __init__(self) -> None:
        ckpt = torch.load(W2_CKPT, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        self.model = build_model(cfg)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.scaler = ckpt["scaler"]
        self.threshold: float = float(ckpt.get("threshold", 0.5))
        self.window: int = int(cfg.get("window", 16))

    def trajectory_proba(self, states: np.ndarray, actions: Sequence[int]) -> np.ndarray:
        """Danger probability for every step of one trajectory.

        Args:
            states: Shape (T, 4) — CartPole state observations.
            actions: Length-T action sequence.

        Returns:
            np.ndarray: Per-step danger probability, shape (T,).
        """
        df = pd.DataFrame(states, columns=["x", "x_dot", "theta", "theta_dot"])
        df.insert(0, "t", np.arange(len(df)))
        df.insert(0, "episode", 0)
        df["action"] = np.asarray(actions, dtype=int)
        from w2.sequences import build_sequences
        X, lengths, _, _ = build_sequences(df, [0], window=self.window,
                                           scaler=self.scaler, fit_scaler=False)
        with torch.no_grad():
            logits = self.model(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(lengths, dtype=torch.long),
            )
            proba = torch.sigmoid(logits).squeeze(-1).numpy()
        return proba
