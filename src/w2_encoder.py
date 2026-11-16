"""
Frozen Workshop 2 model (deep LSTM) used as the SUPERVISED reference in Workshop 4.

Machine Learning - Workshop 4

The Workshop 2 network (``model.FailurePredictor``: 2-layer LSTM with 64
hidden units over past-only windows of 16 raw steps, followed by an MLP head)
is loaded from ``checkpoints/w2_lstm_best.pt`` (a copy of the Workshop 2
``checkpoints/model_best.pt``) and is NOT retrained.

Two outputs are extracted for every row of every partition:
    * ``z``     : final hidden state of the last LSTM layer (64-D). This is the
                  frozen *supervised encoder* whose representation is compared
                  with the autoencoder latent spaces.
    * ``proba`` : predicted probability of "failure imminent" (sigmoid of the
                  head), used for the Workshop 2 row of the cumulative table.

Rows are produced in the same (episode, t) order as ``w4_common.load_w4_data``
and the alignment is checked explicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence

from model import FailurePredictor, build_model
from w4_common import PARTITIONS, W2_CKPT, W4Data, w2_windows


def load_w2(path: Path = W2_CKPT) -> Tuple[FailurePredictor, Dict]:
    """Load the Workshop 2 checkpoint (weights + scaler + window settings).

    Args:
        path (Path): Checkpoint written by the Workshop 2 ``train.py``.

    Returns:
        Tuple[FailurePredictor, Dict]: Model in eval mode and the checkpoint metadata.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["model_cfg"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    meta = {k: v for k, v in ckpt.items() if k != "model_state"}
    return model, meta


@torch.no_grad()
def encode_windows(model: FailurePredictor, X: np.ndarray, lengths: np.ndarray,
                   batch_size: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    """Frozen hidden state and failure probability for a set of windows.

    Mirrors ``FailurePredictor.forward`` but also returns the hidden state that
    feeds the head.

    Args:
        model (FailurePredictor): Trained Workshop 2 model.
        X (np.ndarray): Windows, shape (n, window, F).
        lengths (np.ndarray): True window lengths.
        batch_size (int): Inference batch size.

    Returns:
        Tuple[np.ndarray, np.ndarray]: ``z`` (n, H) and ``proba`` (n,).
    """
    zs, ps = [], []
    for s in range(0, len(X), batch_size):
        x = torch.as_tensor(X[s:s + batch_size])
        lens = torch.as_tensor(lengths[s:s + batch_size]).clamp(min=1)
        packed = pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        _, h_n = model.rnn(packed)
        if isinstance(h_n, tuple):  # LSTM -> (h_n, c_n)
            h_n = h_n[0]
        state = torch.cat([h_n[-2], h_n[-1]], dim=-1) if model.bidirectional else h_n[-1]
        zs.append(state.numpy())
        ps.append(torch.sigmoid(model.head(state).squeeze(-1)).numpy())
    return np.concatenate(zs), np.concatenate(ps)


def w2_infer(data: W4Data, path: Path = W2_CKPT) -> Dict[str, Dict[str, np.ndarray]]:
    """Encode every partition with the frozen Workshop 2 model.

    Args:
        data (W4Data): Workshop 4 data (used to verify the row alignment).
        path (Path): Workshop 2 checkpoint.

    Returns:
        Dict[str, Dict[str, np.ndarray]]: ``{partition: {"z": ..., "proba": ...}}``.

    Raises:
        AssertionError: If the Workshop 2 rows are not aligned with Workshop 4.
    """
    model, meta = load_w2(path)
    out = {}
    for name in PARTITIONS:
        X, lengths, episode, t = w2_windows(meta, name)
        assert np.array_equal(episode, data[name].episode) and np.array_equal(t, data[name].t), \
            f"row order mismatch between Workshop 2 windows and Workshop 4 data ({name})"
        z, proba = encode_windows(model, X, lengths)
        out[name] = {"z": z, "proba": proba}
    return out
