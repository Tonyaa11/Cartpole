"""
Deep sequence model for the CartPole failure-prediction task.

Machine Learning - Workshop 2 (Advanced Learning Models, Part A)

Architecture selection
----------------------
The data are *sequences*: each episode is a time series of states
(x, x_dot, theta, theta_dot) plus the action taken at every step. The label of
a row depends on how the system evolves over the next few steps, so the model
must read the temporal context, not only the current row. Recurrent networks
(LSTM / GRU) maintain a hidden state across time steps and are the natural
choice for this domain; CNNs (images), Transformers (long NLP sequences) and
value-based networks (action-value estimation in RL) do not match the problem
structure. No pretrained model exists for this domain, so the network is
trained from scratch with careful initialisation, weight decay and a learning
rate schedule.

Model
-----
    input  (B, T, F)  -- window of T steps, F features per step
    pack_padded_sequence (right padding is masked out)
    LSTM/GRU (1-2 layers, hidden H, optional bidirectional)
    head: Linear -> ReLU -> Dropout -> Linear -> logit (B,)

Usage (smoke test: forward pass + parameter count)
--------------------------------------------------
    python src/model.py
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class FailurePredictor(nn.Module):
    """Recurrent encoder + MLP head that predicts failure-imminence logits.

    The network reads a window of past steps and outputs one logit per window
    (``logit >= 0`` -> class 1 "failure imminent" after the sigmoid).

    Args:
        input_size (int): Features per time step (5 raw or 16 engineered).
        hidden_size (int): Hidden units of the recurrent layer.
        num_layers (int): Number of stacked recurrent layers.
        dropout (float): Dropout inside the head (and between RNN layers when
            ``num_layers > 1``).
        cell (str): ``"lstm"`` or ``"gru"``.
        bidirectional (bool): If True, the encoder reads the window in both
            directions (allowed here: the window only contains PAST steps with
            respect to the predicted row, so there is no future leakage).
        head_dims (tuple): Width of the hidden layers of the MLP head.

    Attributes:
        rnn (nn.LSTM | nn.GRU): Recurrent encoder.
        head (nn.Sequential): MLP head mapping the final hidden state to a logit.
    """

    CELLS = {"lstm": nn.LSTM, "gru": nn.GRU}

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        cell: str = "lstm",
        bidirectional: bool = False,
        head_dims: tuple = (32,),
    ) -> None:
        super().__init__()
        if cell not in self.CELLS:
            raise ValueError(f"cell must be one of {list(self.CELLS)}, got {cell!r}")

        self.cell = cell
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        rnn_cls = self.CELLS[cell]
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        encoder_out = hidden_size * (2 if bidirectional else 1)

        layers: list[nn.Module] = []
        prev = encoder_out
        for dim in head_dims:
            layers += [nn.Linear(prev, dim), nn.ReLU(), nn.Dropout(dropout)]
            prev = dim
        layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal recurrent initialisation + small uniform head weights.

        Orthogonal initialisation is the standard for recurrent gates (it keeps
        gradients stable over the sequence); the head uses Xavier-uniform.

        Returns:
            None
        """
        for name, param in self.rnn.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Forget-gate bias = 1 helps the cell retain information (LSTM)
                if self.cell == "lstm":
                    n = param.shape[0]
                    param.data[n // 4 : n // 2].fill_(1.0)
        for module in self.head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Run the encoder over a batch of padded windows.

        Args:
            x (torch.Tensor): ``(B, T, F)`` padded windows (right padding).
            lengths (torch.Tensor): ``(B,)`` true lengths (number of real steps).

        Returns:
            torch.Tensor: ``(B,)`` raw logits (apply sigmoid for probabilities).

        Raises:
            ValueError: If ``x`` is not 3-D or lengths do not match the batch.
        """
        if x.dim() != 3:
            raise ValueError(f"expected x of shape (B, T, F), got {tuple(x.shape)}")
        if lengths.shape[0] != x.shape[0]:
            raise ValueError("lengths must have one entry per batch element")

        # Right padding is ignored by the packed representation; the final
        # hidden state therefore corresponds to the last REAL step of each row.
        packed = pack_padded_sequence(
            x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.rnn(packed)
        if isinstance(h_n, tuple):  # LSTM returns (h_n, c_n), GRU returns h_n
            h_n = h_n[0]
        # h_n: (num_layers * num_directions, B, H)

        if self.bidirectional:
            # Last layer, both directions
            state = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            state = h_n[-1]  # last layer, forward direction

        return self.head(state).squeeze(-1)

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count the parameters of the model.

        Args:
            trainable_only (bool): Count only parameters with ``requires_grad``.

        Returns:
            int: Number of parameters.
        """
        params = (p for p in self.parameters() if p.requires_grad or not trainable_only)
        return sum(p.numel() for p in params)


def build_model(cfg: dict) -> FailurePredictor:
    """Instantiate a model from a configuration dictionary.

    Args:
        cfg (dict): Must contain ``input_size`` and may contain ``hidden_size``,
            ``num_layers``, ``dropout``, ``cell``, ``bidirectional``.

    Returns:
        FailurePredictor: Uninitialised (weights initialised in ``__init__``) model.
    """
    return FailurePredictor(
        input_size=int(cfg["input_size"]),
        hidden_size=int(cfg.get("hidden_size", 64)),
        num_layers=int(cfg.get("num_layers", 2)),
        dropout=float(cfg.get("dropout", 0.2)),
        cell=str(cfg.get("cell", "lstm")),
        bidirectional=bool(cfg.get("bidirectional", False)),
    )


def _smoke_test() -> None:
    """Verify the model with a forward pass on a dummy batch (shapes + params)."""
    torch.manual_seed(42)
    batch, steps, feats = 8, 16, 5
    x = torch.randn(batch, steps, feats)
    lengths = torch.tensor([16, 7, 3, 16, 10, 1, 16, 5])

    for cell in ("lstm", "gru"):
        model = FailurePredictor(input_size=feats, cell=cell)
        logits = model(x, lengths)
        assert logits.shape == (batch,), logits.shape
        print(
            f"{cell:>4}: output {tuple(logits.shape)} | "
            f"parameters {model.count_parameters():,}"
        )

    bidir = FailurePredictor(input_size=feats, bidirectional=True)
    logits = bidir(x, lengths)
    assert logits.shape == (batch,)
    print(f"lstm(bi): output {tuple(logits.shape)} | parameters {bidir.count_parameters():,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CartPole failure predictor (smoke test)")
    parser.add_argument("--input-size", type=int, default=5, help="features per step")
    parser.add_argument("--hidden", type=int, default=64, help="hidden units")
    parser.add_argument("--layers", type=int, default=2, help="RNN layers")
    parser.add_argument("--cell", choices=["lstm", "gru"], default="lstm")
    parser.add_argument("--bidirectional", action="store_true")
    args = parser.parse_args()

    if args.input_size == 5 and not args.bidirectional and args.hidden == 64 and args.layers == 2 and args.cell == "lstm":
        _smoke_test()  # default configuration: full smoke test of both cells
    else:
        torch.manual_seed(42)
        m = FailurePredictor(
            input_size=args.input_size,
            hidden_size=args.hidden,
            num_layers=args.layers,
            cell=args.cell,
            bidirectional=args.bidirectional,
        )
        out = m(torch.randn(4, 16, args.input_size), torch.tensor([16, 8, 3, 1]))
        print(f"output {tuple(out.shape)} | parameters {m.count_parameters():,}")
