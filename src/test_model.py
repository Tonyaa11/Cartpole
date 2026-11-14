"""
Inference smoke test: load the saved checkpoint and predict on NEW data.

Machine Learning - Workshop 2 (Advanced Learning Models, Part A)

The script verifies that ``checkpoints/model_best.pt`` is self-contained: it
rebuilds the scaler and the windows from the raw CSV (data never seen during
training is used by default: the TEST episodes), loads the weights and runs a
forward pass. It prints the predicted probability of failure-imminence for a
few sample windows and a summary over the whole set.

Usage
-----
    python src/test_model.py                     # 5 example windows (test)
    python src/test_model.py --n 20 --split val   # more examples, val split
    python src/test_model.py --interactive        # type your own state row
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from dataset import EPISODE_COL, RAW_FEATURE_COLS, ROOT_DIR, load_raw, load_splits
from model import build_model
from sequences import compute_features, window_episodes
from train import set_global_seed

CKPT_PATH = ROOT_DIR / "checkpoints" / "model_best.pt"


def predict_examples(
    ckpt: dict, split: str = "test", n: int = 5, seed: int = 42
) -> None:
    """Run inference on ``n`` example windows of the chosen split.

    The scaler stored in the checkpoint (fitted on TRAIN during training) is
    reused, so the numbers here match exactly what training/evaluation saw.

    Args:
        ckpt (dict): Loaded checkpoint (from ``train.py``).
        split (str): ``"train"``, ``"val"`` or ``"test"`` (default ``"test"``:
            the point is to verify predictions on unseen data).
        n (int): Number of examples to print.
        seed (int): Seed for the example sampling.

    Returns:
        None
    """
    set_global_seed(seed)
    df = load_raw()
    splits = load_splits()

    part_df = df[df[EPISODE_COL].isin(splits[split])].sort_values([EPISODE_COL, "t"])

    # Scale with the TRAINING scaler stored in the checkpoint (no refitting)
    feats = compute_features(part_df, str(ckpt["features"]))
    mean = np.asarray(ckpt["scaler_mean"], dtype=np.float32)
    scale = np.asarray(ckpt["scaler_scale"], dtype=np.float32)
    scaled = (feats.to_numpy(dtype=np.float32) - mean) / scale

    X, lengths = window_episodes(
        scaled, part_df[EPISODE_COL].to_numpy(), int(ckpt["window"])
    )
    raw = part_df[RAW_FEATURE_COLS].to_numpy(dtype=np.float32)  # for display
    y = part_df["label"].to_numpy(dtype=np.float32)

    model = build_model(ckpt["model_cfg"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    thr = float(ckpt["threshold"])

    with torch.no_grad():
        logits = model(torch.from_numpy(X), torch.from_numpy(lengths))
        probs = torch.sigmoid(logits).numpy()

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=min(n, len(y)), replace=False)
    print("=" * 78)
    print(f"INFERENCE | split={split} | window={ckpt['window']} | "
          f"threshold={thr:.3f} | examples={len(idx)}")
    print("=" * 78)
    print(f"{'#':>4} {'len':>4} {'prob':>7} {'pred':>5} {'true':>5}  state (x, x_dot, theta, theta_dot, action)")
    for i in sorted(idx):
        r = raw[i]  # current step (last real step of the window)
        print(
            f"{i:>4} {int(lengths[i]):>4} {probs[i]:>7.3f} "
            f"{int(probs[i] >= thr):>5} {int(y[i]):>5}  "
            f"[{r[0]:+.3f}, {r[1]:+.3f}, {r[2]:+.3f}, {r[3]:+.3f}, {int(r[4])}]"
        )

    pred = probs >= thr
    acc = float((pred == y).mean())
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    print("-" * 78)
    print(f"accuracy {acc:.3f} | detected {tp}/{tp + fn} imminent failures "
          f"({tp / max(tp + fn, 1):.1%} recall) on {len(y)} {split} windows")
    print(f"model: {ckpt['model_cfg']} | params "
          f"{build_model(ckpt['model_cfg']).count_parameters():,}")


def predict_state_row(ckpt: dict, row: dict) -> float:
    """Predict from a single user-provided state (interactive mode).

    Args:
        ckpt (dict): Loaded checkpoint.
        row (dict): Must contain the raw feature columns of ONE time step.

    Returns:
        float: Probability of failure-imminence.
    """
    import pandas as pd

    from sequences import compute_features

    df = pd.DataFrame([row])
    feats = compute_features(df, str(ckpt["features"]))
    mean = np.asarray(ckpt["scaler_mean"], dtype=np.float32)
    scale = np.asarray(ckpt["scaler_scale"], dtype=np.float32)
    scaled = ((feats.to_numpy(dtype=np.float32) - mean) / scale)
    # A single row -> a window of length 1
    X, lengths = window_episodes(scaled, np.array([0]), int(ckpt["window"]))

    model = build_model(ckpt["model_cfg"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.from_numpy(X), torch.from_numpy(lengths)))[0]
    return float(prob)


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description="Verify the saved CartPole model")
    p.add_argument("--checkpoint", type=Path, default=CKPT_PATH)
    p.add_argument("--n", type=int, default=5, help="number of examples")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--interactive", action="store_true",
                   help="type one state row and get a prediction")
    args = p.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Train one first:  python src/train.py"
        )
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    if args.interactive:
        print("Enter one CartPole state (press Enter after each value):")
        row = {}
        for col in RAW_FEATURE_COLS:
            if col == "action":
                row[col] = int(float(input(f"  {col} (0/1): ")))
            else:
                row[col] = float(input(f"  {col}: "))
        prob = predict_state_row(ckpt, row)
        thr = float(ckpt["threshold"])
        print(f"\nP(failure imminent) = {prob:.3f}  ->  "
              f"{'FAILURE IMMINENT' if prob >= thr else 'safe'} (threshold {thr:.3f})")
    else:
        predict_examples(ckpt, split=args.split, n=args.n, seed=args.seed)


if __name__ == "__main__":
    main()
