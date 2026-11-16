"""
Training script for the CartPole failure-prediction deep model (Workshop 2).

Machine Learning - Workshop 2 (Advanced Learning Models, Part A)

What it does
------------
1. Loads the raw CSV and the EPISODE split of ``data/splits.json`` (the same
   split used by the Workshop 1 baselines - never re-split).
2. Builds past-only sliding windows (``src/sequences.py``) with a z-score
   scaler fitted on TRAIN only.
3. Trains the recurrent model of ``src/model.py`` with:
   * loss: BCE-with-logits POSITION-WEIGHTED for the 17.2:1 imbalance
     (equivalent to class weighting in the loss, as recommended by the brief);
   * optimizer: AdamW (decoupled weight decay);
   * LR schedule: cosine annealing with warm restarts-free decay;
   * regularisation: dropout, weight decay and early stopping;
   * gradient clipping (RNNs are prone to exploding gradients).
4. Logs every epoch to TensorBoard (``runs/tb/<run_name>``) AND to a CSV
   (``runs/train_log_<run_name>.csv``) so the learning curves can be
   reproduced without TensorBoard.
5. Saves the BEST model (highest validation F1 at the validation-selected
   threshold) to ``checkpoints/model_best.pt`` together with the scaler, the
   window configuration and the selected threshold - everything needed to
   reload it in ``src/evaluate.py`` / ``src/test_model.py``.

Usage
-----
    python src/train.py                       # default LSTM run
    python src/train.py --cell gru --window 32
    python src/train.py --features engineered  # 16 engineered features
    python src/train.py --epochs 60 --lr 3e-3
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from dataset import EPISODE_COL, ROOT_DIR, SEED, get_or_create_splits, load_raw, load_splits
from model import FailurePredictor, build_model
from sequences import (
    MODE_ENGINEERED,
    MODE_RAW,
    build_all_partitions,
    compute_features,
    describe_partition,
    feature_count,
)

RUNS_DIR = ROOT_DIR / "runs"
CKPT_DIR = ROOT_DIR / "checkpoints"
TB_DIR = RUNS_DIR / "tb"


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_global_seed(seed: int = SEED) -> None:
    """Fix Python, NumPy and PyTorch random seeds (and cuDNN determinism).

    Args:
        seed (int): Seed value.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def make_loaders(
    parts: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    batch_size: int,
    seed: int = SEED,
) -> Dict[str, DataLoader]:
    """Wrap the numpy partitions into DataLoaders.

    Args:
        parts (dict): ``{"train": (X, lengths, y), ...}``.
        batch_size (int): Batch size.
        seed (int): Seed of the train-shuffle generator.

    Returns:
        Dict[str, DataLoader]: Loaders per partition (train is shuffled).
    """
    loaders = {}
    for name, (X, L, y) in parts.items():
        ds = TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(L.astype(np.int64)),
            torch.from_numpy(y),
        )
        shuffle = name == "train"
        gen = torch.Generator().manual_seed(seed) if shuffle else None
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=gen,
            num_workers=0,  # Windows-safe
            drop_last=False,
        )
    return loaders


def positive_weight(y_train: np.ndarray) -> torch.Tensor:
    """Compute the BCE pos_weight = n_negative / n_positive on TRAIN only.

    Args:
        y_train (np.ndarray): Training labels.

    Returns:
        torch.Tensor: Scalar tensor with the weight of the positive class.
    """
    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    return torch.tensor(n_neg / max(n_pos, 1.0), dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Metrics helpers (same protocol as Workshop 1)
# --------------------------------------------------------------------------- #
def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Probability threshold that maximises F1 (use on VALIDATION).

    Args:
        y_true (np.ndarray): True labels.
        scores (np.ndarray): Positive-class probabilities.

    Returns:
        float: Threshold with the highest F1.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if thresholds.size == 0:
        return 0.5
    f1 = (
        2 * precision[:-1] * recall[:-1]
        / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    )
    return float(thresholds[int(np.argmax(f1))])


def f1_at_threshold(y_true: np.ndarray, scores: np.ndarray, thr: float) -> float:
    """F1 score for a given decision threshold.

    Args:
        y_true (np.ndarray): True labels.
        scores (np.ndarray): Positive-class probabilities.
        thr (float): Decision threshold.

    Returns:
        float: F1 score.
    """
    return float(f1_score(y_true, scores >= thr, zero_division=0))


# --------------------------------------------------------------------------- #
# Epoch loops
# --------------------------------------------------------------------------- #
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> float:
    """Train for one epoch and return the mean loss.

    Args:
        model (nn.Module): Model (train mode).
        loader (DataLoader): Training loader.
        criterion (nn.Module): Loss function (receives logits and targets).
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Compute device.
        max_grad_norm (float): Gradient-clipping norm (0 disables it).

    Returns:
        float: Mean training loss of the epoch.
    """
    model.train()
    total, n_batches = 0.0, 0
    for X, L, y in loader:
        X, L, y = X.to(device), L.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X, L)
        loss = criterion(logits, y)
        loss.backward()
        if max_grad_norm and max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total += float(loss.item())
        n_batches += 1
    return total / max(n_batches, 1)


@torch.no_grad()
def evaluate_loader(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run the model over a loader and collect labels, probabilities, loss.

    Args:
        model (nn.Module): Model (eval mode).
        loader (DataLoader): Loader to iterate.
        criterion (nn.Module): Loss function.
        device (torch.device): Compute device.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]: labels, probabilities, mean loss.
    """
    model.eval()
    labels, scores, total, n_batches = [], [], 0.0, 0
    for X, L, y in loader:
        X, L, y = X.to(device), L.to(device), y.to(device)
        logits = model(X, L)
        loss = criterion(logits, y)
        total += float(loss.item())
        n_batches += 1
        labels.append(y.cpu().numpy())
        scores.append(torch.sigmoid(logits).cpu().numpy())
    return (
        np.concatenate(labels),
        np.concatenate(scores),
        total / max(n_batches, 1),
    )


# --------------------------------------------------------------------------- #
# Training experiment
# --------------------------------------------------------------------------- #
def _beats_best(path: Path, val_f1: float) -> bool:
    """Check whether a new validation F1 beats the checkpoint stored at ``path``.

    Used to decide if the current run deserves to be promoted to
    ``checkpoints/model_best.pt``. Only the VALIDATION F1 is compared (the
    test set never decides). A missing or unreadable checkpoint is always
    beaten.

    Args:
        path (Path): Candidate checkpoint path (may not exist).
        val_f1 (float): Validation F1 of the current run.

    Returns:
        bool: True if ``path`` should be overwritten with the new run.
    """
    if not Path(path).exists():
        return True
    try:
        stored = torch.load(path, map_location="cpu", weights_only=False)
        stored_f1 = float(stored.get("val_metrics", {}).get("val_f1", -1.0))
    except Exception:
        return True
    return val_f1 > stored_f1


def train(args: argparse.Namespace) -> dict:
    """Full training run: data -> model -> loop -> logs -> best checkpoint.

    Args:
        args (argparse.Namespace): CLI arguments (see ``parse_args``).

    Returns:
        dict: Summary with best epoch, best validation metrics, threshold,
        paths of the generated artifacts and the test metrics of the best
        checkpoint.
    """
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Data ------------------------------------------------------------- #
    df = load_raw()
    splits = load_splits() if Path(ROOT_DIR / "data" / "splits.json").exists() else get_or_create_splits(df)
    parts, scaler = build_all_partitions(
        df, splits, window=args.window, mode=args.features, augment=args.augment
    )
    print(f"device: {device} | features: {args.features} | window: {args.window} | cell: {args.cell}")
    for name, (X, L, y) in parts.items():
        print("  " + describe_partition(name, X, L, y))

    loaders = make_loaders(parts, args.batch_size, seed=args.seed)
    pos_w = positive_weight(parts["train"][2]).to(device)
    feature_cols = list(
        compute_features(df[df[EPISODE_COL].isin(splits["train"])], args.features).columns
    )

    # ---- Model ------------------------------------------------------------ #
    model_cfg = {
        "input_size": feature_count(args.features),
        "hidden_size": args.hidden,
        "num_layers": args.layers,
        "dropout": args.dropout,
        "cell": args.cell,
        "bidirectional": args.bidirectional,
    }
    model = build_model(model_cfg).to(device)
    print(f"model parameters: {model.count_parameters():,}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ---- Logging ---------------------------------------------------------- #
    # Suffix only for non-default capacities so the default run keeps the
    # stable name used in the README/report (and runs with different hidden
    # sizes do not clobber each other's logs/checkpoints).
    capacity = ""
    if args.hidden != 64:
        capacity += f"_{args.hidden}h"
    if args.layers != 2:
        capacity += f"_L{args.layers}"
    run_name = (
        f"w2_{args.cell}_w{args.window}_{args.features}{capacity}"
        f"{'_bi' if args.bidirectional else ''}"
        f"{'_aug' if args.augment else ''}"
    )
    tb = SummaryWriter(log_dir=str(TB_DIR / run_name))
    log_path = RUNS_DIR / f"train_log_{run_name}.csv"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            [
                "epoch", "train_loss", "val_loss", "val_f1", "val_precision",
                "val_recall", "val_roc_auc", "val_pr_auc", "val_acc",
                "threshold", "lr", "epoch_time_s",
            ]
        )

    # ---- Loop ------------------------------------------------------------- #
    best = {"val_f1": -1.0, "epoch": -1, "threshold": 0.5}
    best_path = CKPT_DIR / "model_best.pt"
    epochs_without_improvement = 0
    promoted = False  # did THIS run win the global model_best.pt slot?
    history = []

    print("=" * 72)
    print(f"TRAINING {run_name} | epochs={args.epochs} | lr={args.lr} | "
          f"batch={args.batch_size} | patience={args.patience}")
    print("=" * 72)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, args.grad_clip
        )
        scheduler.step()

        y_va, s_va, val_loss = evaluate_loader(
            model, loaders["val"], criterion, device
        )
        thr = best_f1_threshold(y_va, s_va)
        pred = s_va >= thr
        val_metrics = {
            "val_f1": f1_score(y_va, pred, zero_division=0),
            "val_precision": precision_score(y_va, pred, zero_division=0),
            "val_recall": recall_score(y_va, pred, zero_division=0),
            "val_roc_auc": roc_auc_score(y_va, s_va),
            "val_pr_auc": average_precision_score(y_va, s_va),
            "val_acc": accuracy_score(y_va, pred),
        }
        epoch_time = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        row = [
            epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
            *[f"{val_metrics[k]:.6f}" for k in (
                "val_f1", "val_precision", "val_recall",
                "val_roc_auc", "val_pr_auc", "val_acc")],
            f"{thr:.6f}", f"{lr_now:.6e}", f"{epoch_time:.2f}",
        ]
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        tb.add_scalar("loss/train", train_loss, epoch)
        tb.add_scalar("loss/val", val_loss, epoch)
        for k, v in val_metrics.items():
            tb.add_scalar(f"metrics/{k}", v, epoch)
        tb.add_scalar("threshold/val", thr, epoch)
        tb.add_scalar("lr", lr_now, epoch)

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, **val_metrics, "threshold": thr})

        improved = val_metrics["val_f1"] > best["val_f1"]
        if improved:
            best = {"val_f1": val_metrics["val_f1"], "epoch": epoch,
                    "threshold": thr, **val_metrics}
            epochs_without_improvement = 0
            ckpt_payload = {
                "model_state": model.state_dict(),
                "model_cfg": model_cfg,
                "window": args.window,
                "features": args.features,
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "feature_cols": feature_cols,
                "threshold": thr,
                "val_metrics": val_metrics,
                "best_epoch": epoch,
                "seed": args.seed,
                "run_name": run_name,
            }
            # Per-run checkpoint (so parallel/sequential runs do not clobber
            # each other) ...
            torch.save(ckpt_payload, CKPT_DIR / f"model_{run_name}.pt")
            # ... and promote to model_best.pt only if this run has the best
            # VALIDATION F1 of all runs so far (workshop rule: highest
            # validation metric; the test set never decides).
            if _beats_best(best_path, val_metrics["val_f1"]):
                torch.save(ckpt_payload, best_path)
                promoted = True
        else:
            epochs_without_improvement += 1

        mark = "  *BEST*" if improved else ""
        print(
            f"epoch {epoch:3d}/{args.epochs} | train {train_loss:.4f} | "
            f"val {val_loss:.4f} | F1 {val_metrics['val_f1']:.3f} "
            f"(P {val_metrics['val_precision']:.3f} R {val_metrics['val_recall']:.3f}) "
            f"| AUC {val_metrics['val_roc_auc']:.3f} | PR-AUC {val_metrics['val_pr_auc']:.3f} "
            f"| thr {thr:.3f} | {epoch_time:.1f}s{mark}"
        )

        if args.patience and epochs_without_improvement >= args.patience:
            print(f"early stopping at epoch {epoch} "
                  f"(no val F1 improvement for {args.patience} epochs)")
            break

    tb.close()

    # ---- Test evaluation of THIS run's best checkpoint -------------------- #
    run_ckpt_path = CKPT_DIR / f"model_{run_name}.pt"
    if not run_ckpt_path.exists():
        # No epoch improved on the first epoch (val F1 <= 0): save the final
        # state so the artifact always exists.
        torch.save(
            {
                "model_state": model.state_dict(),
                "model_cfg": model_cfg,
                "window": args.window,
                "features": args.features,
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "feature_cols": feature_cols,
                "threshold": best["threshold"],
                "val_metrics": {"val_f1": best["val_f1"]},
                "best_epoch": best["epoch"],
                "seed": args.seed,
                "run_name": run_name,
            },
            run_ckpt_path,
        )

    ckpt = torch.load(run_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    y_te, s_te, test_loss = evaluate_loader(model, loaders["test"], criterion, device)
    thr = ckpt["threshold"]
    pred = s_te >= thr
    test_metrics = {
        "threshold": thr,
        "accuracy": accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred, zero_division=0),
        "recall": recall_score(y_te, pred, zero_division=0),
        "f1": f1_score(y_te, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_te, s_te),
        "pr_auc": average_precision_score(y_te, s_te),
        "test_loss": test_loss,
    }
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
    test_metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})

    # Persist results next to the Workshop 1 baselines
    results = {
        "run_name": run_name,
        "model_cfg": model_cfg,
        "window": args.window,
        "features": args.features,
        "epochs_run": len(history),
        "best_epoch": best["epoch"],
        "best_val_f1": best["val_f1"],
        "seed": args.seed,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "pos_weight": float(pos_w.cpu()),
        "test_metrics": test_metrics,
        "history": history,
    }
    out_json = RUNS_DIR / f"train_summary_{run_name}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print("-" * 72)
    print(f"BEST epoch {best['epoch']} | val F1 {best['val_f1']:.3f} | "
          f"threshold {thr:.3f}")
    print(f"TEST  accuracy {test_metrics['accuracy']:.3f} | "
          f"precision {test_metrics['precision']:.3f} | "
          f"recall {test_metrics['recall']:.3f} | F1 {test_metrics['f1']:.3f} | "
          f"ROC-AUC {test_metrics['roc_auc']:.3f} | PR-AUC {test_metrics['pr_auc']:.3f}")
    print(f"confusion [[TN FP] [FN TP]] = [[{tn} {fp}] [{fn} {tp}]]")
    print(f"checkpoint -> {run_ckpt_path}"
          + (" (promoted to model_best.pt)" if promoted else
             f" (model_best.pt kept: better validation F1 from another run)"))
    print(f"log csv    -> {log_path}")
    print(f"summary    -> {out_json}")
    print(f"tensorboard -> {TB_DIR / run_name}  (tensorboard --logdir runs/tb)")
    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    p = argparse.ArgumentParser(description="Train the CartPole failure predictor")
    p.add_argument("--epochs", type=int, default=40, help="max epochs")
    p.add_argument("--patience", type=int, default=8,
                   help="early-stopping patience on val F1 (0 = disabled)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3, help="max learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--hidden", type=int, default=64, help="RNN hidden units")
    p.add_argument("--layers", type=int, default=2, help="stacked RNN layers")
    p.add_argument("--cell", choices=["lstm", "gru"], default="lstm")
    p.add_argument("--bidirectional", action="store_true")
    p.add_argument("--window", type=int, default=16, help="past steps per sample")
    p.add_argument("--features", choices=[MODE_RAW, MODE_ENGINEERED],
                   default=MODE_RAW, help="per-step features")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--augment", action="store_true",
                   help="append noise-augmented copies of the TRAIN rows only")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
