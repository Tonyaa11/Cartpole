"""
Training script for Workshop 4: AE, VAE (three betas), denoising AE and the
supervised reference MLP on the CartPole failure-prediction data.

Machine Learning - Workshop 4

Experiment plan (all runs use seed 42, Adam, 50 epochs, batch 64)
-----------------------------------------------------------------
Every unsupervised model is trained in two REGIMES:
    * ``all``    : every training row          -> reconstruction & latent analysis
    * ``normal`` : only safe rows (label = 0)   -> anomaly detection

Stage 1  AE, z_dim in {2, 4, 8}                  (both regimes)
         z* = latent size with the best VALIDATION anomaly ROC-AUC (normal regime)
Stage 2  VAE, z_dim = z*, beta in {0.5, 1, 4}     (both regimes, KL annealing)
         beta* = best VALIDATION anomaly ROC-AUC (normal regime)
Stage 3  Denoising AE, z_dim = z*, noise std 0.2   (normal regime, optional extension)
Stage 4  Supervised MLP control (same trunk as the AE, bottleneck width z*);
         the Workshop 2 deep LSTM itself is loaded frozen by ``w2_encoder.py``

Outputs
-------
    runs/workshop4/<run>/history.csv     per-epoch train/val losses
    runs/workshop4/<run>/summary.json    best epoch and validation metrics
    runs/workshop4/<run>/tb/             TensorBoard logs (if installed)
    runs/workshop4/training_summary.csv  one row per run
    runs/workshop4/selection.json        selected z* and beta*
    checkpoints/workshop4/<run>.pt       every trained model
    checkpoints/ae_best.pt               AE  (z*, all rows)
    checkpoints/vae_best.pt              VAE (z*, beta*, all rows)
    checkpoints/ae_normal_best.pt        AE  (z*, normal rows)  - anomaly detection
    checkpoints/vae_normal_best.pt       VAE (z*, beta*, normal rows)
    checkpoints/mlp_control.pt           supervised MLP control (same trunk as the AE)

Usage
-----
    python src/train_ae.py               # full experiment (~5-10 min on CPU)
    python src/train_ae.py --epochs 3    # quick smoke test
"""

from __future__ import annotations

import argparse
import shutil
import time
from typing import Dict, List

import pandas as pd
from sklearn.metrics import roc_auc_score

import autoencoder as ae
from w4_common import (
    CKPT_DIR,
    CKPT_RUNS_DIR,
    REGIME_ALL,
    REGIME_NORMAL,
    RUNS_DIR,
    TrainConfig,
    W4Data,
    load_w4_data,
    per_sample_mse,
    save_json,
    select_by_val_auc,
    set_seed,
)

Z_GRID = [2, 4, 8]
BETA_GRID = [0.5, 1.0, 4.0]
DAE_NOISE = 0.2


# --------------------------------------------------------------------------- #
# One run
# --------------------------------------------------------------------------- #
def validation_summary(model, cfg: TrainConfig, data: W4Data) -> Dict[str, float]:
    """Validation metrics of a trained model, used for model selection.

    For the unsupervised models the anomaly score is the per-sample
    reconstruction MSE (and, for the VAE, also MSE + KL). The ROC-AUC is
    computed on the FULL validation set (normal + anomalous rows).

    Args:
        model: Trained model.
        cfg (TrainConfig): Its configuration.
        data (W4Data): Dataset.

    Returns:
        Dict[str, float]: Validation metrics.
    """
    out = ae.infer(model, data.val.X)
    y = data.val.y
    if cfg.model == "sup":
        return {"val_auc": float(roc_auc_score(y, out["proba"]))}
    mse = per_sample_mse(data.val.X, out["recon"])
    res = {
        "val_auc": float(roc_auc_score(y, mse)),
        "val_mse_all": float(mse.mean()),
        "val_mse_normal": float(mse[y == 0].mean()),
    }
    if cfg.model == "vae":
        d = data.val.X.shape[1]
        res["val_auc_kl"] = float(roc_auc_score(y, mse * d + out["kl"]))
        res["val_kl"] = float(out["kl"].mean())
    return res


def train_run(cfg: TrainConfig, data: W4Data) -> Dict:
    """Train, log and checkpoint one configuration.

    Args:
        cfg (TrainConfig): Run configuration.
        data (W4Data): Dataset.

    Returns:
        Dict: Run summary (configuration, best epoch, validation metrics).
    """
    run_dir = RUNS_DIR / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = ae.make_tb_writer(run_dir / "tb")

    if cfg.model == "sup":
        X_tr, X_va = data.train.X, data.val.X
        y_tr, y_va = data.train.y, data.val.y
    else:
        X_tr, X_va = data.training_matrix(cfg.regime), data.validation_matrix(cfg.regime)
        y_tr = y_va = None

    def log_fn(epoch: int, rec: Dict[str, float]) -> None:
        if writer is not None:
            for k, v in rec.items():
                if k != "epoch":
                    writer.add_scalar(k, v, epoch)
        if epoch % 10 == 0 or epoch == cfg.epochs - 1:
            shown = {k: round(v, 4) for k, v in rec.items()
                     if k in ("train_loss", "val_loss", "val_recon_mse", "val_kl", "val_pr_auc")}
            print(f"    epoch {epoch:3d} {shown}")

    print(f"\n>>> {cfg.name}  (train rows = {len(X_tr)}, val rows = {len(X_va)})")
    t0 = time.time()
    model, history, best_epoch = ae.fit(cfg, X_tr, X_va, y_tr, y_va, log_fn=log_fn)
    train_time = time.time() - t0
    if writer is not None:
        writer.close()

    summary = {
        "name": cfg.name, "model": cfg.model, "regime": cfg.regime,
        "z_dim": cfg.z_dim, "beta": cfg.beta if cfg.model == "vae" else None,
        "noise_std": cfg.noise_std, "best_epoch": best_epoch,
        "train_rows": int(len(X_tr)), "train_time_s": train_time,
        **{f"best_{k}": v for k, v in history[best_epoch].items() if k != "epoch"},
        **validation_summary(model, cfg, data),
    }
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    save_json(cfg.to_dict(), run_dir / "config.json")
    save_json(summary, run_dir / "summary.json")
    ae.save_checkpoint(model, cfg, data.train.X.shape[1], CKPT_RUNS_DIR / f"{cfg.name}.pt",
                       extra={"best_epoch": best_epoch, "summary": summary,
                              "feature_names": data.feature_names,
                              "scaler_mean": data.scaler_mean.tolist(),
                              "scaler_scale": data.scaler_scale.tolist()})
    print(f"    best epoch {best_epoch} | val AUC {summary['val_auc']:.4f} | {train_time:.1f}s")
    return summary


# --------------------------------------------------------------------------- #
# Experiment plan
# --------------------------------------------------------------------------- #
def run_experiment(epochs: int) -> None:
    """Run the four stages described in the module docstring.

    Args:
        epochs (int): Number of training epochs for every run.

    Returns:
        None
    """
    set_seed()
    data = load_w4_data()
    print("=" * 72)
    print("WORKSHOP 4 - AUTOENCODERS & VAEs ON CARTPOLE (failure = anomaly)")
    print("=" * 72)
    for name in ("train", "val", "test"):
        p = data[name]
        print(f"{name:>5}: {len(p.y):6d} rows | anomalies {int(p.y.sum()):4d} "
              f"({100 * p.y.mean():.2f}%) | features {p.X.shape[1]}")

    summaries: List[Dict] = []

    # Stage 1: AE latent-size sweep
    ae_normal = []
    for z in Z_GRID:
        for regime in (REGIME_ALL, REGIME_NORMAL):
            s = train_run(TrainConfig(model="ae", regime=regime, z_dim=z, epochs=epochs), data)
            summaries.append(s)
            if regime == REGIME_NORMAL:
                ae_normal.append(s)
    z_star = select_by_val_auc(ae_normal)["z_dim"]
    print(f"\n*** selected z* = {z_star} (best validation anomaly ROC-AUC, normal regime)")

    # Stage 2: VAE beta sweep
    vae_normal = []
    for beta in BETA_GRID:
        for regime in (REGIME_ALL, REGIME_NORMAL):
            s = train_run(TrainConfig(model="vae", regime=regime, z_dim=z_star, beta=beta,
                                      epochs=epochs), data)
            summaries.append(s)
            if regime == REGIME_NORMAL:
                vae_normal.append(s)
    beta_star = select_by_val_auc(vae_normal)["beta"]
    print(f"\n*** selected beta* = {beta_star} (best validation anomaly ROC-AUC, normal regime)")

    # Stage 3: denoising AE (optional extension)
    summaries.append(train_run(TrainConfig(model="ae", regime=REGIME_NORMAL, z_dim=z_star,
                                           noise_std=DAE_NOISE, epochs=epochs), data))

    # Stage 4: supervised MLP control (same architecture as the encoder, with labels)
    summaries.append(train_run(TrainConfig(model="sup", regime=REGIME_ALL, z_dim=z_star,
                                           epochs=epochs), data))

    # Best checkpoints with the names required by the workshop
    names = {
        "ae_best": TrainConfig(model="ae", regime=REGIME_ALL, z_dim=z_star).name,
        "vae_best": TrainConfig(model="vae", regime=REGIME_ALL, z_dim=z_star, beta=beta_star).name,
        "ae_normal_best": TrainConfig(model="ae", regime=REGIME_NORMAL, z_dim=z_star).name,
        "vae_normal_best": TrainConfig(model="vae", regime=REGIME_NORMAL, z_dim=z_star,
                                       beta=beta_star).name,
        "dae_normal": TrainConfig(model="ae", regime=REGIME_NORMAL, z_dim=z_star,
                                  noise_std=DAE_NOISE).name,
        "mlp_control": TrainConfig(model="sup", z_dim=z_star).name,
    }
    for alias, run in names.items():
        if alias != "dae_normal":
            shutil.copyfile(CKPT_RUNS_DIR / f"{run}.pt", CKPT_DIR / f"{alias}.pt")

    selection = {"z_star": z_star, "beta_star": beta_star, "z_grid": Z_GRID,
                 "beta_grid": BETA_GRID, "dae_noise": DAE_NOISE, "epochs": epochs,
                 "rule": "max validation anomaly ROC-AUC (normal regime); ties -> simpler model",
                 "runs": names}
    save_json(selection, RUNS_DIR / "selection.json")
    pd.DataFrame(summaries).to_csv(RUNS_DIR / "training_summary.csv", index=False)
    print(f"\nSelection saved in {RUNS_DIR / 'selection.json'}")
    print(f"Best checkpoints copied to {CKPT_DIR}/(ae_best|vae_best|...).pt")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train the Workshop 4 AE / VAE models")
    parser.add_argument("--epochs", type=int, default=50, help="epochs per run (default 50)")
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args().epochs)
