"""
Evaluation of the Workshop 4 models: reconstruction quality, anomaly detection,
denoising and generative properties.

Machine Learning - Workshop 4

Run AFTER ``train_ae.py``. Everything is computed on the TEST episodes with the
checkpoints in ``checkpoints/``; thresholds are taken from VALIDATION.

Sections
--------
1. Training curves              -> fig-w4-training-curves.png
2. Reconstruction quality       -> runs/workshop4/reconstruction_results.json
   (MSE, R^2, mean Pearson correlation; every AE z and every VAE beta)
3. Sample reconstructions       -> fig-w4-reconstructions.png
   (8 test rows: original | AE | VAE | |error| maps)
4. Error analysis               -> fig-w4-error-analysis.png
   (which rows / features are hardest to reconstruct)
5. Anomaly detection            -> runs/workshop4/anomaly_results.json,
                                   runs/workshop4/cumulative_results.csv,
                                   fig-w4-anomaly.png
   (AE / VAE / DAE trained on normal rows; AUC-ROC compared with the
   Workshop 1 baselines and the Workshop 2 deep LSTM)
6. Denoising (optional)         -> part of reconstruction_results.json
7. Generative properties (VAE)  -> fig-w4-vae-samples.png

Usage
-----
    python src/w4/evaluate_ae.py
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from w4 import autoencoder as ae
from w1.features import THETA_LIMIT
from w4.w2_encoder import w2_infer
from w4.w4_common import (
    CKPT_DIR,
    CKPT_RUNS_DIR,
    COLORS,
    RUNS_DIR,
    TrainConfig,
    W4Data,
    evaluate_scores,
    load_json,
    load_w1_baselines,
    load_w4_data,
    per_sample_mse,
    reconstruction_metrics,
    save_fig,
    save_json,
    set_seed,
    setup_matplotlib,
)

setup_matplotlib()
import matplotlib.pyplot as plt  # noqa: E402

# Short labels of the 16 engineered features (same order as features.py)
FEATURE_LABELS = [
    r"$x$", r"$\dot{x}$", r"$\theta$", r"$\dot{\theta}$", "act",
    r"$|x|$", r"$|\theta|$", r"$m_x$", r"$m_\theta$",
    r"$\Delta x$", r"$\Delta\theta$", r"$\Delta\dot{\theta}$",
    r"$\overline{|\theta|}_3$", r"$\overline{|\dot\theta|}_3$",
    r"ttl$_\theta$", r"ttl$_x$",
]
STATE_IDX = {"x": 0, "x_dot": 1, "theta": 2, "theta_dot": 3}


def load_model(run: str):
    """Load a run checkpoint from ``checkpoints/workshop4``."""
    model, cfg, _ = ae.load_checkpoint(CKPT_RUNS_DIR / f"{run}.pt")
    return model, cfg


# --------------------------------------------------------------------------- #
# 1. Training curves
# --------------------------------------------------------------------------- #
def plot_training_curves(sel: Dict) -> None:
    """Train/validation reconstruction loss per epoch (AE sweep and VAE betas)."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.7))
    shades = plt.cm.Blues(np.linspace(0.45, 0.95, len(sel["z_grid"])))
    for c, z in zip(shades, sel["z_grid"]):
        h = pd.read_csv(RUNS_DIR / TrainConfig(model="ae", z_dim=z).name / "history.csv")
        axes[0].plot(h["epoch"] + 1, h["train_recon_mse"], ls="--", color=c, lw=1)
        axes[0].plot(h["epoch"] + 1, h["val_recon_mse"], color=c, lw=1.4, label=f"z = {z}")
    axes[0].set(title="(a) AE: reconstruction MSE", xlabel="epoch", ylabel="MSE (std. units)",
                yscale="log")
    axes[0].legend(title="solid: validation, dashed: train", title_fontsize=6, frameon=False,
                   loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)

    shades = plt.cm.Oranges(np.linspace(0.45, 0.95, len(sel["beta_grid"])))
    for c, b in zip(shades, sel["beta_grid"]):
        h = pd.read_csv(RUNS_DIR / TrainConfig(model="vae", z_dim=sel["z_star"], beta=b).name
                        / "history.csv")
        axes[1].plot(h["epoch"] + 1, h["train_recon_mse"], ls="--", color=c, lw=1)
        axes[1].plot(h["epoch"] + 1, h["val_recon_mse"], color=c, lw=1.4, label=rf"$\beta$ = {b:g}")
        axes[2].plot(h["epoch"] + 1, h["val_kl"], color=c, lw=1.4, label=rf"$\beta$ = {b:g}")
    for ax in axes[1:]:
        ax.axvspan(1, 10, color="0.92", zorder=0)
        ax.text(5.5, 0.97, "KL warm-up", transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=6, color="0.4")
    axes[1].set(title=f"(b) VAE (z = {sel['z_star']}): reconstruction MSE", xlabel="epoch",
                ylabel="MSE (std. units)", yscale="log")
    axes[2].set(title="(c) VAE: KL divergence (validation)", xlabel="epoch",
                ylabel="KL (nats / sample)")
    for ax in axes[1:]:
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)
    fig.tight_layout()
    save_fig(fig, "fig-w4-training-curves.png")


# --------------------------------------------------------------------------- #
# 2-4. Reconstruction quality, samples and error analysis
# --------------------------------------------------------------------------- #
def reconstruction_section(data: W4Data, sel: Dict) -> Dict:
    """Reconstruction metrics on the test set for every all-rows model."""
    X, y = data.test.X, data.test.y
    results: Dict[str, Dict] = {}
    runs = [TrainConfig(model="ae", z_dim=z).name for z in sel["z_grid"]]
    runs += [TrainConfig(model="vae", z_dim=sel["z_star"], beta=b).name for b in sel["beta_grid"]]
    for run in runs:
        model, _ = load_model(run)
        out = ae.infer(model, X)
        m = reconstruction_metrics(X, out["recon"])
        err = per_sample_mse(X, out["recon"])
        m["mse_normal"] = float(err[y == 0].mean())
        m["mse_anomaly"] = float(err[y == 1].mean())
        if "kl" in out:
            m["kl"] = float(out["kl"].mean())
            var_mu = out["z"].var(axis=0)
            m["active_units"] = int((var_mu > 0.01).sum())  # Burda et al. (2016)
            m["kl_per_dim"] = [float(v) for v in
                               (-0.5 * (1 + out["logvar"] - out["z"] ** 2
                                        - np.exp(out["logvar"]))).mean(axis=0)]
        # Hardest rows: the 1% with the largest error
        top = err >= np.quantile(err, 0.99)
        m["top1pct"] = state_profile(data, top)
        results[run] = m
        print(f"  {run:<22} test MSE {m['mse']:.4f} | R2 {m['r2']:.3f} | "
              f"corr {m['mean_corr']:.3f} | normal {m['mse_normal']:.4f} | "
              f"anomaly {m['mse_anomaly']:.4f}")
    results["overall"] = state_profile(data, np.ones_like(y, dtype=bool))
    return results


def state_profile(data: W4Data, mask: np.ndarray) -> Dict[str, float]:
    """Describe a subset of test states in physical units (used for the hardest 1%)."""
    raw, y = data.test.raw, data.test.y
    return {
        "anomaly_rate": float(y[mask].mean()),
        "mean_abs_x": float(raw["abs_x"][mask].mean()),
        "mean_abs_x_dot": float(raw["x_dot"].abs()[mask].mean()),
        "mean_abs_theta_deg": float(np.degrees(raw["abs_theta"][mask].mean())),
        "mean_abs_theta_dot": float(raw["theta_dot"].abs()[mask].mean()),
        "frac_abs_x_gt_1_5": float((raw["abs_x"][mask] > 1.5).mean()),
    }


def plot_reconstruction_panel(data: W4Data, ae_out: Dict, vae_out: Dict) -> Dict:
    """Side-by-side heatmaps of 8 test rows: original, AE, VAE and |error| maps."""
    rng = np.random.RandomState(42)
    y = data.test.y
    idx = np.concatenate([rng.choice(np.where(y == 0)[0], 4, replace=False),
                          rng.choice(np.where(y == 1)[0], 4, replace=False)])
    X = data.test.X[idx]
    A, V = ae_out["recon"][idx], vae_out["recon"][idx]
    panels = [("Original", X), ("AE reconstruction", A), ("VAE reconstruction", V),
              (r"$|x-\hat{x}_{AE}|$", np.abs(X - A)), (r"$|x-\hat{x}_{VAE}|$", np.abs(X - V))]
    vlim = 3.0
    emax = max(np.abs(X - A).max(), np.abs(X - V).max())
    fig, axes = plt.subplots(1, 5, figsize=(13, 2.6), sharey=True)
    row_labels = [f"#{i} ({'fail' if y[i] else 'safe'})" for i in idx]
    for k, (ax, (title, M)) in enumerate(zip(axes, panels)):
        is_err = k >= 3
        im = ax.imshow(M, aspect="auto", cmap="magma" if is_err else "RdBu_r",
                       vmin=0 if is_err else -vlim, vmax=emax if is_err else vlim)
        ax.set_title(title)
        ax.set_xticks(range(len(FEATURE_LABELS)))
        ax.set_xticklabels(FEATURE_LABELS, rotation=90, fontsize=6)
        ax.axhline(3.5, color="w" if not is_err else "0.7", lw=1)
        if k == 0:
            ax.set_yticks(range(len(idx)))
            ax.set_yticklabels(row_labels, fontsize=6)
        if k in (2, 4):
            cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
            cb.ax.tick_params(labelsize=6)
            cb.set_label("|error| (std. units)" if is_err else "value (std. units)", fontsize=6)
    fig.tight_layout()
    save_fig(fig, "fig-w4-reconstructions.png")
    return {"rows": [int(i) for i in idx],
            "ae_mse": [float(v) for v in per_sample_mse(X, A)],
            "vae_mse": [float(v) for v in per_sample_mse(X, V)],
            "labels": [int(v) for v in y[idx]]}


def plot_error_analysis(data: W4Data, ae_all: Dict, vae_all: Dict, ae_norm: Dict) -> Dict:
    """Per-feature error, error vs. distance to failure, error by class."""
    X, y = data.test.X, data.test.y
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.7), gridspec_kw={"width_ratios": [1.6, 1, 1]})

    # (a) per-feature MSE
    pa = ((X - ae_all["recon"]) ** 2).mean(axis=0)
    pv = ((X - vae_all["recon"]) ** 2).mean(axis=0)
    pos = np.arange(len(FEATURE_LABELS))
    axes[0].bar(pos - 0.2, pa, 0.4, color=COLORS["ae"], label="AE")
    axes[0].bar(pos + 0.2, pv, 0.4, color=COLORS["vae"], label="VAE")
    axes[0].set_xticks(pos)
    axes[0].set_xticklabels(FEATURE_LABELS, rotation=90, fontsize=6)
    axes[0].set(title="(a) Test MSE per feature (all-rows models)", ylabel="MSE (std. units)",
                yscale="log")
    axes[0].legend(frameon=False)

    # (b) error vs. steps to the end of FAILED episodes
    failed_eps = np.unique(data.test.episode[y == 1])
    in_failed = np.isin(data.test.episode, failed_eps)
    steps = np.arange(0, 21)
    stats = {}
    for name, out, color in [("AE (all rows)", ae_all, COLORS["ae"]),
                             ("AE (normal rows only)", ae_norm, COLORS["dae"])]:
        err = per_sample_mse(X, out["recon"])
        curve = [np.median(err[in_failed & (data.test.steps_to_end == s)]) for s in steps]
        axes[1].plot(steps, curve, marker="o", ms=2.5, lw=1.3, color=color, label=name)
        stats[name] = [float(c) for c in curve]
    axes[1].axvspan(-0.5, 4.5, color=COLORS["anomaly"], alpha=0.12, lw=0)
    axes[1].text(2, 0.96, "label = 1", transform=axes[1].get_xaxis_transform(), ha="center",
                 va="top", fontsize=6, color=COLORS["anomaly"])
    axes[1].invert_xaxis()
    axes[1].set(yscale="log", xlabel="steps before the failure", ylabel="median MSE",
                title="(b) Error as failure approaches")
    axes[1].legend(frameon=False, fontsize=6)

    # (c) distribution of the error by class
    for k, out in enumerate([ae_all, vae_all]):
        err = per_sample_mse(X, out["recon"])
        bp = axes[2].boxplot([err[y == 0], err[y == 1]], positions=[k * 3, k * 3 + 1],
                             widths=0.8, patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], [COLORS["normal"], COLORS["anomaly"]]):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        for med in bp["medians"]:
            med.set_color("k")
    axes[2].set_xticks([0.5, 3.5])
    axes[2].set_xticklabels(["AE", "VAE"])
    axes[2].set(yscale="log", ylabel="per-row MSE", title="(c) Error by class (all-rows models)")
    axes[2].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COLORS["normal"], alpha=0.7),
                            plt.Rectangle((0, 0), 1, 1, color=COLORS["anomaly"], alpha=0.7)],
                   labels=["safe (0)", "failure (1)"], frameon=False, fontsize=6)
    fig.tight_layout()
    save_fig(fig, "fig-w4-error-analysis.png")
    return {"steps": steps.tolist(), "median_mse_vs_steps": stats,
            "per_feature_mse_ae": pa.tolist(), "per_feature_mse_vae": pv.tolist()}


# --------------------------------------------------------------------------- #
# 5. Anomaly detection
# --------------------------------------------------------------------------- #
def anomaly_section(data: W4Data, sel: Dict) -> Dict:
    """Anomaly scores of every model, thresholds from validation, metrics on test."""
    yv, yt = data.val.y, data.test.y
    d = data.test.X.shape[1]
    scores: Dict[str, Dict[str, np.ndarray]] = {}

    def add(label: str, run: str, kind: str = "mse") -> None:
        model, _ = load_model(run)
        ov, ot = ae.infer(model, data.val.X), ae.infer(model, data.test.X)
        if kind == "mse":
            sv, st = per_sample_mse(data.val.X, ov["recon"]), per_sample_mse(data.test.X, ot["recon"])
        elif kind == "mse+kl":
            sv = per_sample_mse(data.val.X, ov["recon"]) * d + ov["kl"]
            st = per_sample_mse(data.test.X, ot["recon"]) * d + ot["kl"]
        else:  # supervised probability
            sv, st = ov["proba"], ot["proba"]
        scores[label] = {"val": sv, "test": st, "run": run, "kind": kind}

    z, b = sel["z_star"], sel["beta_star"]
    add("AE", TrainConfig(model="ae", regime="normal", z_dim=z).name)
    add("VAE (recon)", TrainConfig(model="vae", regime="normal", z_dim=z, beta=b).name)
    add("VAE (recon + KL)", TrainConfig(model="vae", regime="normal", z_dim=z, beta=b).name, "mse+kl")
    add("Denoising AE", TrainConfig(model="ae", regime="normal", z_dim=z,
                                    noise_std=sel["dae_noise"]).name)
    # Workshop 2 model (frozen deep LSTM): supervised baseline of the cumulative table
    w2 = w2_infer(data)
    scores["LSTM (W2)"] = {"val": w2["val"]["proba"], "test": w2["test"]["proba"],
                           "run": "w2_lstm_best", "kind": "proba"}

    results = {k: {**evaluate_scores(yv, s["val"], yt, s["test"]), "run": s["run"], "score": s["kind"]}
               for k, s in scores.items()}

    # Sweeps and ablations (threshold-free metrics only)
    sweep = {}
    for zz in sel["z_grid"]:
        for regime in ("normal", "all"):
            run = TrainConfig(model="ae", regime=regime, z_dim=zz).name
            model, _ = load_model(run)
            st = per_sample_mse(data.test.X, ae.infer(model, data.test.X)["recon"])
            sv = per_sample_mse(data.val.X, ae.infer(model, data.val.X)["recon"])
            sweep[run] = {**evaluate_scores(yv, sv, yt, st)["test_f1"],
                          "val_auc": float(roc_auc_score(yv, sv))}
    for bb in sel["beta_grid"]:
        for regime in ("normal", "all"):
            run = TrainConfig(model="vae", regime=regime, z_dim=z, beta=bb).name
            model, _ = load_model(run)
            ov, ot = ae.infer(model, data.val.X), ae.infer(model, data.test.X)
            for kind in ("mse", "mse+kl"):
                sv = per_sample_mse(data.val.X, ov["recon"]) * (d if kind == "mse+kl" else 1)
                st = per_sample_mse(data.test.X, ot["recon"]) * (d if kind == "mse+kl" else 1)
                if kind == "mse+kl":
                    sv, st = sv + ov["kl"], st + ot["kl"]
                sweep[f"{run} [{kind}]"] = {**evaluate_scores(yv, sv, yt, st)["test_f1"],
                                            "val_auc": float(roc_auc_score(yv, sv))}

    for k, r in results.items():
        t = r["test_f1"]
        print(f"  {k:<26} ROC-AUC {t['roc_auc']:.4f} | PR-AUC {t['pr_auc']:.4f} | "
              f"P {t['precision']:.3f} R {t['recall']:.3f} F1 {t['f1']:.3f}")

    plot_anomaly(scores, yt)
    return {"main": results, "sweep": sweep}


def plot_anomaly(scores: Dict, yt: np.ndarray) -> None:
    """ROC and PR curves on test, plus the AE score distribution by class."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    styles = {"AE": (COLORS["ae"], "-"), "VAE (recon)": (COLORS["vae"], "-"),
              "VAE (recon + KL)": (COLORS["vae"], "--"), "Denoising AE": (COLORS["dae"], "-"),
              "LSTM (W2)": (COLORS["sup"], ":")}
    for k, s in scores.items():
        c, ls = styles[k]
        fpr, tpr, _ = roc_curve(yt, s["test"])
        prec, rec, _ = precision_recall_curve(yt, s["test"])
        axes[0].plot(fpr, tpr, color=c, ls=ls, lw=1.3, label=k)
        axes[1].plot(rec, prec, color=c, ls=ls, lw=1.3, label=k)
    axes[0].plot([0, 1], [0, 1], color="0.7", lw=0.8)
    axes[0].set(title="(a) ROC curve (test)", xlabel="false positive rate", ylabel="true positive rate")
    axes[1].axhline(yt.mean(), color="0.7", lw=0.8)
    axes[1].set(title="(b) Precision-recall curve (test)", xlabel="recall", ylabel="precision")
    axes[0].legend(frameon=False, fontsize=6, loc="lower right")

    s = scores["AE"]["test"]
    bins = np.logspace(np.log10(max(s.min(), 1e-5)), np.log10(s.max()), 50)
    axes[2].hist(s[yt == 0], bins=bins, color=COLORS["normal"], alpha=0.6, density=True, label="safe (0)")
    axes[2].hist(s[yt == 1], bins=bins, color=COLORS["anomaly"], alpha=0.6, density=True, label="failure (1)")
    axes[2].set(xscale="log", title="(c) AE anomaly score (test)", xlabel="reconstruction MSE",
                ylabel="density")
    axes[2].legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, "fig-w4-anomaly.png")


def cumulative_table(anom: Dict) -> pd.DataFrame:
    """Cumulative results table: Workshop 1 baselines + W2 reference + Workshop 4."""
    rows = []
    for _, r in load_w1_baselines().iterrows():
        rows.append({"workshop": "W1", "model": r["model"], "supervision": "supervised",
                     "roc_auc": r["roc_auc"], "pr_auc": r["pr_auc"], "precision": r["precision"],
                     "recall": r["recall"], "f1": r["f1"]})
    for k, r in anom["main"].items():
        t = r["test_f1"]
        is_sup = r["score"] == "proba"
        rows.append({"workshop": "W2" if is_sup else "W4", "model": k,
                     "supervision": "supervised" if is_sup else "unsupervised (normal rows)",
                     "roc_auc": t["roc_auc"], "pr_auc": t["pr_auc"], "precision": t["precision"],
                     "recall": t["recall"], "f1": t["f1"]})
    df = pd.DataFrame(rows)
    df.to_csv(RUNS_DIR / "cumulative_results.csv", index=False)
    return df


# --------------------------------------------------------------------------- #
# 6. Denoising
# --------------------------------------------------------------------------- #
def denoising_section(data: W4Data, sel: Dict) -> Dict:
    """Compare AE and denoising AE (both normal rows) on clean and corrupted inputs."""
    X = data.test.X
    out = {}
    for label, run in [("AE", TrainConfig(model="ae", regime="normal", z_dim=sel["z_star"]).name),
                       ("Denoising AE", TrainConfig(model="ae", regime="normal", z_dim=sel["z_star"],
                                                    noise_std=sel["dae_noise"]).name)]:
        model, _ = load_model(run)
        clean = ae.infer(model, X)["recon"]
        noisy = ae.infer(model, X, noise_std=sel["dae_noise"], seed=7)["recon"]
        out[label] = {"mse_clean_input": float(((X - clean) ** 2).mean()),
                      "mse_noisy_input": float(((X - noisy) ** 2).mean())}
    out["identity_on_noisy_input"] = float(sel["dae_noise"] ** 2)  # E[noise^2] if x_hat = x_noisy
    return out


# --------------------------------------------------------------------------- #
# 7. Generative properties
# --------------------------------------------------------------------------- #
def generative_section(data: W4Data, sel: Dict) -> Dict:
    """Sample z ~ N(0, I) from the VAE (and, as a contrast, the AE) and compare
    the generated state variables with the real test distribution."""
    n = 5000
    vae_model, _ = load_model(TrainConfig(model="vae", z_dim=sel["z_star"], beta=sel["beta_star"]).name)
    ae_model, _ = load_model(TrainConfig(model="ae", z_dim=sel["z_star"]).name)
    gen_vae = data.inverse_scale(ae.generate(vae_model, n, seed=0))
    z = np.random.RandomState(0).normal(size=(n, sel["z_star"])).astype(np.float32)
    gen_ae = data.inverse_scale(ae.decode(ae_model, z))
    real = data.test.raw.to_numpy()

    res = {}
    fig, axes = plt.subplots(1, 5, figsize=(12, 2.4))
    names = [("x", r"cart position $x$ (m)"), ("x_dot", r"cart velocity $\dot{x}$ (m/s)"),
             ("theta", r"pole angle $\theta$ (rad)"), ("theta_dot", r"angular velocity $\dot\theta$ (rad/s)")]
    for ax, (col, label) in zip(axes, names):
        j = STATE_IDX[col]
        lo, hi = np.quantile(real[:, j], [0.005, 0.995])
        span = hi - lo
        bins = np.linspace(lo - 0.25 * span, hi + 0.25 * span, 50)
        ax.hist(real[:, j], bins=bins, density=True, color="0.6", alpha=0.6, label="real (test)")
        ax.hist(np.clip(gen_vae[:, j], bins[0], bins[-1]), bins=bins, density=True, histtype="step",
                lw=1.4, color=COLORS["vae"], label=r"VAE, $z\sim N(0,I)$")
        ax.hist(np.clip(gen_ae[:, j], bins[0], bins[-1]), bins=bins, density=True, histtype="step",
                lw=1.0, ls="--", color=COLORS["ae"], label=r"AE, $z\sim N(0,I)$")
        ax.set(xlabel=label, yticks=[])
        sd = real[:, j].std()
        res[col] = {"w1_vae_over_std": float(wasserstein_distance(real[:, j], gen_vae[:, j]) / sd),
                    "w1_ae_over_std": float(wasserstein_distance(real[:, j], gen_ae[:, j]) / sd)}
    axes[0].legend(frameon=False, fontsize=6)
    # Physical consistency: the generated |theta| feature should equal |generated theta|
    j_t, j_abs = STATE_IDX["theta"], 6
    ax = axes[4]
    ax.scatter(np.abs(gen_vae[:2000, j_t]), gen_vae[:2000, j_abs], s=2, alpha=0.4, color=COLORS["vae"],
               label="VAE samples")
    lim = max(np.abs(real[:, j_t]).max(), 0.25)
    ax.plot([0, lim], [0, lim], color="k", lw=0.8, label="consistent")
    ax.axvline(THETA_LIMIT, color=COLORS["anomaly"], lw=0.8, ls=":")
    ax.set(xlabel=r"$|\theta|$ of the generated angle", ylabel=r"generated $|\theta|$ feature",
           xlim=(0, lim), ylim=(-0.02, lim))
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    save_fig(fig, "fig-w4-vae-samples.png")

    def consistency(G: np.ndarray) -> float:
        return float(np.mean(np.abs(np.abs(G[:, j_t]) - G[:, j_abs])))

    res["abs_theta_consistency_mae_rad"] = {"vae": consistency(gen_vae), "ae": consistency(gen_ae),
                                            "real": consistency(real)}
    res["generated_theta_beyond_limit"] = {
        "vae": float(np.mean(np.abs(gen_vae[:, j_t]) > THETA_LIMIT)),
        "real": float(np.mean(np.abs(real[:, j_t]) > THETA_LIMIT))}
    return res


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    """Run every evaluation section and save the results."""
    set_seed()
    data = load_w4_data()
    sel = load_json(RUNS_DIR / "selection.json")
    print("=" * 72)
    print(f"WORKSHOP 4 - EVALUATION (z* = {sel['z_star']}, beta* = {sel['beta_star']})")
    print("=" * 72)

    print("\n[1] Training curves")
    plot_training_curves(sel)

    print("\n[2] Reconstruction quality (test, models trained on all rows)")
    recon = reconstruction_section(data, sel)

    ae_all_m, _ = ae.load_checkpoint(CKPT_DIR / "ae_best.pt")[:2]
    vae_all_m, _ = ae.load_checkpoint(CKPT_DIR / "vae_best.pt")[:2]
    ae_norm_m, _ = ae.load_checkpoint(CKPT_DIR / "ae_normal_best.pt")[:2]
    ae_all, vae_all = ae.infer(ae_all_m, data.test.X), ae.infer(vae_all_m, data.test.X)
    ae_norm = ae.infer(ae_norm_m, data.test.X)

    print("\n[3] Sample reconstructions")
    panel = plot_reconstruction_panel(data, ae_all, vae_all)
    print("\n[4] Error analysis")
    err_analysis = plot_error_analysis(data, ae_all, vae_all, ae_norm)

    print("\n[5] Anomaly detection (models trained on normal rows only)")
    anom = anomaly_section(data, sel)
    cum = cumulative_table(anom)
    print("\n" + cum.round(3).to_string(index=False))

    print("\n[6] Denoising")
    den = denoising_section(data, sel)
    print("  ", den)

    print("\n[7] Generative properties of the VAE")
    gen = generative_section(data, sel)
    print("  ", {k: v for k, v in gen.items() if k.startswith("abs") or k.startswith("gen")})

    save_json({"selection": sel, "reconstruction": recon, "sample_panel": panel,
               "error_analysis": err_analysis, "denoising": den, "generation": gen},
              RUNS_DIR / "reconstruction_results.json")
    save_json(anom, RUNS_DIR / "anomaly_results.json")
    print(f"\nResults saved in {RUNS_DIR}")


if __name__ == "__main__":
    main()
