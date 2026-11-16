"""
Latent-space analysis for Workshop 4: what structure do the autoencoders find
without labels, and how does it compare with a supervised representation?

Machine Learning - Workshop 4

Representations compared (all computed on the TEST episodes)
------------------------------------------------------------
    * Input space   : the 16 standardised features (reference, no learning)
    * AE            : bottleneck code of ``checkpoints/ae_best.pt``
    * VAE           : posterior mean mu of ``checkpoints/vae_best.pt`` (not a sample)
    * W2 LSTM       : frozen final hidden state (64-D) of the Workshop 2 deep
                      LSTM (``checkpoints/w2_lstm_best.pt``, see ``w2_encoder.py``)
    * MLP control   : bottleneck of a supervised MLP with the SAME trunk and
                      input as the AE (isolates the effect of the labels from the
                      temporal context the LSTM reads); table only

Analyses
--------
1. Silhouette score of the two classes in each representation (full test set
   and a class-balanced subsample, because 94.6% of the rows are "safe").
2. Linear probe: a logistic regression trained on the TRAIN representations and
   evaluated on TEST (ROC-AUC, PR-AUC). It measures how much failure information
   is *linearly* available in each space -> usefulness for downstream tasks.
3. t-SNE and UMAP 2-D maps coloured by the label and by |theta|.
4. Correlation between each latent dimension and the physical variables.
5. Effect of beta on the VAE latent space (active units, separability).

Outputs: ``runs/workshop4/latent_results.json`` and the figures
``fig-w4-latent-tsne.png``, ``fig-w4-latent-umap.png``,
``fig-w4-latent-correlation.png`` and ``fig-w4-beta.png``.

Usage
-----
    python src/analyze_latent.py
    python src/analyze_latent.py --no-umap      # skip UMAP (umap-learn missing)
"""

from __future__ import annotations

import argparse
from typing import Dict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score, roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler

import autoencoder as ae
from w2_encoder import w2_infer
from w4_common import (
    CKPT_DIR,
    CKPT_RUNS_DIR,
    COLORS,
    RUNS_DIR,
    SEED,
    TrainConfig,
    W4Data,
    load_json,
    load_w4_data,
    save_fig,
    save_json,
    set_seed,
    setup_matplotlib,
)

setup_matplotlib()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

W2_NAME = "W2 LSTM (supervised)"
MLP_NAME = "MLP control (supervised)"
REP_ORDER = ["Input space (16-D)", "AE", "VAE", W2_NAME, MLP_NAME]
PLOT_ORDER = REP_ORDER[:4]  # 2-D maps: input, AE, VAE and the Workshop 2 model


# --------------------------------------------------------------------------- #
# Representations
# --------------------------------------------------------------------------- #
def representations(data: W4Data, ckpts: Dict[str, str]) -> Dict[str, Dict[str, np.ndarray]]:
    """Encode the train and test partitions with every model.

    Args:
        data (W4Data): Dataset.
        ckpts (Dict[str, str]): Representation name -> checkpoint path.

    Returns:
        Dict[str, Dict[str, np.ndarray]]: ``{name: {"train": Z, "test": Z}}``.
    """
    reps = {REP_ORDER[0]: {"train": data.train.X, "test": data.test.X}}
    for name, path in ckpts.items():
        model, _, _ = ae.load_checkpoint(path)
        reps[name] = {p: ae.infer(model, data[p].X)["z"] for p in ("train", "test")}
    return reps


# --------------------------------------------------------------------------- #
# Separability metrics
# --------------------------------------------------------------------------- #
def silhouettes(Z: np.ndarray, y: np.ndarray, n_repeats: int = 10) -> Dict[str, float]:
    """Silhouette of the label partition, on all rows and on balanced subsamples.

    The vectors are z-scored per dimension first so that every representation
    is measured on the same scale (an AE code has an arbitrary scale).

    Args:
        Z (np.ndarray): Representation, shape (n, k).
        y (np.ndarray): Labels.
        n_repeats (int): Balanced subsamples to average.

    Returns:
        Dict[str, float]: ``full`` and ``balanced`` (mean, std) silhouettes.
    """
    Zs = StandardScaler().fit_transform(Z)
    rng = np.random.RandomState(SEED)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    bal = []
    for _ in range(n_repeats):
        idx = np.concatenate([pos, rng.choice(neg, len(pos), replace=False)])
        bal.append(silhouette_score(Zs[idx], y[idx]))
    return {"full": float(silhouette_score(Zs, y)),
            "balanced": float(np.mean(bal)), "balanced_std": float(np.std(bal))}


def linear_probe(Ztr: np.ndarray, ytr: np.ndarray, Zte: np.ndarray, yte: np.ndarray) -> Dict[str, float]:
    """Logistic-regression probe trained on TRAIN representations, tested on TEST."""
    sc = StandardScaler().fit(Ztr)
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)
    clf.fit(sc.transform(Ztr), ytr)
    p = clf.predict_proba(sc.transform(Zte))[:, 1]
    return {"roc_auc": float(roc_auc_score(yte, p)), "pr_auc": float(average_precision_score(yte, p))}


# --------------------------------------------------------------------------- #
# 2-D embeddings
# --------------------------------------------------------------------------- #
def embed_2d(Z: np.ndarray, method: str) -> np.ndarray:
    """2-D embedding with t-SNE or UMAP (z-scored input, fixed seed)."""
    Zs = StandardScaler().fit_transform(Z)
    if method == "tsne":
        return TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto",
                    max_iter=1000, random_state=SEED).fit_transform(Zs)
    import umap  # pip install umap-learn

    return umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                     random_state=SEED).fit_transform(Zs)


def plot_embeddings(emb: Dict[str, np.ndarray], data: W4Data, sil: Dict, method: str) -> None:
    """Two rows of scatter plots: coloured by label (top) and by |theta| (bottom)."""
    y = data.test.y
    abs_theta = np.degrees(data.test.raw["abs_theta"].to_numpy())
    title = "t-SNE" if method == "tsne" else "UMAP"
    fig, axes = plt.subplots(2, len(emb), figsize=(2.6 * len(emb), 4.9))
    order = np.argsort(y, kind="stable")  # failures drawn on top
    cmap = ListedColormap([COLORS["normal"], COLORS["anomaly"]])
    for k, (name, E) in enumerate(emb.items()):
        ax = axes[0, k]
        ax.scatter(E[order, 0], E[order, 1], c=y[order], cmap=cmap, s=np.where(y[order] == 1, 4, 1.5),
                   alpha=0.6, linewidths=0)
        ax.set_title(f"{name}\nsilhouette (2-D) = {sil[name]:.2f}", fontsize=7.5)
        ax2 = axes[1, k]
        o2 = np.argsort(abs_theta)
        sc = ax2.scatter(E[o2, 0], E[o2, 1], c=abs_theta[o2], cmap="viridis", s=1.5,
                         vmin=0, vmax=12, linewidths=0)
        for a in (ax, ax2):
            a.set_xticks([])
            a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(True)
                s.set_color("0.8")
    axes[0, 0].set_ylabel("coloured by label", fontsize=7)
    axes[1, 0].set_ylabel(r"coloured by $|\theta|$", fontsize=7)
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS["normal"], label="safe (0)"),
               plt.Line2D([], [], marker="o", ls="", color=COLORS["anomaly"], label="failure (1)")]
    axes[0, -1].legend(handles=handles, frameon=False, fontsize=6, loc="best")
    cb = fig.colorbar(sc, ax=axes[1, :].tolist(), fraction=0.015, pad=0.01)
    cb.set_label(r"$|\theta|$ (deg)", fontsize=7)
    fig.suptitle(f"{title} of the test-set representations", fontsize=9, y=0.995)
    save_fig(fig, f"fig-w4-latent-{method}.png")


# --------------------------------------------------------------------------- #
# Interpretation and beta study
# --------------------------------------------------------------------------- #
def plot_latent_correlation(reps: Dict, data: W4Data) -> Dict:
    """|Pearson r| between latent dimensions and interpretable variables.

    The AE and VAE codes are shown dimension by dimension. The 64-D LSTM state
    is summarised by its first two principal components (PC1, PC2), so the
    three panels have the same size.
    """
    raw = data.test.raw
    variables = {r"$x$": raw["x"], r"$\dot x$": raw["x_dot"], r"$\theta$": raw["theta"],
                 r"$\dot\theta$": raw["theta_dot"], r"$|\theta|$": raw["abs_theta"],
                 r"$|x|$": raw["abs_x"], "action": raw["action"], "label": data.test.y}
    V = np.column_stack([np.asarray(v, dtype=float) for v in variables.values()])
    names = ["AE", "VAE", W2_NAME]
    mats = {}
    for name in names:
        Z = reps[name]["test"]
        if Z.shape[1] > 4:  # summarise wide representations by their top-2 PCs
            Z = PCA(n_components=2, random_state=SEED).fit_transform(StandardScaler().fit_transform(Z))
        mats[name] = Z
    fig, axes = plt.subplots(1, len(names), figsize=(8, 2.6),
                             gridspec_kw={"width_ratios": [mats[n].shape[1] for n in names]})
    out = {}
    for ax, name in zip(axes, names):
        Z = mats[name]
        prefix = "PC" if reps[name]["test"].shape[1] > 4 else "z"
        C = np.zeros((V.shape[1], Z.shape[1]))
        for i in range(V.shape[1]):
            for j in range(Z.shape[1]):
                if Z[:, j].std() > 1e-8:
                    C[i, j] = abs(np.corrcoef(V[:, i], Z[:, j])[0, 1])
        im = ax.imshow(C, cmap="Greys", vmin=0, vmax=1, aspect="auto")
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                        color="w" if C[i, j] > 0.55 else "k")
        ax.set_xticks(range(Z.shape[1]))
        ax.set_xticklabels([f"{prefix}{j + 1}" for j in range(Z.shape[1])])
        ax.set_yticks(range(len(variables)))
        ax.set_yticklabels(list(variables.keys()) if name == names[0] else [])
        ax.set_title(name)
        out[name] = C.tolist()
    fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.01).set_label("|Pearson r|", fontsize=7)
    save_fig(fig, "fig-w4-latent-correlation.png")
    return {"variables": list(variables.keys()), "abs_corr": out}


def beta_study(data: W4Data, sel: Dict) -> Dict:
    """Latent statistics of the three VAEs (all rows) as a function of beta."""
    recon = load_json(RUNS_DIR / "reconstruction_results.json")["reconstruction"]
    anom = load_json(RUNS_DIR / "anomaly_results.json")["sweep"]
    res = {}
    for b in sel["beta_grid"]:
        run = TrainConfig(model="vae", z_dim=sel["z_star"], beta=b).name
        model, _, _ = ae.load_checkpoint(CKPT_RUNS_DIR / f"{run}.pt")
        Ztr, Zte = ae.infer(model, data.train.X)["z"], ae.infer(model, data.test.X)["z"]
        norm_run = TrainConfig(model="vae", regime="normal", z_dim=sel["z_star"], beta=b).name
        res[f"{b:g}"] = {
            "run": run,
            "test_mse": recon[run]["mse"],
            "kl": recon[run]["kl"],
            "active_units": recon[run]["active_units"],
            "kl_per_dim": recon[run]["kl_per_dim"],
            "silhouette": silhouettes(Zte, data.test.y),
            "probe": linear_probe(Ztr, data.train.y, Zte, data.test.y),
            "anomaly_auc_recon": anom[f"{norm_run} [mse]"]["roc_auc"],
            "anomaly_auc_recon_kl": anom[f"{norm_run} [mse+kl]"]["roc_auc"],
        }
        print(f"  beta={b:<4g} MSE {res[f'{b:g}']['test_mse']:.4f} | KL {res[f'{b:g}']['kl']:.2f} | "
              f"active {res[f'{b:g}']['active_units']}/{sel['z_star']} | "
              f"sil(bal) {res[f'{b:g}']['silhouette']['balanced']:.3f} | "
              f"probe AUC {res[f'{b:g}']['probe']['roc_auc']:.3f}")

    betas = sel["beta_grid"]
    x = np.arange(len(betas))
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.5))
    axes[0].bar(x - 0.2, [res[f"{b:g}"]["test_mse"] for b in betas], 0.4, color=COLORS["vae"],
                label="test MSE")
    ax0b = axes[0].twinx()
    ax0b.plot(x, [res[f"{b:g}"]["kl"] for b in betas], "k-o", ms=3, lw=1, label="KL")
    ax0b.set_ylabel("KL (nats, line)")
    axes[0].set(ylabel="test MSE (bars)", title="(a) Rate-distortion trade-off")
    kl_dims = np.array([res[f"{b:g}"]["kl_per_dim"] for b in betas])
    axes[1].imshow(kl_dims.T, cmap="Oranges", aspect="auto")
    for i in range(kl_dims.shape[0]):
        for j in range(kl_dims.shape[1]):
            axes[1].text(i, j, f"{kl_dims[i, j]:.2f}", ha="center", va="center", fontsize=6)
    axes[1].set_yticks(range(kl_dims.shape[1]))
    axes[1].set_yticklabels([f"z{j + 1}" for j in range(kl_dims.shape[1])])
    axes[1].set_title("(b) KL per latent dimension (nats)")
    axes[2].plot(x, [res[f"{b:g}"]["silhouette"]["balanced"] for b in betas], "-o", ms=3,
                 color=COLORS["vae"], label="silhouette (balanced)")
    axes[2].set_ylabel("silhouette (balanced)", color=COLORS["vae"])
    ax2b = axes[2].twinx()
    ax2b.plot(x, [res[f"{b:g}"]["anomaly_auc_recon"] for b in betas], "--s", ms=3, color="k")
    ax2b.set_ylabel("anomaly ROC-AUC (test)")
    axes[2].legend(handles=[plt.Line2D([], [], color=COLORS["vae"], marker="o", ms=3,
                                       label="silhouette (left axis)"),
                            plt.Line2D([], [], color="k", ls="--", marker="s", ms=3,
                                       label="anomaly AUC (right axis)")],
                   frameon=False, fontsize=6, loc="upper left")
    axes[2].set_title(r"(c) Separability vs. $\beta$")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([rf"$\beta$={b:g}" for b in betas])
    fig.tight_layout()
    save_fig(fig, "fig-w4-beta.png")
    return res


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(use_umap: bool) -> None:
    """Run the latent-space analysis."""
    set_seed()
    data = load_w4_data()
    sel = load_json(RUNS_DIR / "selection.json")
    print("=" * 72)
    print("WORKSHOP 4 - LATENT SPACE ANALYSIS")
    print("=" * 72)
    ckpts = {"AE": CKPT_DIR / "ae_best.pt", "VAE": CKPT_DIR / "vae_best.pt",
             MLP_NAME: CKPT_RUNS_DIR / f"{TrainConfig(model='sup', z_dim=sel['z_star']).name}.pt"}
    reps = representations(data, ckpts)
    w2 = w2_infer(data)  # frozen Workshop 2 encoder
    reps[W2_NAME] = {"train": w2["train"]["z"], "test": w2["test"]["z"]}
    y = data.test.y

    print("\n[1-2] Silhouette and linear probe")
    metrics = {}
    for name in REP_ORDER:
        Z = reps[name]
        metrics[name] = {"dim": int(Z["test"].shape[1]), "silhouette": silhouettes(Z["test"], y),
                         "probe": linear_probe(Z["train"], data.train.y, Z["test"], y)}
        m = metrics[name]
        print(f"  {name:<22} dim {m['dim']:2d} | silhouette full {m['silhouette']['full']:.3f} "
              f"balanced {m['silhouette']['balanced']:.3f} | probe ROC-AUC {m['probe']['roc_auc']:.3f} "
              f"PR-AUC {m['probe']['pr_auc']:.3f}")

    embeddings = {}
    for method in (["tsne", "umap"] if use_umap else ["tsne"]):
        print(f"\n[3] {method.upper()} embeddings")
        try:
            emb = {name: embed_2d(reps[name]["test"], method) for name in PLOT_ORDER}
        except ImportError:
            print("  umap-learn is not installed -> UMAP skipped (pip install umap-learn)")
            continue
        sil2d = {name: float(silhouette_score(E, y)) for name, E in emb.items()}
        for name in PLOT_ORDER:
            metrics[name][f"silhouette_{method}_2d"] = sil2d[name]
        plot_embeddings(emb, data, sil2d, method)
        embeddings[method] = True

    print("\n[4] Latent dimension <-> physical variable correlation")
    corr = plot_latent_correlation(reps, data)

    print("\n[5] Effect of beta on the VAE latent space")
    beta = beta_study(data, sel)

    save_json({"representations": metrics, "correlation": corr, "beta_study": beta,
               "embeddings_computed": list(embeddings)}, RUNS_DIR / "latent_results.json")
    print(f"\nResults saved in {RUNS_DIR / 'latent_results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent-space analysis (Workshop 4)")
    parser.add_argument("--no-umap", action="store_true", help="skip the UMAP embeddings")
    main(use_umap=not parser.parse_args().no_umap)
