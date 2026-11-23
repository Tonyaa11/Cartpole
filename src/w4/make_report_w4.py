"""
make_report_w4.py - Export the Workshop 4 results to LaTeX.

Machine Learning - Workshop 4

Reads the JSON/CSV files written by ``train_ae.py``, ``evaluate_ae.py`` and
``analyze_latent.py`` and writes, in ``runs/workshop4/generated/``:

    results.tex        one \\newcommand per number quoted in the text
    tab-recon.tex      reconstruction quality (all AE z and VAE beta runs)
    tab-anomaly.tex    cumulative comparison W1 / W2 (deep LSTM) / W4 (anomaly detection)
    tab-sweep.tex      anomaly ROC-AUC per latent size / beta and training regime
    tab-latent.tex     separability of the four representations

``reports/group5_workshop4.tex`` \\input's these files, so after re-running the
experiments the report is updated without editing any number by hand.

Usage
-----
    python src/w4/make_report_w4.py
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from w4.w4_common import GEN_DIR, RUNS_DIR, TrainConfig, load_json


def f3(v: float) -> str:
    """Format with three decimals."""
    return f"{v:.3f}"


def f4(v: float) -> str:
    """Format with four decimals."""
    return f"{v:.4f}"


def pct(v: float) -> str:
    """Format a fraction as a LaTeX percentage."""
    return f"{100 * v:.1f}\\%"


def bold_max(values: List[float], fmt=f3, minimize: bool = False) -> List[str]:
    """Format a column and bold its best value."""
    best = min(values) if minimize else max(values)
    return [f"\\textbf{{{fmt(v)}}}" if abs(v - best) < 1e-12 else fmt(v) for v in values]


def main() -> None:
    """Write every generated LaTeX file."""
    sel = load_json(RUNS_DIR / "selection.json")
    rec = load_json(RUNS_DIR / "reconstruction_results.json")
    anom = load_json(RUNS_DIR / "anomaly_results.json")
    lat = load_json(RUNS_DIR / "latent_results.json")
    train = pd.read_csv(RUNS_DIR / "training_summary.csv")
    cum = pd.read_csv(RUNS_DIR / "cumulative_results.csv")
    GEN_DIR.mkdir(parents=True, exist_ok=True)

    z, b = sel["z_star"], sel["beta_star"]
    ae_run = TrainConfig(model="ae", z_dim=z).name
    vae_run = TrainConfig(model="vae", z_dim=z, beta=b).name
    R = rec["reconstruction"]
    M = anom["main"]
    S = anom["sweep"]
    L = lat["representations"]
    B = lat["beta_study"]
    D = rec["denoising"]
    G = rec["generation"]

    def tf(key: str) -> Dict:
        return M[key]["test_f1"]

    w1 = cum[cum["workshop"] == "W1"].set_index("model")
    macros: Dict[str, str] = {
        "zstar": str(z), "betastar": f"{b:g}",
        "epochs": str(sel["epochs"]), "daenoise": f"{sel['dae_noise']:g}",
        # Reconstruction (all-rows models, test)
        "aeMSE": f4(R[ae_run]["mse"]), "aeRsq": f3(R[ae_run]["r2"]), "aeCorr": f3(R[ae_run]["mean_corr"]),
        "vaeMSE": f4(R[vae_run]["mse"]), "vaeRsq": f3(R[vae_run]["r2"]), "vaeCorr": f3(R[vae_run]["mean_corr"]),
        "aeMSEsafe": f4(R[ae_run]["mse_normal"]), "aeMSEfail": f4(R[ae_run]["mse_anomaly"]),
        "vaeMSEsafe": f4(R[vae_run]["mse_normal"]), "vaeMSEfail": f4(R[vae_run]["mse_anomaly"]),
        "aeTopAnom": pct(R[ae_run]["top1pct"]["anomaly_rate"]),
        "vaeTopAnom": pct(R[vae_run]["top1pct"]["anomaly_rate"]),
        "aeTopXdot": f"{R[ae_run]['top1pct']['mean_abs_x_dot']:.2f}",
        "vaeTopXdot": f"{R[vae_run]['top1pct']['mean_abs_x_dot']:.2f}",
        "allXdot": f"{R['overall']['mean_abs_x_dot']:.2f}",
        "aeTopX": f"{R[ae_run]['top1pct']['mean_abs_x']:.2f}",
        "vaeTopX": f"{R[vae_run]['top1pct']['mean_abs_x']:.2f}",
        "allX": f"{R['overall']['mean_abs_x']:.2f}",
        "aeTopFar": pct(R[ae_run]['top1pct']['frac_abs_x_gt_1_5']),
        "vaeTopFar": pct(R[vae_run]['top1pct']['frac_abs_x_gt_1_5']),
        "allFar": pct(R['overall']['frac_abs_x_gt_1_5']),
        "aeTopThetadot": f"{R[ae_run]['top1pct']['mean_abs_theta_dot']:.2f}",
        "vaeTopThetadot": f"{R[vae_run]['top1pct']['mean_abs_theta_dot']:.2f}",
        "allThetadot": f"{R['overall']['mean_abs_theta_dot']:.2f}",
        "testAnomRate": pct(R["overall"]["anomaly_rate"]),
        "vaeKL": f"{R[vae_run]['kl']:.2f}", "vaeActive": str(R[vae_run]["active_units"]),
        # Anomaly detection (normal-rows models, test, validation threshold)
        "aeAUC": f3(tf("AE")["roc_auc"]), "aePR": f3(tf("AE")["pr_auc"]), "aeFone": f3(tf("AE")["f1"]),
        "aeRec": f3(tf("AE")["recall"]), "aePrec": f3(tf("AE")["precision"]),
        "vaeAUC": f3(tf("VAE (recon)")["roc_auc"]), "vaePR": f3(tf("VAE (recon)")["pr_auc"]),
        "vaeFone": f3(tf("VAE (recon)")["f1"]),
        "vaeklAUC": f3(tf("VAE (recon + KL)")["roc_auc"]), "vaeklPR": f3(tf("VAE (recon + KL)")["pr_auc"]),
        "vaeklFone": f3(tf("VAE (recon + KL)")["f1"]),
        "daeAUC": f3(tf("Denoising AE")["roc_auc"]), "daePR": f3(tf("Denoising AE")["pr_auc"]),
        "supAUC": f3(tf("LSTM (W2)")["roc_auc"]),
        "supPR": f3(tf("LSTM (W2)")["pr_auc"]),
        "supFone": f3(tf("LSTM (W2)")["f1"]),
        "gbAUC": f3(w1.loc["Gradient Boosting", "roc_auc"]), "gbPR": f3(w1.loc["Gradient Boosting", "pr_auc"]),
        "gbFone": f3(w1.loc["Gradient Boosting", "f1"]),
        "lrAUC": f3(w1.loc["Logistic Regression", "roc_auc"]), "lrPR": f3(w1.loc["Logistic Regression", "pr_auc"]),
        "aeQrec": f3(M["AE"]["test_q99"]["recall"]), "aeQprec": f3(M["AE"]["test_q99"]["precision"]),
        "vaeklQrec": f3(M["VAE (recon + KL)"]["test_q99"]["recall"]),
        "vaeklQprec": f3(M["VAE (recon + KL)"]["test_q99"]["precision"]),
        "aeAllAUC": f3(S[TrainConfig(model="ae", regime="all", z_dim=z).name]["roc_auc"]),
        # Latent space
        "silRaw": f3(L["Input space (16-D)"]["silhouette"]["balanced"]),
        "silAE": f3(L["AE"]["silhouette"]["balanced"]), "silVAE": f3(L["VAE"]["silhouette"]["balanced"]),
        "silSup": f3(L["W2 LSTM (supervised)"]["silhouette"]["balanced"]),
        "probeRaw": f3(L["Input space (16-D)"]["probe"]["roc_auc"]),
        "probeAE": f3(L["AE"]["probe"]["roc_auc"]), "probeVAE": f3(L["VAE"]["probe"]["roc_auc"]),
        "probeSup": f3(L["W2 LSTM (supervised)"]["probe"]["roc_auc"]),
        "probePRraw": f3(L["Input space (16-D)"]["probe"]["pr_auc"]),
        "probePRAE": f3(L["AE"]["probe"]["pr_auc"]), "probePRVAE": f3(L["VAE"]["probe"]["pr_auc"]),
        "probePRSup": f3(L["W2 LSTM (supervised)"]["probe"]["pr_auc"]),
        "silMlp": f3(L["MLP control (supervised)"]["silhouette"]["balanced"]),
        "probeMlp": f3(L["MLP control (supervised)"]["probe"]["roc_auc"]),
        "probePRMlp": f3(L["MLP control (supervised)"]["probe"]["pr_auc"]),
        "dimSup": str(L["W2 LSTM (supervised)"]["dim"]),
        # Denoising
        "aeNoisyMSE": f4(D["AE"]["mse_noisy_input"]), "daeNoisyMSE": f4(D["Denoising AE"]["mse_noisy_input"]),
        "aeCleanMSE": f4(D["AE"]["mse_clean_input"]), "daeCleanMSE": f4(D["Denoising AE"]["mse_clean_input"]),
        "noiseVar": f4(D["identity_on_noisy_input"]),
        # Generation
        "genWvae": f3(sum(G[c]["w1_vae_over_std"] for c in ("x", "x_dot", "theta", "theta_dot")) / 4),
        "genWae": f3(sum(G[c]["w1_ae_over_std"] for c in ("x", "x_dot", "theta", "theta_dot")) / 4),
        "genConsVae": f"{G['abs_theta_consistency_mae_rad']['vae']:.3f}",
        "genConsAe": f"{G['abs_theta_consistency_mae_rad']['ae']:.3f}",
        "genBeyond": pct(G["generated_theta_beyond_limit"]["vae"]),
    }
    for bb in sel["beta_grid"]:
        tag = {0.5: "Half", 1.0: "One", 4.0: "Four"}.get(bb, str(bb).replace(".", ""))
        macros[f"active{tag}"] = str(B[f"{bb:g}"]["active_units"])
        macros[f"kl{tag}"] = f"{B[f'{bb:g}']['kl']:.2f}"
        macros[f"mse{tag}"] = f4(B[f"{bb:g}"]["test_mse"])
        macros[f"silbeta{tag}"] = f3(B[f"{bb:g}"]["silhouette"]["balanced"])
    # training rows / times
    macros["trainRowsAll"] = f"{int(train[train['regime'] == 'all']['train_rows'].iloc[0]):,}"
    macros["trainRowsNormal"] = f"{int(train[train['regime'] == 'normal']['train_rows'].iloc[0]):,}"
    for bb in sel["beta_grid"]:
        tag = {0.5: "Half", 1.0: "One", 4.0: "Four"}.get(bb, str(bb).replace(".", ""))
        macros[f"aucbeta{tag}"] = f3(B[f"{bb:g}"]["anomaly_auc_recon"])
    vae_all = TrainConfig(model="vae", regime="all", z_dim=z, beta=b).name + " [mse]"
    macros["vaeAllAUC"] = f3(S[vae_all]["roc_auc"])
    curves = rec["error_analysis"]["median_mse_vs_steps"]
    macros["errZeroNormal"] = f"{curves['AE (normal rows only)'][0]:.3f}"
    macros["errTenNormal"] = f"{curves['AE (normal rows only)'][10]:.3f}"
    macros["errZeroAll"] = f"{curves['AE (all rows)'][0]:.3f}"
    macros["errTenAll"] = f"{curves['AE (all rows)'][10]:.3f}"
    for zz in sel["z_grid"]:
        tag = {2: "Two", 4: "Four", 8: "Eight"}.get(zz, str(zz))
        macros[f"aeRsqz{tag}"] = f3(R[TrainConfig(model="ae", z_dim=zz).name]["r2"])
        macros[f"aeMSEz{tag}"] = f4(R[TrainConfig(model="ae", z_dim=zz).name]["mse"])
        macros[f"aeAUCz{tag}"] = f3(S[TrainConfig(model="ae", regime="normal", z_dim=zz).name]["roc_auc"])

    # ---- Extension: monitoring of the Workshop 3 controllers (optional) ---- #
    mon_path = RUNS_DIR / "w3_monitoring.json"
    if mon_path.exists():
        mon = load_json(mon_path)
        dets = {"Vae": "W4 VAE (no labels)", "Wone": "W1 Gradient Boosting", "Wtwo": "W2 LSTM"}
        det_all = mon["detection"]["all"]
        for tag, det in dets.items():
            m = det_all[det]
            macros[f"mon{tag}AUC"] = f3(m["roc_auc"])
            macros[f"mon{tag}PR"] = f3(m["pr_auc"])
            macros[f"mon{tag}Caught"] = f"{100 * m['failed_episodes_caught']:.0f}\\%"
            macros[f"mon{tag}False"] = f"{100 * m['false_alarm_rate_safe_episodes']:.1f}\\%"
            for pol, ptag in (("DQN (W3 agent)", "Dqn"), ("PD rule (heuristic)", "Pd"), ("Random", "Rand")):
                macros[f"alarm{ptag}{tag}"] = f"{100 * mon['alarm_rate']['nominal'][pol][det]:.1f}\\%"
        ep = mon["episodes"]
        macros["monStates"] = f"{det_all['n_states']:,}"
        macros["monPreRate"] = f"{100 * det_all['pre_failure_rate']:.1f}\\%"
        macros["monFailedEps"] = str(det_all[dets["Vae"]]["n_failed_episodes"])
        macros["dqnFailNominal"] = str(ep["nominal"]["DQN (W3 agent)"]["failures"])
        macros["dqnFailNoisy"] = str(ep["noisy"]["DQN (W3 agent)"]["failures"])
        macros["dqnTheta"] = f"{ep['nominal']['DQN (W3 agent)']['mean_abs_theta_deg']:.1f}"
        macros["pdTheta"] = f"{ep['nominal']['PD rule (heuristic)']['mean_abs_theta_deg']:.1f}"
        macros["dataTheta"] = f"{mon['dataset_reference']['mean_abs_theta_deg']:.1f}"
        rows = []
        names = {"W4 VAE (no labels)": "W4 VAE (no labels)", "W1 Gradient Boosting": "W1 Gradient Boost.",
                 "W2 LSTM": "W2 LSTM"}
        for det in dets.values():
            m = det_all[det]
            rows.append(f"{names[det]} & {f3(m['roc_auc'])} & {f3(m['pr_auc'])} & "
                        f"{100 * m['failed_episodes_caught']:.0f}\\% & "
                        f"{100 * m['false_alarm_rate_safe_episodes']:.1f}\\% \\\\")
        with open(GEN_DIR / "tab-w3.tex", "w") as f:
            f.write("\\begin{tabular}{lcccc}\n\\hline\n"
                    "\\textbf{Detector} & \\textbf{ROC} & \\textbf{PR} & \\textbf{Caught} & "
                    "\\textbf{False al.} \\\\ \\hline\n")
            f.write("\n".join(rows) + "\n\\hline\n\\end{tabular}\n")

    with open(GEN_DIR / "results.tex", "w") as f:
        f.write("% Auto-generated by src/w4/make_report_w4.py - do not edit by hand\n")
        for k, v in macros.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

    # ---- Table: reconstruction ------------------------------------------- #
    runs = [TrainConfig(model="ae", z_dim=zz).name for zz in sel["z_grid"]]
    runs += [TrainConfig(model="vae", z_dim=z, beta=bb).name for bb in sel["beta_grid"]]
    labels = [f"AE, $z={zz}$" for zz in sel["z_grid"]] + \
             [f"VAE, $z={z}$, $\\beta={bb:g}$" for bb in sel["beta_grid"]]
    mse = bold_max([R[r]["mse"] for r in runs], f4, minimize=True)
    r2 = bold_max([R[r]["r2"] for r in runs])
    corr = bold_max([R[r]["mean_corr"] for r in runs])
    lines = []
    for i, (r, lab) in enumerate(zip(runs, labels)):
        sel_mark = "$^\\star$" if r in (ae_run, vae_run) else ""
        kl = f"{R[r]['kl']:.2f}" if "kl" in R[r] else "--"
        au = f"{R[r]['active_units']}/{z}" if "active_units" in R[r] else "--"
        lines.append(f"{lab}{sel_mark} & {mse[i]} & {r2[i]} & {corr[i]} & {f4(R[r]['mse_normal'])} & "
                     f"{f4(R[r]['mse_anomaly'])} & {kl} & {au} \\\\")
        if i == len(sel["z_grid"]) - 1:
            lines.append("\\hline")
    with open(GEN_DIR / "tab-recon.tex", "w") as f:
        f.write("\\begin{tabular}{lccccccc}\n\\hline\n"
                "\\textbf{Model} & \\textbf{MSE} & $\\mathbf{R^2}$ & $\\bar{\\boldsymbol{\\rho}}$ & "
                "\\textbf{MSE$_0$} & \\textbf{MSE$_1$} & \\textbf{KL} & \\textbf{AU} \\\\ \\hline\n")
        f.write("\n".join(lines) + "\n\\hline\n\\end{tabular}\n")

    # ---- Table: cumulative anomaly comparison ----------------------------- #
    order = ["Logistic Regression", "Random Forest", "Gradient Boosting", "LSTM (W2)",
             "AE", "VAE (recon)", "VAE (recon + KL)", "Denoising AE"]
    c = cum.set_index("model").loc[order].reset_index()
    names = {"LSTM (W2)": "Deep LSTM", "VAE (recon)": "VAE: recon.",
             "VAE (recon + KL)": "VAE: recon.\\,+\\,KL", "Denoising AE": "Denoising AE",
             "Logistic Regression": "Logistic Reg.", "Random Forest": "Random Forest",
             "Gradient Boosting": "Gradient Boost.", "AE": "AE"}
    sup_mask = c["supervision"] == "supervised"
    rows = []
    unsup_best = {m: c.loc[~sup_mask, m].max() for m in ("roc_auc", "pr_auc", "precision", "recall", "f1")}
    for i, r in c.iterrows():
        cells = []
        for m in ("roc_auc", "pr_auc", "precision", "recall", "f1"):
            v = f3(r[m])
            if not sup_mask[i] and abs(r[m] - unsup_best[m]) < 1e-12:
                v = f"\\textbf{{{v}}}"
            cells.append(v)
        ws = r["workshop"]
        rows.append(f"{ws} & {names[r['model']]} & " + " & ".join(cells) + " \\\\")
        if i == 3:
            rows.append("\\hline")
    with open(GEN_DIR / "tab-anomaly.tex", "w") as f:
        f.write("\\begin{tabular}{llccccc}\n\\hline\n"
                "& \\textbf{Model} & \\textbf{ROC} & \\textbf{PR} & \\textbf{P} & \\textbf{R} & "
                "\\textbf{F1} \\\\ \\hline\n")
        f.write("\n".join(rows) + "\n\\hline\n\\end{tabular}\n")

    # ---- Table: sweep (regimes) ------------------------------------------ #
    rows = []
    for zz in sel["z_grid"]:
        n = S[TrainConfig(model="ae", regime="normal", z_dim=zz).name]
        a = S[TrainConfig(model="ae", regime="all", z_dim=zz).name]
        mark = "$^\\star$" if zz == z else ""
        rows.append(f"AE, $z={zz}${mark} & {f3(n['val_auc'])} & {f3(n['roc_auc'])} & {f3(a['roc_auc'])} \\\\")
    rows.append("\\hline")
    for bb in sel["beta_grid"]:
        n = S[TrainConfig(model="vae", regime="normal", z_dim=z, beta=bb).name + " [mse]"]
        nk = S[TrainConfig(model="vae", regime="normal", z_dim=z, beta=bb).name + " [mse+kl]"]
        a = S[TrainConfig(model="vae", regime="all", z_dim=z, beta=bb).name + " [mse]"]
        mark = "$^\\star$" if bb == b else ""
        rows.append(f"VAE, $\\beta={bb:g}${mark} & {f3(n['val_auc'])} & {f3(n['roc_auc'])} "
                    f"({f3(nk['roc_auc'])}) & {f3(a['roc_auc'])} \\\\")
    with open(GEN_DIR / "tab-sweep.tex", "w") as f:
        f.write("\\begin{tabular}{lccc}\n\\hline\n"
                "& \\multicolumn{2}{c}{\\textbf{Normal rows only}} & \\textbf{All rows} \\\\\n"
                "\\textbf{Model} & \\textbf{Val} & \\textbf{Test} & \\textbf{Test} \\\\ \\hline\n")
        f.write("\n".join(rows) + "\n\\hline\n\\end{tabular}\n")

    # ---- Table: latent separability --------------------------------------- #
    reps = ["Input space (16-D)", "AE", "VAE", "W2 LSTM (supervised)", "MLP control (supervised)"]
    rnames = {"Input space (16-D)": "Input features", "AE": "AE code", "VAE": "VAE $\\mu$",
              "W2 LSTM (supervised)": "W2 LSTM state", "MLP control (supervised)": "MLP control$^\\ddagger$"}
    rows = []
    for r in reps:
        m = L[r]
        tsne = f3(m["silhouette_tsne_2d"]) if "silhouette_tsne_2d" in m else "--"
        umap = f3(m["silhouette_umap_2d"]) if "silhouette_umap_2d" in m else "--"
        rows.append(f"{rnames[r]} & {m['dim']} & {f3(m['silhouette']['full'])} & "
                    f"{f3(m['silhouette']['balanced'])} & {tsne} & {umap} & {f3(m['probe']['roc_auc'])} & "
                    f"{f3(m['probe']['pr_auc'])} \\\\")
    with open(GEN_DIR / "tab-latent.tex", "w") as f:
        f.write("\\begin{tabular}{lccccccc}\n\\hline\n"
                "& & \\multicolumn{4}{c}{\\textbf{Silhouette}} & \\multicolumn{2}{c}{\\textbf{Linear probe}} \\\\\n"
                "\\textbf{Space} & \\textbf{Dim} & \\textbf{All} & \\textbf{Bal.} & \\textbf{t-SNE} & "
                "\\textbf{UMAP} & \\textbf{ROC} & \\textbf{PR} \\\\ \\hline\n")
        f.write("\n".join(rows) + "\n\\hline\n\\end{tabular}\n")

    print(f"LaTeX files written to {GEN_DIR}")
    for k in ("zstar", "betastar", "aeAUC", "vaeklAUC", "supAUC", "silVAE", "silSup"):
        print(f"  \\{k} = {macros[k]}")


if __name__ == "__main__":
    main()
