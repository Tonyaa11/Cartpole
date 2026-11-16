"""
make_report_assets.py - Figures and LaTeX tables of the Workshop 3 report.

Machine Learning - Workshop 3 (Reinforcement Learning)

Everything here is computed from the files written by ``env.py``, ``train.py`` and
``evaluate.py`` (nothing is typed by hand), so the report always matches the code:

    workshop3/report/figures/
        fig-random-policy.png      random-policy diagnostics (100 episodes)
        fig-render.png             frames drawn with env.render() (agent episode)
        fig-training-curves.png    validation return vs env steps, every config x seed
        fig-exploration.png        epsilon schedules + how validation episodes end
        fig-sample-efficiency.png  learning curve of the selected config vs baselines
        fig-test-comparison.png    test return of every policy, nominal and noisy
        fig-trajectories.png       4 test episodes of the agent (states, actions, return)
    workshop3/report/generated/
        macros.tex                 every number quoted in the text
        tab_sweep_rows.tex         rows of the hyperparameter table
        tab_test_rows.tex          rows of the test comparison table
        tab_seed_rows.tex          per-seed test results of the selected configuration

Usage
-----
    python src/make_report_assets_w3.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIG_DIR, RUNS_DIR, TRAIN_SEEDS, W3_DIR, load_json
from env import CartPoleControlEnv
from train import HALF_THRESHOLD, SOLVED_THRESHOLD

GEN_DIR = W3_DIR / "report" / "generated"
W1_DATASET_STEPS = 35_874   # environment steps stored in the Workshop 1 dataset
W1_TRAIN_ROWS = 24_782      # labelled rows used to train the Workshop 1 models

# Categorical slots (validated reference palette, light mode) + neutral ink
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SEED_COLORS = {42: BLUE, 7: ORANGE, 99: AQUA}
INK, INK2, GRID, GRAY = "#0b0b0b", "#52514e", "#e4e3df", "#9a9993"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": GRAY,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4,
    "savefig.dpi": 220, "figure.dpi": 110,
})


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def save(fig: plt.Figure, name: str) -> None:
    """Save a figure in ``report/figures`` and close it.

    Args:
        fig (plt.Figure): Figure.
        name (str): File name.

    Returns:
        None
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved figures/{name}")


def tex_escape(text: str) -> str:
    """Escape the characters that LaTeX treats specially in plain text.

    Args:
        text (str): Raw text.

    Returns:
        str: LaTeX-safe text.
    """
    return text.replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def fmt_int(v: float) -> str:
    """Format an integer with thousands separators (LaTeX-safe).

    Args:
        v (float): Value.

    Returns:
        str: e.g. "35{,}874".
    """
    return f"{int(round(v)):,}".replace(",", "{,}")


def load_val_curves() -> Dict[str, Dict[int, pd.DataFrame]]:
    """Read every ``validation.csv`` of the sweep.

    Returns:
        Dict[str, Dict[int, pd.DataFrame]]: config -> seed -> validation log.
    """
    sweep_cfg = load_json(W3_DIR / "configs" / "sweep.json")
    curves: Dict[str, Dict[int, pd.DataFrame]] = {}
    for c in sweep_cfg["configs"]:
        for s in TRAIN_SEEDS:
            p = RUNS_DIR / c["name"] / f"seed_{s}" / "validation.csv"
            if p.exists():
                curves.setdefault(c["name"], {})[s] = pd.read_csv(p)
    return curves


def mean_curve(runs: Dict[int, pd.DataFrame], col: str, grid: np.ndarray) -> np.ndarray:
    """Interpolate each seed's curve on a common step grid.

    Args:
        runs (Dict[int, pd.DataFrame]): seed -> validation log.
        col (str): Column to interpolate.
        grid (np.ndarray): Common environment-step grid.

    Returns:
        np.ndarray: Array (n_seeds, len(grid)); NaN beyond a run's last step.
    """
    out = []
    for df in runs.values():
        y = np.interp(grid, df["env_steps"], df[col])
        y[grid > df["env_steps"].max()] = np.nan
        out.append(y)
    return np.array(out)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_random_policy(diag: Dict[str, Any]) -> None:
    """Random-policy diagnostics: episode lengths and action frequency."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.2), gridspec_kw={"width_ratios": [2.2, 1]})
    L = np.array(diag["lengths"])
    axes[0].hist(L, bins=range(0, L.max() + 6, 4), color=BLUE, edgecolor="white", linewidth=1.0)
    axes[0].axvline(L.mean(), color=INK, linestyle="--", linewidth=1)
    axes[0].text(L.mean() + 1, axes[0].get_ylim()[1] * 0.9, f"mean {L.mean():.1f}", color=INK, fontsize=7)
    axes[0].set(xlabel="episode length (steps)", ylabel="episodes",
                title=f"(a) Random policy: episode length ({diag['n_episodes']} episodes)")
    acts = np.array(diag["action_counts"])
    bars = axes[1].bar(["left (0)", "right (1)"], acts, color=BLUE, width=0.55)
    for b, v in zip(bars, acts):
        axes[1].text(b.get_x() + b.get_width() / 2, v, f"{v / acts.sum():.1%}", ha="center",
                     va="bottom", fontsize=7, color=INK)
    axes[1].set(title="(b) Action frequency", ylabel="actions")
    axes[1].set_ylim(0, acts.max() * 1.18)
    fig.tight_layout()
    save(fig, "fig-random-policy.png")


def fig_render(traj: Dict[str, Any]) -> None:
    """Frames drawn with ``env.render()`` along the first test episode."""
    ep = traj["episodes"][0]
    states = np.array(ep["states"])
    idx = np.linspace(0, len(states) - 1, 5).astype(int)
    env = CartPoleControlEnv(render_mode="rgb_array")
    env.reset(seed=0)
    fig, axes = plt.subplots(1, len(idx), figsize=(7.0, 1.15))
    for ax, i in zip(axes, idx):
        env.state = states[i]
        ax.imshow(env.render()[60:, :])
        ax.set_title(f"t = {i}, theta = {np.degrees(states[i, 2]):+.1f} deg", fontsize=6.5)
        ax.axis("off")
    fig.tight_layout(pad=0.2)
    save(fig, "fig-render.png")


def fig_training_curves(curves: Dict[str, Dict[int, pd.DataFrame]]) -> None:
    """Small multiples: validation return vs env steps for every config and seed."""
    names = list(curves)
    ncol = 3
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0, 2.0 * nrow), sharey=True)
    axes = np.atleast_2d(axes)
    xmax = max(df["env_steps"].max() for runs in curves.values() for df in runs.values())
    for k, name in enumerate(names):
        ax = axes[k // ncol, k % ncol]
        for s, df in curves[name].items():
            ax.plot(df["env_steps"] / 1000, df["val_return"], color=SEED_COLORS[s], label=f"seed {s}")
        ax.axhline(SOLVED_THRESHOLD, color=GRAY, linestyle="--", linewidth=0.8)
        ax.set_title(name)
        ax.set_xlim(0, xmax / 1000 * 1.02)
        ax.set_ylim(0, 520)
        if k % ncol == 0:
            ax.set_ylabel("validation return")
        if k // ncol == nrow - 1:
            ax.set_xlabel("environment steps (thousands)")
    for k in range(len(names), nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "fig-training-curves.png")


def fig_exploration(curves: Dict[str, Dict[int, pd.DataFrame]], best_cfg: str) -> None:
    """Epsilon schedules and how the validation episodes end during training."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
    sweep_cfg = load_json(W3_DIR / "configs" / "sweep.json")
    decays = sorted({float({**sweep_cfg["defaults"], **c}["epsilon_decay"]) for c in sweep_cfg["configs"]})
    eps_min = sweep_cfg["defaults"]["epsilon_min"]
    episodes = np.arange(0, sweep_cfg["defaults"]["episodes"] + 1)
    for d, col in zip(decays, [BLUE, ORANGE]):
        eps = np.maximum(eps_min, sweep_cfg["defaults"]["epsilon_start"] * d ** episodes)
        axes[0].plot(episodes, eps, color=col)
        axes[0].text(episodes[-1], eps[-1] + 0.04, f"decay {d}", color=INK, fontsize=7, ha="right")
    axes[0].set(xlabel="training episode", ylabel="epsilon", title="(a) Exploration schedules",
                ylim=(0, 1.05))

    runs = curves[best_cfg]
    grid = np.linspace(0, max(df["env_steps"].max() for df in runs.values()), 60)
    for col_name, label, col in (("val_x_failure_rate", "ends by cart position (|x| > 2.4)", ORANGE),
                                 ("val_theta_failure_rate", "ends by pole angle (|theta| > 12 deg)", BLUE)):
        m = np.nanmean(mean_curve(runs, col_name, grid), axis=0)
        axes[1].plot(grid / 1000, m, color=col, label=label)
    axes[1].set(xlabel="environment steps (thousands)", ylabel="share of validation episodes",
                title=f"(b) How validation episodes end ({best_cfg})", ylim=(-0.03, 1.03))
    axes[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=1, fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig-exploration.png")


def fig_sample_efficiency(curves, best_cfg: str, test: Dict[str, Any]) -> None:
    """Learning curve of the selected configuration against the baselines."""
    runs = curves[best_cfg]
    grid = np.linspace(0, max(df["env_steps"].max() for df in runs.values()), 120)
    Y = mean_curve(runs, "val_smoothed", grid)
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.fill_between(grid / 1000, np.nanmin(Y, 0), np.nanmax(Y, 0), color=BLUE, alpha=0.15, linewidth=0)
    ax.plot(grid / 1000, np.nanmean(Y, 0), color=BLUE, label=f"DQN {best_cfg} (mean, min-max of 3 seeds)")
    pol = {p["policy"]: p for p in test["conditions"]["nominal"]["policies"]}
    for name, col in (("PD rule (heuristic)", ORANGE), ("W1 classifier (look-ahead)", AQUA), ("Random", GRAY)):
        if name in pol:
            ax.axhline(pol[name]["mean_return"], color=col, linestyle="--", linewidth=1.1,
                       label=f"{name}: {pol[name]['mean_return']:.0f} (test, no training)")
    ax.axvline(W1_DATASET_STEPS / 1000, color=INK2, linestyle=":", linewidth=1)
    ax.text(W1_DATASET_STEPS / 1000, 440, "size of the W1 dataset (35,874 steps) ", color=INK2,
            fontsize=6.5, va="top", ha="right")
    ax.set(xlabel="environment steps (thousands)", ylabel="smoothed validation return", ylim=(0, 520),
           xlim=(0, max(grid.max(), W1_DATASET_STEPS) / 1000 * 1.03))
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig-sample-efficiency.png")


def fig_test_comparison(test: Dict[str, Any]) -> None:
    """Mean test return (+- std) of every policy under both test conditions."""
    nom = test["conditions"]["nominal"]["policies"]
    noisy = {p["policy"]: p for p in test["conditions"]["noisy"]["policies"]}
    names = [p["policy"] for p in nom]
    y = np.arange(len(names))
    h = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    for off, data, col, lab in ((-h / 2, {p["policy"]: p for p in nom}, BLUE, "nominal test"),
                                (h / 2, noisy, ORANGE, f"noisy test (force noise {test['noisy_force_std']} N)")):
        means = [data[n]["mean_return"] for n in names]
        stds = [data[n]["std_return"] for n in names]
        ax.barh(y + off, means, height=h - 0.04, color=col, xerr=stds, label=lab,
                error_kw={"elinewidth": 0.8, "ecolor": INK2, "capsize": 2})
        for yy, m, sd in zip(y + off, means, stds):
            ax.text(m + sd + 8, yy, f"{m:.0f}", va="center", fontsize=6.5, color=INK)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set(xlabel="mean test return (steps survived, max 500), error bars = 1 std", xlim=(0, 720))
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    save(fig, "fig-test-comparison.png")


def fig_trajectories(traj: Dict[str, Any]) -> None:
    """Pole angle, cart position, actions and return for example test episodes."""
    eps = traj["episodes"]
    fig, axes = plt.subplots(4, len(eps), figsize=(7.0, 4.6), sharex="col",
                             gridspec_kw={"height_ratios": [1, 1, 0.35, 0.8]})
    for j, ep in enumerate(eps):
        S = np.array(ep["states"])
        t = np.arange(len(S))
        a = np.array(ep["actions"])
        axes[0, j].plot(t, np.degrees(S[:, 2]), color=BLUE)
        axes[0, j].axhline(0, color=GRAY, linewidth=0.6)
        axes[0, j].set_ylim(-12.5, 12.5)
        axes[0, j].set_title(f"test seed {ep['seed']}\n{ep['length']} steps ({ep['outcome'].replace('_', ' ')})",
                             fontsize=6.8)
        axes[1, j].plot(t, S[:, 0], color=ORANGE)
        axes[1, j].axhline(0, color=GRAY, linewidth=0.6)
        axes[1, j].set_ylim(-2.5, 2.5)
        axes[2, j].imshow(a[None, :], aspect="auto", cmap=matplotlib.colors.ListedColormap([GRAY, BLUE]),
                          extent=(0, len(a), 0, 1), interpolation="nearest")
        axes[2, j].set_yticks([])
        axes[2, j].grid(False)
        axes[3, j].plot(t, np.cumsum(ep["rewards"]), color=AQUA)
        axes[3, j].set_xlabel("time step")
        if j == 0:
            axes[0, j].set_ylabel("theta (deg)")
            axes[1, j].set_ylabel("x (m)")
            axes[2, j].set_ylabel("action", rotation=0, ha="right", va="center")
            axes[3, j].set_ylabel("return")
    fig.text(0.99, 0.005, "action strip: blue = push right, gray = push left", ha="right",
             fontsize=6.5, color=INK2)
    fig.tight_layout(h_pad=0.4, w_pad=0.6)
    save(fig, "fig-trajectories.png")


# --------------------------------------------------------------------------- #
# LaTeX snippets
# --------------------------------------------------------------------------- #
def pm(m: float, s: float, d: int = 1) -> str:
    """Format mean +- std for LaTeX.

    Args:
        m (float): Mean.
        s (float): Standard deviation.
        d (int): Decimals.

    Returns:
        str: e.g. "$495.2 \\pm 3.1$".
    """
    return f"${m:.{d}f} \\pm {s:.{d}f}$"


def write_tex(diag, sweep, test, curves) -> None:
    """Write macros.tex, tab_sweep_rows.tex and tab_test_rows.tex."""
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    sweep_cfg = load_json(W3_DIR / "configs" / "sweep.json")
    defaults = sweep_cfg["defaults"]
    best = sweep["best_config"]
    macros: Dict[str, str] = {}

    # ---- environment diagnostics ---------------------------------------- #
    L = np.array(diag["lengths"])
    acts = np.array(diag["action_counts"])
    macros.update({
        "RandMeanReturn": f"{np.mean(diag['task_returns']):.1f}",
        "RandStdReturn": f"{np.std(diag['task_returns']):.1f}",
        "RandShapedReturn": f"{np.mean(diag['shaped_returns']):.1f}",
        "RandMinLen": str(L.min()), "RandMedianLen": f"{np.median(L):.1f}", "RandMaxLen": str(L.max()),
        "RandLeftPct": f"{100 * acts[0] / acts.sum():.1f}", "RandRightPct": f"{100 * acts[1] / acts.sum():.1f}",
        "RandThetaFailPct": f"{100 * diag['outcomes'].get('failure_theta', 0) / diag['n_episodes']:.0f}",
        "ParityDiff": f"{diag['parity_max_abs_diff']:.1e}" if diag["parity_max_abs_diff"] > 0 else "0",
    })

    # ---- sweep table ----------------------------------------------------- #
    table = {t["config"]: t for t in sweep["table"]}
    rows = []
    for c in sweep_cfg["configs"]:
        full = {**defaults, **c}
        t = table.get(c["name"])
        if t is None:
            continue
        steps = "--" if t["steps_to_475_mean"] is None else fmt_int(t["steps_to_475_mean"])
        name = tex_escape(c["name"])
        name = f"\\textbf{{{name}}}" if c["name"] == best else name
        rows.append(
            f"{name} & {full['learning_rate']:g} & {full['gamma']:g} & {full['epsilon_decay']:g} & "
            f"{full['weight_decay']:g} & {full['shaping_scale']:g} & {pm(t['best_val_mean'], t['best_val_std'])} & "
            f"{pm(t['final_val_mean'], t['final_val_std'])} & {t['runs_reaching_475']}/3 & {steps} \\\\ \\hline")
    (GEN_DIR / "tab_sweep_rows.tex").write_text("\n".join(rows) + "\n")

    br = sweep["best_run"]
    macros.update({
        "BestConfig": tex_escape(best), "BestSeed": str(br["seed"]),
        "BestEpisode": str(br["best_episode"]), "BestValReturn": f"{br['best_val_return']:.1f}",
        "BestRunSteps": fmt_int(br["total_env_steps"]),
        "NParams": fmt_int(br["n_parameters"]),
        "BestCfgBestVal": f"{table[best]['best_val_mean']:.1f}",
        "BestCfgFinalVal": f"{table[best]['final_val_mean']:.1f}",
        "TrainEpisodes": str(defaults["episodes"]),
        "MeanRunTime": f"{np.mean([t['train_time_mean_s'] for t in sweep['table']]):.0f}",
    })
    finals = [t["final_val_mean"] for t in sweep["table"]]
    macros.update({"MinFinalVal": f"{min(finals):.1f}", "MaxFinalVal": f"{max(finals):.1f}"})

    # Sample efficiency of the selected configuration
    sel_runs = [r for r in sweep["runs"] if r["config"] == best]
    s195 = [r["steps_to_195"] for r in sel_runs if r["steps_to_195"] is not None]
    s475 = [r["steps_to_475"] for r in sel_runs if r["steps_to_475"] is not None]
    macros.update({
        "StepsHalfMean": fmt_int(np.mean(s195)) if s195 else "--",
        "StepsSolvedMean": fmt_int(np.mean(s475)) if s475 else "--",
        "RunsSolved": f"{len(s475)}/{len(sel_runs)}",
        "WOneDatasetSteps": fmt_int(W1_DATASET_STEPS), "WOneTrainRows": fmt_int(W1_TRAIN_ROWS),
    })
    # mean env steps at which the best checkpoint of each seed was saved
    ck_steps = []
    for r in sel_runs:
        df = curves[best][r["seed"]]
        ck_steps.append(int(df.loc[df["episode"] == r["best_episode"], "env_steps"].iloc[0]))
    macros["BestCheckpointSteps"] = fmt_int(np.mean(ck_steps))

    # ---- test table ------------------------------------------------------ #
    nom = test["conditions"]["nominal"]
    noisy = {p["policy"]: p for p in test["conditions"]["noisy"]["policies"]}
    rows = []
    for p in nom["policies"]:
        q = noisy[p["policy"]]
        name = tex_escape(p["policy"])
        name = f"\\textbf{{{name}}}" if p["policy"].startswith("DQN") else name
        rows.append(
            f"{name} & {pm(p['mean_return'], p['std_return'])} & {100 * p['success_rate']:.0f} & "
            f"{100 * p['x_failure_rate']:.0f} & {100 * p['w1_danger_rate']:.2f} & {p['mean_abs_theta_deg']:.2f} & "
            f"{pm(q['mean_return'], q['std_return'])} & {100 * q['success_rate']:.0f} \\\\ \\hline")
    seeds_n = test["conditions"]["nominal"]
    seeds_z = test["conditions"]["noisy"]
    rows.append(
        f"DQN, 3 seeds of {tex_escape(best)} & {pm(seeds_n['dqn_seed_mean_return'], seeds_n['dqn_seed_std_return'])} & "
        f"{100 * seeds_n['dqn_seed_mean_success']:.0f} & -- & -- & -- & "
        f"{pm(seeds_z['dqn_seed_mean_return'], seeds_z['dqn_seed_std_return'])} & "
        f"{100 * seeds_z['dqn_seed_mean_success']:.0f} \\\\ \\hline")
    (GEN_DIR / "tab_test_rows.tex").write_text("\n".join(rows) + "\n")

    # Per-seed results of the selected configuration (robustness across training seeds)
    z_seeds = {r["seed"]: r for r in seeds_z["seeds"]}
    agent_seed = test["agent_best_info"].get("seed")
    rows = []
    for r in seeds_n["seeds"]:
        q = z_seeds[r["seed"]]
        mark = " (agent\\_best)" if r["seed"] == agent_seed else ""
        rows.append(
            f"{r['seed']}{mark} & {r['mean_return']:.1f} & {100 * r['success_rate']:.0f} & "
            f"{100 * r['w1_danger_rate']:.2f} & {r['mean_abs_theta_deg']:.2f} & "
            f"{q['mean_return']:.1f} & {100 * q['success_rate']:.0f} \\\\ \\hline")
    (GEN_DIR / "tab_seed_rows.tex").write_text("\n".join(rows) + "\n")

    pol = {p["policy"]: p for p in nom["policies"]}
    key = {"Random": "Rand", "Alternating (fixed)": "Alt", "Angle rule (greedy)": "Angle",
           "PD rule (heuristic)": "PD", "W1 classifier (look-ahead)": "WOne", "DQN (agent_best.pt)": "DQN"}
    for name, k in key.items():
        if name in pol:
            p, q = pol[name], noisy[name]
            macros.update({
                f"Test{k}Mean": f"{p['mean_return']:.1f}", f"Test{k}Std": f"{p['std_return']:.1f}",
                f"Test{k}Success": f"{100 * p['success_rate']:.0f}",
                f"Test{k}Danger": f"{100 * p['w1_danger_rate']:.2f}",
                f"Test{k}Theta": f"{p['mean_abs_theta_deg']:.2f}", f"Test{k}AbsX": f"{p['mean_abs_x']:.2f}",
                f"Test{k}XFail": f"{100 * p['x_failure_rate']:.0f}",
                f"Noisy{k}Mean": f"{q['mean_return']:.1f}", f"Noisy{k}Success": f"{100 * q['success_rate']:.0f}",
                f"Noisy{k}XFail": f"{100 * q['x_failure_rate']:.0f}",
                f"Noisy{k}ThetaFail": f"{100 * q['theta_failure_rate']:.0f}",
                f"Noisy{k}Danger": f"{100 * q['w1_danger_rate']:.2f}",
            })
    macros.update({
        "SeedsTestMean": f"{seeds_n['dqn_seed_mean_return']:.1f}", "SeedsTestStd": f"{seeds_n['dqn_seed_std_return']:.1f}",
        "SeedsNoisyMean": f"{seeds_z['dqn_seed_mean_return']:.1f}", "SeedsNoisyStd": f"{seeds_z['dqn_seed_std_return']:.1f}",
        "SeedsTestSuccess": f"{100 * seeds_n['dqn_seed_mean_success']:.0f}",
        "SeedsNoisySuccess": f"{100 * seeds_z['dqn_seed_mean_success']:.0f}",
        "NoiseStd": f"{test['noisy_force_std']:g}", "WOneThreshold": f"{test['w1_threshold']:.3f}",
        "LR": f"{defaults['learning_rate']:g}", "DiscountGamma": f"{defaults['gamma']:g}",
        "EpsDecay": f"{defaults['epsilon_decay']:g}", "EpsMin": f"{defaults['epsilon_min']:g}",
        "Buffer": fmt_int(defaults["buffer_size"]), "Batch": str(defaults["batch_size"]),
        "LearnStarts": fmt_int(defaults["learning_starts"]), "TargetUpdate": str(defaults["target_update"]),
        "GradClip": f"{defaults['grad_clip']:g}", "ShapingK": f"{defaults['shaping_scale']:g}",
        "EvalEvery": str(defaults["eval_every"]),
    })

    # Workshop 1 results (read from the W1 result files of the repository)
    w1_file = RUNS_DIR.parent / "workshop1" / "baseline_results_base.csv"
    if w1_file.exists():
        w1 = pd.read_csv(w1_file)
        gb = w1[(w1["model"] == "Gradient Boosting") & (w1["setting"] == "test @ val-threshold")].iloc[0]
        macros.update({"WOneFOne": f"{gb['f1']:.3f}", "WOneRecall": f"{gb['recall']:.3f}",
                       "WOnePrecision": f"{gb['precision']:.3f}", "WOnePRAUC": f"{gb['pr_auc']:.3f}"})

    lines = ["% Generated by src/make_report_assets_w3.py -- do not edit by hand"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    (GEN_DIR / "macros.tex").write_text("\n".join(lines) + "\n")
    print(f"  wrote generated/macros.tex ({len(macros)} values), tab_sweep_rows.tex, "
          "tab_test_rows.tex, tab_seed_rows.tex")


def main() -> None:
    """Build every figure and LaTeX snippet."""
    diag = load_json(RUNS_DIR / "env_validation.json")
    sweep = load_json(RUNS_DIR / "sweep_results.json")
    test = load_json(RUNS_DIR / "test_results.json")
    traj = load_json(RUNS_DIR / "trajectories.json")
    curves = load_val_curves()
    print("=" * 60)
    print("REPORT ASSETS ->", FIG_DIR.parent)
    print("=" * 60)
    fig_random_policy(diag)
    fig_render(traj)
    fig_training_curves(curves)
    fig_exploration(curves, sweep["best_config"])
    fig_sample_efficiency(curves, sweep["best_config"], test)
    fig_test_comparison(test)
    fig_trajectories(traj)
    write_tex(diag, sweep, test, curves)


if __name__ == "__main__":
    main()
