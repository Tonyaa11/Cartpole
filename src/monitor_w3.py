"""
monitor_w3.py - Label-free monitoring of the Workshop 3 controllers with the
Workshop 4 detector.

Machine Learning - Workshop 4 (extension that connects Workshops 1-4)

Motivation
----------
Workshop 3 trained a DQN agent that controls CartPole and measured how often the
supervised Workshop 1 and Workshop 2 models flag its states as "dangerous". Those
detectors needed failure labels. The Workshop 4 VAE was trained only on SAFE
states and never saw a failure label, so it is the kind of monitor one can deploy
on a NEW controller whose failures have never been observed. This script tests
that use case on the Workshop 3 policies.

Protocol (re-uses the Workshop 3 code in ``workshop3/src``)
----------------------------------------------------------
* Policies: the four Workshop 3 baselines that do not query a failure model
  (random, alternating, angle rule, PD rule) and the selected DQN agent
  (``checkpoints/agent_best.pt``).
* Conditions: nominal dynamics and actuator noise (std 2 N, as in Workshop 3),
  on the 100 held-out Workshop 3 test initial states.
* Detectors, applied to every visited state:
    - W4 VAE (normal rows only, z*, beta*): score = squared error + KL, with the
      threshold selected on the Workshop 4 VALIDATION episodes (max F1);
    - W1 Gradient Boosting and W2 LSTM, through the Workshop 3 bridges, each with
      its own validation threshold.
  No threshold or model is tuned on the agents' trajectories.
* Ground truth on the agents' trajectories: a state is "pre-failure" when the
  episode ends in a failure within the next five steps (the dataset definition).
  Episodes that reach 500 steps contain no pre-failure state.
* Metrics: alarm rate per policy (share of flagged steps), ROC-AUC / PR-AUC /
  precision / recall pooled over all policies (how well each detector transfers
  to state distributions produced by controllers it has never seen), the share
  of failed episodes caught (an alarm in the last five steps) and the false-alarm
  rate in episodes that never fail.

Outputs: ``runs/workshop4/w3_monitoring.json`` and
``report/workshop4/figures/fig-w4-w3-monitoring.png``.

Usage
-----
    python src/monitor_w3.py          # requires gymnasium (Workshop 3 environment)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

# Workshop 3 modules first on the path: they include names (baselines, common)
# that also exist in this folder with a different meaning.
_W3_SRC = Path(__file__).resolve().parent.parent / "workshop3" / "src"
if str(_W3_SRC) not in sys.path:
    sys.path.insert(0, str(_W3_SRC))

from agent import DQNAgent  # noqa: E402  (Workshop 3)
from baselines import make_baselines  # noqa: E402  (Workshop 3)
from common import TEST_SEEDS  # noqa: E402  (Workshop 3)
from env import make_env  # noqa: E402  (Workshop 3)
from w1_bridge import W1DangerModel  # noqa: E402  (Workshop 3 -> Workshop 1)
from w2_bridge import W2DangerModel  # noqa: E402  (Workshop 3 -> Workshop 2)

import autoencoder as ae  # noqa: E402  (Workshop 4)
from features import build_features  # noqa: E402
from w4_common import (  # noqa: E402
    CKPT_DIR,
    COLORS,
    RUNS_DIR,
    detection_metrics,
    load_json,
    load_w4_data,
    per_sample_mse,
    save_fig,
    save_json,
    set_seed,
    setup_matplotlib,
)

setup_matplotlib()
import matplotlib.pyplot as plt  # noqa: E402

NOISY_FORCE_STD = 2.0  # Newtons, the Workshop 3 "noisy" test condition
HORIZON = 5            # pre-failure window (same as the dataset label)
DETECTORS = ["W4 VAE (no labels)", "W1 Gradient Boosting", "W2 LSTM"]
DET_COLORS = {DETECTORS[0]: COLORS["vae"], DETECTORS[1]: COLORS["raw"], DETECTORS[2]: COLORS["sup"]}


class _DQNPolicy:
    """Greedy DQN agent with the interface of the Workshop 3 baselines."""

    name = "DQN (W3 agent)"

    def __init__(self, agent: DQNAgent) -> None:
        self.agent = agent

    def reset(self) -> None:
        """Nothing to reset (greedy policy)."""

    def act(self, obs: np.ndarray) -> int:
        return self.agent.act(obs, epsilon=0.0)


def run_episode(policy, env, seed: int) -> Dict[str, Any]:
    """Play one episode (same loop as ``workshop3/src/evaluate.py``).

    Args:
        policy: Policy with ``reset`` and ``act``.
        env: Workshop 3 environment.
        seed (int): Reset seed (initial state).

    Returns:
        Dict[str, Any]: length, outcome, states (T, 4) and actions (T,).
    """
    policy.reset()
    obs, _ = env.reset(seed=seed)
    states, actions = [], []
    done, info = False, {}
    while not done:
        a = policy.act(obs)
        states.append(np.asarray(obs, dtype=np.float64))
        actions.append(a)
        obs, _, term, trunc, info = env.step(a)
        done = term or trunc
    return {"length": len(states), "outcome": info["outcome"],
            "states": np.array(states), "actions": np.asarray(actions, dtype=int)}


class VAEMonitor:
    """Workshop 4 VAE trained on safe states, used as a label-free alarm."""

    def __init__(self) -> None:
        data = load_w4_data()
        self.columns = data.feature_names
        self.mean, self.scale = data.scaler_mean, data.scaler_scale
        self.model, self.cfg, _ = ae.load_checkpoint(CKPT_DIR / "vae_normal_best.pt")
        anom = load_json(RUNS_DIR / "anomaly_results.json")["main"]["VAE (recon + KL)"]
        self.threshold = float(anom["test_f1"]["threshold"])  # chosen on W4 validation

    def scores(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Anomaly score (squared error + KL) of every step of one trajectory."""
        df = pd.DataFrame(states, columns=["x", "x_dot", "theta", "theta_dot"])
        df.insert(0, "t", np.arange(len(df)))
        df.insert(0, "episode", 0)
        df["action"] = actions
        X = ((build_features(df)[self.columns].to_numpy() - self.mean) / self.scale).astype(np.float32)
        out = ae.infer(self.model, X)
        return per_sample_mse(X, out["recon"]) * X.shape[1] + out["kl"]


def pre_failure_labels(length: int, outcome: str) -> np.ndarray:
    """1 for the last ``HORIZON`` states of an episode that ends in a failure."""
    y = np.zeros(length, dtype=int)
    if outcome.startswith("failure"):
        y[max(0, length - HORIZON):] = 1
    return y


def collect(policies: List, detectors: Dict[str, Any]) -> pd.DataFrame:
    """Roll out every policy in both conditions and score every visited state.

    Returns:
        pd.DataFrame: One row per state with policy, condition, episode, label,
        steps to the end of the episode and one score column per detector.
    """
    vae, w1, w2 = detectors[DETECTORS[0]], detectors[DETECTORS[1]], detectors[DETECTORS[2]]
    rows = []
    for cond, noise in (("nominal", 0.0), ("noisy", NOISY_FORCE_STD)):
        env = make_env(force_noise_std=noise)
        for pol in policies:
            t0 = time.time()
            n_fail = 0
            for k, seed in enumerate(TEST_SEEDS):
                e = run_episode(pol, env, seed)
                T = e["length"]
                n_fail += e["outcome"].startswith("failure")
                rows.append(pd.DataFrame({
                    "condition": cond, "policy": pol.name, "episode": k, "t": np.arange(T),
                    "steps_to_end": T - 1 - np.arange(T), "outcome": e["outcome"],
                    "abs_x": np.abs(e["states"][:, 0]), "abs_theta_deg": np.degrees(np.abs(e["states"][:, 2])),
                    "label": pre_failure_labels(T, e["outcome"]),
                    DETECTORS[0]: vae.scores(e["states"], e["actions"]),
                    DETECTORS[1]: w1.trajectory_proba(e["states"], e["actions"]),
                    DETECTORS[2]: w2.trajectory_proba(e["states"], e["actions"]),
                }))
            print(f"  {cond:<8} {pol.name:<26} failures {n_fail:3d}/{len(TEST_SEEDS)} "
                  f"({time.time() - t0:.0f}s)")
    return pd.concat(rows, ignore_index=True)


def summarise(df: pd.DataFrame, thresholds: Dict[str, float]) -> Dict[str, Any]:
    """Alarm rates per policy and pooled detection metrics per detector."""
    out: Dict[str, Any] = {"thresholds": thresholds, "alarm_rate": {}, "detection": {},
                           "episodes": {}}
    test_raw = load_w4_data().test.raw  # reference: states of the Workshop 1 dataset (test)
    out["dataset_reference"] = {"mean_abs_x": float(test_raw["abs_x"].mean()),
                                "mean_abs_theta_deg": float(np.degrees(test_raw["abs_theta"]).mean())}
    for (cond, pol), g in df.groupby(["condition", "policy"], sort=False):
        out["alarm_rate"].setdefault(cond, {})[pol] = {
            det: float(np.mean(g[det] >= thr)) for det, thr in thresholds.items()}
        ep = g.groupby("episode")["outcome"].first()
        out["episodes"].setdefault(cond, {})[pol] = {
            "n": int(len(ep)), "failures": int(ep.str.startswith("failure").sum()),
            "mean_length": float(g.groupby("episode").size().mean()),
            "mean_abs_x": float(g["abs_x"].mean()), "mean_abs_theta_deg": float(g["abs_theta_deg"].mean())}
    for scope, g in [("all", df), ("nominal", df[df.condition == "nominal"]),
                     ("noisy", df[df.condition == "noisy"])]:
        res = {}
        for det, thr in thresholds.items():
            m = detection_metrics(g["label"].to_numpy(), g[det].to_numpy(), thr)
            # Episode level: failed episodes with an alarm in their last HORIZON steps,
            # and alarm rate inside episodes that never fail.
            failed = g[g.outcome.str.startswith("failure")]
            caught = failed[failed.label == 1].groupby(["condition", "policy", "episode"])[det] \
                .apply(lambda s, thr=thr: bool((s >= thr).any()))
            safe_eps = g[~g.outcome.str.startswith("failure")]
            m["failed_episodes_caught"] = float(caught.mean()) if len(caught) else float("nan")
            m["n_failed_episodes"] = int(len(caught))
            m["false_alarm_rate_safe_episodes"] = (float(np.mean(safe_eps[det] >= thr))
                                                   if len(safe_eps) else float("nan"))
            res[det] = m
        out["detection"][scope] = res
        out["detection"][scope]["n_states"] = int(len(g))
        out["detection"][scope]["pre_failure_rate"] = float(g["label"].mean())
    return out


def plot(df: pd.DataFrame, summary: Dict[str, Any]) -> None:
    """(a, b) alarm rate per policy and detector; (c) pooled ROC curves."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.9), gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    for ax, cond, title in [(axes[0], "nominal", "(a) Alarm rate, nominal dynamics"),
                            (axes[1], "noisy", "(b) Alarm rate, actuator noise (2 N)")]:
        rates = summary["alarm_rate"][cond]
        pols = list(rates)
        x = np.arange(len(pols))
        w = 0.26
        for k, det in enumerate(DETECTORS):
            ax.bar(x + (k - 1) * w, [100 * rates[p][det] for p in pols], w, color=DET_COLORS[det],
                   label=det)
        pre = df[df.condition == cond].groupby("policy", sort=False)["label"].mean()
        ax.scatter(x, [100 * pre[p] for p in pols], marker="_", s=180, color="k", zorder=3,
                   label="true pre-failure rate")
        ax.set_xticks(x)
        ax.set_xticklabels([p.replace(" (heuristic)", "").replace(" policy", "").replace(" (W3 agent)", "")
                            for p in pols], rotation=30, ha="right", fontsize=6.5)
        ax.set(title=title, ylabel="% of visited states")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=6.5, ncol=4, loc="upper center",
               bbox_to_anchor=(0.38, 1.06))
    for det in DETECTORS:
        fpr, tpr, _ = roc_curve(df["label"], df[det])
        auc = summary["detection"]["all"][det]["roc_auc"]
        axes[2].plot(fpr, tpr, color=DET_COLORS[det], lw=1.3, label=f"{det} ({auc:.3f})")
    axes[2].plot([0, 1], [0, 1], color="0.7", lw=0.8)
    axes[2].set(title="(c) Pre-failure detection, all policies", xlabel="false positive rate",
                ylabel="true positive rate")
    axes[2].legend(frameon=False, fontsize=5.5, loc="lower right")
    fig.tight_layout()
    save_fig(fig, "fig-w4-w3-monitoring.png")


def main() -> None:
    """Run the monitoring experiment and save results and figure."""
    set_seed()
    print("=" * 72)
    print("WORKSHOP 4 x WORKSHOP 3 - LABEL-FREE MONITORING OF THE RL CONTROLLERS")
    print("=" * 72)
    print("Loading detectors (W4 VAE, W1 Gradient Boosting, W2 LSTM) and policies...")
    detectors = {DETECTORS[0]: VAEMonitor(), DETECTORS[1]: W1DangerModel(), DETECTORS[2]: W2DangerModel()}
    thresholds = {d: float(m.threshold) for d, m in detectors.items()}
    # make_baselines() without failure models -> random, alternating, angle rule, PD rule
    policies = make_baselines() + [_DQNPolicy(DQNAgent.load(CKPT_DIR / "agent_best.pt"))]

    df = collect(policies, detectors)
    summary = summarise(df, thresholds)
    for scope in ("all", "noisy"):
        print(f"\nPre-failure detection on the agents' states ({scope}, "
              f"{summary['detection'][scope]['n_states']} states, "
              f"{100 * summary['detection'][scope]['pre_failure_rate']:.1f}% pre-failure):")
        for det in DETECTORS:
            m = summary["detection"][scope][det]
            print(f"  {det:<22} ROC-AUC {m['roc_auc']:.3f} | PR-AUC {m['pr_auc']:.3f} | "
                  f"P {m['precision']:.3f} R {m['recall']:.3f} | caught {m['failed_episodes_caught']:.0%} "
                  f"| false alarms (safe eps.) {100 * m['false_alarm_rate_safe_episodes']:.2f}%")
    plot(df, summary)
    save_json(summary, RUNS_DIR / "w3_monitoring.json")
    print(f"\nSaved {RUNS_DIR / 'w3_monitoring.json'}")


if __name__ == "__main__":
    main()
