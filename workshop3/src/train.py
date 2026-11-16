"""
train.py - Training loop, validation and hyperparameter sweep for the DQN agent.

Machine Learning - Workshop 3 (Reinforcement Learning)

One *run* = one configuration trained with one seed:
    * every episode: act epsilon-greedily, store transitions, one gradient step
      per environment step, decay epsilon at the end of the episode;
    * every ``eval_every`` episodes: greedy evaluation (epsilon = 0) on the 10
      fixed VALIDATION initial states; the checkpoint with the highest validation
      return is kept (``checkpoints/workshop3_sweep/<config>_s<seed>.pt``).

The *sweep* trains every configuration in ``configs/sweep.json`` with the three
seeds (42, 7, 99), selects the best configuration on validation only, and copies
its best run to ``checkpoints/agent_best.pt``. The test set is never touched here.

Usage
-----
    python workshop3/src/train.py --sweep                       # full sweep (5 configs x 3 seeds)
    python workshop3/src/train.py --config reference --seed 42  # a single run
    python workshop3/src/train.py --sweep --episodes 50         # quick smoke test
    python workshop3/src/train.py --select-only                 # redo the selection from the logs
"""

from __future__ import annotations

import argparse
import csv
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from agent import DQNAgent, DQNConfig
from common import (CKPT_DIR, CONFIG_FILE, ROOT_DIR, RUNS_DIR, SWEEP_CKPT_DIR, TRAIN_SEED_LOW,
                    TRAIN_SEEDS, VAL_SEEDS, load_json, save_json, set_global_seed)
from env import make_env

# Return thresholds used in the sample-efficiency analysis
SOLVED_THRESHOLD = 475.0   # Gymnasium's official reward threshold for CartPole-v1
HALF_THRESHOLD = 195.0     # historical CartPole-v0 threshold, an "early competence" level

try:  # TensorBoard is optional: CSV logs are always written
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None


def evaluate_policy(agent: DQNAgent, seeds: List[int]) -> Dict[str, Any]:
    """Greedy evaluation (epsilon = 0) on a fixed list of initial states.

    The return reported is the TASK return (+1 per step survived), so every
    configuration is compared on the same quantity whatever its shaping.

    Args:
        agent (DQNAgent): Agent to evaluate.
        seeds (List[int]): Reset seeds, one episode per seed.

    Returns:
        Dict[str, Any]: mean and std of the task return, the list of returns and
        the share of episodes that ended by cart position or by pole angle.
    """
    env = make_env()  # evaluation env: task reward only, no noise
    returns, outcomes = [], []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        done, ret = False, 0.0
        while not done:
            obs, _, term, trunc, info = env.step(agent.act(obs, epsilon=0.0))
            ret += info["task_reward"]
            done = term or trunc
        returns.append(ret)
        outcomes.append(info["outcome"])
    return {"mean": float(np.mean(returns)), "std": float(np.std(returns)), "returns": returns,
            "x_failure_rate": float(np.mean([o == "failure_x" for o in outcomes])),
            "theta_failure_rate": float(np.mean([o == "failure_theta" for o in outcomes]))}


def first_step_reaching(steps: List[int], values: List[float], threshold: float) -> Optional[int]:
    """Environment steps needed until a curve first reaches a threshold.

    Args:
        steps (List[int]): Cumulative environment steps at each evaluation.
        values (List[float]): Curve values (smoothed validation return).
        threshold (float): Target value.

    Returns:
        Optional[int]: First step count with value >= threshold, or None.
    """
    for s, v in zip(steps, values):
        if v >= threshold:
            return int(s)
    return None


def train_run(cfg: DQNConfig, seed: int, verbose: bool = True) -> Dict[str, Any]:
    """Train one configuration with one seed and keep its best checkpoint.

    Args:
        cfg (DQNConfig): Hyperparameters.
        seed (int): Seed (network init, exploration, replay sampling, episode starts).
        verbose (bool): Print a progress line at every validation.

    Returns:
        Dict[str, Any]: Summary of the run (best validation return, final smoothed
        validation return, steps to thresholds, wall-clock time...).
    """
    set_global_seed(seed)
    run_name = f"{cfg.name}_s{seed}"
    log_dir = RUNS_DIR / cfg.name / f"seed_{seed}"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = SWEEP_CKPT_DIR / f"{run_name}.pt"

    env = make_env(shaping_scale=cfg.shaping_scale, gamma=cfg.gamma)
    agent = DQNAgent(cfg, seed=seed)
    episode_seeds = np.random.RandomState(seed).randint(TRAIN_SEED_LOW, 2**31 - 1, size=cfg.episodes)

    writer = SummaryWriter(log_dir=str(log_dir)) if SummaryWriter is not None else None
    ep_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    best_val, best_episode = -np.inf, 0
    total_steps, t0 = 0, time.time()

    for ep in range(1, cfg.episodes + 1):
        obs, _ = env.reset(seed=int(episode_seeds[ep - 1]))
        done, ret_shaped, ret_task, losses = False, 0.0, 0.0, []
        while not done:
            action = agent.act(obs)                                   # epsilon-greedy
            next_obs, reward, term, trunc, info = env.step(action)
            agent.buffer.add(obs, action, reward, next_obs, term)     # only failures are terminal
            obs = next_obs
            ret_shaped += reward
            ret_task += info["task_reward"]
            total_steps += 1
            done = term or trunc
            if total_steps % cfg.train_freq == 0:
                loss = agent.update()
                if loss is not None:
                    losses.append(loss)

        row = {"episode": ep, "env_steps": total_steps, "length": env.steps,
               "task_return": ret_task, "shaped_return": round(ret_shaped, 4),
               "epsilon": round(agent.epsilon, 5), "outcome": info["outcome"],
               "mean_loss": float(np.mean(losses)) if losses else float("nan")}
        ep_rows.append(row)
        if writer:
            writer.add_scalar("train/task_return", ret_task, total_steps)
            writer.add_scalar("train/shaped_return", ret_shaped, total_steps)
            writer.add_scalar("train/episode_length", env.steps, total_steps)
            writer.add_scalar("train/epsilon", agent.epsilon, total_steps)
            if losses:
                writer.add_scalar("train/td_loss", row["mean_loss"], total_steps)
        agent.decay_epsilon()  # per-episode exploration decay

        # ---- periodic validation without exploration ----------------------- #
        if ep % cfg.eval_every == 0 or ep == cfg.episodes:
            val = evaluate_policy(agent, VAL_SEEDS)
            recent = [r["val_return"] for r in val_rows[-2:]] + [val["mean"]]
            smoothed = float(np.mean(recent))   # moving average of the last 3 evaluations
            val_rows.append({"episode": ep, "env_steps": total_steps, "val_return": val["mean"],
                             "val_std": val["std"], "val_smoothed": smoothed,
                             "val_x_failure_rate": val["x_failure_rate"],
                             "val_theta_failure_rate": val["theta_failure_rate"],
                             "epsilon": round(agent.epsilon, 5)})
            if writer:
                writer.add_scalar("val/return", val["mean"], total_steps)
                writer.add_scalar("val/return_smoothed", smoothed, total_steps)
            # ">=" keeps the most trained checkpoint among equally good ones
            if val["mean"] >= best_val:
                best_val, best_episode = val["mean"], ep
                agent.save(ckpt_path, extra={"episode": ep, "env_steps": total_steps,
                                             "val_return": val["mean"], "seed": seed})
            if verbose:
                print(f"  [{run_name}] ep {ep:4d} | steps {total_steps:7d} | eps {agent.epsilon:.3f} | "
                      f"train len {env.steps:3d} | val {val['mean']:6.1f} (smoothed {smoothed:6.1f})")

    elapsed = time.time() - t0
    if writer:
        writer.close()

    # ---- save the logs used by the figures and the report ------------------ #
    for name, rows in (("episodes.csv", ep_rows), ("validation.csv", val_rows)):
        with open(log_dir / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    v_steps = [r["env_steps"] for r in val_rows]
    v_smooth = [r["val_smoothed"] for r in val_rows]
    summary = {
        "config": cfg.name, "seed": seed, "best_val_return": float(best_val),
        "best_episode": best_episode,
        "final_val_smoothed": float(np.mean([r["val_return"] for r in val_rows[-5:]])),
        "steps_to_195": first_step_reaching(v_steps, v_smooth, HALF_THRESHOLD),
        "steps_to_475": first_step_reaching(v_steps, v_smooth, SOLVED_THRESHOLD),
        "total_env_steps": total_steps, "train_time_s": round(elapsed, 1),
        "n_parameters": agent.n_parameters(), "checkpoint": str(ckpt_path.relative_to(ROOT_DIR)),
    }
    save_json({"summary": summary, "config": cfg.to_dict()}, log_dir / "summary.json")
    return summary


def select_best(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Choose the best configuration using VALIDATION results only.

    Criterion: highest mean (over seeds) of the best validation return; ties are
    broken by the mean final smoothed validation return (stability at the end of
    training), then by fewer steps to reach 475.

    Args:
        summaries (List[Dict[str, Any]]): One summary per run.

    Returns:
        Dict[str, Any]: Aggregated table per configuration and the chosen
        configuration / run.
    """
    by_cfg: Dict[str, List[Dict[str, Any]]] = {}
    for s in summaries:
        by_cfg.setdefault(s["config"], []).append(s)

    table = []
    for name, runs in by_cfg.items():
        steps475 = [r["steps_to_475"] for r in runs]
        reached = [s for s in steps475 if s is not None]
        table.append({
            "config": name,
            "best_val_mean": float(np.mean([r["best_val_return"] for r in runs])),
            "best_val_std": float(np.std([r["best_val_return"] for r in runs])),
            "final_val_mean": float(np.mean([r["final_val_smoothed"] for r in runs])),
            "final_val_std": float(np.std([r["final_val_smoothed"] for r in runs])),
            "runs_reaching_475": len(reached),
            "steps_to_475_mean": float(np.mean(reached)) if reached else None,
            "train_time_mean_s": float(np.mean([r["train_time_s"] for r in runs])),
        })
    key = lambda t: (t["best_val_mean"], t["final_val_mean"],
                     -(t["steps_to_475_mean"] or float("inf")))
    best_cfg = max(table, key=key)["config"]
    best_run = max(by_cfg[best_cfg], key=lambda r: (r["best_val_return"], r["final_val_smoothed"]))
    return {"table": table, "best_config": best_cfg, "best_run": best_run}


def run_sweep(config_names: Optional[List[str]] = None, episodes: Optional[int] = None) -> None:
    """Train all configurations x seeds, select the best one, save agent_best.pt.

    Args:
        config_names (Optional[List[str]]): Subset of configurations to run.
        episodes (Optional[int]): Override the number of episodes (smoke tests).

    Returns:
        None
    """
    base = load_json(CONFIG_FILE)
    configs = [DQNConfig.from_dict({**base["defaults"], **c}) for c in base["configs"]]
    if config_names:
        configs = [c for c in configs if c.name in config_names]

    summaries = []
    for cfg in configs:
        if episodes:
            cfg.episodes = episodes
        print("\n" + "=" * 72)
        print(f"CONFIG {cfg.name}: lr={cfg.learning_rate} gamma={cfg.gamma} "
              f"eps_decay={cfg.epsilon_decay} weight_decay={cfg.weight_decay} shaping k={cfg.shaping_scale}")
        print("=" * 72)
        for seed in TRAIN_SEEDS:
            s = train_run(cfg, seed)
            summaries.append(s)
            print(f"  => {cfg.name} seed {seed}: best val {s['best_val_return']:.1f} "
                  f"(ep {s['best_episode']}) | final val {s['final_val_smoothed']:.1f} | "
                  f"{s['total_env_steps']} steps | {s['train_time_s']:.0f}s")

    finalize_sweep([c.name for c in configs])


def finalize_sweep(config_names: Optional[List[str]] = None) -> None:
    """Gather the saved run summaries, select the best configuration and copy
    its best checkpoint to ``checkpoints/agent_best.pt``.

    Args:
        config_names (Optional[List[str]]): Configurations to include (default:
            every configuration listed in ``configs/sweep.json``).

    Returns:
        None
    """
    names = config_names or [c["name"] for c in load_json(CONFIG_FILE)["configs"]]
    summaries = []
    for name in names:
        for seed in TRAIN_SEEDS:
            path = RUNS_DIR / name / f"seed_{seed}" / "summary.json"
            if path.exists():
                summaries.append(load_json(path)["summary"])
    if not summaries:
        raise SystemExit("No run summaries found: run the sweep first.")

    sel = select_best(summaries)
    best_src = SWEEP_CKPT_DIR / Path(sel["best_run"]["checkpoint"]).name
    shutil.copy(best_src, CKPT_DIR / "agent_best.pt")
    save_json({"runs": summaries, **sel}, RUNS_DIR / "sweep_results.json")

    print("\n" + "=" * 72)
    print("SWEEP SUMMARY (validation only)")
    print("=" * 72)
    for t in sel["table"]:
        print(f"{t['config']:>14}: best val {t['best_val_mean']:6.1f} +- {t['best_val_std']:5.1f} | "
              f"final val {t['final_val_mean']:6.1f} +- {t['final_val_std']:5.1f} | "
              f"runs reaching 475: {t['runs_reaching_475']}/3")
    print(f"\nBest configuration: {sel['best_config']} -> checkpoints/agent_best.pt "
          f"(seed {sel['best_run']['seed']}, val {sel['best_run']['best_val_return']:.1f})")


def main() -> None:
    """Command-line entry point."""
    torch.set_num_threads(1)  # small MLP: one thread is faster and more reproducible
    parser = argparse.ArgumentParser(description="Train the DQN agent")
    parser.add_argument("--sweep", action="store_true", help="run every configuration x seed")
    parser.add_argument("--configs", nargs="*", default=None, help="subset of configs for --sweep")
    parser.add_argument("--config", default="reference", help="single run: configuration name")
    parser.add_argument("--seed", type=int, default=42, help="single run: seed")
    parser.add_argument("--episodes", type=int, default=None, help="override episodes per run")
    parser.add_argument("--select-only", action="store_true",
                        help="only re-run the selection from the saved run summaries")
    args = parser.parse_args()

    if args.select_only:
        finalize_sweep(args.configs)
        return
    if args.sweep:
        run_sweep(args.configs, args.episodes)
        return
    base = load_json(CONFIG_FILE)
    match = [c for c in base["configs"] if c["name"] == args.config]
    if not match:
        raise SystemExit(f"Unknown config '{args.config}'. Options: {[c['name'] for c in base['configs']]}")
    cfg = DQNConfig.from_dict({**base["defaults"], **match[0]})
    if args.episodes:
        cfg.episodes = args.episodes
    print(train_run(cfg, args.seed))


if __name__ == "__main__":
    main()
