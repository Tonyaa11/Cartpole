"""
evaluate.py - Test-set evaluation of the best DQN agent against the baselines.

Machine Learning - Workshop 3 (Reinforcement Learning)

Protocol
--------
* Test set: 100 held-out initial states (seeds 20000-20099), never used for
  training, checkpoint selection or configuration selection.
* Two test conditions:
    - nominal : the same dynamics as in training;
    - noisy   : Gaussian actuator noise of std 2 N (20 % of the 10 N force),
                a perturbation the agent never saw during training.
* Policies: the six baselines of ``src/w3/baselines.py``, the selected agent
  (``checkpoints/workshop3/agent_best.pt``) and the best checkpoint of every seed
  of the selected configuration (``checkpoints/workshop3/sweep/``), to report the
  variability across training seeds.
* Metrics per policy: mean / std / median task return (= steps survived),
  success rate (500 steps reached), share of failures by angle and by position,
  W1 danger rate (share of steps the Workshop 1 classifier flags as dangerous),
  W2 danger rate (the same with the Workshop 2 LSTM), mean |theta| (degrees) and
  mean |x| (meters).

Outputs: ``runs/workshop3/test_results.json`` and ``runs/workshop3/trajectories.json``.

Usage
-----
    python src/w3/evaluate.py
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from w3.agent import DQNAgent
from w3.baselines import Policy, make_baselines
from w3.common import CKPT_DIR, RUNS_DIR, SWEEP_CKPT_DIR, TEST_SEEDS, load_json, save_json
from w3.env import MAX_STEPS, make_env
from w3.w1_bridge import W1DangerModel
from w3.w2_bridge import W2DangerModel

NOISY_FORCE_STD = 2.0      # Newtons (20 % of the nominal force)
N_TRAJECTORIES = 4         # example episodes saved for the figures


class AgentPolicy(Policy):
    """Wrap a trained DQN agent so it behaves like the baseline policies (greedy)."""

    def __init__(self, agent: DQNAgent, name: str):
        """Args:
            agent (DQNAgent): Trained agent.
            name (str): Label for tables.
        """
        self.agent = agent
        self.name = name

    def act(self, obs: np.ndarray) -> int:
        """Greedy action of the agent (epsilon = 0, no exploration at test time)."""
        return self.agent.act(obs, epsilon=0.0)


def sweep_checkpoint(run: Dict[str, Any]) -> Path:
    """Locate the best checkpoint of one sweep run.

    The file name (``<config>_s<seed>.pt``) is taken from the stored summary
    and looked up in ``SWEEP_CKPT_DIR``, so moving the checkpoint folder never
    breaks the evaluation (the same rule as ``train.py --select-only``).

    Args:
        run (Dict[str, Any]): One run summary of ``sweep_results.json``.

    Returns:
        Path: Path of the checkpoint file.
    """
    return SWEEP_CKPT_DIR / Path(run["checkpoint"]).name


def run_episode(policy: Policy, env, seed: int) -> Dict[str, Any]:
    """Play one episode and record everything needed for the metrics.

    Args:
        policy (Policy): Policy to run.
        env: Environment instance.
        seed (int): Reset seed (defines the initial state).

    Returns:
        Dict[str, Any]: length, outcome, and the arrays of states, actions and
        task rewards (state i is the one in which action i was taken).
    """
    policy.reset()
    obs, _ = env.reset(seed=seed)
    states, actions, rewards = [], [], []
    done = False
    while not done:
        a = policy.act(obs)
        states.append(np.asarray(obs, dtype=np.float64))
        actions.append(a)
        obs, _, term, trunc, info = env.step(a)
        rewards.append(info["task_reward"])
        done = term or trunc
    return {"seed": seed, "length": env.steps, "outcome": info["outcome"],
            "states": np.array(states), "actions": actions, "rewards": rewards}


def evaluate_policy(policy: Policy, seeds: List[int], force_noise_std: float,
                    danger_model: W1DangerModel, w2_model: W2DangerModel) -> Dict[str, Any]:
    """Evaluate one policy on the test seeds under one condition.

    Args:
        policy (Policy): Policy to evaluate.
        seeds (List[int]): Test seeds (one episode each).
        force_noise_std (float): Actuator noise (0 = nominal condition).
        danger_model (W1DangerModel): Workshop 1 classifier for the danger rate.
        w2_model (W2DangerModel): Workshop 2 LSTM for the W2 danger rate.

    Returns:
        Dict[str, Any]: Aggregated metrics and per-episode lengths.
    """
    env = make_env(force_noise_std=force_noise_std)
    episodes = [run_episode(policy, env, s) for s in seeds]
    lengths = np.array([e["length"] for e in episodes], dtype=float)
    outcomes = [e["outcome"] for e in episodes]
    flagged, flagged_w2, total = 0, 0, 0
    for e in episodes:                      # pooled W1 / W2 danger rates over all steps
        p = danger_model.trajectory_proba(e["states"], e["actions"])
        flagged += int(np.sum(p >= danger_model.threshold))
        p2 = w2_model.trajectory_proba(e["states"], e["actions"])
        flagged_w2 += int(np.sum(p2 >= w2_model.threshold))
        total += len(p)
    all_states = np.concatenate([e["states"] for e in episodes])
    return {
        "policy": policy.name,
        "mean_return": float(lengths.mean()), "std_return": float(lengths.std()),
        "median_return": float(np.median(lengths)),
        "success_rate": float(np.mean(lengths >= MAX_STEPS)),
        "theta_failure_rate": float(np.mean([o == "failure_theta" for o in outcomes])),
        "x_failure_rate": float(np.mean([o == "failure_x" for o in outcomes])),
        "w1_danger_rate": flagged / total,
        "w2_danger_rate": flagged_w2 / total,
        "mean_abs_theta_deg": float(np.degrees(np.mean(np.abs(all_states[:, 2])))),
        "mean_abs_x": float(np.mean(np.abs(all_states[:, 0]))),
        "lengths": lengths.astype(int).tolist(),
    }


def main() -> None:
    """Run the whole test evaluation and save the results."""
    torch.set_num_threads(1)
    t0 = time.time()
    sweep = load_json(RUNS_DIR / "sweep_results.json")
    best_cfg = sweep["best_config"]
    print("=" * 72)
    print(f"TEST EVALUATION | selected configuration: {best_cfg} | {len(TEST_SEEDS)} episodes")
    print("=" * 72)

    print("Rebuilding the Workshop 1 classifier and loading the Workshop 2 LSTM...")
    danger = W1DangerModel()
    w2 = W2DangerModel()

    # Policies: baselines + selected agent + every seed of the selected configuration
    agent_best = DQNAgent.load(CKPT_DIR / "agent_best.pt")
    policies: List[Policy] = make_baselines(danger, w2_model=w2) + [AgentPolicy(agent_best, "DQN (agent_best.pt)")]
    seed_policies = []
    for run in sweep["runs"]:
        if run["config"] == best_cfg:
            agent = DQNAgent.load(sweep_checkpoint(run))
            seed_policies.append((run["seed"], AgentPolicy(agent, f"DQN seed {run['seed']}")))

    results: Dict[str, Any] = {"best_config": best_cfg, "agent_best_info": agent_best.checkpoint_info,
                               "noisy_force_std": NOISY_FORCE_STD, "conditions": {}}
    for cond, noise in (("nominal", 0.0), ("noisy", NOISY_FORCE_STD)):
        print(f"\n--- condition: {cond} (force noise std = {noise} N)")
        rows = []
        for pol in policies:
            r = evaluate_policy(pol, TEST_SEEDS, noise, danger, w2)
            rows.append(r)
            print(f"  {r['policy']:<28} return {r['mean_return']:6.1f} +- {r['std_return']:5.1f} | "
                  f"success {r['success_rate']:5.0%} | x-fail {r['x_failure_rate']:4.0%} | "
                  f"W1 danger {r['w1_danger_rate']:6.2%} | W2 danger {r['w2_danger_rate']:6.2%} | "
                  f"|theta| {r['mean_abs_theta_deg']:.2f} deg")
        seed_rows = []
        for seed, pol in seed_policies:
            r = evaluate_policy(pol, TEST_SEEDS, noise, danger, w2)
            r["seed"] = seed
            seed_rows.append(r)
        means = [r["mean_return"] for r in seed_rows]
        succ = [r["success_rate"] for r in seed_rows]
        print(f"  DQN across seeds ({best_cfg}): return {np.mean(means):.1f} +- {np.std(means):.1f} | "
              f"success {np.mean(succ):.0%} +- {np.std(succ):.0%}")
        results["conditions"][cond] = {
            "policies": rows, "seeds": seed_rows,
            "dqn_seed_mean_return": float(np.mean(means)), "dqn_seed_std_return": float(np.std(means)),
            "dqn_seed_mean_success": float(np.mean(succ)), "dqn_seed_std_success": float(np.std(succ)),
        }

    # Example trajectories of the selected agent (nominal test condition)
    env = make_env()
    traj = []
    for s in TEST_SEEDS[:N_TRAJECTORIES]:
        e = run_episode(AgentPolicy(agent_best, "DQN"), env, s)
        traj.append({"seed": s, "length": e["length"], "outcome": e["outcome"],
                     "states": e["states"].tolist(), "actions": e["actions"], "rewards": e["rewards"]})
    save_json({"episodes": traj}, RUNS_DIR / "trajectories.json")

    results["w1_threshold"] = float(danger.threshold)
    results["w1_train_samples"] = int(danger.n_train_samples)
    results["w2_threshold"] = float(w2.threshold)
    results["w2_run_name"] = w2.run_name
    results["eval_time_s"] = round(time.time() - t0, 1)
    save_json(results, RUNS_DIR / "test_results.json")
    print(f"\nSaved runs/workshop3/test_results.json and trajectories.json ({results['eval_time_s']:.0f}s)")


if __name__ == "__main__":
    main()
