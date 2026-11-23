"""
env.py - Custom Gymnasium environment for the CartPole control task.

Machine Learning - Workshop 3 (Reinforcement Learning)

Why this environment?
---------------------
In Workshops 1 and 2 we *predicted* whether a CartPole episode was about to fail.
Here the agent *acts* on the same physical system to avoid those failures: at every
time step it pushes the cart left or right, and earlier choices change which states
are reachable later. That is a genuine sequential decision problem, so RL fits it
naturally.

MDP summary (details in the report, Section II)
-----------------------------------------------
* State   s_t = (x, x_dot, theta, theta_dot)                       (4 continuous values)
* Actions a_t in {0: push left, 1: push right}                      (force of -10 N / +10 N)
* Transition: frictionless cart-pole equations of motion, Euler step of 0.02 s
  (identical to Gymnasium CartPole-v1, the system that generated our dataset).
  Deterministic given (s_t, a_t); the initial state is random, and an optional
  Gaussian force noise can make the transitions stochastic.
* Reward  r_t = +1 for every step survived (task reward), plus an OPTIONAL
  potential-based shaping term  F = gamma * Phi(s_{t+1}) - Phi(s_t)  with
  Phi(s) = -k * 0.5 * [(x / 2.4)^2 + (theta / 0.2095)^2].
  Phi encodes the main finding of Workshop 1: the distance to the failure limits is
  what predicts a failure. Potential-based shaping provably keeps the optimal policy
  unchanged (Ng et al., 1999), so it only changes how fast the agent learns.
* Termination: |theta| > 12 degrees or |x| > 2.4 m (failure, terminal).
  Truncation: 500 steps (time limit, NOT a failure, the agent must bootstrap).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# --------------------------------------------------------------------------- #
# Physical constants (same values as Gymnasium CartPole-v1)
# --------------------------------------------------------------------------- #
GRAVITY = 9.8
MASS_CART = 1.0
MASS_POLE = 0.1
TOTAL_MASS = MASS_CART + MASS_POLE
HALF_POLE_LENGTH = 0.5
POLE_MASS_LENGTH = MASS_POLE * HALF_POLE_LENGTH
FORCE_MAG = 10.0
TAU = 0.02  # seconds between two decisions

# Failure limits (also used in Workshop 1 to define the "danger" label)
THETA_LIMIT = 12 * 2 * math.pi / 360  # 0.2095 rad
X_LIMIT = 2.4                         # meters
MAX_STEPS = 500                       # time limit (truncation)

ACTION_NAMES = {0: "push left", 1: "push right"}


def cartpole_dynamics(state: np.ndarray, force: float) -> np.ndarray:
    """Advance the cart-pole system by one Euler step of ``TAU`` seconds.

    This is the transition function s_{t+1} = f(s_t, a_t). It is a pure function
    (no side effects), so the baselines can also use it for one-step look-ahead.

    Args:
        state (np.ndarray): Current state (x, x_dot, theta, theta_dot).
        force (float): Horizontal force applied to the cart in Newtons.

    Returns:
        np.ndarray: Next state (x, x_dot, theta, theta_dot), float64.
    """
    x, x_dot, theta, theta_dot = state
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    # Equations of motion of the frictionless cart-pole (Florian, 2007)
    temp = (force + POLE_MASS_LENGTH * theta_dot ** 2 * sin_t) / TOTAL_MASS
    theta_acc = (GRAVITY * sin_t - cos_t * temp) / (
        HALF_POLE_LENGTH * (4.0 / 3.0 - MASS_POLE * cos_t ** 2 / TOTAL_MASS)
    )
    x_acc = temp - POLE_MASS_LENGTH * theta_acc * cos_t / TOTAL_MASS

    # Explicit Euler integration (same order as Gymnasium's "euler" integrator)
    x = x + TAU * x_dot
    x_dot = x_dot + TAU * x_acc
    theta = theta + TAU * theta_dot
    theta_dot = theta_dot + TAU * theta_acc
    return np.array((x, x_dot, theta, theta_dot), dtype=np.float64)


def is_failure(state: np.ndarray) -> bool:
    """Check whether a state is beyond the failure limits.

    Args:
        state (np.ndarray): State (x, x_dot, theta, theta_dot).

    Returns:
        bool: True if |x| > 2.4 or |theta| > 0.2095 rad.
    """
    return bool(abs(state[0]) > X_LIMIT or abs(state[2]) > THETA_LIMIT)


def danger_potential(state: np.ndarray, scale: float) -> float:
    """Potential function Phi(s) used for reward shaping.

    Phi is 0 at the upright centred position and becomes more negative as the
    system approaches a failure limit, in normalised units (1.0 = at the limit).

    Args:
        state (np.ndarray): State (x, x_dot, theta, theta_dot).
        scale (float): Shaping strength k (0 disables shaping).

    Returns:
        float: Phi(s) = -k * 0.5 * [(x / X_LIMIT)^2 + (theta / THETA_LIMIT)^2].
    """
    x_norm = state[0] / X_LIMIT
    theta_norm = state[2] / THETA_LIMIT
    return -scale * 0.5 * (x_norm ** 2 + theta_norm ** 2)


class CartPoleControlEnv(gym.Env):
    """CartPole control environment with optional potential-based reward shaping.

    The ``info`` dictionary returned by ``step`` always contains the TASK reward
    (``task_reward``, +1 per step) so that every agent is evaluated on the same
    quantity regardless of the shaping used for training.
    """

    metadata = {"render_modes": ["rgb_array", "ansi"], "render_fps": 50}

    def __init__(
        self,
        shaping_scale: float = 0.0,
        gamma: float = 0.99,
        force_noise_std: float = 0.0,
        max_steps: int = MAX_STEPS,
        render_mode: Optional[str] = None,
    ) -> None:
        """Create the environment.

        Args:
            shaping_scale (float): Strength k of the potential-based shaping
                (0 = task reward only).
            gamma (float): Discount factor used in the shaping term; it must be the
                same gamma used by the agent for the policy-invariance guarantee.
            force_noise_std (float): Std (N) of Gaussian noise added to the force
                (0 = deterministic transitions, used for training/validation/test).
            max_steps (int): Time limit after which the episode is truncated.
            render_mode (Optional[str]): "rgb_array", "ansi" or None.
        """
        super().__init__()
        self.shaping_scale = float(shaping_scale)
        self.gamma = float(gamma)
        self.force_noise_std = float(force_noise_std)
        self.max_steps = int(max_steps)
        self.render_mode = render_mode

        # Action space: two discrete actions (push left / push right)
        self.action_space = spaces.Discrete(2)

        # Observation space: limits at twice the failure thresholds so that the
        # final (failing) observation is still inside the box (as in Gymnasium)
        high = np.array([X_LIMIT * 2, np.inf, THETA_LIMIT * 2, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.state: Optional[np.ndarray] = None
        self.steps = 0
        self.last_action: Optional[int] = None
        self.last_reward = 0.0

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Start a new episode from a random near-upright state.

        Each component of the initial state is drawn from U(-0.05, 0.05), exactly
        like Gymnasium CartPole-v1 (same seed -> same initial state).

        Args:
            seed (Optional[int]): Seed of the environment's random generator.
            options (Optional[Dict[str, Any]]): Unused, kept for API compatibility.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: First observation and an info dict.
        """
        super().reset(seed=seed)
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,)).astype(np.float64)
        self.steps = 0
        self.last_action = None
        self.last_reward = 0.0
        return self.state.astype(np.float32), {"steps": 0}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Apply one action and return the transition.

        Args:
            action (int): 0 = push left, 1 = push right.

        Returns:
            Tuple: (observation, reward, terminated, truncated, info) where
            ``reward`` = task reward + shaping, and ``info`` has ``task_reward``,
            ``shaping`` and ``outcome`` ("running", "failure_theta", "failure_x",
            "time_limit").
        """
        assert self.action_space.contains(action), f"invalid action {action!r}"
        assert self.state is not None, "call reset() before step()"

        # 1) Transition: force of +-10 N (+ optional actuator noise)
        force = FORCE_MAG if action == 1 else -FORCE_MAG
        if self.force_noise_std > 0:
            force += float(self.np_random.normal(0.0, self.force_noise_std))
        prev_state = self.state
        self.state = cartpole_dynamics(prev_state, force)
        self.steps += 1

        # 2) Termination (failure) and truncation (time limit)
        terminated = is_failure(self.state)
        truncated = (not terminated) and self.steps >= self.max_steps

        # 3) Reward: +1 per step (Gymnasium convention: the failing step also gets +1)
        task_reward = 1.0

        # Potential-based shaping; Phi(terminal) = 0 keeps the optimal policy unchanged
        shaping = 0.0
        if self.shaping_scale > 0:
            phi_prev = danger_potential(prev_state, self.shaping_scale)
            phi_next = 0.0 if terminated else danger_potential(self.state, self.shaping_scale)
            shaping = self.gamma * phi_next - phi_prev

        reward = task_reward + shaping

        if terminated:
            outcome = "failure_theta" if abs(self.state[2]) > THETA_LIMIT else "failure_x"
        elif truncated:
            outcome = "time_limit"
        else:
            outcome = "running"

        self.last_action = int(action)
        self.last_reward = float(reward)
        info = {"task_reward": task_reward, "shaping": shaping, "outcome": outcome, "steps": self.steps}
        return self.state.astype(np.float32), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Rendering (debugging and figures, no extra dependency needed)
    # ------------------------------------------------------------------ #
    def render(self):
        """Render the current state.

        Returns:
            np.ndarray | str | None: an RGB image (H x W x 3, uint8) for
            ``render_mode="rgb_array"``, a one-line text summary for "ansi",
            and None when no render mode was selected.
        """
        if self.render_mode is None or self.state is None:
            return None
        if self.render_mode == "ansi":
            x, x_dot, theta, theta_dot = self.state
            act = "-" if self.last_action is None else ACTION_NAMES[self.last_action]
            return (f"t={self.steps:3d} | x={x:+.3f} x_dot={x_dot:+.3f} "
                    f"theta={math.degrees(theta):+6.2f}deg theta_dot={theta_dot:+.3f} "
                    f"| action={act} | reward={self.last_reward:+.3f}")
        return self._render_rgb()

    def _render_rgb(self, width: int = 600, height: int = 240) -> np.ndarray:
        """Draw the cart-pole with plain NumPy (track, limits, cart and pole).

        Args:
            width (int): Image width in pixels.
            height (int): Image height in pixels.

        Returns:
            np.ndarray: RGB image of shape (height, width, 3).
        """
        img = np.full((height, width, 3), 255, dtype=np.uint8)
        scale = width / (2 * X_LIMIT * 1.2)          # pixels per meter
        track_y = int(height * 0.75)
        to_px = lambda xm: int(width / 2 + xm * scale)

        img[track_y:track_y + 2, :] = (60, 60, 60)    # track
        for lim in (-X_LIMIT, X_LIMIT):               # failure limits (red)
            px = to_px(lim)
            img[track_y - 60:track_y + 10, max(px - 1, 0):px + 2] = (200, 40, 40)

        x, _, theta, _ = self.state
        cx = to_px(x)
        cart_w, cart_h = 50, 26
        img[track_y - cart_h:track_y, max(cx - cart_w // 2, 0):min(cx + cart_w // 2, width)] = (40, 90, 160)

        # Pole: a thick line drawn by sampling points along it
        pole_len = int(2 * HALF_POLE_LENGTH * scale)
        base_y = track_y - cart_h
        for s in np.linspace(0, 1, pole_len * 2):
            px = int(cx + s * pole_len * math.sin(theta))
            py = int(base_y - s * pole_len * math.cos(theta))
            img[max(py - 2, 0):py + 3, max(px - 2, 0):px + 3] = (200, 140, 40)
        return img

    def close(self) -> None:
        """Release resources (nothing to free in this implementation)."""
        return None


def make_env(shaping_scale: float = 0.0, gamma: float = 0.99, force_noise_std: float = 0.0,
             render_mode: Optional[str] = None) -> CartPoleControlEnv:
    """Factory used by every script, so all of them build the same environment.

    Args:
        shaping_scale (float): Shaping strength k (0 = task reward only).
        gamma (float): Discount factor used in the shaping term.
        force_noise_std (float): Actuator noise std in Newtons.
        render_mode (Optional[str]): Render mode.

    Returns:
        CartPoleControlEnv: A new environment instance.
    """
    return CartPoleControlEnv(shaping_scale=shaping_scale, gamma=gamma,
                              force_noise_std=force_noise_std, render_mode=render_mode)


# --------------------------------------------------------------------------- #
# Environment validation (run: python src/w3/env.py)
# --------------------------------------------------------------------------- #
def check_parity_with_gymnasium(n_episodes: int = 20, seed: int = 0) -> float:
    """Compare our transitions with Gymnasium CartPole-v1 step by step.

    Both environments are reset with the same seed and receive the same random
    actions; the largest absolute difference between their states is returned.

    Args:
        n_episodes (int): Number of episodes to compare.
        seed (int): Base seed.

    Returns:
        float: Maximum absolute state difference over all compared steps
        (0.0 means identical dynamics). Returns NaN if Gymnasium's CartPole-v1
        cannot be created.
    """
    try:
        ref = gym.make("CartPole-v1").unwrapped
    except Exception:  # pragma: no cover - only if gymnasium lacks classic_control
        return float("nan")
    ours = CartPoleControlEnv()
    rng = np.random.RandomState(seed)
    max_diff = 0.0
    for ep in range(n_episodes):
        s_ref, _ = ref.reset(seed=seed + ep)
        s_our, _ = ours.reset(seed=seed + ep)
        max_diff = max(max_diff, float(np.max(np.abs(np.asarray(s_ref) - s_our))))
        for _ in range(MAX_STEPS):
            a = int(rng.randint(2))
            s_ref, _, term_ref, _, _ = ref.step(a)
            s_our, _, term_our, trunc_our, _ = ours.step(a)
            max_diff = max(max_diff, float(np.max(np.abs(np.asarray(s_ref) - s_our))))
            assert term_ref == term_our, "termination differs from Gymnasium"
            if term_ref or trunc_our:
                break
    return max_diff


def random_policy_diagnostics(n_episodes: int = 100, shaping_scale: float = 1.0,
                              seed: int = 0) -> Dict[str, Any]:
    """Run ``n_episodes`` with a uniformly random policy (required sanity check).

    Args:
        n_episodes (int): Number of episodes (the workshop asks for 100).
        shaping_scale (float): Shaping strength used to report the shaped return.
        seed (int): Base seed for the environment and the random actions.

    Returns:
        Dict[str, Any]: Per-episode lengths and returns, action counts and the
        distribution of episode outcomes.
    """
    env = CartPoleControlEnv(shaping_scale=shaping_scale)
    env.action_space.seed(seed)
    lengths, task_returns, shaped_returns = [], [], []
    action_counts = np.zeros(2, dtype=int)
    outcomes: Dict[str, int] = {}
    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        done, ret_task, ret_shaped = False, 0.0, 0.0
        while not done:
            a = int(env.action_space.sample())
            action_counts[a] += 1
            _, r, term, trunc, info = env.step(a)
            ret_task += info["task_reward"]
            ret_shaped += r
            done = term or trunc
        lengths.append(env.steps)
        task_returns.append(ret_task)
        shaped_returns.append(ret_shaped)
        outcomes[info["outcome"]] = outcomes.get(info["outcome"], 0) + 1
    return {
        "n_episodes": n_episodes,
        "lengths": lengths,
        "task_returns": task_returns,
        "shaped_returns": shaped_returns,
        "action_counts": action_counts.tolist(),
        "outcomes": outcomes,
    }


def main() -> None:
    """Validate the environment and save the diagnostics for the report."""
    import sys
    from pathlib import Path

    from gymnasium.utils.env_checker import check_env

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/ (run as a script)
    from w3.common import RUNS_DIR, save_json

    print("=" * 64)
    print("ENVIRONMENT VALIDATION - CartPoleControlEnv")
    print("=" * 64)

    # 1) Gymnasium API compliance (spaces, reset/step signatures, dtypes)
    import warnings
    with warnings.catch_warnings():
        # Gymnasium warns about the infinite velocity bounds; CartPole-v1 uses the same bounds
        warnings.simplefilter("ignore", UserWarning)
        check_env(CartPoleControlEnv(shaping_scale=1.0), skip_render_check=True)
    print("Gymnasium API check      : OK")

    # 2) Same physics as CartPole-v1 (the system that generated our dataset)
    diff = check_parity_with_gymnasium()
    print(f"Max |state diff| vs CartPole-v1 over 20 episodes: {diff:.2e}")

    # 3) 100 random-policy episodes
    diag = random_policy_diagnostics()
    L = np.array(diag["lengths"])
    acts = np.array(diag["action_counts"])
    print(f"\nRandom policy, {diag['n_episodes']} episodes")
    print(f"  mean return (task, +1/step): {np.mean(diag['task_returns']):.2f} "
          f"+- {np.std(diag['task_returns']):.2f}")
    print(f"  mean return (shaped, k=1)  : {np.mean(diag['shaped_returns']):.2f}")
    print(f"  episode length min/median/max: {L.min()} / {np.median(L):.1f} / {L.max()}")
    print(f"  action frequency: left {acts[0]} ({acts[0] / acts.sum():.1%}) | "
          f"right {acts[1]} ({acts[1] / acts.sum():.1%})")
    print(f"  outcomes: {diag['outcomes']}")

    # 4) Rendering smoke test
    env = CartPoleControlEnv(render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    env.render_mode = "ansi"
    print(f"\nrender(): rgb_array frame {frame.shape} | ansi: {env.render()}")

    diag["parity_max_abs_diff"] = diff
    save_json(diag, RUNS_DIR / "env_validation.json")
    print(f"\nSaved {RUNS_DIR / 'env_validation.json'}")


if __name__ == "__main__":
    main()
