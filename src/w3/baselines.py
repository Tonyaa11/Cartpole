"""
baselines.py - Non-learning reference policies for the CartPole control task.

Machine Learning - Workshop 3 (Reinforcement Learning)

The RL agent is compared with policies that need no training interactions:

* ``RandomPolicy``     : uniform random actions (lower bound, also used to
                         validate the environment).
* ``AlternatingPolicy``: fixed strategy, left / right / left / ... (zero mean force).
* ``AngleRule``        : greedy heuristic, push towards the side the pole leans
                         (uses theta only).
* ``PDRule``           : classical control heuristic, push towards the sign of
                         theta + c * theta_dot (angle plus its trend). The gain
                         c = 0.5 was chosen on the VALIDATION episodes only.
* ``W1GuidedPolicy``   : uses the Workshop 1 classifier as a policy: simulate
                         both actions one step ahead and choose the one whose next
                         state has the lower predicted probability of failure.
* ``W2GuidedPolicy``   : the same look-ahead rule with the Workshop 2 LSTM, which
                         reads the last 16 steps of the episode.

All policies share the interface ``reset()`` + ``act(obs) -> int``.

These classes belong to Workshop 3 (they need the RL environment), so they live
here and not in ``src/w1/baselines.py``: the Workshop 1 code stays independent of
Gymnasium.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from w3.env import FORCE_MAG, cartpole_dynamics  # noqa: E402


class Policy:
    """Base class: a policy maps the current observation to an action."""

    name = "policy"

    def reset(self) -> None:
        """Called at the start of every episode (stateless policies ignore it)."""
        return None

    def act(self, obs: np.ndarray) -> int:
        """Choose an action.

        Args:
            obs (np.ndarray): Current state (x, x_dot, theta, theta_dot).

        Returns:
            int: 0 (push left) or 1 (push right).
        """
        raise NotImplementedError


class RandomPolicy(Policy):
    """Uniform random actions."""

    name = "Random"

    def __init__(self, seed: int = 0):
        """Args:
            seed (int): Seed of the action generator.
        """
        self.rng = np.random.RandomState(seed)

    def act(self, obs: np.ndarray) -> int:
        """Ignore the state and pick 0 or 1 with probability 1/2."""
        return int(self.rng.randint(2))


class AlternatingPolicy(Policy):
    """Fixed open-loop strategy: alternate left and right at every step."""

    name = "Alternating (fixed)"

    def __init__(self):
        self.t = 0

    def reset(self) -> None:
        """Restart the left / right sequence at every episode."""
        self.t = 0

    def act(self, obs: np.ndarray) -> int:
        """Return 1, 0, 1, 0, ... regardless of the state."""
        self.t += 1
        return self.t % 2


class AngleRule(Policy):
    """Greedy heuristic: push right if the pole leans right (theta > 0)."""

    name = "Angle rule (greedy)"

    def act(self, obs: np.ndarray) -> int:
        """Push towards the side the pole leans to (sign of theta)."""
        return int(obs[2] > 0)


class PDRule(Policy):
    """Proportional-derivative style rule on the pole angle.

    Pushes right if theta + c * theta_dot > 0, i.e. towards where the pole is
    going, not only where it is now.
    """

    name = "PD rule (heuristic)"

    def __init__(self, gain: float = 0.5):
        """Args:
            gain (float): Weight c of the angular velocity (chosen on validation).
        """
        self.gain = gain

    def act(self, obs: np.ndarray) -> int:
        """Push towards the sign of theta + c * theta_dot."""
        return int(obs[2] + self.gain * obs[3] > 0)


class W1GuidedPolicy(Policy):
    """Workshop 1 classifier used as a one-step look-ahead policy.

    For each action a, the known dynamics give the next state s'_a; the W1
    Gradient Boosting model scores P(danger | s'_a) and the action with the lower
    probability is chosen. When both probabilities are (almost) equal, the
    angle rule breaks the tie. Note that this policy uses the simulator as a
    model of the dynamics, an advantage the model-free DQN agent does not have.
    """

    name = "W1 classifier (look-ahead)"

    def __init__(self, danger_model, tie_tol: float = 1e-3):
        """Args:
            danger_model (W1DangerModel): Rebuilt Workshop 1 classifier.
            tie_tol (float): Probability difference below which actions tie.
        """
        self.model = danger_model
        self.tie_tol = tie_tol
        self.history: List[np.ndarray] = []

    def reset(self) -> None:
        """Forget the states of the previous episode (the W1 features use the past)."""
        self.history = []

    def act(self, obs: np.ndarray) -> int:
        """Simulate both actions and keep the one with the lower W1 danger probability."""
        state = np.asarray(obs, dtype=np.float64)
        candidates = [cartpole_dynamics(state, -FORCE_MAG), cartpole_dynamics(state, FORCE_MAG)]
        p = self.model.danger_proba_candidates(self.history + [state], candidates, [0, 1])
        self.history.append(state)
        if abs(p[0] - p[1]) < self.tie_tol:
            return int(state[2] > 0)          # tie-break with the angle rule
        return int(np.argmin(p))


class W2GuidedPolicy(Policy):
    """Workshop 2 LSTM used as a one-step look-ahead policy.

    Same rule as ``W1GuidedPolicy``: simulate both actions with the known
    dynamics and choose the one whose next state has the lower probability of
    danger, here predicted by the W2 sequence model from the recent history
    (states and the actions actually taken). Ties fall back to the angle rule.
    """

    name = "W2 LSTM (look-ahead)"

    def __init__(self, danger_model, tie_tol: float = 1e-3):
        """Args:
            danger_model (W2DangerModel): Workshop 2 model loaded from its checkpoint.
            tie_tol (float): Probability difference below which actions tie.
        """
        self.model = danger_model
        self.tie_tol = tie_tol
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []

    def reset(self) -> None:
        """Forget the states and actions of the previous episode."""
        self.states, self.actions = [], []

    def act(self, obs: np.ndarray) -> int:
        """Simulate both actions and keep the one with the lower W2 danger probability."""
        state = np.asarray(obs, dtype=np.float64)
        self.states.append(state)
        candidates = [cartpole_dynamics(state, -FORCE_MAG), cartpole_dynamics(state, FORCE_MAG)]
        p = self.model.danger_proba_candidates(self.states, self.actions, candidates, [0, 1])
        action = int(state[2] > 0) if abs(p[0] - p[1]) < self.tie_tol else int(np.argmin(p))
        self.actions.append(action)   # remembered so the next window has the real actions
        return action


def make_baselines(danger_model: Optional[object] = None, seed: int = 0,
                   w2_model: Optional[object] = None) -> List[Policy]:
    """Instantiate every baseline policy.

    Args:
        danger_model (Optional[object]): W1 model; the W1-guided policy is
            skipped if it is None.
        seed (int): Seed of the random policy.
        w2_model (Optional[object]): W2 model; the W2-guided policy is skipped
            if it is None.

    Returns:
        List[Policy]: Baseline policies in report order.
    """
    policies: List[Policy] = [RandomPolicy(seed), AlternatingPolicy(), AngleRule(), PDRule()]
    if danger_model is not None:
        policies.append(W1GuidedPolicy(danger_model))
    if w2_model is not None:
        policies.append(W2GuidedPolicy(w2_model))
    return policies
