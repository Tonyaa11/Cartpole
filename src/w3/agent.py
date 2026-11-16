"""
agent.py - Deep Q-Network (DQN) agent implemented from scratch in PyTorch.

Machine Learning - Workshop 3 (Reinforcement Learning)

Why DQN?
--------
The action set is discrete (push left / push right) and the CartPole state is
Markovian (position, velocities and angle fully describe the system), which is the
setting DQN was designed for (Mnih et al., 2015).

Components required by the workshop
-----------------------------------
1. Q-network  Q(s, a; w): an MLP that outputs one value per action.
2. Experience replay: transitions are stored in a circular buffer and sampled
   uniformly, which breaks the correlation between consecutive steps.
3. Target network Q(s, a; w-): a frozen copy updated every ``target_update``
   gradient steps, which keeps the TD target from moving at every update.
4. Epsilon-greedy exploration with multiplicative decay per episode.
5. Gradient clipping (global L2 norm) and Huber loss for stable updates.

TD target (terminal = failure only; time-limit truncation still bootstraps):
    y = r + gamma * (1 - terminated) * max_a' Q(s', a'; w-)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from w3.env import THETA_LIMIT, X_LIMIT


@dataclass
class DQNConfig:
    """Hyperparameters of one agent configuration (one row of the sweep).

    Attributes:
        name (str): Configuration label used in logs and tables.
        learning_rate (float): Adam step size.
        gamma (float): Discount factor (also used by the reward shaping).
        epsilon_start (float): Initial exploration rate.
        epsilon_min (float): Lower bound of the exploration rate.
        epsilon_decay (float): Multiplicative decay applied after each episode.
        weight_decay (float): L2 regularisation strength in Adam.
        shaping_scale (float): Potential-based shaping strength k (0 = off).
        hidden_sizes (List[int]): Hidden layer widths of the Q-network.
        buffer_size (int): Replay buffer capacity (transitions).
        batch_size (int): Mini-batch size per gradient step.
        learning_starts (int): Environment steps before the first update.
        train_freq (int): Environment steps between two gradient steps.
        target_update (int): Gradient steps between two target-network copies.
        double_dqn (bool): Use the Double-DQN target (online net picks a',
            target net evaluates it), which reduces Q-value over-estimation.
        grad_clip (float): Maximum global L2 norm of the gradients.
        episodes (int): Training episodes per run.
        eval_every (int): Episodes between two validation evaluations.
    """

    name: str = "reference"
    learning_rate: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.99
    weight_decay: float = 0.0
    shaping_scale: float = 1.0
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 128])
    buffer_size: int = 50_000
    batch_size: int = 64
    learning_starts: int = 1_000
    train_freq: int = 1
    target_update: int = 500
    double_dqn: bool = False
    grad_clip: float = 10.0
    episodes: int = 300
    eval_every: int = 10

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DQNConfig":
        """Build a config from a dictionary, ignoring unknown keys.

        Args:
            d (Dict[str, Any]): Values to override the defaults.

        Returns:
            DQNConfig: The configuration.
        """
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def to_dict(self) -> Dict[str, Any]:
        """Return the configuration as a plain dictionary.

        Returns:
            Dict[str, Any]: All hyperparameters.
        """
        return asdict(self)


class QNetwork(nn.Module):
    """MLP that maps a state to one Q-value per action.

    The input is first divided by fixed physical scales (failure limits for x and
    theta, typical magnitudes for the velocities). These constants come from the
    environment definition, not from data, so no information leaks from any split.
    """

    def __init__(self, obs_dim: int = 4, n_actions: int = 2, hidden_sizes: Optional[List[int]] = None):
        """Create the network.

        Args:
            obs_dim (int): State dimension (4 for CartPole).
            n_actions (int): Number of discrete actions (2).
            hidden_sizes (Optional[List[int]]): Hidden layer widths.
        """
        super().__init__()
        hidden_sizes = hidden_sizes or [128, 128]
        self.register_buffer("obs_scale", torch.tensor([X_LIMIT, 3.0, THETA_LIMIT, 3.5]))
        layers: List[nn.Module] = []
        prev = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_actions))  # linear output: Q-values are unbounded
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute Q(s, .) for a batch of states.

        Args:
            obs (torch.Tensor): States, shape (batch, 4).

        Returns:
            torch.Tensor: Q-values, shape (batch, n_actions).
        """
        return self.net(obs / self.obs_scale)


class ReplayBuffer:
    """Fixed-size circular buffer of transitions (s, a, r, s', terminated)."""

    def __init__(self, capacity: int, obs_dim: int, rng: np.random.RandomState):
        """Allocate the buffer.

        Args:
            capacity (int): Maximum number of stored transitions.
            obs_dim (int): State dimension.
            rng (np.random.RandomState): Generator used for sampling (seeded).
        """
        self.capacity = capacity
        self.rng = rng
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, obs, action: int, reward: float, next_obs, terminated: bool) -> None:
        """Store one transition, overwriting the oldest when full.

        Args:
            obs: State s_t.
            action (int): Action a_t.
            reward (float): Reward r_t (shaped reward used for training).
            next_obs: State s_{t+1}.
            terminated (bool): True only for a failure (not for the time limit).

        Returns:
            None
        """
        i = self.pos
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.terminated[i] = next_obs, float(terminated)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a uniform random mini-batch.

        Args:
            batch_size (int): Number of transitions.

        Returns:
            Dict[str, torch.Tensor]: Batched tensors obs, actions, rewards,
            next_obs and terminated.
        """
        idx = self.rng.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idx]),
            "actions": torch.as_tensor(self.actions[idx]),
            "rewards": torch.as_tensor(self.rewards[idx]),
            "next_obs": torch.as_tensor(self.next_obs[idx]),
            "terminated": torch.as_tensor(self.terminated[idx]),
        }


class DQNAgent:
    """DQN agent: online network, target network, replay buffer and optimiser."""

    def __init__(self, cfg: DQNConfig, seed: int = 0):
        """Create the agent.

        Args:
            cfg (DQNConfig): Hyperparameters.
            seed (int): Seed for network initialisation and replay sampling.
        """
        self.cfg = cfg
        torch.manual_seed(seed)
        self.rng = np.random.RandomState(seed)
        self.q = QNetwork(hidden_sizes=cfg.hidden_sizes)
        self.q_target = QNetwork(hidden_sizes=cfg.hidden_sizes)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()
        # Adam's weight_decay adds an L2 penalty on the weights (regularisation)
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=cfg.learning_rate,
                                          weight_decay=cfg.weight_decay)
        self.buffer = ReplayBuffer(cfg.buffer_size, obs_dim=4, rng=self.rng)
        self.epsilon = cfg.epsilon_start
        self.grad_steps = 0

    def n_parameters(self) -> int:
        """Count trainable parameters of the Q-network.

        Returns:
            int: Number of parameters.
        """
        return sum(p.numel() for p in self.q.parameters() if p.requires_grad)

    @torch.no_grad()
    def act(self, obs: np.ndarray, epsilon: Optional[float] = None) -> int:
        """Epsilon-greedy action selection.

        Args:
            obs (np.ndarray): Current state.
            epsilon (Optional[float]): Exploration rate; None uses the agent's
                current epsilon, 0.0 gives the greedy policy (evaluation).

        Returns:
            int: Chosen action.
        """
        eps = self.epsilon if epsilon is None else epsilon
        if eps > 0 and self.rng.rand() < eps:
            return int(self.rng.randint(2))            # exploration
        q_values = self.q(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        return int(torch.argmax(q_values, dim=1).item())  # exploitation

    def decay_epsilon(self) -> None:
        """Apply the per-episode multiplicative epsilon decay (bounded below)."""
        self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)

    def update(self) -> Optional[float]:
        """Run one gradient step on a replay mini-batch.

        Returns:
            Optional[float]: Huber TD loss of the step, or None while the buffer
            holds fewer than ``learning_starts`` transitions.
        """
        if self.buffer.size < max(self.cfg.learning_starts, self.cfg.batch_size):
            return None
        b = self.buffer.sample(self.cfg.batch_size)

        # Q(s, a) for the actions actually taken
        q_sa = self.q(b["obs"]).gather(1, b["actions"].unsqueeze(1)).squeeze(1)

        # TD target with the frozen target network (no gradient flows through it)
        with torch.no_grad():
            if self.cfg.double_dqn:
                best_next = self.q(b["next_obs"]).argmax(dim=1, keepdim=True)
                q_next = self.q_target(b["next_obs"]).gather(1, best_next).squeeze(1)
            else:
                q_next = self.q_target(b["next_obs"]).max(dim=1).values
            target = b["rewards"] + self.cfg.gamma * (1.0 - b["terminated"]) * q_next

        loss = F.smooth_l1_loss(q_sa, target)   # Huber loss: robust to large TD errors
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip)
        self.optimizer.step()

        # Hard copy of the online weights into the target network
        self.grad_steps += 1
        if self.grad_steps % self.cfg.target_update == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return float(loss.item())

    def save(self, path: Path, extra: Optional[Dict[str, Any]] = None) -> None:
        """Save the Q-network weights and the configuration.

        Args:
            path (Path): Output ``.pt`` file.
            extra (Optional[Dict[str, Any]]): Metadata (episode, validation return...).

        Returns:
            None
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.q.state_dict(), "config": self.cfg.to_dict(),
                    "extra": extra or {}}, path)

    @classmethod
    def load(cls, path: Path) -> "DQNAgent":
        """Load an agent saved with ``save`` (greedy use: epsilon = 0).

        Args:
            path (Path): Checkpoint file.

        Returns:
            DQNAgent: Agent with the stored weights.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        agent = cls(DQNConfig.from_dict(ckpt["config"]))
        agent.q.load_state_dict(ckpt["state_dict"])
        agent.q_target.load_state_dict(ckpt["state_dict"])
        agent.epsilon = 0.0
        agent.checkpoint_info = ckpt.get("extra", {})
        return agent
