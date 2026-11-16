"""
common.py - Shared paths, seeds and helpers for Workshop 3.

Machine Learning - Workshop 3 (Reinforcement Learning)

Keeping these values in one place guarantees that training, validation and
testing always use the same, disjoint sets of initial states.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # repository root
RUNS_DIR = ROOT_DIR / "runs" / "workshop3"            # logs (required location)
CKPT_DIR = ROOT_DIR / "checkpoints" / "workshop3"     # checkpoints/workshop3/agent_best.pt
SWEEP_CKPT_DIR = CKPT_DIR / "sweep"                   # per-run best checkpoints
CONFIG_FILE = ROOT_DIR / "configs" / "sweep.json"

# --------------------------------------------------------------------------- #
# Seeds: three disjoint sets of initial states
# --------------------------------------------------------------------------- #
TRAIN_SEEDS = [42, 7, 99]                  # one independent training run per seed
VAL_SEEDS = list(range(10_000, 10_010))    # 10 validation episodes (checkpoint selection)
TEST_SEEDS = list(range(20_000, 20_100))   # 100 held-out test episodes (final report)
TRAIN_SEED_LOW = 100_000                   # training episodes use seeds >= this value


def set_global_seed(seed: int) -> None:
    """Fix Python, NumPy and PyTorch seeds.

    Args:
        seed (int): Seed value.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:  # torch is only needed for the agent
        pass


def save_json(obj: Dict[str, Any], path: Path) -> None:
    """Write a dictionary to a JSON file (creating parent folders).

    Args:
        obj (Dict[str, Any]): Data to save.
        path (Path): Output file.

    Returns:
        None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path) -> Any:
    """Read a JSON file.

    Args:
        path (Path): File to read.

    Returns:
        Any: Parsed content.
    """
    with open(path) as f:
        return json.load(f)
