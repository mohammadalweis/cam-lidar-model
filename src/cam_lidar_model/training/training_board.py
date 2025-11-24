"""Centralized training configuration board.

Edit this file to change dataset paths (e.g., move between Colab/local),
training hyperparameters, loss weights, and device defaults. All values can
still be overridden by CLI flags in ``train.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingBoard:
    # Paths
    data_root: str = "/content/nuPlan-mini_dataset"
    checkpoints_dir: str = "checkpoints"

    # Training
    epochs: int = 1
    batch_size: int = 2
    max_batches: int = -1  # -1 = use full epoch
    num_workers: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4

    # Loss weights
    lambda_pred: float = 0.1

    # Device
    device: str = "cuda"  # or "cpu"

    # Misc
    seed: int = 42


def get_default_board() -> TrainingBoard:
    """
    Returns a default TrainingBoard object.
    Edit this file directly to change defaults.
    """

    return TrainingBoard()
