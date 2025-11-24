"""Placeholder metric computations aligned with docs/architecture/06."""

from __future__ import annotations

from typing import Dict

import torch


def compute_perception_metrics(predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Return placeholder perception metrics."""

    return {"mAP": 0.0, "mIoU": 0.0, "occupancy_acc": 0.0}


def compute_prediction_metrics(prediction_outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Return placeholder prediction metrics."""

    return {"ADE": 0.0, "FDE": 0.0, "miss_rate": 0.0}
