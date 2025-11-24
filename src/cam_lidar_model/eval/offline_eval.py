"""Offline evaluation harness for nuPlan-mini."""

from __future__ import annotations

from typing import Dict

import torch

from cam_lidar_model.training.metrics import compute_perception_metrics, compute_prediction_metrics


def run_offline_eval(model: torch.nn.Module, dataset: torch.utils.data.Dataset) -> Dict[str, float]:
    """Run placeholder offline evaluation and return aggregate metrics."""

    perception_metrics = compute_perception_metrics({}, {})
    prediction_metrics = compute_prediction_metrics({}, {})
    return {**perception_metrics, **prediction_metrics}
