"""Trajectory scoring utilities."""

from __future__ import annotations

from typing import Dict

import torch


class TrajectoryScorer:
    """Evaluates candidate trajectories using weighted cost terms."""

    def __init__(self, weights: Dict[str, float] | None = None) -> None:
        self.weights = weights or {
            "collision": 10.0,
            "lane": 1.0,
            "comfort": 0.5,
            "route": 0.5,
        }

    def score(
        self,
        candidates: torch.Tensor,
        prediction_outputs: Dict[str, torch.Tensor],
        occupancy: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scalar costs for each candidate.

        Args:
            candidates: ``[N, T, 4]`` tensor of ego trajectories.
            prediction_outputs: Dict containing predicted agent trajectories.
            occupancy: BEV occupancy grid aligning with candidates.

        Returns:
            ``torch.Tensor`` of shape ``[N]`` with scalar costs.
        """

        num_candidates = candidates.shape[0]
        return torch.zeros(num_candidates)
