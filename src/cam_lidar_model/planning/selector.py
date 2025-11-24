"""Trajectory selection logic."""

from __future__ import annotations

import torch


class TrajectorySelector:
    """Chooses the best trajectory given candidate costs."""

    def select(self, candidates: torch.Tensor, costs: torch.Tensor) -> torch.Tensor:
        """Return the minimum-cost feasible trajectory."""

        best_idx = torch.argmin(costs)
        return candidates[best_idx]
