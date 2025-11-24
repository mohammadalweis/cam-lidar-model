"""Trajectory candidate generation for planning."""

from __future__ import annotations

from typing import List

import torch

from .route_context import RouteContext


class CandidateGenerator:
    """Produces kinematic trajectory candidates based on reference path."""

    def __init__(self, horizon_s: float = 4.0, num_candidates: int = 24, dt: float = 0.2) -> None:
        self.horizon_s = horizon_s
        self.num_candidates = num_candidates
        self.dt = dt

    def generate(self, route_context: RouteContext, ego_state: torch.Tensor) -> torch.Tensor:
        """Return a tensor of candidate trajectories.

        Args:
            route_context: Reference path information.
            ego_state: Current ego pose/velocity vector.

        Returns:
            ``torch.Tensor`` of shape ``[N_cand, T, 4]`` (x, y, heading, speed).
        """

        timesteps = int(self.horizon_s / self.dt)
        return torch.zeros(self.num_candidates, timesteps, 4)
