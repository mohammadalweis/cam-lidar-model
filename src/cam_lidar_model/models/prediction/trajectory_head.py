"""Trajectory heads producing future predictions per agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn


@dataclass
class PredictionOutputs:
    """Container for per-agent trajectories and mode scores."""

    trajectories: torch.Tensor  # [B, N, K, T, 4]
    mode_scores: torch.Tensor  # [B, N, K]


class TrajectoryHead(nn.Module):
    """Predicts multi-modal future trajectories for each agent."""

    def __init__(self, embed_dim: int = 128, num_modes: int = 2, horizon: int = 8) -> None:
        super().__init__()
        self.num_modes = num_modes
        self.horizon = horizon
        self.traj_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, num_modes * horizon * 4),
        )
        self.score_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, num_modes),
        )

    def forward(self, actor_embeddings: torch.Tensor) -> PredictionOutputs:
        """Predict trajectories and scores.

        Args:
            actor_embeddings: ``[B, N, D]`` tensor after interaction modeling.

        Returns:
            :class:`PredictionOutputs` with trajectories and mode scores.
        """

        b, n, d = actor_embeddings.shape
        traj = self.traj_mlp(actor_embeddings).view(b, n, self.num_modes, self.horizon, 4)
        scores = self.score_mlp(actor_embeddings)
        return PredictionOutputs(trajectories=traj, mode_scores=scores)
