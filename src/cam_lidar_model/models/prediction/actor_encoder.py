"""Actor encoder for per-agent embeddings."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorEncoder(nn.Module):
    """Pools local BEV context and agent attributes into fixed-size embeddings."""

    def __init__(self, context_channels: int = 128, hidden_dim: int = 128, geom_dim: int = 6) -> None:
        super().__init__()
        self.conv = nn.Conv2d(context_channels, hidden_dim, kernel_size=1)
        self.geom_dim = geom_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + geom_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, scene_context: torch.Tensor, actor_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode actors.

        Args:
            scene_context: ``[B, C, H, W]`` tensor from :class:`SceneEncoder`.
            actor_features: Dict containing at least:
                * ``boxes`` – ``[B, N, 4]`` (x, y, length, width).
                * ``velocities`` – ``[B, N, 2]``.

        Returns:
            ``torch.Tensor`` with shape ``[B, N, hidden_dim]``.
        """

        b, n, _ = actor_features["boxes"].shape
        pooled = F.adaptive_avg_pool2d(self.conv(scene_context), (1, 1)).flatten(2).transpose(1, 2)
        pooled = pooled.expand(-1, n, -1)
        velocities = actor_features.get("velocities")
        if velocities is None:
            velocities = torch.zeros(*actor_features["boxes"].shape[:2], 2, device=actor_features["boxes"].device)
        geom = torch.cat([actor_features["boxes"], velocities], dim=-1)
        if geom.shape[-1] != self.geom_dim:
            raise ValueError(f"Actor geometry features must have {self.geom_dim} dims, got {geom.shape[-1]}")
        combined = torch.cat([pooled, geom], dim=-1)
        return self.mlp(combined)
