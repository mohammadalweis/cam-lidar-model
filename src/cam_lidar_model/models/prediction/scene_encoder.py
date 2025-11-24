"""Scene encoder for prediction, operating on BEV context tensors."""

from __future__ import annotations

import torch
import torch.nn as nn


class SceneEncoder(nn.Module):
    """Shallow CNN that produces context features for prediction heads."""

    def __init__(self, in_channels: int = 132, context_channels: int = 128) -> None:
        """Initialize the scene encoder.

        Args:
            in_channels: Channels from perception BEV plus occupancy/lane rasters.
            context_channels: Number of output channels to share with actor encoders.
        """

        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, context_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(context_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(context_channels, context_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(context_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, bev_context: torch.Tensor) -> torch.Tensor:
        """Encode BEV context into features shared with actor encoders."""

        return self.encoder(bev_context)
