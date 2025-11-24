"""Lightweight lidar encoder projecting point-based inputs into BEV space."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class LidarEncoder(nn.Module):
    """Placeholder lidar encoder operating on voxel or BEV tensors.

    The MVP assumes lidar data has already been binned into a pseudo-image (pillars
    or voxels). This module simply applies a shallow CNN to produce BEV-aligned
    features. Future work can swap in sparse convolution engines or PointPillars.
    """

    def __init__(self, in_channels: int = 4, bev_channels: int = 128) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, bev_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bev_channels, bev_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, lidar_tensor: torch.Tensor) -> torch.Tensor:
        """Encode lidar features into BEV.

        Args:
            lidar_tensor: Tensor with shape ``[B, C, H, W]`` (already rasterized).

        Returns:
            ``torch.Tensor`` with shape ``[B, C_bev, H, W]``.
        """

        if lidar_tensor.ndim != 4:
            raise ValueError("Lidar encoder expects a 4-D tensor [B, C, H, W].")

        return self.block(lidar_tensor)
