"""Lightweight camera encoder for six-camera RGB inputs."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CameraEncoder(nn.Module):
    """Simple convolutional encoder projecting six camera images into BEV space.

    This is intentionally lightweight: it applies a shared-weight 2D CNN to each
    camera frame, aggregates the resulting features, and resamples them into a BEV
    grid. Future iterations can replace this with a true lift-splat or transformer
    encoder; for Milestone 2 we only need a runnable scaffold.
    """

    def __init__(
        self,
        stem_channels: int = 32,
        bev_channels: int = 128,
        bev_resolution: Tuple[int, int] = (200, 200),
    ) -> None:
        super().__init__()
        self.bev_resolution = bev_resolution

        self.backbone = nn.Sequential(
            nn.Conv2d(3, stem_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_channels, stem_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )
        self.projection = nn.Conv2d(stem_channels, bev_channels, kernel_size=1)

    def forward(self, camera_tensor: torch.Tensor) -> torch.Tensor:
        """Encode six RGB cameras into a BEV feature grid.

        Args:
            camera_tensor: Tensor with shape ``[B, 6, 3, H, W]``.

        Returns:
            ``torch.Tensor`` with shape ``[B, C_bev, H_bev, W_bev]``.
        """

        if camera_tensor.ndim != 5 or camera_tensor.shape[1] != 6:
            raise ValueError(f"Expected camera tensor of shape [B, 6, 3, H, W], got {camera_tensor.shape}")

        batch_size = camera_tensor.shape[0]
        flattened = camera_tensor.view(batch_size * 6, 3, *camera_tensor.shape[-2:])
        features = self.backbone(flattened)
        features = self.projection(features)
        features = features.view(batch_size, 6, features.shape[1], *features.shape[-2:])
        aggregated = features.mean(dim=1)  # simple average over cameras
        bev = F.interpolate(aggregated, size=self.bev_resolution, mode="bilinear", align_corners=False)
        return bev
