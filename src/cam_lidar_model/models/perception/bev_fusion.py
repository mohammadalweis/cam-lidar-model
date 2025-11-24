"""BEV fusion block combining camera and lidar features."""

from __future__ import annotations

import torch
import torch.nn as nn


class BEVFusion(nn.Module):
    """Simple channel-concatenation fusion with a refinement block.

    The MVP version keeps fusion cheap: concatenate camera and lidar BEV tensors
    along the channel dimension and pass them through a shallow residual block to
    mix modalities. More advanced approaches (attention, gating) can drop in later.
    """

    def __init__(self, cam_channels: int = 128, lidar_channels: int = 128, fused_channels: int = 128) -> None:
        super().__init__()
        in_channels = cam_channels + lidar_channels
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, fused_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fused_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, camera_bev: torch.Tensor, lidar_bev: torch.Tensor) -> torch.Tensor:
        """Fuse camera and lidar BEV tensors.

        Args:
            camera_bev: ``[B, C_cam, H, W]`` tensor from :class:`CameraEncoder`.
            lidar_bev: ``[B, C_lidar, H, W]`` tensor from :class:`LidarEncoder`.

        Returns:
            ``torch.Tensor`` of shape ``[B, C_fused, H, W]``.
        """

        if camera_bev.shape[-2:] != lidar_bev.shape[-2:]:
            raise ValueError("Camera and lidar BEV tensors must share spatial resolution.")
        x = torch.cat([camera_bev, lidar_bev], dim=1)
        return self.fuse(x)
