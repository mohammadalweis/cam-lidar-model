"""Task heads operating on fused BEV features."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionHead(nn.Module):
    """Anchor-free detection head producing heatmaps and box regressions."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.heatmap_logits = nn.Conv2d(in_channels, 1, kernel_size=1)
        # bbox channels: [length_px, width_px, yaw, height_proxy]
        self.bbox = nn.Conv2d(in_channels, 4, kernel_size=1)
        self.velocity = nn.Conv2d(in_channels, 2, kernel_size=1)

    def forward(self, bev: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute detection outputs."""

        feats = self.shared(bev)
        heatmap_logits = self.heatmap_logits(feats)
        return {
            "heatmap": torch.sigmoid(heatmap_logits),
            "heatmap_logits": heatmap_logits,
            "bbox": self.bbox(feats),
            "velocity": self.velocity(feats),
        }


class LaneSegmentationHead(nn.Module):
    """Predicts lane masks / centerline rasters from BEV."""

    def __init__(self, in_channels: int, num_channels: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_channels, kernel_size=1)

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.conv(bev))


class OccupancyHead(nn.Module):
    """Binary occupancy / free-space prediction head."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.conv(bev))


class EgoStateHead(nn.Module):
    """Predicts ego-vehicle dynamics features (velocity, yaw rate)."""

    def __init__(self, in_channels: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4),  # vx, vy, yaw_rate, acceleration
        )

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        flattened = self.pool(bev).flatten(1)
        return self.fc(flattened)


class PerceptionHeads(nn.Module):
    """Wrapper aggregating all perception task heads."""

    def __init__(
        self,
        bev_channels: int = 128,
        lane_channels: int = 3,
    ) -> None:
        super().__init__()
        self.detection = DetectionHead(bev_channels)
        self.lane = LaneSegmentationHead(bev_channels, lane_channels)
        self.occupancy = OccupancyHead(bev_channels)
        self.ego_state = EgoStateHead(bev_channels)

    def forward(self, bev: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run all task heads and return a dictionary of outputs."""

        detection_outputs = self.detection(bev)
        outputs = {
            "detection": detection_outputs,
            "lane": self.lane(bev),
            "occupancy": self.occupancy(bev),
            "ego_state": self.ego_state(bev),
        }
        return outputs
