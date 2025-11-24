"""Lightweight BEV backbone applied after fusion."""

from __future__ import annotations

import torch
import torch.nn as nn


class _ResidualBlock(nn.Module):
    """Simple residual block with two conv layers."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + x)
        return out


class BEVBackbone(nn.Module):
    """Shallow ResNet-style BEV backbone."""

    def __init__(self, in_channels: int, hidden_channels: int = 128, num_layers: int = 4) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[_ResidualBlock(hidden_channels) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return x
