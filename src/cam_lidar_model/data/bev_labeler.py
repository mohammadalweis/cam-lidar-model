"""Utilities for rasterizing 3D boxes into BEV supervision tensors."""

from __future__ import annotations

import math
from typing import Tuple

import torch


def boxes_to_bev_heatmap(
    boxes_3d: torch.Tensor,
    bev_h: int = 200,
    bev_w: int = 200,
    x_range: Tuple[float, float] = (-50.0, 50.0),
    y_range: Tuple[float, float] = (-50.0, 50.0),
    sigma: float = 2.0,
) -> torch.Tensor:
    """Rasterize 3D boxes into a BEV Gaussian heatmap.

    Args:
        boxes_3d: Tensor ``[N, 7]`` with (x, y, z, dx, dy, dz, yaw) in meters (global or ego-aligned).
        bev_h: Height of the BEV grid.
        bev_w: Width of the BEV grid.
        x_range: (min, max) meters along x covered by the grid.
        y_range: (min, max) meters along y covered by the grid.
        sigma: Standard deviation for the Gaussian bump in grid cells.

    Returns:
        Tensor ``[1, bev_h, bev_w]`` heatmap.
    """

    device = boxes_3d.device
    heatmap = torch.zeros((1, bev_h, bev_w), device=device, dtype=torch.float32)
    if boxes_3d.numel() == 0:
        return heatmap

    xs = boxes_3d[:, 0]
    ys = boxes_3d[:, 1]

    x_min, x_max = x_range
    y_min, y_max = y_range
    valid = (xs >= x_min) & (xs <= x_max) & (ys >= y_min) & (ys <= y_max)
    if valid.sum() == 0:
        return heatmap

    xs = xs[valid]
    ys = ys[valid]

    # Convert meters to grid indices
    gx = (xs - x_min) / (x_max - x_min) * bev_w
    gy = (ys - y_min) / (y_max - y_min) * bev_h
    gx = gx.long().clamp(0, bev_w - 1)
    gy = gy.long().clamp(0, bev_h - 1)

    # Create Gaussian bumps at each center
    sigma2 = sigma * sigma
    for x_idx, y_idx in zip(gx, gy):
        # Define a small window around the center to avoid full-image loops
        x0 = max(int(x_idx - 3 * sigma), 0)
        x1 = min(int(x_idx + 3 * sigma) + 1, bev_w)
        y0 = max(int(y_idx - 3 * sigma), 0)
        y1 = min(int(y_idx + 3 * sigma) + 1, bev_h)
        xx = torch.arange(x0, x1, device=device, dtype=torch.float32)
        yy = torch.arange(y0, y1, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(yy, xx, indexing="ij")
        dx2 = (xx - float(x_idx)) ** 2
        dy2 = (yy - float(y_idx)) ** 2
        patch = torch.exp(-(dx2 + dy2) / (2 * sigma2))
        heatmap[0, y0:y1, x0:x1] = torch.maximum(heatmap[0, y0:y1, x0:x1], patch)

    return heatmap
