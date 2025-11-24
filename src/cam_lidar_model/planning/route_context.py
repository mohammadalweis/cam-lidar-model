"""Route and lane context utilities for the planning module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass
class RouteContext:
    """Container describing ego-lane geometry and optional global route."""

    centerline: torch.Tensor  # [M, 2]
    curvature: torch.Tensor  # [M]
    waypoint_speed: torch.Tensor  # [M]


class RouteContextModule:
    """Prepares reference paths/curvatures for downstream trajectory generation."""

    def __init__(self, lookahead: float = 50.0) -> None:
        self.lookahead = lookahead

    def build_context(self, lane_geometry: Dict[str, torch.Tensor]) -> RouteContext:
        """Construct a :class:`RouteContext` from perception outputs."""

        return RouteContext(
            centerline=lane_geometry["centerline"],
            curvature=lane_geometry.get("curvature", torch.zeros_like(lane_geometry["centerline"][:, 0])),
            waypoint_speed=lane_geometry.get("speed_limit", torch.ones_like(lane_geometry["centerline"][:, 0])),
        )
