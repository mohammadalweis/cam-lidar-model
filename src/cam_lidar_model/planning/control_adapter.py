"""Control adapter translating planned trajectories into actuator commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class ControlCommand:
    """Simple Ackermann-like command."""

    steering: float
    throttle: float
    brake: float


class ControlAdapter:
    """Applies PID control to follow a planned trajectory."""

    def __init__(self, kp_lat: float = 0.5, kp_lon: float = 0.5) -> None:
        self.kp_lat = kp_lat
        self.kp_lon = kp_lon

    def compute(self, trajectory: torch.Tensor, ego_state: torch.Tensor) -> ControlCommand:
        """Compute control command for the next time step."""

        return ControlCommand(steering=0.0, throttle=0.0, brake=0.0)
