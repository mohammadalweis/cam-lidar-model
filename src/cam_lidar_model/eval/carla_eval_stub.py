"""CARLA closed-loop evaluation stub."""

from __future__ import annotations

from typing import Dict


def run_carla_closed_loop(model_path: str, config: Dict) -> Dict[str, float]:
    """Placeholder interface for CARLA closed-loop evaluation."""

    return {"route_completion": 0.0, "infractions": 0}
