#!/usr/bin/env python
"""Simple smoke test for MVPAutonomyModel."""

from __future__ import annotations

import torch

from cam_lidar_model.data import SensorBundle
from cam_lidar_model.models.mvp_model import MVPAutonomyModel


def build_dummy_bundle() -> SensorBundle:
    """Construct a minimal SensorBundle with random tensors."""

    camera_tensor = torch.randn(1, 6, 3, 384, 640)
    lidar_tensor = torch.randn(1, 128, 200, 200)
    ego_pose = torch.eye(4)
    calibrations = {
        "camera_intrinsics": [torch.eye(3) for _ in range(6)],
        "camera_extrinsics": [torch.eye(4) for _ in range(6)],
        "lidar_extrinsic": torch.eye(4),
    }
    metadata = {"frame_id": 0}
    return SensorBundle(
        timestamp=0.0,
        camera_tensor=camera_tensor,
        lidar_tensor=lidar_tensor,
        lidar_points_raw=None,
        ego_pose=ego_pose,
        calibrations=calibrations,
        metadata=metadata,
    )


def main() -> None:
    model = MVPAutonomyModel()
    bundle = build_dummy_bundle()
    outputs = model(bundle)

    print("Perception outputs:")
    for key, value in outputs["perception"].items():
        print(f"  {key}: shape={tuple(value.shape)}")

    pred = outputs["prediction"]
    print("Prediction outputs:")
    print(f"  trajectories: shape={tuple(pred.trajectories.shape)}")
    print(f"  mode_scores: shape={tuple(pred.mode_scores.shape)}")


if __name__ == "__main__":
    main()
