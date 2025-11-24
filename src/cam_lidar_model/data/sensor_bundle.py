"""Definitions of shared data containers that travel through the autonomy stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class SensorBundle:
    """Collection of synchronized sensor tensors and metadata for a single timestamp.

    Attributes:
        timestamp: Simulation or dataset timestamp in seconds.
        camera_tensor: Stacked RGB camera tensor with shape ``[6, 3, H, W]`` corresponding
            to front, front-left, front-right, back, back-left, and back-right cameras.
        lidar_tensor: Processed lidar volume (e.g., voxel or pillar grid) with shape
            ``[Vx, Vy, Vz, F]`` representing spatial bins and feature channels.
        lidar_points_raw: Optional raw lidar point cloud with shape ``[N, 5]`` storing
            ``(x, y, z, intensity, ring)``. ``None`` when raw points are not retained.
        ego_pose: Pose of the ego vehicle in SE(3). For now this is a
            ``torch.Tensor`` of shape ``[4, 4]`` representing a homogeneous transform.
        calibrations: Dictionary of calibration data, expected keys:
            * ``camera_intrinsics`` – list of 6 ``3x3`` tensors.
            * ``camera_extrinsics`` – list of 6 ``4x4`` SE(3) tensors mapping from camera
              to vehicle frame.
            * ``lidar_extrinsic`` – single ``4x4`` SE(3) tensor mapping lidar to vehicle.
        metadata: Arbitrary dictionary for bookkeeping (e.g., ``frame_id``, ``drop_flags``).
    """

    timestamp: float
    camera_tensor: torch.Tensor
    lidar_tensor: torch.Tensor
    lidar_points_raw: Optional[torch.Tensor]
    ego_pose: torch.Tensor
    calibrations: Dict[str, List[torch.Tensor]]
    metadata: Dict[str, object] = field(default_factory=dict)

    def validate_shapes(self) -> None:
        """Lightweight validation hook for downstream modules.

        Raises:
            ValueError: If key tensor shapes are incompatible with expectations.
        """

        if self.camera_tensor.ndim != 4 or self.camera_tensor.shape[0] != 6:
            raise ValueError(
                f"camera_tensor must have shape [6, 3, H, W]; got {tuple(self.camera_tensor.shape)}"
            )
        if self.ego_pose.shape != (4, 4):
            raise ValueError(f"ego_pose must be 4x4 SE(3) matrix; got {tuple(self.ego_pose.shape)}")
        if "camera_intrinsics" not in self.calibrations or len(self.calibrations["camera_intrinsics"]) != 6:
            raise ValueError("calibrations must include six camera_intrinsics matrices.")
        if "camera_extrinsics" not in self.calibrations or len(self.calibrations["camera_extrinsics"]) != 6:
            raise ValueError("calibrations must include six camera_extrinsics transforms.")
        if "lidar_extrinsic" not in self.calibrations:
            raise ValueError("calibrations must include lidar_extrinsic transform.")
