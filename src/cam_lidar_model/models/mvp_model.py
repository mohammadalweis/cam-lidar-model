"""Top-level MVP autonomy model wiring perception and prediction stacks."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from cam_lidar_model.data import SensorBundle
from cam_lidar_model.models.perception import BEVFusion, CameraEncoder, LidarEncoder, PerceptionHeads
from cam_lidar_model.models.perception.bev_backbone import BEVBackbone
from cam_lidar_model.models.prediction import ActorEncoder, InteractionModule, SceneEncoder, TrajectoryHead


class MVPAutonomyModel(nn.Module):
    """Wrapper that consumes a :class:`SensorBundle` and outputs perception/prediction results."""

    def __init__(self, lidar_channels: int = 4) -> None:
        super().__init__()
        self.camera_encoder = CameraEncoder()
        self.lidar_encoder = LidarEncoder(in_channels=lidar_channels)
        self.bev_fusion = BEVFusion()
        self.bev_backbone = BEVBackbone(in_channels=128, hidden_channels=128, num_layers=4)
        self.perception_heads = PerceptionHeads()

        self.scene_encoder = SceneEncoder()
        self.actor_encoder = ActorEncoder()
        self.interaction = InteractionModule()
        self.trajectory_head = TrajectoryHead(num_modes=1)

    def forward(self, sensor_bundle: SensorBundle) -> Dict[str, torch.Tensor]:
        """Run the end-to-end MVP stack."""

        camera_tensor = self._ensure_camera_batch(sensor_bundle.camera_tensor)
        lidar_tensor = self._ensure_lidar_batch(sensor_bundle.lidar_tensor)
        camera_bev = self.camera_encoder(camera_tensor)
        lidar_bev = self.lidar_encoder(lidar_tensor)
        fused_bev = self.bev_fusion(camera_bev, lidar_bev)
        bev_features = self.bev_backbone(fused_bev)
        perception_outputs = self.perception_heads(bev_features)

        bev_context = torch.cat([bev_features, perception_outputs["occupancy"], perception_outputs["lane"]], dim=1)
        scene_context = self.scene_encoder(bev_context)

        batch = fused_bev.shape[0]
        num_agents = 1
        actor_features = {
            "boxes": torch.zeros(batch, num_agents, 4, device=fused_bev.device),
            "velocities": torch.zeros(batch, num_agents, 2, device=fused_bev.device),
        }
        actor_embeddings = self.actor_encoder(scene_context, actor_features)
        interacted = self.interaction(actor_embeddings)
        prediction_outputs = self.trajectory_head(interacted)

        traj = prediction_outputs.trajectories
        if traj.ndim == 5 and traj.shape[0] == 1:
            traj = traj[0]

        return {
            "perception": perception_outputs,
            "prediction": {"traj": traj},
        }

    @staticmethod
    def _ensure_camera_batch(camera_tensor: torch.Tensor) -> torch.Tensor:
        """Ensure camera tensor has leading batch dimension."""

        if camera_tensor.ndim == 4:
            camera_tensor = camera_tensor.unsqueeze(0)
        if camera_tensor.ndim != 5 or camera_tensor.shape[1] != 6:
            raise ValueError(f"Expected camera tensor shape [B, 6, 3, H, W], got {tuple(camera_tensor.shape)}")
        return camera_tensor

    @staticmethod
    def _ensure_lidar_batch(lidar_tensor: torch.Tensor) -> torch.Tensor:
        """Ensure lidar tensor has leading batch dimension for encoder."""

        if lidar_tensor.ndim == 3:
            lidar_tensor = lidar_tensor.unsqueeze(0)
        return lidar_tensor
