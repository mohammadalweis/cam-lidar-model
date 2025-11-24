"""Dataset scaffolding for loading nuPlan-mini samples as SensorBundle instances."""

from __future__ import annotations

import bisect
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from cam_lidar_model.data import SensorBundle
from cam_lidar_model.data.bev_labeler import boxes_to_bev_heatmap

# Ensure the bundled nuPlan devkit can be imported when running from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEVKIT_PATH = PROJECT_ROOT / "nuplan-devkit"
if str(DEVKIT_PATH) not in sys.path:
    sys.path.append(str(DEVKIT_PATH))

try:
    from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB
    from nuplan.database.nuplan_db_orm.ego_pose import EgoPose
except ImportError as exc:  # pragma: no cover - import guard for runtime environments only
    raise ImportError(
        "nuplan-devkit is expected to be available in the repository. "
        "Ensure the repository root is on the Python path."
    ) from exc


class NuPlanMiniDataset(Dataset):
    """Skeleton PyTorch dataset yielding synthetic ``SensorBundle`` entries for now."""

    def __init__(self, data_root: str, split: str, config: Optional[Dict] = None) -> None:
        """Initialize the dataset scaffold.

        Args:
            data_root: Path to the nuPlan-mini dataset directory (sensor blobs, maps, DB).
            split: Name of the split to load (e.g., ``"train"``, ``"val"``, ``"test"``).
            config: Optional configuration dictionary for overrides (camera resize, etc.).
        """

        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.config = config or {}
        self.camera_root = self.data_root / "nuplan-v1.1_mini_camera_0"
        self.db_root = self.data_root / "nuplan-v1.1_mini" / "data" / "cache" / "mini"
        self.camera_channels = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0", "CAM_L1", "CAM_R1"]
        camera_hw = self.config.get("camera_resolution", (384, 640))
        self.camera_height = int(camera_hw[0])
        self.camera_width = int(camera_hw[1])
        bev_resolution = self.config.get("bev_resolution", (200, 200))
        self.lidar_channels = int(self.config.get("lidar_channels", 4))
        self.bev_height = int(bev_resolution[0])
        self.bev_width = int(bev_resolution[1])
        self._db_cache: Dict[str, Any] = {}
        all_samples = self._build_index()
        split_idx = int(0.9 * len(all_samples))
        if split == "train":
            self._samples = all_samples[:split_idx]
        elif split == "val":
            self._samples = all_samples[split_idx:]
        else:
            self._samples = all_samples
        self.length = len(self._samples)
        self._identity_calibrations = {
            "camera_intrinsics": [torch.eye(3, dtype=torch.float32) for _ in range(6)],
            "camera_extrinsics": [torch.eye(4, dtype=torch.float32) for _ in range(6)],
            "lidar_extrinsic": torch.eye(4, dtype=torch.float32),
        }
        self.future_horizon_s = 4.0
        self.future_dt_s = 0.5
        self.future_steps = int(self.future_horizon_s / self.future_dt_s)
        self._vehicle_category_cache: Dict[str, str] = {}

    def __len__(self) -> int:
        """Return the number of synthetic samples."""

        return self.length

    def __getitem__(self, index: int) -> Dict[str, object]:
        """Return a minimal batch dict containing ``SensorBundle`` and labels."""

        sample = self._samples[index]
        camera_tensor = self._load_camera_tensor(sample["image_paths"])
        lidar_tensor, lidar_missing = self._load_lidar_bev(sample)
        ego_pose = torch.eye(4, dtype=torch.float32)
        metadata = {
            "frame_id": index,
            "split": self.split,
            "timestamp": sample["timestamp"],
            "log_db": sample["db_name"],
            "lidar_missing": lidar_missing,
        }
        detection_labels = self._load_detection_labels(sample)
        prediction_labels = self._load_prediction_labels(sample)
        bev_heatmap = boxes_to_bev_heatmap(detection_labels["boxes"])
        sensor_bundle = SensorBundle(
            timestamp=sample["timestamp"],
            camera_tensor=camera_tensor,
            lidar_tensor=lidar_tensor,
            lidar_points_raw=None,
            ego_pose=ego_pose,
            calibrations=self._identity_calibrations,
            metadata=metadata,
        )
        return {
            "sensor_bundle": sensor_bundle,
            "labels": {"detection": detection_labels, "prediction": prediction_labels, "bev_heatmap": bev_heatmap},
        }

    def _build_index(self) -> List[Dict[str, object]]:
        """Pre-compute image paths for all frames across available log DBs."""

        samples: List[Dict[str, object]] = []
        kept = 0
        skipped = 0
        if not self.db_root.exists():
            return samples

        db_paths = sorted(self.db_root.glob("*.db"))
        for db_path in db_paths:
            db = self._get_db(db_path)
            channel_to_images: Dict[str, List[object]] = {channel: [] for channel in self.camera_channels}
            for img in db.image:
                channel = img.camera.channel
                if channel in channel_to_images:
                    channel_to_images[channel].append(img)

            if any(len(images) == 0 for images in channel_to_images.values()):
                db.remove_ref()
                continue

            for images in channel_to_images.values():
                images.sort(key=lambda x: x.timestamp)

            frame_count = min(len(images) for images in channel_to_images.values())
            lidar_pcs_sorted = sorted(db.lidar_pc, key=lambda pc: pc.timestamp)
            lidar_timestamps = [pc.timestamp for pc in lidar_pcs_sorted]
            for frame_idx in range(frame_count):
                image_paths = [
                    self.camera_root / channel_to_images[channel][frame_idx].filename_jpg
                    for channel in self.camera_channels
                ]
                timestamp = float(channel_to_images[self.camera_channels[0]][frame_idx].timestamp) / 1e6
                if not image_paths[0].exists():
                    skipped += 1
                    continue
                lidar_pc_token = None
                if lidar_pcs_sorted:
                    lidar_pc_token = self._find_closest_lidar_pc(timestamp, lidar_pcs_sorted, lidar_timestamps).token
                samples.append(
                    {
                        "image_paths": image_paths,
                        "timestamp": timestamp,
                        "db_name": db_path.name,
                        "db_path": str(db_path),
                        "lidar_pc_token": lidar_pc_token,
                    }
                )

                kept += 1

        if skipped > 0:
            print(f"NuPlanMiniDataset index: kept {kept} samples, skipped {skipped} missing camera frames.")

        return samples

    def _load_camera_tensor(self, image_paths: List[Path]) -> torch.Tensor:
        """Load and stack the six camera images into a ``[6, 3, H, W]`` tensor."""

        images: List[torch.Tensor] = []
        for image_path in image_paths:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                if (img.height, img.width) != (self.camera_height, self.camera_width):
                    img = img.resize((self.camera_width, self.camera_height), resample=Image.BILINEAR)
                array = np.asarray(img, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1)
            images.append(tensor)

        return torch.stack(images, dim=0)

    def _get_db(self, db_path: Path) -> NuPlanDB:
        """Lazily load and cache nuPlan DB objects."""

        key = str(db_path)
        if key not in self._db_cache:
            self._db_cache[key] = NuPlanDB(data_root=str(self.data_root), load_path=str(db_path), maps_db=None, verbose=False)
        return self._db_cache[key]

    def _vehicle_category_token(self, db: NuPlanDB) -> Optional[str]:
        """Return cached vehicle category token for a DB."""

        key = db.load_path
        if key not in self._vehicle_category_cache:
            category = db.category.select_one(name="vehicle")
            self._vehicle_category_cache[key] = category.token if category is not None else ""
        token = self._vehicle_category_cache[key]
        return token or None

    def _find_closest_lidar_pc(
        self, timestamp_s: float, lidar_pcs: List[object], lidar_timestamps: List[int]
    ) -> object:
        """Return lidar_pc entry whose timestamp is closest to the given timestamp (seconds)."""

        target_us = int(timestamp_s * 1e6)
        idx = bisect.bisect_left(lidar_timestamps, target_us)
        candidates: List[Tuple[int, object]] = []
        if idx < len(lidar_timestamps):
            candidates.append((abs(lidar_timestamps[idx] - target_us), lidar_pcs[idx]))
        if idx > 0:
            candidates.append((abs(lidar_timestamps[idx - 1] - target_us), lidar_pcs[idx - 1]))
        if not candidates:
            return lidar_pcs[0]
        return min(candidates, key=lambda x: x[0])[1]

    def _load_detection_labels(self, sample: Dict[str, object]) -> Dict[str, torch.Tensor]:
        """Fetch vehicle boxes for the sample's lidar sweep and map to BEV grid pixels."""

        lidar_pc_token = sample.get("lidar_pc_token")
        db_path_str = sample.get("db_path")
        if lidar_pc_token is None or db_path_str is None:
            return {"boxes": torch.zeros((0, 7), dtype=torch.float32)}

        db = self._get_db(Path(db_path_str))
        vehicle_token = self._vehicle_category_token(db)
        if vehicle_token is None:
            return {"boxes": torch.zeros((0, 7), dtype=torch.float32)}

        lidar_boxes = db.lidar_box.select_many(lidar_pc_token=lidar_pc_token)
        entries: List[List[float]] = []

        bev_width = 200
        bev_height = 200
        res = 0.5  # meters per pixel
        ego_x = 0.0
        ego_y = 0.0

        for box in lidar_boxes:
            if box.category.token != vehicle_token:
                continue

            grid_x = (float(box.x) - ego_x) / res + bev_width / 2.0
            grid_y = (float(box.y) - ego_y) / res + bev_height / 2.0

            if grid_x < 0 or grid_x >= bev_width or grid_y < 0 or grid_y >= bev_height:
                continue

            width_px = float(box.width) / res
            length_px = float(box.length) / res

            entries.append(
                [
                    grid_x,
                    grid_y,
                    float(box.z),
                    width_px,
                    length_px,
                    float(box.height),
                    float(box.yaw),
                ]
            )

        if not entries:
            return {"boxes": torch.zeros((0, 7), dtype=torch.float32)}

        return {"boxes": torch.tensor(entries, dtype=torch.float32)}

    def _load_lidar_bev(self, sample: Dict[str, object]) -> Tuple[torch.Tensor, bool]:
        """Load lidar sweep and rasterize into a BEV pseudo-image."""

        lidar_pc_token = sample.get("lidar_pc_token")
        db_path_str = sample.get("db_path")
        bev = np.zeros((self.lidar_channels, self.bev_height, self.bev_width), dtype=np.float32)
        if lidar_pc_token is None or db_path_str is None:
            return torch.from_numpy(bev), True

        try:
            db = self._get_db(Path(db_path_str))
            lidar_pc = db.lidar_pc.get(lidar_pc_token)
            if lidar_pc is None:
                return torch.from_numpy(bev), True

            pc = lidar_pc.load(db)
            points = pc.points  # [4, N] (x, y, z, intensity)
            x, y, z, intensity = points[0], points[1], points[2], points[3]

            x_min, x_max = -50.0, 50.0
            y_min, y_max = -50.0, 50.0
            res = 0.5

            mask = (x >= x_min) & (x < x_max) & (y >= y_min) & (y < y_max)
            if mask.sum() == 0:
                return torch.from_numpy(bev), False

            x = x[mask]
            y = y[mask]
            z = z[mask]
            intensity = intensity[mask]

            gx = np.floor((x - x_min) / res).astype(np.int64)
            gy = np.floor((y - y_min) / res).astype(np.int64)
            gx = np.clip(gx, 0, self.bev_width - 1)
            gy = np.clip(gy, 0, self.bev_height - 1)
            flat = gy * self.bev_width + gx
            num_cells = self.bev_height * self.bev_width

            counts = np.bincount(flat, minlength=num_cells).astype(np.float32).reshape(self.bev_height, self.bev_width)
            sum_height = np.bincount(flat, weights=z, minlength=num_cells).astype(np.float32).reshape(
                self.bev_height, self.bev_width
            )
            sum_intensity = np.bincount(flat, weights=intensity, minlength=num_cells).astype(np.float32).reshape(
                self.bev_height, self.bev_width
            )
            max_height = np.full(num_cells, -np.inf, dtype=np.float32)
            np.maximum.at(max_height, flat, z.astype(np.float32))
            max_height = max_height.reshape(self.bev_height, self.bev_width)
            max_height[counts == 0] = 0.0

            mean_height = np.divide(
                sum_height, counts, out=np.zeros_like(sum_height, dtype=np.float32), where=counts > 0
            )
            mean_intensity = np.divide(
                sum_intensity, counts, out=np.zeros_like(sum_intensity, dtype=np.float32), where=counts > 0
            )

            bev[0] = max_height
            bev[1] = mean_height
            bev[2] = counts
            bev[3] = mean_intensity
            return torch.from_numpy(bev), False
        except Exception:
            return torch.from_numpy(bev), True

    def _to_ego_frame(
        self, x_g: float, y_g: float, heading_g: float, vel_g: float, base_x: float, base_y: float, base_yaw: float
    ) -> torch.Tensor:
        """Convert a global pose to ego-centric frame at t0."""

        dx = x_g - base_x
        dy = y_g - base_y
        cos_yaw = math.cos(base_yaw)
        sin_yaw = math.sin(base_yaw)
        x_ego = cos_yaw * dx + sin_yaw * dy
        y_ego = -sin_yaw * dx + cos_yaw * dy
        heading_ego = heading_g - base_yaw
        return torch.tensor([x_ego, y_ego, heading_ego, vel_g], dtype=torch.float32)

    def _collect_future_from_lidar_box(self, box: object, base_x: float, base_y: float, base_yaw: float) -> torch.Tensor:
        """Collect future states from the linked list of lidar boxes for a track."""

        traj = torch.zeros((self.future_steps, 4), dtype=torch.float32)
        target_ts = int(getattr(box, "timestamp"))
        next_box = box
        dt_us = int(self.future_dt_s * 1e6)

        for step in range(self.future_steps):
            target_ts += dt_us
            while next_box is not None and getattr(next_box, "timestamp") < target_ts:
                next_box = getattr(next_box, "next")
            if next_box is None:
                break
            speed = math.hypot(float(getattr(next_box, "vx", 0.0) or 0.0), float(getattr(next_box, "vy", 0.0) or 0.0))
            traj[step] = self._to_ego_frame(
                float(next_box.x), float(next_box.y), float(next_box.yaw), speed, base_x, base_y, base_yaw
            )

        return traj

    def _collect_future_from_ego(
        self, lidar_pc_token: str, db: NuPlanDB, base_x: float, base_y: float, base_yaw: float
    ) -> Optional[torch.Tensor]:
        """Collect ego future poses aligned to the lidar timestamp."""

        lidar_pc = db.lidar_pc.get(lidar_pc_token)
        if lidar_pc is None:
            return None

        base_ts = int(lidar_pc.timestamp)
        dt_us = int(self.future_dt_s * 1e6)
        target_times = [base_ts + (i + 1) * dt_us for i in range(self.future_steps)]
        future_poses: List[EgoPose] = (
            db.session.query(EgoPose)
            .filter(EgoPose.log_token == lidar_pc.lidar.log_token, EgoPose.timestamp > base_ts)
            .order_by(EgoPose.timestamp.asc())
            .all()
        )
        if not future_poses:
            return None

        traj = torch.zeros((self.future_steps, 4), dtype=torch.float32)
        pose_idx = 0
        for step, target_ts in enumerate(target_times):
            while pose_idx < len(future_poses) and future_poses[pose_idx].timestamp < target_ts:
                pose_idx += 1
            if pose_idx >= len(future_poses):
                break
            pose = future_poses[pose_idx]
            heading = float(pose.quaternion.yaw_pitch_roll[0])
            speed = math.hypot(float(pose.vx or 0.0), float(pose.vy or 0.0))
            traj[step] = self._to_ego_frame(float(pose.x), float(pose.y), heading, speed, base_x, base_y, base_yaw)

        return traj

    def _load_prediction_labels(self, sample: Dict[str, object]) -> Dict[str, object]:
        """Build future trajectory labels for tracks and ego at the sample timestamp."""

        lidar_pc_token = sample.get("lidar_pc_token")
        db_path_str = sample.get("db_path")
        if lidar_pc_token is None or db_path_str is None:
            return {
                "trajectories": torch.zeros((0, self.future_steps, 4), dtype=torch.float32),
                "object_ids": [],
            }

        db = self._get_db(Path(db_path_str))
        lidar_pc = db.lidar_pc.get(lidar_pc_token)
        if lidar_pc is None or lidar_pc.ego_pose is None:
            return {
                "trajectories": torch.zeros((0, self.future_steps, 4), dtype=torch.float32),
                "object_ids": [],
            }

        base_pose = lidar_pc.ego_pose
        base_x = float(base_pose.x)
        base_y = float(base_pose.y)
        base_yaw = float(base_pose.quaternion.yaw_pitch_roll[0])

        vehicle_token = self._vehicle_category_token(db)
        lidar_boxes = db.lidar_box.select_many(lidar_pc_token=lidar_pc_token)

        trajectories: List[torch.Tensor] = []
        object_ids: List[str] = []

        for box in lidar_boxes:
            if vehicle_token is not None and box.category.token != vehicle_token:
                continue
            if box.track_token is None:
                continue
            traj = self._collect_future_from_lidar_box(box, base_x, base_y, base_yaw)
            trajectories.append(traj)
            object_ids.append(str(box.track_token))

        ego_traj = self._collect_future_from_ego(lidar_pc_token, db, base_x, base_y, base_yaw)
        if ego_traj is not None:
            trajectories.append(ego_traj)
            object_ids.append("ego")

        if trajectories:
            traj_tensor = torch.stack(trajectories, dim=0)
        else:
            traj_tensor = torch.zeros((0, self.future_steps, 4), dtype=torch.float32)

        return {"trajectories": traj_tensor, "object_ids": object_ids}
