"""Simple inference runner for a single nuPlan-mini sample."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch

from cam_lidar_model.data.nuplan_mini_dataset import NuPlanMiniDataset
from cam_lidar_model.models.mvp_model import MVPAutonomyModel


def _default_data_root() -> Path:
    return Path(__file__).resolve().parents[3] / "nuPlan-mini_dataset"


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> Dict[str, Any]:
    state = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    return state


def summarize_outputs(outputs: Dict[str, Any], sample: Dict[str, Any]) -> None:
    detection = outputs["perception"]["detection"]
    heatmap = detection["heatmap"].detach().cpu()
    heat_shape = tuple(heatmap.shape)
    num_peaks = int((heatmap > 0.5).sum().item())
    gt_boxes = sample["labels"]["detection"]["boxes"]
    num_gt = int(gt_boxes.shape[0]) if isinstance(gt_boxes, torch.Tensor) else 0

    traj = outputs["prediction"]["traj"]
    traj_shape = tuple(traj.shape)
    first_coords = None
    first_heading = None
    first_speed = None
    with torch.no_grad():
        if traj.numel() > 0:
            t = traj
            if t.ndim == 4:  # [N, K, T, 4]
                if t.shape[0] > 0 and t.shape[1] > 0:
                    first_coords = t[0, 0, :3, :2].detach().cpu().numpy()
                    first_heading = t[0, 0, :3, 2].detach().cpu().numpy()
                    first_speed = t[0, 0, :3, 3].detach().cpu().numpy()
            elif t.ndim == 3:  # [N, T, 4]
                if t.shape[0] > 0:
                    first_coords = t[0, :3, :2].detach().cpu().numpy()
                    first_heading = t[0, :3, 2].detach().cpu().numpy()
                    first_speed = t[0, :3, 3].detach().cpu().numpy()

    print("Detection heatmap shape:", heat_shape)
    print(f"GT boxes: {num_gt}, heatmap peaks (>0.5): {num_peaks}")
    print("Prediction traj shape:", traj_shape)
    if first_coords is not None:
        print("First traj coords (first 3 steps, xy):", first_coords)
        print("First traj heading (first 3 steps):", first_heading)
        print("First traj speed (first 3 steps):", first_speed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MVP inference on a single nuPlan-mini sample.")
    parser.add_argument("--data-root", type=Path, default=_default_data_root(), help="Path to nuPlan-mini dataset root.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint (.pt).")
    parser.add_argument("--index", type=int, default=0, help="Dataset index to run inference on.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = NuPlanMiniDataset(data_root=str(args.data_root), split="val")
    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"Index {args.index} out of range for dataset of length {len(dataset)}")

    sample = dataset[args.index]
    sensor_bundle = sample["sensor_bundle"]
    sensor_bundle.camera_tensor = sensor_bundle.camera_tensor.to(device)
    sensor_bundle.lidar_tensor = sensor_bundle.lidar_tensor.to(device)

    model = MVPAutonomyModel().to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    with torch.no_grad():
        outputs = model(sensor_bundle)

    summarize_outputs(outputs, sample)


if __name__ == "__main__":
    main()
