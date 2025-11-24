"""Training loop scaffolding for the camera + lidar stack."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
from torch.utils.data import DataLoader

from cam_lidar_model.data import SensorBundle
from cam_lidar_model.data.nuplan_mini_dataset import NuPlanMiniDataset
from cam_lidar_model.models.mvp_model import MVPAutonomyModel
from cam_lidar_model.training.losses import detection_loss, prediction_loss
from cam_lidar_model.training.training_board import TrainingBoard, get_default_board


class Trainer:
    """Lightweight trainer orchestrating dataset/model/loss wiring."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        log_every: int = 10,
        checkpoints_dir: Path | str = Path("checkpoints"),
        lambda_pred: float = 0.1,
    ) -> None:
        self.model = model.to(device)
        self.loader = train_loader
        self.val_loader = val_loader
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.log_every = max(1, log_every)
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss: Optional[float] = None
        self.lambda_pred = lambda_pred

    def train_step(self, batch: Dict[str, SensorBundle]) -> Dict[str, float]:
        """Single training step placeholder using SensorBundle inputs."""

        sensor_bundle = batch["sensor_bundle"]
        sensor_bundle.camera_tensor = sensor_bundle.camera_tensor.to(self.device)
        sensor_bundle.lidar_tensor = sensor_bundle.lidar_tensor.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(sensor_bundle)
        detection_targets = {k: v.to(self.device) for k, v in batch["labels"]["detection"].items()}
        prediction_targets_raw = batch["labels"].get("prediction", {})
        prediction_targets = {
            "trajectories": prediction_targets_raw.get("trajectories", torch.zeros(0)).to(self.device)
            if isinstance(prediction_targets_raw.get("trajectories"), torch.Tensor)
            else torch.zeros(0, device=self.device),
            "object_ids": prediction_targets_raw.get("object_ids", []),
        }
        detection_outputs = outputs["perception"]["detection"]
        loss_det = detection_loss(detection_outputs["heatmap_logits"], batch["labels"]["bev_heatmap"].to(self.device))
        loss_pred, ade, fde, pos_loss, heading_loss, speed_loss = prediction_loss(outputs["prediction"], prediction_targets)
        total_loss = loss_det + self.lambda_pred * loss_pred
        total_loss.backward()
        self.optimizer.step()
        return {
            "loss": float(total_loss),
            "det_loss": float(loss_det),
            "pred_loss": float(loss_pred),
            "ade": float(ade),
            "fde": float(fde),
            "pos_loss": float(pos_loss),
            "heading_loss": float(heading_loss),
            "speed_loss": float(speed_loss),
        }

    def validate(self, max_batches: int = 20) -> Dict[str, float]:
        """Evaluate detection + prediction losses on a small validation subset."""

        if self.val_loader is None:
            return {
                "total": float("nan"),
                "det": float("nan"),
                "pred": float("nan"),
                "ade": float("nan"),
                "fde": float("nan"),
                "heading": float("nan"),
                "speed": float("nan"),
            }

        self.model.eval()
        losses = []
        det_losses = []
        pred_losses = []
        ade_losses = []
        fde_losses = []
        heading_losses = []
        speed_losses = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                sensor_bundle = batch["sensor_bundle"]
                sensor_bundle.camera_tensor = sensor_bundle.camera_tensor.to(self.device)
                sensor_bundle.lidar_tensor = sensor_bundle.lidar_tensor.to(self.device)
                detection_targets = {k: v.to(self.device) for k, v in batch["labels"]["detection"].items()}
                prediction_targets_raw = batch["labels"].get("prediction", {})
                prediction_targets = {
                    "trajectories": prediction_targets_raw.get("trajectories", torch.zeros(0)).to(self.device)
                    if isinstance(prediction_targets_raw.get("trajectories"), torch.Tensor)
                    else torch.zeros(0, device=self.device),
                    "object_ids": prediction_targets_raw.get("object_ids", []),
                }
                outputs = self.model(sensor_bundle)
                detection_outputs = outputs["perception"]["detection"]
                loss_det = detection_loss(detection_outputs["heatmap_logits"], batch["labels"]["bev_heatmap"].to(self.device))
                loss_pred, ade, fde, pos_loss, heading_loss, speed_loss = prediction_loss(outputs["prediction"], prediction_targets)
                total_loss = loss_det + loss_pred
                losses.append(float(total_loss))
                det_losses.append(float(loss_det))
                pred_losses.append(float(loss_pred))
                ade_losses.append(float(ade))
                fde_losses.append(float(fde))
                heading_losses.append(float(heading_loss))
                speed_losses.append(float(speed_loss))
                if batch_idx + 1 >= max_batches:
                    break
        self.model.train()
        if not losses:
            return {
                "total": float("nan"),
                "det": float("nan"),
                "pred": float("nan"),
                "ade": float("nan"),
                "fde": float("nan"),
                "heading": float("nan"),
                "speed": float("nan"),
            }
        return {
            "total": sum(losses) / len(losses),
            "det": sum(det_losses) / len(det_losses),
            "pred": sum(pred_losses) / len(pred_losses),
            "ade": sum(ade_losses) / len(ade_losses),
            "fde": sum(fde_losses) / len(fde_losses),
            "heading": sum(heading_losses) / len(heading_losses),
            "speed": sum(speed_losses) / len(speed_losses),
        }

    def fit(self, epochs: int = 1, max_batches: Optional[int] = None) -> None:
        """Run training for a number of epochs."""

        for epoch in range(epochs):
            train_losses = []
            for batch_idx, batch in enumerate(self.loader):
                metrics = self.train_step(batch)
                train_losses.append(metrics["loss"])
                if (batch_idx + 1) % self.log_every == 0:
                    print(
                        f"Epoch {epoch + 1} batch {batch_idx + 1}: "
                        f"loss={metrics['loss']:.4f} "
                        f"(det={metrics['det_loss']:.4f}, pred={metrics['pred_loss']:.4f}, "
                        f"ADE={metrics['ade']:.4f}, FDE={metrics['fde']:.4f}, "
                        f"heading={metrics['heading_loss']:.4f}, speed={metrics['speed_loss']:.4f})"
                    )
                if max_batches is not None and batch_idx + 1 >= max_batches:
                    break
            avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else float("nan")
            val_metrics = (
                self.validate()
                if self.val_loader is not None
                else {
                    "total": float("nan"),
                    "det": float("nan"),
                    "pred": float("nan"),
                    "ade": float("nan"),
                    "fde": float("nan"),
                    "heading": float("nan"),
                    "speed": float("nan"),
                }
            )
            print(
                f"Epoch {epoch + 1} done: train_loss={avg_train_loss:.4f}, "
                f"val_loss={val_metrics['total']:.4f}, val_det={val_metrics['det']:.4f}, "
                f"val_pred={val_metrics['pred']:.4f}, val_ADE={val_metrics['ade']:.4f}, val_FDE={val_metrics['fde']:.4f}, "
                f"val_heading={val_metrics['heading']:.4f}, val_speed={val_metrics['speed']:.4f}"
            )
            ckpt_path = self.checkpoints_dir / f"mvp_epoch_{epoch + 1}_val_{val_metrics['total']:.4f}.pt"
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": val_metrics["total"],
                },
                ckpt_path,
            )
            if self.best_val_loss is None or (not torch.isnan(torch.tensor(val_metrics["total"]))) and val_metrics["total"] < self.best_val_loss:
                self.best_val_loss = val_metrics["total"]
                best_path = self.checkpoints_dir / "mvp_best.pt"
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "epoch": epoch + 1,
                        "train_loss": avg_train_loss,
                        "val_loss": val_metrics["total"],
                    },
                    best_path,
                )


def _default_data_root() -> Path:
    """Resolve repository root and point to the bundled nuPlan-mini dataset."""

    return Path(__file__).resolve().parents[3] / "nuPlan-mini_dataset"


def main() -> None:
    """Tiny smoke-training entrypoint over real nuPlan-mini camera data."""

    board: TrainingBoard = get_default_board()

    parser = argparse.ArgumentParser(description="Run a tiny training loop on nuPlan-mini camera data.")
    parser.add_argument("--data-root", type=str, default=None, help="Path to nuPlan-mini root.")
    parser.add_argument("--batch-size", type=int, default=None, help="Dataloader batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Dataloader workers.")
    parser.add_argument("--max-batches", type=int, default=None, help="Number of batches to run per epoch.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs to train.")
    parser.add_argument("--device", type=str, default=None, help="Device string (e.g., 'cuda' or 'cpu').")
    args = parser.parse_args()

    if args.data_root is not None:
        board.data_root = args.data_root
    if args.batch_size is not None:
        board.batch_size = args.batch_size
    if args.num_workers is not None:
        board.num_workers = args.num_workers
    if args.max_batches is not None:
        board.max_batches = args.max_batches
    if args.epochs is not None:
        board.epochs = args.epochs
    if args.device is not None:
        board.device = args.device

    torch.manual_seed(board.seed)

    device_cfg = torch.device(board.device)
    print(f"Training on device: {device_cfg}")

    train_dataset = NuPlanMiniDataset(data_root=str(board.data_root), split="train")
    val_dataset = NuPlanMiniDataset(data_root=str(board.data_root), split="val")

    # Limit the dataset so the smoke pass stays fast.
    if board.max_batches is not None and board.max_batches > 0:
        subset_size = min(len(train_dataset), board.max_batches * max(board.batch_size, 1))
        train_dataset = torch.utils.data.Subset(train_dataset, list(range(subset_size)))  # type: ignore[arg-type]
        val_subset_size = min(len(val_dataset), max(1, board.max_batches))
        val_dataset = torch.utils.data.Subset(val_dataset, list(range(val_subset_size)))  # type: ignore[arg-type]

    collate_fn = lambda batch: batch[0]  # noqa: E731 - simple pass-through for single-sample batches
    train_loader = DataLoader(
        train_dataset, batch_size=board.batch_size, num_workers=board.num_workers, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=board.batch_size, num_workers=board.num_workers, collate_fn=collate_fn
    )
    model = MVPAutonomyModel().to(device_cfg)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device_cfg,
        log_every=1,
        checkpoints_dir=Path(board.checkpoints_dir),
        lr=board.lr,
        weight_decay=board.weight_decay,
        lambda_pred=board.lambda_pred,
    )
    max_batches = None if board.max_batches is None or board.max_batches < 0 else board.max_batches
    trainer.fit(epochs=board.epochs, max_batches=max_batches)


if __name__ == "__main__":
    main()
