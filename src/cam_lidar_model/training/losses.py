"""Placeholder loss functions for multi-task training."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def focal_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """Compute focal loss for heatmaps as in CenterNet."""

    pos_inds = target.eq(1)
    neg_inds = target.lt(1)
    pos_loss = torch.log(pred.clamp(min=1e-6)) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = torch.log((1 - pred).clamp(min=1e-6)) * torch.pow(pred, alpha) * torch.pow(1 - target, beta) * neg_inds
    num_pos = pos_inds.float().sum()
    if num_pos > 0:
        loss = -(pos_loss.sum() + neg_loss.sum()) / num_pos
    else:
        loss = -neg_loss.sum()
    return loss


def _build_detection_targets(
    predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create dense targets (heatmap, offsets, sizes, yaw, mask) from sparse box annotations."""

    heatmap = predictions["heatmap"]
    b, _, h, w = heatmap.shape
    device = heatmap.device
    dtype = heatmap.dtype

    target_heatmap = torch.zeros((b, 1, h, w), device=device, dtype=dtype)
    target_offsets = torch.zeros((b, 2, h, w), device=device, dtype=dtype)
    target_sizes = torch.zeros((b, 2, h, w), device=device, dtype=dtype)
    target_yaw = torch.zeros((b, 1, h, w), device=device, dtype=dtype)
    mask = torch.zeros((b, 1, h, w), device=device, dtype=dtype)

    boxes = targets.get("boxes")
    if boxes is None:
        return target_heatmap, target_offsets, target_sizes, target_yaw, mask

    if boxes.ndim == 2:
        boxes = boxes.unsqueeze(0)

    for i in range(b):
        boxes_i = boxes[i] if i < boxes.shape[0] else boxes[0]
        if boxes_i.numel() == 0:
            continue
        centers_x = boxes_i[:, 0]
        centers_y = boxes_i[:, 1]
        sizes = boxes_i[:, 3:5]  # width, length
        yaw = boxes_i[:, 6]

        grid_x = torch.clamp(centers_x, min=0, max=w - 1 - 1e-3)
        grid_y = torch.clamp(centers_y, min=0, max=h - 1 - 1e-3)
        gx_int = grid_x.long()
        gy_int = grid_y.long()
        offsets = torch.stack([grid_x - gx_int.float(), grid_y - gy_int.float()], dim=1)

        for j in range(boxes_i.shape[0]):
            x_idx = gx_int[j]
            y_idx = gy_int[j]
            target_heatmap[i, 0, y_idx, x_idx] = 1.0
            target_offsets[i, :, y_idx, x_idx] = offsets[j]
            target_sizes[i, :, y_idx, x_idx] = sizes[j]
            target_yaw[i, 0, y_idx, x_idx] = yaw[j]
            mask[i, 0, y_idx, x_idx] = 1.0

    return target_heatmap, target_offsets, target_sizes, target_yaw, mask


def detection_loss(predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Compute BEV detection loss using heatmap BCE."""

    pred_heatmap = predictions if isinstance(predictions, torch.Tensor) else predictions["heatmap_logits"]
    target_heatmap = targets
    if isinstance(targets, dict):
        target_heatmap = targets.get("bev_heatmap", None)
    if target_heatmap is None:
        return torch.tensor(0.0, device=pred_heatmap.device)
    if target_heatmap.dim() == 3:
        target_heatmap = target_heatmap.unsqueeze(1)
    target_heatmap = target_heatmap.to(dtype=pred_heatmap.dtype)
    if pred_heatmap.shape != target_heatmap.shape:
        raise ValueError(f"Pred heatmap shape {pred_heatmap.shape} != target {target_heatmap.shape}")
    return F.binary_cross_entropy_with_logits(pred_heatmap, target_heatmap)


def segmentation_loss(predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Compute lane segmentation loss placeholder."""

    return torch.tensor(0.0, device=predictions["lane"].device)


def occupancy_loss(predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Compute occupancy loss placeholder."""

    return torch.tensor(0.0, device=predictions["occupancy"].device)


def _wrap_angle_diff(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute wrapped angle difference."""

    diff = pred - target
    return torch.atan2(torch.sin(diff), torch.cos(diff))


def prediction_loss(
    prediction_outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute trajectory losses (position, heading, speed) plus ADE/FDE in ego frame."""

    if "traj" not in prediction_outputs:
        device = next(iter(prediction_outputs.values())).device if prediction_outputs else torch.device("cpu")
        zero = torch.tensor(0.0, device=device)
        return zero, zero, zero, zero, zero, zero

    traj_pred = prediction_outputs["traj"]
    device = traj_pred.device

    if traj_pred.ndim == 5:
        traj_pred = traj_pred[0]
    if traj_pred.ndim == 4:
        traj_pred = traj_pred[:, 0]  # use first mode (K=1)

    target_traj = targets.get("trajectories")
    if target_traj is None:
        zero = torch.tensor(0.0, device=device)
        return zero, zero, zero, zero, zero, zero
    if target_traj.ndim == 2:
        target_traj = target_traj.unsqueeze(0)
    if target_traj.ndim == 4:
        target_traj = target_traj[0]

    if traj_pred.numel() == 0 or target_traj.numel() == 0:
        zero = torch.tensor(0.0, device=device)
        return zero, zero, zero, zero, zero, zero

    target_traj = target_traj.to(device)
    num = min(traj_pred.shape[0], target_traj.shape[0])
    if num == 0:
        zero = torch.tensor(0.0, device=device)
        return zero, zero, zero, zero, zero, zero

    T = min(traj_pred.shape[1], target_traj.shape[1])
    if T == 0:
        zero = torch.tensor(0.0, device=device)
        return zero, zero, zero, zero, zero, zero

    traj_pred = traj_pred[:num, :T]
    target_traj = target_traj[:num, :T]

    dist = torch.norm(traj_pred[..., :2] - target_traj[..., :2], dim=-1)
    ade = dist.mean() if dist.numel() > 0 else torch.tensor(0.0, device=device)
    fde = dist[:, -1].mean() if dist.shape[1] > 0 else torch.tensor(0.0, device=device)

    pos_loss = dist.mean() if dist.numel() > 0 else torch.tensor(0.0, device=device)
    heading_err = _wrap_angle_diff(traj_pred[..., 2], target_traj[..., 2]).abs()
    heading_loss = heading_err.mean() if heading_err.numel() > 0 else torch.tensor(0.0, device=device)
    speed_loss = (traj_pred[..., 3] - target_traj[..., 3]).abs().mean() if traj_pred.numel() > 0 else torch.tensor(
        0.0, device=device
    )

    total_loss = pos_loss * 1.0 + heading_loss * 0.5 + speed_loss * 0.5
    return total_loss, ade, fde, pos_loss, heading_loss, speed_loss
