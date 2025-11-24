# Training Pipeline
Status: Draft

## Purpose
Provide a reproducible, resource-aware pipeline for training and validating the perception and prediction modules (with hooks for planning logic evaluation) so the MVP stack can be iterated quickly on a single GPU.

## Scope
Covers dataset ingest (nuPlan-mini), preprocessing into `SensorBundle`s, multi-task training loops, checkpointing/evaluation, and minimal automation for CARLA closed-loop regression; excludes large-scale distributed training and full-scale data lake management (future work).

## Key responsibilities
- Convert nuPlan-mini raw samples (six cameras, one lidar, ego pose, ground-truth trajectories) into standardized training inputs.
- Run multi-task training spanning detection, segmentation, occupancy, and trajectory prediction heads with balanced losses.
- Monitor key metrics (mAP, mIoU, ADE/FDE, closed-loop success) and checkpoint models with sufficient metadata for replay.
- Maintain consistent configs, logging, and reproducibility so experiments can be compared across iterations.

## Inputs and outputs
- **Inputs:** nuPlan-mini scenarios (sensor logs + annotations), configuration files (YAML), pretrained backbone weights (optional).
- **Outputs:** model checkpoints (weights, optimizer state, config snapshot, git commit), training logs (metrics per step), lightweight evaluation reports, CARLA replay artifacts.
- Diagram: `Raw Dataset → SensorBundle Builder → Batching → Encoders/Heads → Loss Functions → Optimizer/Checkpoint`.

## Dependencies and interfaces
- Depends on sensor interface definitions (`SensorBundle` schemas) and perception/prediction model implementations.
- Interfaces with experiment tracking (e.g., TensorBoard, Weights & Biases) for metrics; interacts with CARLA harness for closed-loop evaluations triggered periodically.
- Uses filesystem structure `data/nuplan_mini/…` for raw/logged data; outputs stored in `checkpoints/<experiment>` with symlinks to latest.

## Constraints / assumptions
- Single 24 GB GPU; batch size 2–4 due to six high-res cameras and BEV memory footprint.
- Mixed-precision (FP16) training and gradient accumulation (if batch<2) to utilize GPU efficiently.
- Data loader must keep GPU fed: ≥4 worker processes, pinned memory, caching precomputed intrinsics/extrinsics.
- Preprocessing performed on-the-fly but cached for repeated epochs when feasible (e.g., store lidar pillars, depth bins).

## Open questions
- Should lidar-specific augmentations (noise, dropout) be enabled early or after baseline convergence?
- How will we scale from nuPlan-mini to full nuPlan or custom datasets—do we need sharding utilities now?
- Do we incorporate planning imitation losses (e.g., expert trajectory matching) in the training loop, or keep planning evaluation separate?
- What cadence should CARLA closed-loop regression run (every N epochs vs. manual trigger)?

## Notes for implementation
- Recommend PyTorch Lightning or a lightweight custom trainer with clear hooks for multi-task losses and mixed-precision.
- Configuration-driven setup (YAML + CLI overrides) to capture model architecture, optimizer, loss weights, and data paths.
- Set deterministic seeds, control `torch.backends.cudnn` flags when reproducibility is critical; log random seeds in checkpoints.
- Provide scripts for dataset download/verification, dataset-to-`SensorBundle` conversion, and training entrypoints.

### Dataset structure and loading
- nuPlan-mini provides six camera streams, one 32-beam lidar, ego pose, and annotations for agents/lanes.
- Raw samples converted to `SensorBundle` objects via preprocessing pipeline (undistortion, resize, voxelization) stored either on-disk (LMDB/NPZ) or generated on-the-fly.
- Scenario-based iteration: each scenario (~20 s) sliced into overlapping windows; training split uses ~80%, validation 20% with city diversity.

### Preprocessing pipeline
- **Images:** resize to 960×540 (or 640×384 for memory), normalize per-channel mean/std, apply light augmentations (color jitter, horizontal flip for symmetric cases, random brightness).
- **Lidar:** pillarization at 0.2 m grid, intensity normalization, optional random point dropout; convert to BEV pseudo-image.
- **BEV grid:** align camera- and lidar-derived features to 0.5 m resolution grid (200×200). Precompute depth bins or frustum data for lift-splat to speed up training.
- Augmentation toggles stored in config; ensure deterministic augmentations for validation.

### Training loop responsibilities
- Multi-task objectives:
  - Detection (focal loss + L1 for box params).
  - Lane/segmentation (binary cross-entropy + dice).
  - Occupancy/free-space (binary cross-entropy).
  - Prediction (ADE/FDE losses, Gaussian NLL for modes).
- Loss balancing via static weights initially (configurable), with option to enable uncertainty-based weighting later.
- Mixed-precision training (AMP) and gradient accumulation (e.g., accumulate 2 steps when batch=2) to stabilize updates.
- Scheduler (cosine or step) and optimizer (AdamW) configured via YAML.

### Batching and performance
- Batch size 2–4 depending on image resolution; scenes grouped by similar length to minimize padding.
- DataLoader: 4–8 workers, prefetch factor 2, persistent workers for long runs.
- Cache heavy preprocessing outputs (e.g., camera intrinsics/extrinsics, voxelized lidar) in RAM/disk to reduce CPU bottlenecks.
- Monitor GPU utilization; record throughput (samples/s) per experiment for regression.

### Checkpoints and versioning
- Each checkpoint includes: model state dict, optimizer/scheduler states, scaler state (if AMP), config snapshot, git hash, dataset manifest hash.
- Save every N epochs or when validation metric improves; keep top-K checkpoints.
- Run lightweight validation (perception/prediction metrics) automatically post-checkpoint.
- Integrate hooks to trigger CARLA closed-loop tests using latest checkpoint (manual or scheduled) and archive results.

### Evaluation metrics
- Perception: mAP for detection, mIoU for lane/segmentation, occupancy accuracy/recall.
- Prediction: ADE/FDE, miss rate at 2 m, mode recall/precision, collision rate vs. ground truth.
- Closed-loop: CARLA route completion %, infractions/interventions count, ego comfort metrics (max accel/jerk).
- Metrics logged per epoch and surfaced on dashboards; validation scripts shared with CI.

### Distributed training (future)
- MVP runs single-GPU; design code with hooks for PyTorch DDP (e.g., `Trainer` with optional `accelerator=ddp`).
- Data loaders should support distributed sampling; ensure `SensorBundle` caching works across workers.
- Keep configuration fields for multi-GPU world size, gradient synchronization, etc., even if unused initially.
