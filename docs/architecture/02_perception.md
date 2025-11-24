# Perception
Status: Draft

## Purpose
Transform six-camera RGB images and a 32-beam lidar sweep into a unified bird’s-eye-view (BEV) representation plus lightweight task heads so prediction/planning receive consistent semantic cues without reprocessing raw sensor data.

## Scope
Covers camera and lidar encoders, fusion strategy, BEV construction, and multi-head outputs (objects, lanes, occupancy, ego features) tuned for a single-GPU MVP; excludes long-term tracking, map building, or model scaling beyond nuPlan-mini–sized datasets.

## Key responsibilities
- Provide modular subcomponents—camera encoder, lidar encoder, fusion module, and downstream heads—that can be profiled and iterated independently.
- Deliver a BEV tensor with reasonable spatial resolution (~0.5 m grid over 100 m × 100 m) that prediction and planning can consume directly.
- Emit explicit detections, lane/ego geometry, and occupancy grids to minimize duplicated computation downstream.
- Maintain tight latency (≤50 ms perception budget) and memory usage so the entire stack fits on one GPU.

## Inputs and outputs
- Inputs: `SensorBundle` camera tensor (6×3×H×W), lidar voxel/point tensor, calib transforms, and ego pose.
- Outputs to prediction/planning:
  - BEV feature tensor: `float32[C_bev, H_bev, W_bev]` (e.g., C=128, H=W=200 for 100 m × 100 m @ 0.5 m).
  - Object list: bounding boxes with class, heading, velocity estimates.
  - Ego-lane geometry: centerline polyline plus curvature estimates.
  - Free-space/occupancy grid: binary/probabilistic map aligned with BEV resolution.

## Dependencies and interfaces
- Consumes `SensorBundle` defined by sensor interfaces, leveraging shared calibration utilities.
- Shares BEV feature tensor with prediction (for context) and planning (for ego alignment); object outputs follow a common schema with confidence scores and covariance approximations.
- Training pipeline must support multi-task losses; evaluation pipeline reads detection/occupancy metrics from this module.

## Constraints / assumptions
- Operates within a single 24 GB GPU: encoders use lightweight backbones (e.g., ResNet-18/34 variants for cameras, sparse convolution or PointPillars-style encoder for lidar).
- BEV grid limited to 200×200 (0.5 m) initially; can shrink to 0.75 m if memory requires.
- Batch size for training likely 2–4; inference runs per-frame with mixed precision.
- Camera images downsampled (≤960×540) and lidar voxels coarse (0.2–0.3 m) to stay within budget.

## Open questions
- What BEV resolution and channel count best balance fidelity and GPU memory (0.5 m vs. 0.75 m, 64 vs. 128 channels)?
- Should we bootstrap with camera-only training before adding lidar to simplify debugging?
- How many object classes do we support in the MVP (vehicles only vs. vehicles + pedestrians/cyclists)?
- Do we need temporal smoothing (feature queue) before prediction, or is single-frame perception sufficient?

## Notes for implementation
- Reuse backbone weights from open-source nuScenes/nuPlan-style models to accelerate convergence, but strip unused layers to keep parameter counts low.
- Keep fusion interfaces generic (e.g., `bev_features = fuse(camera_bev, lidar_bev)`) so alternative strategies can swap in later.
- Log per-module latency and memory usage during CARLA replay to catch regressions early.

### Submodules
1. **Camera encoder**
   - Six images processed by a shared-weight CNN (e.g., ResNet-34) with lightweight FPN.
   - Projection: features lifted to BEV via camera intrinsics/extrinsics and depth hypotheses (simple learned depth bins, 4–6 layers).
   - Diagram: `6 × Image → Shared CNN → Feature Pyramid → Lift-Splat (intrinsics/extrinsics) → Camera BEV tensor`.
2. **Lidar encoder**
   - Use PointPillars-style pillarization (0.2 m × 0.2 m) to convert point cloud into pseudo-image.
   - Apply sparse CNN or 2D CNN to produce lidar BEV tensor aligned with camera BEV grid.
   - Diagram: `Point Cloud → Pillarization → Pseudo-image → Sparse/2D CNN → Lidar BEV tensor`.
3. **Fusion module**
   - MVP strategy: aligned BEV tensors are concatenated along channels, then passed through a shallow residual block to mix modalities.
   - Justification: avoids complex attention mechanisms while still leveraging complementary cues; simple channel concat keeps compute predictable.
   - Diagram: `Camera BEV || Lidar BEV → Residual BEV Fusion Block → Unified BEV`.
4. **Head modules**
   - **Object detection head:** anchor-free BEV detection (center heatmap + size/velocity regressions).
   - **Lane segmentation head:** predicts ego-lane raster plus centerline offsets; outputs polylines via post-processing.
   - **BEV occupancy head:** binary/probabilistic free-space grid with dilation for safety margins.
   - **Ego-state head:** estimates ego velocity, yaw rate, and uncertainties for planners lacking direct CAN feeds.
   - Diagram: `Unified BEV → {Detection Head, Lane Head, Occupancy Head, Ego-state Head}`.

### BEV representation
- Spatial extent: 100 m forward/back, 50 m lateral each side (200×200 grid at 0.5 m resolution).
- Channels: start with 128 shared channels post-fusion; individual heads derive task-specific logits/features.
- Coordinate frame: ego-centered, x-forward, y-left, consistent with prediction/planning expectations.
- Normalization: store normalization parameters (mean/std) for each channel to ease training/inference alignment.
