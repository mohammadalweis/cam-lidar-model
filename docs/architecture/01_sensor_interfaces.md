# Sensor Interfaces
Status: Draft

## Purpose
Define how raw camera and lidar streams are captured, aligned, calibrated, and converted into compact tensors so downstream perception models can focus on learning rather than bookkeeping.

## Scope
Covers the MVP sensor hardware (or CARLA equivalents), approximate time synchronization, image/point-cloud preprocessing, calibration handling, and the structure of the unified packet delivered to perception; excludes low-level driver development and large-scale logging infrastructure.

## Key responsibilities
- Ensure six synchronized RGB cameras (front, front-left, front-right, back, back-left, back-right) and a single lidar stream remain approximately aligned (target: ±30–50 ms skew) using lightweight timestamp matching suitable for a single-GPU setup.
- Apply consistent preprocessing pipelines per sensor so training/offline replay match live execution.
- Produce a normalized data packet (`SensorBundle`) containing tensors, calibration blobs, and metadata required by perception/prediction.
- Surface sanity metrics (dropped frames, latency) for monitoring but defer advanced health checks to future iterations.

## Inputs and outputs
- **Camera inputs:** Six synchronized RGB streams (front, front-left, front-right, back, back-left, back-right) at ~12 Hz (as in nuPlan-mini). Native resolution is on the order of ~1600×900; for the MVP we downsample to something like 960×540 or 640×384 to control memory and compute.
  - Diagram: `6 × Camera Frame → Undistort/Rectify → Resize (e.g., 960×540) → Normalize (mean/std) → Tensor (6, 3, H, W)`.
- **Lidar inputs:** Single 64-beam spinning lidar at 10 Hz with intensity values.
  - Diagram: `Raw Point Cloud → Motion Comp (optional) → Ground Removal (ROI crop) → Voxelize (0.2 m) → Tensor (Vx, Vy, Vz, features)`.
- **Unified output (`SensorBundle`):**
  ```
  {
    timestamp: float64,
    camera_tensor: float32[6,3,H,W],
    lidar_tensor: float32[Vx,Vy,Vz,F],
    lidar_points_raw: float32[N,5]  # optional passthrough
    ego_pose: SE3,
    calibrations: {
      camera_intrinsics: [6][3x3],
      camera_extrinsics: [6]SE3_vehicle,
      lidar_extrinsic: SE3_vehicle
    },
    metadata: {frame_id, drop_flags}
  }
  ```

## Dependencies and interfaces
- Upstream: sensor drivers or CARLA bridges provide timestamped frames and calibration blobs; dataset loaders must serialize the same structures.
- Downstream: perception expects standardized tensor shapes and SE(3) transforms; training pipelines consume archived `SensorBundle` sequences; diagnostics read metadata for frame-drop detection.
- Calibration store (YAML/JSON) provides intrinsics/extrinsics; versioning handled outside this module but referenced here.

## Constraints / assumptions
- Approximate synchronization uses software-based message filters—no hardware trigger wiring assumed.
- Preprocessing must fit in the single GPU budget: lightweight CUDA kernels or CPU operations that keep per-frame latency under ~10 ms for cameras and ~25 ms for lidar.
- Six-camera preprocessing must remain lightweight; we will batch operations and aggressively downsample images (e.g., to 960×540 or 640×384) to stay within a single-GPU budget.
- Intrinsics/extrinsics treated as static during a run; recalibration happens offline.
- CARLA-specific sensors map 1:1 to this design via blueprint configs; deviations (e.g., different beam counts) require padding/cropping to maintain tensor shapes.

## Open questions
- Do we require motion-compensated lidar sweeps, or is raw spinning data adequate for the MVP?
- Should we include additional modalities (e.g., radar, IMU) in `SensorBundle` placeholders now or wait until the MVP works?
- What tolerances trigger resync/restart (e.g., camera-lidar skew > 50 ms) and how do we report that upstream?
- Is the voxel grid resolution (0.2 m) sufficient for close-proximity planning, or do we need adaptive schemes?

## Notes for implementation
- Implement preprocessing as deterministic functions callable from both online bridge and offline dataset builder to avoid drift.
- Store calibration in a versioned manifest (e.g., `calib/<date>/sensor.yaml`) and embed a hash in each `SensorBundle` to trace provenance.
- For CARLA, script sensor spawning so poses and intrinsics mirror our target rig; log blueprint parameters alongside each run for reproducibility.
- nuPlan-mini provides per-sensor calibration (intrinsics/extrinsics) for all six cameras and the lidar; the dataset loader is responsible for reading these files and packaging them into the `SensorBundle` calibration fields.
