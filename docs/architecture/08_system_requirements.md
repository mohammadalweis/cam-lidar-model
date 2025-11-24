# System Requirements
Status: Draft

## Purpose
Capture the concrete hardware, software, and dataset requirements needed to run the MVP autonomy stack (sensor interface → perception → prediction → planning → CARLA) on top of the nuPlan devkit and nuPlan-mini dataset currently checked into this repository.

## Scope
Includes single-GPU training/evaluation targets, data/layout expectations (nuPlan-mini folders, env vars), CARLA simulation support, and devkit compatibility constraints; excludes future large-scale distributed training or on-vehicle hardware specifications beyond the MVP.

## Key responsibilities
- Ensure the stack can be trained and evaluated on a single 24 GB GPU workstation with SSD storage.
- Align perception/prediction models with nuPlan-mini sensor formats (six cameras + one lidar) while acknowledging current devkit defaults (three-camera configs, `include_cameras=false`).
- Define required software versions (Python 3.9, PyTorch+CUDA, CARLA build) and environment variables (`NUPLAN_DATA_ROOT`, `NUPLAN_MAPS_ROOT`, etc.).
- Outline data/logging expectations so outputs (checkpoints, CARLA runs) can be reproduced and shared.

## Inputs and outputs
- **Inputs:** Workstation with ≥24 GB VRAM GPU (e.g., RTX 3090/4090), 16+ CPU cores, 64+ GB RAM, fast SSD with ≥400 GB free (nuPlan-mini sensor blobs + maps + checkpoints); OS with CUDA 11.7+ drivers; nuPlan devkit cloned (current repo), CARLA installation (v0.9.14+), dataset directories (`dataset/nuplan-v1.1_mini*`).
- **Outputs:** Functioning training/eval pipeline producing checkpoints and metric reports, CARLA closed-loop harness capable of executing the stack, and artifact storage for logs/videos under `reports/`.

## Dependencies and interfaces
- **nuPlan devkit:** `nuplan-devkit` directory shipped with repo (v1.2.x per docs/README). Uses Hydra configs such as `nuplan/planning/script/config/common/scenario_builder/nuplan_mini.yaml`, which expects env vars and currently sets `include_cameras: false`.
- **Dataset:** `dataset/nuplan-v1.1_mini`, `nuplan-v1.1_mini_camera_0`, `nuplan-v1.1_mini_lidar_0`, and `nuplan-maps-v1.0`; configs like `configs/data/nuplan_mini_multisensor.yaml` specify camera channels (`CAM_F0`, `CAM_L0`, `CAM_R0`), resize (384×640), BEV ranges, batch size (2). Requirements must bridge from these defaults to the 6-camera design (need to extend configs to include `CAM_B0`, `CAM_BL0`, `CAM_BR0` and set `include_cameras=true`).
- **CARLA:** Version aligned with planning module (target v0.9.14 or later). Control adapter must speak CARLA `VehicleControl`. Sensor rig configured via CARLA Python API to mirror nuPlan extrinsics (approximate transforms acceptable for MVP).
- **Software stack:** Python 3.9 (per devkit docs), PyTorch ≥1.13 with CUDA 11.7, cuDNN 8.x, Hydra/OmegaConf versions pinned by devkit `requirements*.txt`.

## Constraints / assumptions
- Entire training/inference workload fits within a single GPU with ≤24 GB VRAM; typical batch size 2–4 due to six-camera tensors.
- CPU-bound preprocessing relies on 16+ logical cores; DataLoader uses 4–8 workers and pinned memory.
- nuPlan-mini remains the canonical dataset; additional datasets optional but not required for MVP.
- CARLA regression suite limited to ~10 routes per nightly run to keep runtime manageable (<2 hours).
- Devkit APIs assumed stable (v1.2.x); major upgrades require revalidation of configs and dataset schema.
- SSD storage required for acceptable I/O; HDD not supported due to throughput constraints when streaming six cameras + lidar.

## Open questions
- How do we handle devkit updates (e.g., nuPlan v1.3) that may change scenario builder defaults or sensor schema?
- Do we need to support alternative datasets (e.g., nuScenes) in the same pipeline, or is nuPlan-mini sufficient for MVP validation?
- Should collaborators with smaller GPUs (≤16 GB) have a reduced-resolution configuration, or do we enforce 24 GB minimum?
- Is CARLA v0.9.14 sufficient long term, or do we need compatibility with v0.9.15+ for features like Vulkan rendering?

## Notes for implementation
- Document hardware/software requirements in `README.md` or `docs/setup.md`, explicitly listing GPU/CPU/RAM/disk expectations and required env vars (`NUPLAN_DATA_ROOT`, `NUPLAN_MAPS_ROOT`, `NUPLAN_DB_FILES`).
- Provide `requirements.txt`/`environment.yml` derived from `nuplan-devkit/requirements*.txt` plus CARLA/PyTorch pins.
- Author an `env_check.py` script verifying CUDA availability, GPU memory, nuPlan dataset paths, map folders, and CARLA install/version.
- Extend nuPlan configs to enable camera blobs (`include_cameras=true`) and all six camera channels; mirror these in training configs for BEV pipeline.
- Maintain a compatibility matrix (devkit commit, dataset version, CARLA version) in the docs to track validated combinations.
