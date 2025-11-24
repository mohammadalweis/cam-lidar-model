# System Overview
Status: Draft

## Purpose
Establish a first-principles view of the camera+lidar autonomous driving stack, clarify what “working” means for the MVP, and align the team on a pragmatic path from CARLA testing to future on-vehicle deployment.

## Scope
Covers the entire online loop from synchronized sensors through perception, prediction, planning, lightweight control, and their training/evaluation support systems; excludes low-level vehicle actuation firmware and large-scale data operations that will come after the MVP succeeds in CARLA.

## Key responsibilities
- Deliver a cohesive architecture where camera and lidar features are fused early enough to exploit complementary strengths (texture/color vs. geometry/range).
- Define module boundaries so perception, prediction, and planning can be iterated independently without breaking the single-GPU constraint.
- Provide a phased roadmap: Sensors → Perception → Prediction → Planning → Control → CARLA/Real Vehicle, with explicit handoffs and monitoring points.
- Capture success criteria (e.g., completes CARLA routes safely) and non-goals (no lidar-only or camera-only baselines, no giant fleet-scale training yet).

## Inputs and outputs
- **Inputs:** Time-synchronized camera frames (RGB fisheye pair for overlap) and lidar sweeps (64-beam range/intensity) plus ego pose and HD map priors.
- **Outputs:** Planned ego trajectory and high-level control commands (throttle/brake/steer) suitable for CARLA today and a vehicle interface later, along with telemetry for evaluation and retraining.
- Text diagram: `Sensors (Camera + Lidar + Pose) → Sensor Fusion & Perception → Prediction → Planning → Lightweight Control Adapter → CARLA / Real Vehicle`.

## Dependencies and interfaces
- Perception consumes calibrated sensor packets and exports object lists, drivable area, and occupancy grids to prediction.
- Prediction ingests perception tracks plus map context and emits multi-modal trajectory distributions for nearby agents.
- Planning consumes prediction results, road rules, and mission goals to generate a short-horizon trajectory; a control adapter converts that to CARLA’s API and later to CAN/proprietary vehicle commands.
- Training/evaluation pipelines provide shared feature encoders and checkpoint management but stay modular to enable swapping individual heads.

## Constraints / assumptions
- MVP must train and run inference on a single local GPU (≈16 GB), so models favor compact backbones, mixed precision, and limited sequence length.
- Initial dataset is modest (few dozen driving hours); data pipelines emphasize reproducibility over scale.
- Simplicity and clarity beat state-of-the-art metrics—the goal is “works reliably in CARLA” rather than leaderboard dominance.
- Roadmap assumes CARLA-first validation, then longer training runs with larger datasets, then hardware-in-the-loop, and finally controlled real-car testing.

## Open questions
- Exact sensor configuration in CARLA scenarios (number of cameras, lidar characteristics) and how closely we must mirror future real-car hardware.
- Criteria for “lightly trained” before moving to CARLA—number of epochs, target metrics, or qualitative behaviors.
- How much of the control interface can be shared between CARLA and the eventual vehicle (drive-by-wire specifics).
- Data governance for future real-world logs and how that will alter the single-GPU constraint.

## Notes for implementation
- Document concrete serialization formats for sensor packets early to avoid downstream refactors.
- Build module APIs with stubs/tests even before full models exist to enable parallel work by perception, prediction, and planning sub-teams.
- Keep CARLA integration scripts version-controlled so scenario regressions can be reproduced as the stack evolves.
