# Planning and Control
Status: Draft

## Purpose
Convert perception/prediction insights into executable ego trajectories and low-level control commands that keep the vehicle safe, smooth, and aligned with the intended route in CARLA and future real-car deployments.

## Scope
Covers short-horizon behavior planning (lane keeping, car-following, light braking/overtaking), trajectory generation and evaluation, emergency fallback logic, and a simple control adapter; excludes long-horizon mission planning or advanced model-predictive control (MPC) that may arrive later.

## Key responsibilities
- Maintain contextual awareness using ego-lane geometry, predicted agents, occupancy, and optional global route cues.
- Generate a compact library of candidate trajectories (3–5 s horizon) and score them for safety/comfort.
- Select a final trajectory that balances collision avoidance, lane adherence, and smooth control, with a deterministic fallback (stop-in-lane).
- Produce throttle, brake, and steering commands via a lightweight PID-based controller compatible with CARLA and future drive-by-wire interfaces.

## Inputs and outputs
- **Inputs:**
  - Predicted agent trajectories (`[N_agents, K, T, 4]`).
  - Occupancy grid / free-space map (aligned with BEV).
  - Ego-lane geometry and optional global route waypoints.
  - Ego state (pose, velocity, yaw rate) from localization/estimation.
- **Outputs:**
  - Final ego trajectory (`[T, 4]` states over 3–5 s).
  - Low-level control commands (steering, throttle, brake) at control frequency.
  - Debug metadata: candidate set, selected index, cost breakdown.
- Diagram: `Predictions + Occupancy + Lane → Route/Lane Context → Candidate Generator → Trajectory Scorer → Selector → PID Control`.

## Dependencies and interfaces
- Consumes prediction outputs and sensor-derived lane/occupancy from perception.
- Interacts with CARLA bridge (or future vehicle interface) via a control adapter that issues throttle/brake/steer commands.
- Feeds evaluation/logging modules with chosen trajectory plus cost components for offline analysis.
- Training/validation harness uses nuPlan-mini scenarios to benchmark closed-loop behavior; CARLA scripts run the same logic online.

## Constraints / assumptions
- Compute budget ~10–20 ms per planning cycle; candidate generation/scoring must remain lightweight (e.g., 16–32 trajectories).
- Rely primarily on ego-lane centerline and occupancy grid; map semantics beyond lane boundaries are optional for MVP.
- Planning ticks at 10 Hz (matching lidar) with control loop upsampled to 20–30 Hz via interpolation.
- Deterministic logic favored for reproducibility; stochastic components (e.g., random candidate seeding) disabled.

## Open questions
- Use polynomial quintic trajectories or spline-based templates for the candidate library?
- Is a simple PID controller sufficient for CARLA and initial vehicle testing, or should we plan to upgrade to MPC soon?
- Should we enable lane-change behavior in the first MVP, or focus purely on lane-following/car-following?
- What thresholds trigger emergency stop-in-lane vs. slower deceleration (e.g., occupancy probability, predicted collision time)?

## Notes for implementation
- Candidate generator can follow nuPlan-style kinematic rollout: sample target accelerations/curvatures, integrate with bicycle model, and store `[x, y, speed, heading]`.
- Trajectory scoring uses weighted sum: `J = w_collision * cost_collision + w_lane * cost_lane + w_accel * cost_accel + w_jerk * cost_jerk + w_route * cost_route`.
- Emergency fallback reuses zero-acceleration candidate clamped to lane center; ensure it is always available.
- Control adapter: simple PID on lateral error (steering) and longitudinal speed error (throttle/brake). For CARLA, convert to `VehicleControl` commands; for real car, map to CAN/Ackermann signals.
- Log per-candidate costs and final decision each tick for CARLA debugging; enable replay in nuPlan-mini offline evaluator.

### Submodules
1. **Route/lane context module**
   - Consumes ego-lane polyline + optional global waypoint queue; outputs reference path, curvature, and lookahead points.
   - Diagram: `Lane Geometry + Route → Context Module → Reference Path`.
2. **Candidate generator**
   - Produces 16–32 ego-centric trajectories using kinematic bicycle or polynomial fits over 3–5 s horizon.
   - Trajectories categorized (keep lane, slow down, slight lateral offset) to cover typical maneuvers.
   - Diagram: `Reference Path + Ego State → Generator → Candidate Set`.
3. **Trajectory scorer**
   - Evaluates each candidate against occupancy grid, predicted agent trajectories (nearest predicted positions), lane deviation, curvature, jerk.
   - Outputs scalar cost and diagnostics per candidate.
   - Diagram: `Candidate + Predictions + Occupancy → Cost Functions → Scores`.
4. **Selector**
   - Chooses minimum-cost feasible trajectory; if all infeasible, fallback to stop-in-lane.
   - Diagram: `Scores → Argmin → Selected Trajectory`.
5. **Control interface**
   - Converts selected trajectory into time-indexed setpoints; applies PID to compute steering/throttle/brake commands.
   - Diagram: `Selected Trajectory + Ego State → PID → Control Commands`.
