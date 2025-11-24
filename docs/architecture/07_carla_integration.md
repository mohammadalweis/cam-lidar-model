# CARLA Integration
Status: Draft

## Purpose
Describe how the end-to-end MVP stack (sensor interface → perception → prediction → planning → control) operates inside CARLA in closed loop, ensuring we can validate functionality before moving toward real-vehicle testing.

## Scope
Covers CARLA sensor configuration, data bridging, model inference, control command generation, closed-loop timing, logging, and scenario orchestration; excludes physical vehicle drivers or large-scale cloud simulation farms.

## Key responsibilities
- Spawn CARLA sensors matching the six-camera + one-lidar nuPlan-mini rig and feed them into the `SensorBundle` pipeline.
- Run the perception/prediction/planning stack online at ~10–20 Hz, producing control commands every tick.
- Handle synchronization, dropped frames, and emergency fallbacks (stop-in-lane) within the simulator.
- Capture logs (sensor data, BEV outputs, predictions, commands, infractions) for offline evaluation.

## Inputs and outputs
- **Inputs (from CARLA):** RGB frames (6 sensors), lidar point clouds, ego pose, map/lane topology, traffic manager state.
- **Outputs (to CARLA):** throttle, brake, steering commands (Ackermann-like control), optional debug overlays, recorded logs (video, BEV tensors, events).
- Diagram: `CARLA Sensors → Bridge → SensorBundle → Perception → Prediction → Planning → Control Adapter → CARLA Vehicle`.

## Dependencies and interfaces
- Relies on CARLA Python API for synchronous stepping, sensor callbacks, and vehicle control.
- Uses sensor interface code to convert CARLA data into `SensorBundle` objects (same as training input format).
- Integrates with model runtime (PyTorch) for perception/prediction inference and planning logic for trajectory/control output.
- Interfaces with evaluation pipeline to hand over logs/metrics after each run.

## Constraints / assumptions
- Real-time target: 10 Hz planning pipeline, 20–30 Hz control actuation via interpolation; total latency per tick ≤100 ms.
- Sensors approximate nuPlan-mini but need not match perfectly—CARLA extrinsics/intrinsics manually configured and stored per scenario.
- Ego vehicle controlled solely by our stack; NPC traffic uses CARLA traffic manager/autopilot.
- Focus on lane following and collision avoidance; high-level routing limited to pre-defined waypoint lists.

## Open questions
- Should we build a custom CARLA scenario suite mirroring nuPlan-mini scenario structure for apples-to-apples evaluation?
- How much lidar fidelity (noise, beam divergence) do we need to emulate nuPlan-mini? Is CARLA’s default 32/64-beam sufficient?
- Do we replay collected real-world logs inside CARLA (sensor replay mode) to test consistency?
- Which CARLA stepping mode (synchronous vs. asynchronous) provides the best trade-off between determinism and throughput for MVP?

## Notes for implementation
- Use CARLA Python API with synchronous stepping to ensure deterministic sensor timestamps; fall back to async with buffering if real-time rate is insufficient.
- Implement bridge as set of sensor callbacks pushing data into async queues; a central loop pops synchronized frames, builds `SensorBundle`, and runs inference.
- Provide script entrypoint `run_carla_closed_loop.py` with CLI params for town, weather, route file, checkpoint path, and logging directory.
- Utilize CARLA’s `VehicleControl` (steer, throttle, brake) interface; for planning output in SE(2), convert to desired curvature/speed then feed PID to produce commands.
- Logging: store sensor video, BEV snapshots, prediction trajectories, selected trajectories, control commands, infractions, and system latency traces for each run.

### CARLA sensor bridge
- Spawn six RGB cameras roughly matching nuPlan rig (front, front-left/right, back, back-left/right). Use `sensor.camera.rgb` with resolution ~1280×720 (downsampled to 960×540) and FOV similar to nuPlan sensors.
- Lidar: `sensor.lidar.ray_cast` configured for 32 beams (nuPlan-mini style) at 10 Hz; record intensity and range.
- Each sensor callback attaches timestamp and sensor transform; calibration stored in config file and embedded in `SensorBundle`.
- Handle timestamp alignment via CARLA synchronous mode; intrinsics derived from CARLA camera parameters (focal length, principal point) and stored alongside extrinsics relative to ego chassis.

### Real-time data flow
- Sensor callbacks enqueue raw frames/point clouds → bridge thread waits until data from all sensors with matching tick arrives (±1 tick tolerance).
- Build `SensorBundle`, run preprocessing (undistort/resize/pillarize), and feed to perception model.
- Pipeline target: perception (50 ms) + prediction (20 ms) + planning/control (20 ms) to maintain 10 Hz; control commands interpolated to 20–30 Hz using short-horizon trajectory.
- Diagram: `Camera/Lidar Streams → Sync Queue → Preprocess → SensorBundle → Perception → Prediction → Planning → Control Adapter → CARLA`.

### Model integration
- Load perception/prediction checkpoints from disk (config-driven). Models run in eval mode, mixed precision if supported.
- Each tick: pass `SensorBundle` through perception to get BEV + detections, feed into prediction for trajectories, hand to planning for candidate evaluation and control generation.
- Control adapter translates final trajectory + ego state into throttle/brake/steer via PID and sends commands to CARLA vehicle.
- Monitor inference latency; if backlog occurs, drop frames strategically (e.g., skip camera frame) but keep lidar aligned.

### Closed-loop operation
- Planning loop runs at sensor tick (10 Hz). Control loop runs at 20–30 Hz using interpolated setpoints from selected trajectory.
- If frames drop or latency > threshold, trigger “safe mode”: maintain last known command or command gentle braking.
- Emergency fallback: stop-in-lane trajectory enforced when prediction indicates imminent collision or data stale (>200 ms).
- Keep watchdog timer to ensure CARLA synchronous stepping doesn’t stall; abort run if simulator lags.

### Evaluation and logging
- Record: camera feeds (compressed video), lidar sweeps (binary), BEV tensors, predictions, selected trajectories, control commands, and CARLA event logs (infractions).
- After each run, package logs + metrics (route completion, infractions, latency stats) for evaluation pipeline in `reports/carla/<date>/<scenario>`.
- Optional video overlay (perception visualization) for debugging; store only for failed runs to save space.

### Simulation configuration
- Supported towns: Town03, Town05, Town10 for variety; weather presets (ClearNoon, HardRainNoon) defined per regression suite.
- Traffic manager parameters: moderate traffic density, autopilot speeds capped (e.g., 30 mph) to keep scenarios manageable.
- Sensor spawn transforms tuned to approximate nuPlan rig; store configuration file per town/vehicle.
- Use deterministic seeds for CARLA world, traffic manager, and sensor noise to reproduce results.
