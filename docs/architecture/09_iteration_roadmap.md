# Iteration Roadmap
Status: Draft

## Purpose
Lay out the phased plan for taking the nuPlan-mini + CARLA-based autonomy stack from initial setup to a functional MVP and eventually toward real-car readiness, while respecting our single-GPU constraint and modular architecture.

## Scope
Covers six phases (environment setup, perception MVP, prediction MVP, planning/CARLA MVP, stabilization, future real-car work), each with entry/exit criteria, risks, and ties to the architecture docs; excludes broader company milestones or large-scale fleet deployment planning.

## Key responsibilities
- Provide a shared timeline with clear deliverables so teams can work in parallel (e.g., training pipeline vs. CARLA integration).
- Define measurable entry/exit criteria for each phase to avoid scope creep.
- Surface risks early (e.g., GPU limits, CARLA integration) and outline mitigation plans.
- Reference relevant design docs (00–08) to keep implementation aligned.

## Inputs and outputs
- **Inputs:** Architecture documents 00–08, nuPlan devkit/dataset, CARLA simulator, available hardware/resources.
- **Outputs:** Phase-by-phase status, Git milestones/tags, checklists, and retrospective notes feeding back into docs and requirements.

## Dependencies and interfaces
- Dependencies between phases: each phase builds on prior deliverables (e.g., prediction requires perception outputs, CARLA MVP requires planning).
- Interfaces with training pipeline (`docs/architecture/05_training_pipeline.md`), evaluation metrics (`06_evaluation_metrics.md`), and CARLA integration (`07_carla_integration.md`) to define success criteria.
- Coordination with dataset/requirements (08) ensures environment readiness before development starts.

## Constraints / assumptions
- Each phase should be achievable on a single 24 GB GPU workstation within days to a couple of weeks—short iterations favored over long waterfall cycles.
- Prioritize MVP functionality first, then quality improvements; avoid premature optimization.
- CARLA is the only closed-loop environment until Phase 5; real-car work only begins after CARLA MVP is stable.
- Resource availability: small research/engineering team with limited compute; no multi-node training.

## Open questions
- Do we need a lightweight interim milestone between Phase 3 and 4 (e.g., CARLA autopilot comparison) to de-risk stabilization?
- How do we handle unexpected nuPlan devkit updates mid-roadmap—freeze version or adapt per phase?
- What criteria trigger Phase 5 (real-car) kickoff—CARLA metrics, stakeholder approval, or hardware availability?
- Should we schedule time for documentation/tooling debt between phases?

## Notes for implementation
- Track progress via Git milestones/tags (e.g., `phase0_setup_complete`, `phase1_perception_mvp`, `phase3_carla_mvp`).
- Maintain a checklist in `docs/roadmap_checklist.md` or issue tracker with entry/exit items per phase.
- After each phase, hold a lightweight retro and update relevant docs (00–08) with lessons learned.
- Keep a single source of truth for phase status (e.g., README badge or tracker board) to align collaborators.

### Phase 0 – Environment & data setup
- **Entry criteria:** Hardware procured, repo cloned, team aligned on architecture docs.
- **Exit criteria:** nuPlan devkit installed, `NUPLAN_DATA_ROOT`/`MAPS_ROOT` set, nuPlan-mini (sensor blobs, maps) downloaded; `SensorBundle` builder and basic DataLoader implemented; sanity test runs (load 1 scenario, visualize all six cameras + lidar, confirm BEV grid).
- **Risks/mitigation:** Large dataset download delays (mitigate via checksums and staged downloads); environment drift (pin versions via `environment.yml`).
- **Documents:** `01_sensor_interfaces.md`, `05_training_pipeline.md`, `07_carla_integration.md`.

### Phase 1 – Perception MVP
- **Entry criteria:** Phase 0 complete, `SensorBundle` loader stable, training pipeline skeleton running.
- **Exit criteria:** Camera + lidar encoders, BEV fusion, and detection/occupancy heads implemented per `02_perception.md`; perception model trained on nuPlan-mini (batch size 2–4) with basic mAP/mIoU/occupancy metrics logged; inference runs in ≤50 ms on single GPU.
- **Risks/mitigation:** GPU memory pressure (mitigate via downsampling, mixed precision); incomplete camera coverage in devkit configs (update to include all six channels).
- **Deliverable:** “camera+lidar BEV baseline” checkpoint + evaluation report.
- **Documents:** `02_perception.md`, `05_training_pipeline.md`, `06_evaluation_metrics.md`.

### Phase 2 – Prediction MVP
- **Entry criteria:** Perception model producing stable BEV features/detections; training pipeline supports multi-task losses.
- **Exit criteria:** Prediction module (scene/actor encoders, interaction layer, trajectory heads) implemented per `03_prediction.md`; joint or staged training producing ADE/FDE < initial threshold (e.g., ADE < 2.0 m) on nuPlan-mini validation; predictions exported in agreed format.
- **Risks/mitigation:** Limited data may overfit (use regularization, data augmentation); integration complexity (start with frozen perception).
- **Deliverable:** Prediction head generating usable short-horizon forecasts with confidence scores.
- **Documents:** `03_prediction.md`, `05_training_pipeline.md`, `06_evaluation_metrics.md`.

### Phase 3 – Planning & CARLA closed-loop MVP
- **Entry criteria:** Perception + prediction modules operational; planning design finalized per `04_planning_control.md`; CARLA environment available.
- **Exit criteria:** Planning/control stack implemented (candidate generator, scorer, selector, PID control); full stack integrated into CARLA per `07_carla_integration.md`; runs on a curated set of routes completing ≥50% without severe infractions; logs/metrics collected.
- **Risks/mitigation:** CARLA synchronization issues (use synchronous stepping, watchdogs); control tuning challenges (start with simple lane-following scenarios).
- **Deliverable:** CARLA vehicle that can drive simple routes without constant crashing.
- **Documents:** `04_planning_control.md`, `07_carla_integration.md`, `06_evaluation_metrics.md`.

### Phase 4 – Stabilization & quality pass
- **Entry criteria:** CARLA MVP functional, baseline metrics recorded.
- **Exit criteria:** Tightened thresholds (perception mAP target x, prediction ADE target y, CARLA route completion ≥80%); regression suite automated (nightly CARLA runs); improved loss balancing, tuned cost weights, bug fixes logged and triaged.
- **Risks/mitigation:** Diminishing returns on quality (focus on biggest regressions first); runtime creep (monitor latency).
- **Deliverable:** Stable stack ready for broader testing, with dashboards tracking key metrics.
- **Documents:** `05_training_pipeline.md`, `06_evaluation_metrics.md`, `07_carla_integration.md`.

### Phase 5 – Toward real-car readiness (future)
- **Entry criteria:** Stabilized CARLA performance, stakeholder approval to proceed, preliminary hardware plan.
- **Exit criteria:** Roadmap for porting stack to real hardware (sensor rig, compute box), data collection loop defined (real-car logs → training pipeline), safety/logging requirements drafted; no actual on-road deployment yet, but plan + prototypes in place.
- **Risks/mitigation:** Hardware availability (plan early), regulatory considerations (engage safety stakeholders).
- **Deliverable:** Real-car readiness plan and prioritized backlog of tasks (calibration, hardware integration, safety).
- **Documents:** `00_system_overview.md`, `01_sensor_interfaces.md`, `08_system_requirements.md`.
