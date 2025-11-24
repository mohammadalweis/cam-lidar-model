# Evaluation and Metrics
Status: Draft

## Purpose
Define the quantitative framework that determines whether perception, prediction, and closed-loop planning meet MVP expectations, enabling objective comparisons between iterations and guiding future improvements.

## Scope
Covers offline metrics on nuPlan-mini validation splits (perception + prediction), online/closed-loop metrics from CARLA regression routes, and reporting infrastructure that aggregates results for gating releases; excludes large-scale fleet analytics.

## Key responsibilities
- Compute standardized perception metrics (detection, lanes, occupancy) after each training run.
- Evaluate prediction quality (ADE/FDE, collision rate, uncertainty calibration) on held-out scenarios.
- Collect closed-loop CARLA metrics (route completion, infractions, comfort, latency) and track regressions.
- Aggregate metrics into dashboards/JSON reports with pass/fail thresholds for MVP milestones.

## Inputs and outputs
- **Inputs:** model checkpoints, nuPlan-mini validation data, CARLA simulation logs, configuration metadata.
- **Outputs:** metrics database/JSON, plots (precision-recall, ADE distributions), CARLA run summaries, alerts when thresholds are violated.
- Diagram: `Checkpoint + Validation Data → Metric Computation → Aggregator → Dashboards/Reports`.

## Dependencies and interfaces
- Interfaces with training pipeline to fetch latest checkpoints and dataset manifests.
- Hooks into CARLA harness to collect runtime stats (infractions, latency) post-run.
- Uses experiment tracking tools (TensorBoard/W&B) or custom dashboards for visualization.
- CI pipeline may trigger evaluation scripts after nightly training jobs.

## Constraints / assumptions
- Validation splits must be representative: diverse cities/time-of-day; run metrics with fixed seeds for determinism.
- CARLA evaluations limited to a manageable number of routes (e.g., 5–10) to keep runtime under 1–2 hours per regression suite.
- Metrics should be comparable across checkpoints; avoid changing definitions without versioning.
- Resource constraints mean closed-loop tests run less frequently than offline metrics (e.g., nightly vs. per-commit).

## Open questions
- Should planning evaluation include imitation-style metrics (e.g., similarity to expert trajectories) alongside closed-loop results?
- Do we add probabilistic risk metrics (expected collision risk) in MVP or defer to later?
- What criteria promote a model from offline-only approval to CARLA closed-loop testing (e.g., detection mAP threshold)?
- How do we weigh CARLA infractions vs. route completion when deciding release readiness?

## Notes for implementation
- Provide CLI scripts (`eval_perception.py`, `eval_prediction.py`, `eval_carla.py`) with consistent interfaces and JSON output.
- Ensure perception/prediction outputs follow standardized formats so evaluators can plug in new checkpoints without code changes.
- Use fixed random seeds and deterministic dataloaders when computing metrics to ensure reproducibility.
- Store evaluation artifacts (plots, JSON) under `reports/<date>/<experiment>` with metadata for traceability.

### Perception metrics
- **Object detection:** mean average precision (mAP) at multiple IoU thresholds (0.5, 0.7), center distance error (L2), orientation error (abs yaw difference), velocity error (L1 between predicted and GT velocities).
- **Lane/segmentation:** mean IoU across lane classes, boundary F1 score to capture centerline adherence.
- **Occupancy:** binary accuracy, precision, recall, plus calibration plots (reliability diagrams) to assess probability quality.
- Metrics computed per scenario and aggregated; include breakdowns by weather/time-of-day when available.

### Prediction metrics
- **Average Displacement Error (ADE)** and **Final Displacement Error (FDE)** over 4 s horizon.
- **Miss rate @ 2 m** (percentage of agents whose predicted trajectory stays within 2 m of ground truth).
- **Collision rate vs. ground truth** (predicted future overlapping GT occupancy).
- **Uncertainty metrics:** negative log-likelihood (if Gaussian heads) and calibration error between predicted variance and actual residuals; per-mode ADE/FDE where multi-modal outputs exist.

### Closed-loop CARLA metrics
- **Route completion rate:** % of predefined routes completed without major failure.
- **Infractions:** count of collisions, red-light violations, lane invasions per km.
- **Interventions/disengagements:** number of manual overrides or emergency stops triggered.
- **Comfort:** maximum/minimum acceleration, jerk, lateral acceleration; flag values beyond comfort thresholds.
- **Latency:** end-to-end control loop latency (average/max) to ensure real-time operation.
- Metrics logged per scenario and summarized across regression suites; store video snippets for failures when possible.

### Metric pipeline design
- Offline: iterate over nuPlan-mini validation scenarios, run perception/prediction models, collect metrics, and write JSON summaries; run after each checkpoint or nightly.
- Closed-loop: run CARLA scenarios via automation script, collect telemetry + event logs, compute metrics offline after the run.
- Aggregation: merge offline + closed-loop metrics into dashboards (e.g., Grafana or static HTML) and publish daily/weekly reports; maintain threshold tables for gating (e.g., mAP>0.2, ADE<1.5 m).

### Evaluation datasets and splits
- **nuPlan-mini validation set:** 20% of scenarios, stratified by city/time-of-day.
- **Held-out splits:** additional subsets focusing on high-interaction scenarios (e.g., intersections) to stress prediction.
- **CARLA regression routes:** curated set of 5–10 town/route combinations covering urban, suburban, and highway patterns; keep consistent across releases.
