# Prediction
Status: Draft

## Purpose
Provide lightweight, reliable forecasts of dynamic agents and ego candidates so the planner can reason about near-future interactions without reprocessing raw perception outputs.

## Scope
Covers short-horizon prediction (3–5 s) for vehicles and vulnerable road users detected by perception, plus optional ego-trajectory proposal scoring; excludes map-level behavior planning or long-horizon intent prediction.

## Key responsibilities
- Generate per-agent future trajectories (positions + heading/velocity estimates) with uncertainty scores.
- Provide ego trajectory candidates or score planner-generated proposals against predicted traffic flow.
- Model agent-agent and agent-ego interactions using minimal compute (≤20–30 ms) so overall stack stays single-GPU friendly.
- Expose structured outputs (tensors/lists) consumable by planning without additional feature engineering.

## Inputs and outputs
- **Inputs:**
  - BEV feature tensor (`C_bev × H_bev × W_bev`) from perception.
  - Object list with bounding boxes, classes, velocities.
  - Ego-lane geometry and occupancy grid for context.
  - Optional short history buffer (last 2–3 frames of object states) if latency permits.
- **Outputs to planning:**
  - `trajectories`: shape `[N_agents, K_modes, T_steps, 4]` (x, y, heading, speed) in ego frame.
  - `mode_scores`: `[N_agents, K_modes]` confidence logits or probabilities.
  - `ego_candidates`: `[M, T_steps, 4]` optional ego paths with scores, or scoring metrics for planner-generated candidates.
  - Metadata: per-agent covariance estimates, validity flags, timestamps.
- Diagram: `Perception BEV + Objects → Scene/Actor Encoders → Interaction Module → Trajectory Heads → Predicted Futures`.

## Dependencies and interfaces
- Consumes perception outputs (BEV tensor, objects, lanes, occupancy) defined in `docs/architecture/02_perception.md`.
- Feeds planning with trajectories + scores and, if enabled, ego candidate evaluations; planning returns feedback for training (e.g., closed-loop loss).
- Training pipeline aligns with nuPlan-mini scenario format for ground-truth future states; CARLA closed-loop harness injects prediction outputs into planner at every simulation tick.

## Constraints / assumptions
- Must finish within ~20 ms per frame (post-perception) on a single 24 GB GPU; batch-size 1 inference is common.
- Limited dataset initially, so models favor shallow architectures (small CNNs for scene encoding, GRU/MLP for actors).
- Time horizon limited to 4 s at 0.5 s steps (T=8) for MVP; multi-modality capped at K=2 modes per agent.
- Focus on common nuPlan-style urban driving; rare behaviors (U-turns, reversing) may be deprioritized initially.
- History length minimal: either single frame or 0.5 s buffer to control memory.

## Open questions
- Start with single-mode trajectories (K=1) and add a second mode later, or launch with K=2 from the beginning?
- Should ego trajectory generation live entirely in planning, with prediction only scoring third-party agents?
- How many historical frames are necessary for stable velocity estimates—is single-frame velocity from perception sufficient?
- Do we require explicit pedestrian/cyclist handling in MVP, or can we start with vehicles-only and extend later?

## Notes for implementation
- Suggested tensors:
  - Scene encoder input: concatenate BEV features (C=128) with occupancy (1) and lane rasters (e.g., 3 channels) → `[132, H_bev, W_bev]`.
  - Actor features: pooled from BEV via ROI Align around each detection + object attributes (size, velocity).
  - Interaction module: 1–2 layers of multi-head attention (dim 128) or social pooling grid sized 5×5 around ego.
  - Trajectory head: small MLP or GRU producing `[K, T, 4]` plus log-variance.
- Baseline: start with MLP per agent using concatenated features (position, velocity, lane orientation) to produce single-mode trajectory; upgrade to interaction-aware transformer once baseline is stable.
- Integrate with nuPlan-mini rollouts by sampling ground-truth futures from dataset sequences; for CARLA closed-loop, write adapter to push `trajectories` into the planner each tick and log discrepancies for training.

### Subcomponents
1. **Scene encoder**
   - Consumes BEV tensor + occupancy + lane rasters.
   - Uses a shallow UNet/ResNet to produce context features (`scene_context`).
   - Diagram: `BEV stack → Shallow CNN → Scene Context`.
2. **Actor encoder**
   - For each detected agent: ROI Align scene context + feed object attributes through MLP/GRU to produce actor embedding.
   - Optionally stack past 2–3 states to encode motion history.
   - Diagram: `{Object Box + History} + Scene Context → Actor Encoder → Actor Embedding`.
3. **Interaction module**
   - Lightweight transformer or pooling to capture agent-agent influence, plus special token for ego.
   - Diagram: `Actor Embeddings → Interaction Layer → Interaction-aware Embeddings`.
4. **Trajectory head(s)**
   - Per actor, predict K trajectories and scores via MLP; incorporate uncertainty via Gaussian log-variance.
   - Ego candidate head (optional) uses same scene context to score planner proposals.
   - Diagram: `Interaction-aware Embedding → Trajectory Head → {Traj, Scores}`.
