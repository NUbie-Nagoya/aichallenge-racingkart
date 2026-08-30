# Racing Maneuver Imitation Learning Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Build a new `racing_maneuver_il` workspace and a new `racing-maneuver` Git branch for a real multi-car racing policy trained by temporal behavioral cloning from **manual expert maneuver demonstrations** in AWSIM.

**Architecture:** This is imitation learning, not RL. The policy learns causal mappings from 2-D LiDAR history, ego motion state, and fresh opponent/V2X state to the human expert’s final Ackermann command. It adopts End2Race’s useful temporal multi-agent learning pattern—LiDAR observations, recurrent memory, follow/pass scenarios, and closed-loop evaluation—but is trained from racing-kart manual demonstrations and outputs racing-kart physical actions `[steering_tire_angle_rad, longitudinal_acceleration_mps2]`, not F1TENTH desired speed. The policy remains shadow-only until it passes offline and AWSIM safety gates.

**Tech Stack:** ROS 2 Humble, MCAP/zstd rosbag2, Python 3.10, `rosbags`, NumPy, PyTorch, TorchScript, pytest, ruff, Docker Compose, AWSIM.

---

## Decisions locked by this plan

1. **New workspace:** Create `aichallenge/ml_workspace/racing_maneuver_il/`. Do not modify or depend on `tiny_lidar_net`.
2. **New branch:** Perform implementation on `racing-maneuver`, preserving the current controllers and old Pose-BC work.
3. **Learning type:** supervised temporal behavioral cloning / imitation learning.
4. **Expert source:** manual racing maneuvers. Labels come only from the final `/control/command/control_cmd` message issued while manual expert control is active; raw command and joystick streams are diagnostics/provenance only.
5. **Primary physical observation:** `/sensing/lidar/scan`.
6. **Opponent source:** record `/v2x/vehicle_positions` whenever available. It will be used for explicit opponent-relative features and runtime safety/freshness checks; LiDAR remains the independent physical observation.
7. **Action contract:** `[steering_tire_angle_rad, longitudinal_acceleration_mps2]` in physical units. No desired-speed output and no unvalidated speed-to-acceleration conversion.
8. **Safety:** default ROS deployment is shadow-only. Only one node may publish `/control/command/control_cmd` in an actuating run.

---

## Data contract

### Raw topics to record

The recording script must preflight and record the following when available:

```text
# Time
/clock

# Primary policy observation
/sensing/lidar/scan

# Ego localization and dynamics
/localization/kinematic_state
/localization/pose
/localization/twist
/localization/acceleration
/sensing/imu/imu_raw
/vehicle/status/velocity_status
/vehicle/status/steering_status
/vehicle/status/control_mode

# Opponent information
/v2x/vehicle_positions

# Final label and provenance/diagnostics
/control/command/control_cmd
/control/command/control_cmd_raw
/control/command/actuation_cmd
/joy
/control/control_mode_request_topic
/awsim/control_mode_request_topic

# Episode and simulator context
/awsim/state
/awsim/status
/initialpose
```

### Per-recording metadata

Write an immutable JSON file next to every finalized bag:

```json
{
  "schema_version": 1,
  "recording_id": "timestamp-or-uuid",
  "expert_source": "manual",
  "ego_vehicle_id": "required-id",
  "scenario_id": "required-scenario-name",
  "maneuver_class": "follow|pass_left|pass_right|side_by_side|abort|recover|free_lap",
  "opponent_count": 1,
  "opponent_behavior": "human|scripted|constant_speed",
  "seed": 0,
  "notes": "optional concise annotation"
}
```

A bag without this metadata must be rejected for final training. It may remain available for diagnostic replay.

### Processed training sample

Use a 10-step causal history at 20 Hz, ordered oldest to newest:

```text
LiDAR:
  canonical_lidar_ranges_m[360]

Ego state:
  longitudinal_velocity_mps
  measured_steering_rad
  longitudinal_acceleration_mps2
  previous_safe_steering_command_rad
  previous_safe_longitudinal_command

Opponent summary:
  opponent_present
  opponent_relative_x_body_m
  opponent_relative_y_body_m
  opponent_relative_vx_body_mps
  opponent_relative_vy_body_mps
  opponent_age_s

Target at current timestep:
  steering_tire_angle_rad
  longitudinal_acceleration_mps2
```

All rows must be built causally: latest valid source message at or before the label time, with maximum-age rejection. Never match a future sensor/V2X message merely because it is closest in time.

---

## Task 1: Create the implementation branch

**Objective:** Isolate new multi-car IL work from the existing controller experiments.

**Files:**
- No code files changed.

**Step 1: Inspect worktree before branching**

```bash
cd /home/ucluser/aichallenge-racingkart
git status --short
git branch --show-current
git log -1 --oneline
```

Expected: identify existing uncommitted work and avoid mixing it into this effort.

**Step 2: Create/switch to the branch**

```bash
git switch -c racing-maneuver
```

If the branch already exists:

```bash
git switch racing-maneuver
```

**Step 3: Verify branch and clean scope**

```bash
git branch --show-current
git status --short
```

Expected: current branch is `racing-maneuver`; no accidental staging.

---

## Task 2: Initialize the standalone workspace

**Objective:** Create a new self-contained package with no import from `tiny_lidar_net` or `pose_behavior_cloning`.

**Files:**
- Create: `aichallenge/ml_workspace/racing_maneuver_il/README.md`
- Create: `aichallenge/ml_workspace/racing_maneuver_il/requirements.txt`
- Create: `aichallenge/ml_workspace/racing_maneuver_il/pyproject.toml`
- Create: `aichallenge/ml_workspace/racing_maneuver_il/config/base.yaml`
- Create: `aichallenge/ml_workspace/racing_maneuver_il/racing_maneuver_il/__init__.py`
- Create: `aichallenge/ml_workspace/racing_maneuver_il/racing_maneuver_il/schema.py`
- Create: `aichallenge/ml_workspace/racing_maneuver_il/tests/test_schema.py`

**Step 1: Write failing schema tests**

```python
def test_feature_names_are_unique(): ...
def test_target_names_are_unique(): ...
def test_history_contract_is_10_frames_at_20_hz(): ...
def test_canonical_lidar_contract_is_360_rays_over_270_degrees(): ...
```

**Step 2: Implement an immutable schema**

```python
CANONICAL_LIDAR_RAYS = 360
CANONICAL_LIDAR_FOV_DEG = 270.0
HISTORY_LENGTH = 10
CONTROL_RATE_HZ = 20.0
TARGET_NAMES = (
    "steering_tire_angle_rad",
    "longitudinal_acceleration_mps2",
)
```

Keep raw physical units at the data boundary. Scaling/normalization is a model export responsibility.

**Step 3: Run tests**

```bash
cd aichallenge/ml_workspace/racing_maneuver_il
python -m pytest -q tests/test_schema.py
```

Expected: PASS.

**Step 4: Commit**

```bash
git add aichallenge/ml_workspace/racing_maneuver_il
git commit -m "feat: initialize racing maneuver IL workspace"
```

---

## Task 3: Prove the real AWSIM LiDAR contract

**Objective:** Define how raw `/sensing/lidar/scan` becomes the End2Race-inspired 360-ray input without assuming the F1TENTH scan layout.

**Files:**
- Create: `racing_maneuver_il/lidar.py`
- Create: `racing_maneuver_il/inspect_scan_contract.py`
- Create: `racing_maneuver_il/docs/scan_contract.md`
- Create: `racing_maneuver_il/tests/test_lidar.py`

**Step 1: Write failing preprocessing tests**

Test these cases:

```python
def test_canonicalizer_returns_360_float32_rays(): ...
def test_canonicalizer_uses_physical_angles_not_array_position(): ...
def test_positive_infinity_maps_to_declared_max_range(): ...
def test_nan_uses_declared_invalid_return_policy(): ...
def test_rejects_source_scan_with_insufficient_angular_coverage(): ...
```

**Step 2: Implement angle-aware canonicalization**

```python
def canonicalize_scan(
    ranges, angle_min_rad, angle_increment_rad, range_min_m, range_max_m, *,
    target_fov_deg=270.0, target_rays=360, model_max_range_m=30.0,
) -> np.ndarray:
    """Return 360 raw meter ranges on a fixed forward-centered angular grid."""
```

Requirements:

- Resample by angle, not by assuming a source ray count.
- Document the ordering and zero-angle direction.
- Handle invalid values deterministically.
- Do not normalize to `[0, 1]` in this function.

**Step 3: Inspect an active AWSIM scan**

When AWSIM is running, capture and document:

```bash
ros2 topic type /sensing/lidar/scan
ros2 topic echo --once /sensing/lidar/scan
ros2 topic hz /sensing/lidar/scan
```

Record source `angle_min`, `angle_max`, `angle_increment`, ray count, range bounds, rate, and invalid-return statistics in `docs/scan_contract.md`.

**Step 4: Run unit tests**

```bash
python -m pytest -q tests/test_lidar.py
```

Expected: PASS.

**Step 5: Commit**

```bash
git add aichallenge/ml_workspace/racing_maneuver_il
git commit -m "feat: define canonical lidar contract for maneuver IL"
```

---

## Task 4: Build a manual-expert multi-car recorder

**Objective:** Record full-quality manual expert demonstrations with maneuver provenance.

**Files:**
- Create: `aichallenge/ml_workspace/racing_maneuver_il/record_maneuver_data.bash`
- Create: `aichallenge/ml_workspace/racing_maneuver_il/scripts/write_recording_metadata.py`
- Create: `aichallenge/ml_workspace/racing_maneuver_il/docs/recording_protocol.md`
- Create: `aichallenge/ml_workspace/racing_maneuver_il/tests/test_recording_metadata.py`

**Step 1: Write metadata validation tests**

Require every metadata JSON file to have a supported maneuver class, `expert_source="manual"`, a non-empty scenario ID, and an ego vehicle ID.

**Step 2: Implement recording preflight**

The script must:

- source the ROS workspace safely under Bash strict mode;
- list required and optional topics before recording;
- refuse recording if `/sensing/lidar/scan`, final command labels, key ego state, or `/clock` are absent;
- require `/v2x/vehicle_positions` for multi-car recordings unless an explicit `--allow-no-v2x` diagnostic mode is selected;
- show topic type and publisher information for `/control/command/control_cmd`;
- make the expert manually confirm the scenario/maneuver metadata before capture;
- record MCAP with zstd compression;
- interrupt and wait for the recorder cleanly;
- require `metadata.yaml` and passing `ros2 bag info` before declaring the bag valid.

**Step 3: Preserve manual provenance**

Record `/joy` and `/vehicle/status/control_mode`, but do not use `/joy` as an ML feature. They are evidence that labels were produced under manual control.

**Step 4: Publish the manual driving protocol**

`docs/recording_protocol.md` must instruct the expert to collect balanced examples of:

```text
free_lap
following at multiple gaps
pass_left
pass_right
side_by_side hold
pass abort / yield
merge-back after pass
recovery to lane after an interrupted maneuver
```

For every category, vary track location, speed, initial relative gap, and opponent behavior. Never record an unsafe maneuver just to balance a class.

**Step 5: Smoke-test the script**

Run it for a short parked or low-risk manual scenario, then verify topic counts and metadata. Do not begin long data collection until all essential streams have nonzero counts.

**Step 6: Commit**

```bash
git add aichallenge/ml_workspace/racing_maneuver_il
git commit -m "feat: add manual multi-car maneuver recorder"
```

---

## Task 5: Inventory and reject incomplete recording sessions

**Objective:** Prevent command-only, structurally incomplete, or mislabeled bags from entering training.

**Files:**
- Create: `racing_maneuver_il/inventory.py`
- Create: `racing_maneuver_il/inventory_bags.py`
- Create: `racing_maneuver_il/tests/test_inventory.py`

**Step 1: Write failing tests**

Cover complete bags, missing scan, missing final command, missing V2X, missing metadata, incorrect message type, empty topic counts, and truncated MCAP detection.

**Step 2: Implement JSON inventory reports**

For each bag report:

```text
accept/reject status and reason
topic names/types/message counts
scan geometry and finite/invalid statistics
final command ranges
control-mode and `/joy` evidence
V2X IDs/message counts/age range
scenario metadata
source time range
```

**Step 3: Run tests and commit**

```bash
python -m pytest -q tests/test_inventory.py
git add aichallenge/ml_workspace/racing_maneuver_il
git commit -m "feat: validate manual maneuver recording inventory"
```

---

## Task 6: Implement causal multi-topic extraction

**Objective:** Convert accepted raw bags into immutable, source-stamped maneuver-learning examples.

**Files:**
- Create: `racing_maneuver_il/extraction.py`
- Create: `racing_maneuver_il/extract_dataset.py`
- Create: `racing_maneuver_il/opponents.py`
- Create: `racing_maneuver_il/tests/test_extraction.py`
- Create: `racing_maneuver_il/tests/test_opponents.py`

**Step 1: Write failing causal synchronization tests**

```python
def test_uses_latest_scan_at_or_before_label_time(): ...
def test_does_not_use_future_observation(): ...
def test_rejects_stale_scan_or_ego_state(): ...
def test_rejects_stale_v2x_for_multi_car_sample(): ...
def test_labels_come_from_final_control_command_only(): ...
def test_episode_break_resets_prior_action_history(): ...
def test_runtime_prior_action_is_previous_safe_command_not_future_label(): ...
```

**Step 2: Implement source-stamped alignment**

For each 20 Hz label time:

1. take the final `/control/command/control_cmd` action label;
2. select latest scan at or before the action time and canonicalize it to 360 rays;
3. select latest ego dynamics/steering/IMU state at or before that time;
4. select latest V2X report at or before that time;
5. reject the sample if any required source age exceeds its configured threshold;
6. break episodes on scenario boundaries, reset/initial pose, time discontinuities, or stale intervals;
7. store source timestamps and ages for auditability.

**Step 3: Implement body-frame opponent features**

Use position of the nearest relevant fresh forward/corridor opponent. Estimate velocity only from previous V2X samples with bounded finite differences. Transform relative position and velocity from map frame to ego body frame.

If no relevant fresh opponent exists, set the fixed feature vector to zero and `opponent_present=0`. Do not fabricate an opponent.

**Step 4: Write outputs atomically**

Write a versioned `.npz` dataset plus a JSON extraction report containing input rows, rejected rows and reasons, episode IDs, maneuver counts, command distributions, and source-age distributions.

**Step 5: Run tests and commit**

```bash
python -m pytest -q tests/test_extraction.py tests/test_opponents.py
git add aichallenge/ml_workspace/racing_maneuver_il
git commit -m "feat: extract causal manual maneuver demonstrations"
```

---

## Task 7: Build grouped temporal datasets and train-only normalizers

**Objective:** Produce leakage-resistant 10-frame causal sequences for temporal BC.

**Files:**
- Create: `racing_maneuver_il/dataset.py`
- Create: `racing_maneuver_il/normalization.py`
- Create: `racing_maneuver_il/tests/test_dataset.py`
- Create: `racing_maneuver_il/tests/test_normalization.py`

**Step 1: Write failing tests**

Verify:

- histories are oldest-to-newest and never cross episode boundaries;
- entire `recording_id + episode_id` groups appear in only one split;
- normalizer fits only training groups;
- every split manifest identifies maneuver-class counts;
- test frames cannot appear in train through overlapping time histories.

**Step 2: Create session/episode-grouped splits**

Do not randomize adjacent frames. Split complete recordings/episodes into train/validation/test with maneuver-class coverage reported, then save a locked `split_manifest.json`.

**Step 3: Fit raw feature normalizers on train only**

Normalize LiDAR, ego values, and V2X values using train-only statistics. Preserve target labels in their physical units. Store normalizer metadata in every checkpoint.

**Step 4: Run tests and commit**

```bash
python -m pytest -q tests/test_dataset.py tests/test_normalization.py
git add aichallenge/ml_workspace/racing_maneuver_il
git commit -m "feat: add grouped temporal maneuver datasets"
```

---

## Task 8: Implement an End2Race-inspired temporal policy

**Objective:** Learn maneuver behavior from recent perception and state without importing End2Race weights or code.

**Files:**
- Create: `racing_maneuver_il/model.py`
- Create: `racing_maneuver_il/checkpoint.py`
- Create: `racing_maneuver_il/tests/test_model.py`

**Step 1: Write failing model tests**

```python
def test_model_accepts_batch_time_lidar_and_aux_features(): ...
def test_model_returns_batch_two_actions(): ...
def test_hidden_state_round_trip_is_causal_and_shape_stable(): ...
def test_model_has_no_nan_output_for_bounded_input(): ...
def test_export_wrapper_owns_normalization(): ...
```

**Step 2: Implement a compact temporal architecture**

Initial architecture:

```text
360 LiDAR rays per step
  → small 1-D CNN scan encoder

Ego + V2X features per step
  → auxiliary MLP

concatenate scan embedding + auxiliary embedding
  → single-layer GRU
  → shared action MLP
  → steering head and longitudinal-acceleration head
```

Use a modest hidden size first and measure CPU inference under the Docker ROS environment. Do not start with End2Race’s approximately 11.3M-parameter recurrent model.

**Step 3: Preserve physical action units**

The network may use bounded/scaled internal representation, but its wrapper output must explicitly be:

```text
steering_tire_angle_rad
longitudinal_acceleration_mps2
```

with limits serialized into the artifact. The ROS safety core applies an additional independent clip/slew envelope.

**Step 4: Run tests and commit**

```bash
python -m pytest -q tests/test_model.py
git add aichallenge/ml_workspace/racing_maneuver_il
git commit -m "feat: add temporal lidar maneuver BC policy"
```

---

## Task 9: Train and evaluate with maneuver-specific metrics

**Objective:** Select a policy using held-out maneuver behavior, not only average action loss.

**Files:**
- Create: `racing_maneuver_il/train.py`
- Create: `racing_maneuver_il/evaluate.py`
- Create: `racing_maneuver_il/metrics.py`
- Create: `racing_maneuver_il/config/train_baseline.yaml`
- Create: `racing_maneuver_il/tests/test_metrics.py`

**Step 1: Write failing metric tests**

Require physical-unit metrics:

```text
steering MAE/RMSE [rad and degrees]
longitudinal acceleration MAE/RMSE [m/s²]
action-change error
```

and slice them by:

```text
free_lap
follow
pass_left
pass_right
side_by_side
abort
recover
opponent present / absent
closing-speed bins
```

**Step 2: Implement training**

Use weighted Huber losses for steering and longitudinal acceleration plus a small action-change loss. Save:

```text
best.pt / last.pt
resolved config
normalizer
locked split manifest
training/validation curves
dataset and code provenance
```

**Step 3: Use autoregressive offline evaluation**

Evaluate two modes:

- teacher-forced: historical prior action uses the demonstration command;
- autoregressive: historical prior action uses prior safely limited policy output.

Promotion decisions must use the autoregressive result because it matches runtime feedback.

**Step 4: Run a deterministic synthetic smoke training test**

Verify a checkpoint, config, and metrics report are generated. Do not claim racing quality from the synthetic test.

**Step 5: Commit**

```bash
git add aichallenge/ml_workspace/racing_maneuver_il
git commit -m "feat: train and evaluate manual maneuver imitation policy"
```

---

## Task 10: Export a verified TorchScript artifact

**Objective:** Deploy the exact trained preprocessing and action contract, not an approximate reimplementation.

**Files:**
- Create: `racing_maneuver_il/export_torchscript.py`
- Create: `racing_maneuver_il/tests/test_export.py`

**Step 1: Write a failing eager/export parity test**

Feed one fixed raw sequence into eager and exported models; verify same bounded physical outputs within tolerance.

**Step 2: Embed deployment metadata**

Include in artifact/checkpoint metadata:

```text
schema version
canonical LiDAR geometry
history length/rate
full feature ordering
normalizer statistics
output units and limits
training split/dataset provenance
```

**Step 3: Verify in both environments**

- Training virtual environment: export/load/parity.
- Actual ROS Docker Python: load and infer a raw fixture.

**Step 4: Commit**

```bash
git add aichallenge/ml_workspace/racing_maneuver_il
git commit -m "feat: export verified maneuver policy TorchScript"
```

---

## Task 11: Integrate a separate shadow-only ROS controller

**Objective:** Validate live inputs and model behavior without allowing the learned policy to drive the kart.

**Files:**
- Create: `aichallenge/workspace/src/aichallenge_submit/racing_maneuver_controller/` package
- Create: `.../racing_maneuver_controller/controller_core.py`
- Create: `.../racing_maneuver_controller/node.py`
- Create: `.../config/racing_maneuver_controller.param.yaml`
- Create: `.../launch/racing_maneuver.launch.xml`
- Create: `.../test/test_controller_core.py`
- Modify: `aichallenge_submit_launch/launch/reference.launch.xml` only to add an explicit `control_method:=racing_maneuver` route; keep default `mpc`.

**Step 1: Write pure-core tests first**

Test:

```text
causal history warm-up/reset
scan/ego/V2X freshness rejection
V2X feature construction
TorchScript raw input contract
finite output check
output clipping and rate limiting
previous-safe-command feedback
mode permissions
shadow-vs-real publisher separation
```

**Step 2: Subscribe to live inputs**

Use the confirmed topic types/QoS for:

```text
/sensing/lidar/scan
/localization/kinematic_state
/localization/kinematic_state/twist
/vehicle/status/steering_status
/v2x/vehicle_positions
```

**Step 3: Default to debug-only publications**

Always expose:

```text
/racing_maneuver/debug/control_cmd
/racing_maneuver/debug/features
/racing_maneuver/debug/safety_status
```

Set:

```yaml
publish_commands: false
```

The real command topic is never written in the first deployment milestone.

**Step 4: Build/test/lint in the actual container**

```bash
colcon build --packages-select racing_maneuver_controller
pytest -q src/aichallenge_submit/racing_maneuver_controller/test
ruff check src/aichallenge_submit/racing_maneuver_controller
```

**Step 5: Commit**

```bash
git add aichallenge/workspace/src/aichallenge_submit/racing_maneuver_controller \
        aichallenge/workspace/src/aichallenge_submit/aichallenge_submit_launch
git commit -m "feat: add shadow racing maneuver controller"
```

---

## Task 12: Validate against manual multi-car scenarios in shadow mode

**Objective:** Prove that the model receives correct live data and behaves coherently before any live actuation decision.

**Files:**
- Create: `racing_maneuver_il/docs/shadow_validation.md`
- Create: `racing_maneuver_il/scripts/validate_shadow_run.bash`

**Step 1: Launch explicitly without learned commands**

```bash
CONTROL_METHOD=racing_maneuver RACING_MANEUVER_PUBLISH_COMMANDS=false make dev
```

Use the exact final environment/launch names implemented in Task 11.

**Step 2: Gather live proof**

```bash
ros2 param get /racing_maneuver_controller publish_commands
ros2 topic hz /racing_maneuver/debug/control_cmd
ros2 topic echo --once /racing_maneuver/debug/control_cmd
ros2 topic echo --once /racing_maneuver/debug/safety_status
ros2 topic info /control/command/control_cmd -v
```

**Step 3: Replay all maneuver classes**

At minimum validate free-lap, following, left/right pass, side-by-side, abort, recovery, stale V2X, and missing scan cases. Capture scenario ID, artifact hash, debug traces, and observed fallback reason.

**Step 4: Gate any future actuation**

Do not turn on `publish_commands=true` unless all of these are true:

- exact artifact parity was verified inside the ROS container;
- test split is grouped by recordings/episodes;
- autoregressive offline results are acceptable in every maneuver slice;
- shadow output remains stable and bounded at 20 Hz;
- stale scan, ego state, and V2X each produce deterministic safe fallback;
- no competing controller publisher exists on `/control/command/control_cmd`;
- commanded-versus-measured steering/acceleration route is independently verified in low-speed tests.

**Step 5: Commit**

```bash
git add aichallenge/ml_workspace/racing_maneuver_il
git commit -m "docs: define racing maneuver shadow validation gates"
```

---

## Data collection checklist for the manual expert

Before each session:

- confirm manual control is active and the final command topic reflects the expert’s commands;
- verify LiDAR and V2X are actually publishing, not merely listed;
- declare exactly one scenario and maneuver class for the bag;
- ensure a safe, repeatable initial position and opponent behavior;
- collect a whole episode, including approach, decision, maneuver, and recovery—not just the steering peak;
- stop and discard/relabel a run if it contains reset glitches, a wrong command owner, topic dropouts, or an unsafe incident.

After each session:

- run bag inventory;
- inspect topic counts and maximum source ages;
- inspect LiDAR geometry and command distributions;
- visually replay a small sample of every maneuver class;
- preserve raw bags immutably; only processed outputs may be regenerated.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Manual data has too few pass/abort examples | Use a scripted scenario matrix and deliberate balanced collection, not opportunistic laps. |
| Expert behavior is inconsistent | Record scenario metadata, retain per-session grouping, review video/bag samples, and reject low-quality episodes. |
| Multi-car visibility differs between LiDAR and V2X | Record both, require freshness, and bias runtime fallback conservatively on disagreement. |
| Dataset leaks nearly identical frames into test | Split complete recordings/episodes, never random adjacent frames. |
| Policy learns shortcuts from position or scenario timing | Do not include lap/time/scenario ID as model features. Use them only for splits and metrics. |
| Stable predicted actions are mistaken for safe driving | Keep strict shadow-only deployment until closed-loop safety and command-route evidence exists. |

## Definition of done: first milestone

The first milestone is a reproducible **manual-expert multi-car imitation-learning pipeline**, not autonomous racing:

1. manual expert bags contain final command labels, LiDAR, ego state, V2X, and complete metadata;
2. causal extractor produces valid 10-frame / 20 Hz sequences with audit reports;
3. temporal LiDAR/V2X policy trains and reports held-out maneuver-slice metrics;
4. TorchScript carries its preprocessing and output contract;
5. a ROS controller publishes stable debug actions and safety state in AWSIM; and
6. it never publishes real commands by default.
