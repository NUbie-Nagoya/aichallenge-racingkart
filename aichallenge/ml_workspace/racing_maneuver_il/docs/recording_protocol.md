# Manual Multi-Car Recording Protocol

## Before recording

1. Start AWSIM with the intended ego domain and opponent configuration.
2. Confirm manual control is active. Do not use a learned controller as the label owner.
3. Check that `/control/command/control_cmd` has exactly the intended publisher and that its values follow manual input.
4. Verify messages are flowing on `/sensing/lidar/scan`, ego state/status, `/v2x/vehicle_positions`, `/clock`, and `/awsim/status`.
5. Choose exactly one maneuver class and scenario ID for the bag.
6. Place cars in a safe, reproducible initial arrangement.

Run `record_maneuver_data.bash` with explicit metadata. Example:

```bash
./record_maneuver_data.bash \
  --ego-vehicle-id d1 \
  --scenario-id straight-gap8-slow-opponent-01 \
  --maneuver-class pass_left \
  --opponent-count 1 \
  --opponent-behavior constant_speed \
  --seed 1 \
  --command-owner teleop_manager_node
```

## Episode lifecycle inside one continuous bag

The recorder stays running for the full collection session. The keyboard bridge emits
`std_msgs/msg/String` events on `/racing_maneuver/episode_control`, which the recorder
requires and saves in the same MCAP. It does not affect vehicle control:

- **F9**: `START` a new candidate episode.
- **F10**: `STOP` and retain the current episode.
- **F11**: `DISCARD` the complete active interval; it remains in the immutable raw bag
  for audit but extraction excludes it from training data.

Extraction accepts only closed `START` → `STOP` intervals, resets temporal history at
`START`, and rejects samples before the first start, after a stop, in discarded intervals,
or in an episode left open at finalization. Do not use F9/F10/F11 for vehicle control.

## Maneuver coverage

Collect complete episodes—approach, decision, maneuver, and stabilization—for:

- `free_lap`: no relevant nearby opponent;
- `follow`: several gaps and closing speeds, including choosing not to pass;
- `pass_left` and `pass_right`: several straight/corner contexts;
- `side_by_side`: stable alongside interaction without contact;
- `abort`: begin a pass, reject/abort safely, and return to following;
- `recover`: regain a stable line after an interrupted maneuver.

Vary track location, ego speed, opponent speed/profile, initial gap, and safe lateral arrangement. Never execute an unsafe maneuver merely to balance the dataset.

## Stop and discard or relabel a run when

- the wrong command publisher owns the final command topic;
- control is not manual for any labeled interval;
- a required topic stops publishing;
- AWSIM resets or time jumps unexpectedly;
- the scenario metadata is wrong;
- a collision or unsafe incident makes the expert action unsuitable for imitation.

Do not delete raw bags casually. Mark rejected recordings in a separate review log; training inventory will reject structurally invalid bags automatically.

## After recording

1. Confirm `metadata.yaml` and `recording.json` exist.
2. Run `ros2 bag info <bag-directory>`.
3. Run:

```bash
PYTHONPATH=. python3 inventory_bags.py rawdata --output inventory.json
```

4. Inspect required topic counts, final command ranges, scan geometry, V2X IDs, and source-age statistics produced during extraction.
5. Visually replay representative sections before admitting the recording to a final split.
6. Keep raw bags immutable. Regenerate only processed datasets and reports.

## Provenance rules

- `/control/command/control_cmd` is the sole action label.
- `/joy`, raw control, and actuation topics are provenance/diagnostics, not model inputs or labels.
- `scenario_id`, maneuver class, seed, and notes are used for splits and evaluation, never as policy inputs.
- A complete recording/episode group must belong to exactly one of train, validation, or test.
