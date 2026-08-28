# Shadow Validation

The learned racing-maneuver policy is an experimental controller. A stable debug command is not evidence that AWSIM received a command or that a maneuver is safe.

## Preconditions

- Build and source the runtime container overlay containing `racing_maneuver_controller`.
- Export a verified TorchScript artifact and record its hash.
- Select `CONTROL_METHOD=racing_maneuver` explicitly.
- Keep `RACING_MANEUVER_PUBLISH_COMMANDS=false`.
- Keep a known baseline/manual controller available for actual motion during shadow evaluation, or replay an accepted bag. Do not run two real command publishers.

## Live checks

```bash
ros2 param get /racing_maneuver_controller publish_commands
ros2 param get /racing_maneuver_controller model_path
ros2 topic hz /racing_maneuver/debug/control_cmd
ros2 topic echo --once /racing_maneuver/debug/control_cmd
ros2 topic echo --once /racing_maneuver/debug/features
ros2 topic echo --once /racing_maneuver/debug/safety_status
ros2 topic info /control/command/control_cmd -v
```

Expected:

- `publish_commands` is false;
- debug output begins only after the configured chronological history is warm;
- debug output remains finite, bounded, and near 20 Hz while required inputs are fresh;
- stale scan, ego state, or required opponent/V2X state produces an explicit rejected/fallback safety status and resets history;
- no messages from the learned policy appear on `/control/command/control_cmd`.

## Required scenario strata

Replay or manually execute accepted held-out episodes for:

- no nearby opponent / free lap;
- follow at multiple closing speeds;
- pass left;
- pass right;
- side-by-side;
- abort/yield;
- recovery;
- V2X stale/missing;
- LiDAR stale/missing;
- mode transition and episode reset.

For every run store scenario/recording IDs, dataset split, artifact hash, controller parameters, debug topics, rate/bounds, safety interventions, and qualitative discrepancies from the expert.

## Actuation promotion gate

Do not set `publish_commands=true` until all of the following are demonstrated:

1. Eager and TorchScript outputs agree on raw held-out fixtures in both training and ROS-container Python.
2. Test groups are independent recordings/episodes, not adjacent frames from training runs.
3. Autoregressive held-out metrics pass for every maneuver class.
4. Shadow predictions remain bounded and stable at the configured rate.
5. Stale/missing scan, ego state, and V2X have deterministic safe behavior.
6. Runtime mode constants and periodic/event-driven semantics are confirmed from the deployed ROS messages.
7. Exactly one intended publisher can own `/control/command/control_cmd`.
8. Low-speed fixed-command tests establish commanded-versus-measured steering and acceleration behavior.
9. A deterministic collision-time/freshness supervisor can override the learned action.

First actuation must use one opponent, a low-speed straight segment, strict limits, and an immediately available stop path. Multi-car/high-speed promotion requires separate evidence.
