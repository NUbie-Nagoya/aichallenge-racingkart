# Residual RL fine-tuning

`racing_maneuver_il.residual_rl.ResidualRacingEnv` keeps the schema-v3 behavioral-cloning (BC) policy as the nominal controller and trains PPO only on bounded residual actions:

```text
BC [steering, target speed]
+ PPO residual [-1, 1]^2
→ bounded physical command
→ local target-speed-to-acceleration feedback
→ AWSIM
```

The reward deliberately contains **no reference-velocity matching term**. It rewards forward progress and racing speed, strongly penalizes manual crash/stall termination, and adds a smooth quadratic penalty whenever the **closest LiDAR ray anywhere around the kart** is within configured `danger_clearance_m`. It also rewards braking or steering in that hazard region. This makes side/rear obstacle proximity relevant too, rather than reacting only to a front sector.

## Initial episode termination

The closed AWSIM circuit does not need map-based off-track termination initially. AWSIM has no verified, per-vehicle collision-report topic on the RL route, so the backend intentionally uses measured-state detection:

- `velocity_drop`: a speed loss of at least `2.0 m/s` inside `0.25 s`, after moving at least `1.0 m/s`;
- `prolonged_low_speed`: speed below `0.1 m/s` continuously for `10 s`;
- normal Gym time limit.

These checks run on fresh odometry timestamps in `CrashChecker`; a detected event terminates the episode and the next `reset()` restarts AWSIM. They are collision proxies, not authoritative contact reports, so their thresholds must be calibrated from rollout logs before a long training run.

## Safety contract

- The ROS backend refuses to construct unless `RESIDUAL_RL_ALLOW_CONTROL=1`.
- Use a dedicated simulator-training route. No MPC, teleop, or learned-controller publisher may remain on `/control/command/control_cmd`.
- Keep normal deployment shadow-only. This backend is for AWSIM training only.
- Train from a `best.pt` schema-v3 BC checkpoint, never from an old acceleration-label artifact.

## Live-training status: not yet enabled

The residual core and PPO integration are unit/smoke-tested. **Do not run the live AWSIM command yet.** A live racing backend still needs the proven dual-domain reset sequence (domain-0 `/admin/awsim/reset`, then domain-1 initial-pose/control handover), fresh-observation verification, and exclusive command ownership. Map/centerline progress and off-track termination are not initial blockers for the closed circuit.

Until those pieces are implemented and exercised, use this work only for offline/unit development and retain normal deployment shadow-only.

## Residual PPO configuration

Residual PPO is configured through a versioned YAML file rather than a collection of training CLI flags:

```text
config/residual_ppo_live_v1.yaml
```

The file records the frozen BC checkpoint, rollout/training values, **all active Stable-Baselines3 PPO hyperparameters** (`learning_rate`, `gamma`, `gae_lambda`, `clip_range`, `ent_coef`, `vf_coef`, `max_grad_norm`, `n_epochs`, and optional `target_kl`), residual action bounds, reward weights, crash/stall thresholds, ROS/AWSIM backend limits, and output root. Each launch creates the next sequential `run-N` directory below `output_root` and writes an exact `resolved_config.yaml` there.

## Intended training command (after live backend verification)

After configuring a simulator-only route with exclusive command ownership:

```bash
cd /aichallenge/ml_workspace/racing_maneuver_il
source /opt/ros/humble/setup.bash
source /aichallenge/workspace/install/setup.bash

RESIDUAL_RL_ALLOW_CONTROL=1 ROS_DOMAIN_ID=1 \
python3 train_residual_ppo.py config/residual_ppo_live_v1.yaml
```

Do not launch this command on a shared domain where `mpc_controller` is publishing. First verify exclusive ownership:

```bash
ros2 topic info -v /control/command/control_cmd
```

The only publisher must be `racing_maneuver_residual_rl_backend`.

## Deterministic residual inference

Use `run_residual_ppo.py` to run an SB3 PPO residual together with its matching frozen BC checkpoint. It requires the isolated `rl_train` route, `RESIDUAL_RL_ALLOW_CONTROL=1`, and an explicit finite step bound:

```bash
cd /aichallenge/ml_workspace/racing_maneuver_il
source /opt/ros/humble/setup.bash
source /aichallenge/workspace/install/setup.bash

RESIDUAL_RL_ALLOW_CONTROL=1 ROS_DOMAIN_ID=1 \
python3 run_residual_ppo.py \
  runs/mpc-multicar-100lap-sweep-aicdev/run-4/best.pt \
  runs/residual-rl/live-run-1/residual_ppo.zip \
  --max-steps 2000 \
  --max-episodes 3
```

The runner always uses `PPO.predict(..., deterministic=True)`, combines its normalized residual with the BC physical action using the configured residual bounds, and resets after each terminal crash/stall episode. Do not use `residual_ppo.zip` with the normal TorchScript-only `racing_maneuver_controller`.

### Live telemetry

While a residual PPO training or inference process is running, it publishes numeric telemetry on:

```text
/racing_maneuver/residual_rl/debug
std_msgs/msg/Float32MultiArray
```

Data ordering:

```text
[0] base steering rad        [5] final target speed m/s
[1] base target speed m/s    [6] step progress m
[2] PPO residual steering    [7] cumulative episode progress m
[3] PPO residual speed       [8] measured speed m/s
[4] final steering rad       [9] termination code
```

Termination codes are `0=none`, `1=collision`, `2=off_track`, `3=velocity_drop`, `4=prolonged_low_speed`. The effective AWSIM command remains on `/control/command/control_cmd`.

## Evaluation requirements

Evaluate across held-out AWSIM seeds, initial poses, speed-randomization profiles, and multicar configurations. Compare against frozen BC on lap completion, progress, manual crash/stall termination rate, and command smoothness. Do not accept a model based on training reward alone.
