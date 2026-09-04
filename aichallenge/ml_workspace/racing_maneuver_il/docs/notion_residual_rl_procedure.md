# Racing maneuver IL — BC, residual PPO training, and deterministic inference

This procedure covers the current schema-v3 target-speed BC policy and its bounded residual-PPO refinement. Run training, TorchScript export, and residual PPO work **inside the AIC_DEV / Autoware Docker runtime** so deployed PyTorch/CUDA versions match.

> **Safety requirement:** Residual PPO controls a live AWSIM kart. Use only an isolated `rl_train` route where no MPC, teleop, or normal learned controller publishes `/control/command/control_cmd`.

## Policy contract

```text
10 LiDAR frames × 360 rays + 9 ego/absolute-pose/action-history features
→ frozen BC policy
→ [steering rad, target speed m/s]

PPO predicts normalized [Δ steering, Δ target speed]
→ BC action + bounded residual
→ target-speed feedback controller
→ bounded acceleration command to AWSIM
```

Current residual bounds used for `live-run-1` training:

```text
steering:     ±0.10 rad
target speed: ±0.75 m/s
```

### LiDAR-aware collision-avoidance reward

The racing reward has no `ref_vel` tracking term. It uses the **minimum clearance across all 360 canonical LiDAR rays**, not only the forward sector. When any ray is inside configured `danger_clearance_m`, PPO receives a smooth quadratic proximity penalty (`-proximity_penalty_weight × hazard²`) as well as an incentive for braking below measured speed and/or applying steering. Thus a close side or rear-side object is relevant before collision.

## 1. BC training from MPC-expert data

Inside AIC_DEV:

```bash
cd /aichallenge/ml_workspace/racing_maneuver_il
source /opt/ros/humble/setup.bash
source /aichallenge/workspace/install/setup.bash

python3 -m racing_maneuver_il.train config/train_mpc_within_session.yaml
```

- Training is CUDA-first and fails rather than silently falling back if CUDA is requested but unavailable.
- Each launch creates a sequential `run-N` directory.
- Select `best.pt`, not `last.pt`, using normalized validation score and per-target MAE.
- Current residual-PPO BC initialization:

```text
runs/mpc-multicar-100lap-sweep-aicdev/run-4/best.pt
```

## 2. Optional BC TorchScript export

For normal shadow-only learned-controller work, export inside AIC_DEV:

```bash
PYTHONPATH=/aichallenge/ml_workspace/racing_maneuver_il \
python3 -m racing_maneuver_il.export_torchscript \
  runs/mpc-multicar-100lap-sweep-aicdev/run-4/best.pt \
  runs/mpc-multicar-100lap-sweep-aicdev/run-4/policy.ts
```

This artifact is for the normal BC controller; it is not the PPO residual model.

## 3. Prepare isolated AWSIM RL route

On the host, at `/home/ucluser/aichallenge-racingkart`:

```bash
SIM_MODE=dev HEADLESS=1 make simulator

LOG_DIR=/output/residual-rl \
CONTROL_METHOD=rl_train ROS_DOMAIN_ID=1 RUN_MODE=awsim-no-viz \
docker compose -p 1 up -d --force-recreate autoware
```

Inside the Autoware container, verify reset, sensors, and exclusive ownership:

```bash
source /opt/ros/humble/setup.bash
source /aichallenge/workspace/install/setup.bash

ROS_DOMAIN_ID=0 ros2 topic info /admin/awsim/reset
ROS_DOMAIN_ID=1 ros2 node list | grep /awsim_reset_bridge
ROS_DOMAIN_ID=1 ros2 topic info -v /control/command/control_cmd
```

Required before training/inference:

```text
/control/command/control_cmd
Publisher count: 0
Subscriber: awsim_d1
```

Do not continue if `mpc_controller`, `teleop_manager_node`, or `racing_maneuver_controller` publishes on that topic.

## 4. Train residual PPO

Residual PPO uses a versioned YAML file rather than CLI training flags:

```text
config/residual_ppo_live_v1.yaml
```

It records the frozen BC checkpoint, rollout/training settings, **all active Stable-Baselines3 PPO hyperparameters** (`learning_rate`, `gamma`, `gae_lambda`, `clip_range`, `ent_coef`, `vf_coef`, `max_grad_norm`, `n_epochs`, and optional `target_kl`), residual bounds, reward weights, crash/stall thresholds, backend limits, and output root. The per-episode time limit is set by `training.max_steps` (default `2400` control steps ≈ 120 s at the 20 Hz backend); it ends an episode by truncation rather than termination. Each launch creates the next sequential `run-N` directory and writes its exact `resolved_config.yaml`.

```bash
cd /aichallenge/ml_workspace/racing_maneuver_il
source /opt/ros/humble/setup.bash
source /aichallenge/workspace/install/setup.bash

RESIDUAL_RL_ALLOW_CONTROL=1 ROS_DOMAIN_ID=1 \
python3 train_residual_ppo.py config/residual_ppo_live_v1.yaml
```

The default v1 configuration produces runs below:

```text
runs/residual-rl/run-N/
```

The earlier initial live run is retained at:

```text
runs/residual-rl/live-run-1/residual_ppo.zip
```

## 5. Deterministic residual inference

`residual_ppo.zip` is not standalone. Pair it with the **same frozen BC checkpoint**:

```bash
cd /aichallenge/ml_workspace/racing_maneuver_il
source /opt/ros/humble/setup.bash
source /aichallenge/workspace/install/setup.bash

RESIDUAL_RL_ALLOW_CONTROL=1 ROS_DOMAIN_ID=1 \
python3 run_residual_ppo.py \
  runs/mpc-multicar-100lap-sweep-aicdev/run-4/best.pt \
  runs/residual-rl/live-run-1/residual_ppo.zip \
  --max-steps 2000 \
  --max-episodes 3 \
  --max-steering-residual-rad 0.10 \
  --max-speed-residual-mps 0.75
```

- `--max-steps 2000`: hard control-step bound; at 20 Hz, approximately 100 seconds maximum.
- `--max-episodes 3`: stop after 3 terminal crash/stall episodes; `0` disables the episode-count limit.
- `--max-steering-residual-rad 0.10`: PPO correction cap around BC steering.
- `--max-speed-residual-mps 0.75`: PPO correction cap around BC target speed.

For a near-pure BC route sanity check, use `0.001` for both residual bounds. This is not a fair learned-PPO evaluation because it changes the trained action scale.

## 6. Episode termination and reset

The closed circuit initially uses manual measured-state detection rather than map off-track logic or an unverified AWSIM collision signal:

```text
velocity_drop:       measured speed loss ≥2.0 m/s within 0.25 s,
                     after speed was at least 1.0 m/s
prolonged_low_speed: speed <0.1 m/s continuously for 10 s
time limit:          Gym episode maximum steps
```

A terminal event triggers the next environment reset through the AWSIM reset bridge. These signals are collision/stall proxies and should be calibrated with rollout logs before long campaigns.

For direct telemetry while PPO is running, use:

```bash
ROS_DOMAIN_ID=1 ros2 topic echo /racing_maneuver/residual_rl/debug
```

The message is `std_msgs/msg/Float32MultiArray` with data:

```text
[base steering, base target speed,
 PPO residual steering, PPO residual speed,
 final steering, final target speed,
 step progress, cumulative episode progress,
 measured speed, termination code]
```

Termination codes: `0=none`, `1=collision`, `2=off_track`, `3=velocity_drop`, `4=prolonged_low_speed`. The final effective AWSIM command can also be viewed at `/control/command/control_cmd`.

## 7. Post-run checks

After inference/training exits:

```bash
ROS_DOMAIN_ID=1 ros2 topic info -v /control/command/control_cmd
```

Expected cleanup:

```text
Publisher count: 0
Subscriber: awsim_d1
```

Evaluate residual PPO against frozen BC across initial poses, simulator seeds, speed profiles, and multicar scenarios. Track lap completion, progress, crash/stall termination rate, and command smoothness; do not accept a model based on training reward alone.
