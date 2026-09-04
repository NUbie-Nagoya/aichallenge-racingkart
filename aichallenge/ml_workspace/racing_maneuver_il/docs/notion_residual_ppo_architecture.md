# Residual PPO model — structure, architecture, inputs, and outputs

This page describes the current residual-PPO controller for racing-maneuver IL. It is a **residual policy around a frozen behavioral-cloning (BC) controller**, not a replacement controller.

## 1. Control architecture

```text
Temporal LiDAR + ego-state history
│
├─ Frozen BC policy
│    └─ nominal physical action:
│       [steering_tire_angle_rad, target_longitudinal_speed_mps]
│
├─ PPO residual policy
│    └─ normalized residual action:
│       [delta_steering_normalized, delta_target_speed_normalized]
│
└─ bounded action combiner
     └─ final [steering rad, target speed m/s]
          └─ target speed − measured speed
               └─ bounded proportional acceleration controller
                    └─ AWSIM AckermannControlCommand
```

The BC network remains frozen during PPO training. PPO learns only limited corrections around it.

## 2. Input observation

The Gym/PPO observation is a dictionary with two `float32` tensors:

| Key | Shape | Meaning |
|---|---:|---|
| `lidar` | `[10, 360]` | Ten chronological canonical LiDAR frames, 360 rays each, range 0–30 m. |
| `aux` | `[10, 9]` | Ten chronological ego/pose/action-history feature vectors. |

The 9 auxiliary features per frame are:

```text
1. longitudinal_velocity_mps
2. measured_steering_rad
3. longitudinal_acceleration_mps2
4. absolute_position_x_m
5. absolute_position_y_m
6. sin_yaw
7. cos_yaw
8. previous_safe_steering_command_rad
9. previous_safe_longitudinal_command
```

For PPO, the default Stable-Baselines3 `CombinedExtractor` flattens these inputs:

```text
LiDAR: 10 × 360 = 3600 values
Aux:   10 × 9   =   90 values
--------------------------------
Total PPO feature vector = 3690 float32 values
```

The frozen BC model consumes the same temporal tensors but retains its own temporal CNN+GRU structure.

## 3. Frozen BC nominal policy

The frozen BC policy is a temporal network:

```text
LiDAR frame: 1-D CNN
  Conv1d(1→8, kernel 9, stride 3) + ReLU
  Conv1d(8→16, kernel 7, stride 3) + ReLU
  AdaptiveAvgPool1d(8)
  Linear(128→lidar_embedding) + ReLU

Auxiliary frame: MLP
  Linear(9→32) + ReLU
  Linear(32→aux_embedding) + ReLU

Temporal fusion:
  concatenate LiDAR and auxiliary embeddings per frame
  GRU over 10 frames
  Linear(hidden→32) + ReLU
  Linear(32→2) + tanh
```

The export wrapper normalizes LiDAR/auxiliary inputs and rescales BC output to physical action units:

```text
BC output = [steering rad, target speed m/s]
```

## 4. PPO residual policy architecture

The inspected `live-run-1/residual_ppo.zip` policy uses Stable-Baselines3 `MultiInputPolicy` with the default MLP extractor:

```text
Flattened feature vector: 3690

Actor / policy branch:
  Linear(3690→64) + Tanh
  Linear(64→64)   + Tanh
  Linear(64→2)    → Gaussian action mean
  learned diagonal log_std: shape [2]

Critic / value branch:
  Linear(3690→64) + Tanh
  Linear(64→64)   + Tanh
  Linear(64→1)    → state value
```

PPO output is a two-dimensional normalized continuous action:

```text
[delta_steering_normalized, delta_target_speed_normalized] ∈ [-1, +1]²
```

During deterministic inference the runner uses:

```python
PPO.predict(observation, deterministic=True)
```

so it uses the policy mean rather than sampled exploration noise.

## 5. Action combination and physical output

Residual actions are clipped to `[-1, +1]` and scaled by configured bounds:

```text
final steering = clip(
  BC steering + residual_steering × max_steering_delta_rad,
  -1.0, +1.0
)

final target speed = clip(
  BC target speed + residual_speed × max_target_speed_delta_mps,
  0.0, 15.0
)
```

Current YAML limits for future runs are in:

```text
config/residual_ppo_live_v1.yaml
```

```yaml
residual_action:
  max_steering_delta_rad: 0.10
  max_target_speed_delta_mps: 0.75
```

Thus PPO can initially correct nominal BC output by at most:

```text
steering:     ±0.10 rad
target speed: ±0.75 m/s
```

The final target speed is converted to physical acceleration online:

```text
acceleration = clamp(
  speed_control_kp × (final target speed − measured speed),
  −acceleration_limit_mps2,
  +acceleration_limit_mps2
)
```

The effective vehicle command is published only while the residual backend runs:

```text
/control/command/control_cmd
Type: autoware_auto_control_msgs/msg/AckermannControlCommand
```

## 6. Reward inputs and safety termination

PPO is trained to optimize racing behavior, not MPC `ref_vel` matching.

Reward inputs include:

```text
+ speed/progress proxy
+ braking and steering response when any LiDAR ray is close
− smooth quadratic proximity penalty from min(all 360 LiDAR rays)
− abrupt residual action changes
− collision/stall penalty
```

The configured reward and safety settings live in the same YAML. The current YAML uses:

```yaml
reward:
  speed_weight: 0.3
  danger_clearance_m: 1.0
  proximity_penalty_weight: 3.0
  hazard_braking_weight: 0.1
  hazard_steering_weight: 0.50
```

Initial closed-circuit terminal signals are measured-state proxies:

```text
velocity_drop: speed loss ≥2.0 m/s within 0.25 s, after moving ≥1.0 m/s
prolonged_low_speed: speed <0.1 m/s continuously for 10 s
```

## 7. Runtime telemetry

A residual backend started after telemetry support was added publishes:

```text
/racing_maneuver/residual_rl/debug
Type: std_msgs/msg/Float32MultiArray
```

Array ordering:

```text
[0] base BC steering rad
[1] base BC target speed m/s
[2] PPO steering residual
[3] PPO speed residual
[4] final steering rad
[5] final target speed m/s
[6] step progress m
[7] cumulative episode progress m
[8] measured speed m/s
[9] termination code
```

Termination codes:

```text
0 normal
1 collision
2 off-track
3 velocity drop
4 prolonged low speed
```

Use only the isolated `rl_train` route. No MPC, teleop, or normal learned controller may publish `/control/command/control_cmd` concurrently.
