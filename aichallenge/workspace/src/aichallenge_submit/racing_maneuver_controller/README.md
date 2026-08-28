# Racing Maneuver Controller

A standalone ROS 2 Humble package for **shadow-first** execution of the racing-maneuver TorchScript policy. It does not alter or replace any existing controller launch route.

## Safety contract

- `publish_commands` defaults to `false` in source, YAML, and launch.
- Predictions publish to `/racing_maneuver/debug/control_cmd` only after ten fresh, chronological frames.
- `/control/command/control_cmd` is written only when `publish_commands:=true`, all required inputs (including the permission report) are fresh, the mode is in the narrow `allowed_control_modes`, and model output passes shape/finite/clamp/rate-limit checks.
- Shadow and actuation mode allow-lists are separate. Any continuity break, mode transition, stale/non-causal input, invalid input, or invalid model output resets history and previous-safe-command state.
- An actuating run must disable every other publisher of the real command topic.

## Inputs and model contract

Subscriptions:

- `sensor_msgs/LaserScan` on `/sensing/lidar/scan`
- `nav_msgs/Odometry` on `/localization/kinematic_state` (pose and twist)
- `geometry_msgs/AccelWithCovarianceStamped` on `/localization/acceleration`
- `geometry_msgs/PoseWithCovarianceStamped` on `/initialpose` (history reset event)
- `autoware_auto_vehicle_msgs/SteeringReport` on `/vehicle/status/steering_status`
- `autoware_auto_vehicle_msgs/ControlModeReport` on `/vehicle/status/control_mode`
- `v2x_msgs/V2XVehiclePositionArray` on `/v2x/vehicle_positions`

LiDAR is interpolated to 360 rays over the deployed 179-degree field of view (`[-89.5°, +89.5°]`). V2X velocity is derived causally from the position-only message. The auxiliary vector is:

1. longitudinal velocity
2. measured steering
3. measured longitudinal acceleration
4. previous safe steering command
5. previous safe acceleration command
6. opponent present
7. opponent relative body-frame x
8. opponent relative body-frame y
9. opponent relative body-frame vx
10. opponent relative body-frame vy
11. opponent age

The TorchScript module is called as `policy(lidar_history, auxiliary_history)`, with shapes `[1,10,360]` and `[1,10,11]`. It must return finite `[1,2]` physical `[steering_rad, acceleration_mps2]`. The exporter writes both the model and a required `<model>.metadata.json` sidecar. The node reads the sidecar **before** TorchScript loading and rejects PyTorch major/minor mismatches, preventing a newer artifact from crashing an older ROS runtime. Export with the same PyTorch major/minor installed in the Autoware image, then deploy both files together. Startup also rejects mismatched schema, history/ray counts, feature/target order, units, LiDAR range/FOV, opponent preprocessing, action bounds, or control rate.

## Build, test, and shadow launch

```bash
cd aichallenge/workspace
colcon build --packages-select racing_maneuver_controller
source install/setup.bash
pytest -q src/aichallenge_submit/racing_maneuver_controller/test
ros2 launch racing_maneuver_controller racing_maneuver.launch.xml \
  model_path:=/container/visible/path/racing_maneuver_policy.pt
```

Inspect `/racing_maneuver/debug/features` (371 floats: 360 LiDAR + 11 auxiliary), `/racing_maneuver/debug/safety_status`, and `/racing_maneuver/debug/control_cmd`. Keep `publish_commands` false until simulator validation and a single-publisher launch route have been independently reviewed.
