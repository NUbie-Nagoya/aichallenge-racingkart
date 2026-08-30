# AWSIM LiDAR scan contract

## Canonical policy grid

The model grid contains 360 samples in increasing physical angle from -135° to +135°, inclusive, relative to vehicle forward (+x). Ranges remain metres. The canonicalizer verifies that the source angular interval covers the full target field of view and interpolates by angle rather than array index.

- finite values are clipped to `[range_min_m, min(range_max_m, model_max_range_m)]`;
- positive/negative infinity and NaN use `min(range_max_m, model_max_range_m)`;
- normalization is not performed by LiDAR preprocessing.

## Live AWSIM status

A live simulator scan was not available during standalone implementation. Therefore source ray count, source FOV, bounds, rate, and invalid-return frequencies are **not asserted here**. Before training on AWSIM data, record:

```bash
ros2 topic type /sensing/lidar/scan
ros2 topic echo --once /sensing/lidar/scan
ros2 topic hz /sensing/lidar/scan
```

Save a fixture NPZ with `ranges`, `angle_min_rad`, `angle_increment_rad`, `range_min_m`, `range_max_m`, and optional `rate_hz`, then run:

```bash
racing-maneuver-inspect-scan scan_fixture.npz --output scan_contract.json
```

Review the generated finite/NaN/infinity counts and confirm `canonicalize_scan` accepts the measured coverage. Update this document only with captured evidence; do not infer geometry from ray count.
