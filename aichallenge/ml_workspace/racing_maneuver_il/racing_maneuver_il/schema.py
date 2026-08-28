"""Immutable raw feature and action contract."""

SCHEMA_VERSION = 1
CANONICAL_LIDAR_RAYS = 360
CANONICAL_LIDAR_FOV_DEG = 179.0
MODEL_MAX_RANGE_M = 30.0
OPPONENT_FORWARD_CORRIDOR_HALF_WIDTH_M = 5.0
OPPONENT_MAXIMUM_SPEED_MPS = 50.0
V2X_MAXIMUM_AGE_S = 0.25
HISTORY_LENGTH = 10
CONTROL_RATE_HZ = 20.0
AUX_FEATURE_NAMES = (
    "longitudinal_velocity_mps",
    "measured_steering_rad",
    "longitudinal_acceleration_mps2",
    "previous_safe_steering_command_rad",
    "previous_safe_longitudinal_command",
    "opponent_present",
    "opponent_relative_x_body_m",
    "opponent_relative_y_body_m",
    "opponent_relative_vx_body_mps",
    "opponent_relative_vy_body_mps",
    "opponent_age_s",
)
TARGET_NAMES = (
    "steering_tire_angle_rad",
    "longitudinal_acceleration_mps2",
)
