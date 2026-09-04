"""Immutable raw feature and action contract."""

SCHEMA_VERSION = 3
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
    "absolute_position_x_m",
    "absolute_position_y_m",
    "sin_yaw",
    "cos_yaw",
    "previous_safe_steering_command_rad",
    "previous_safe_longitudinal_command",
)
TARGET_NAMES = (
    "steering_tire_angle_rad",
    "target_longitudinal_speed_mps",
)
