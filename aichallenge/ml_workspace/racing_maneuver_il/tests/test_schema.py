from racing_maneuver_il import schema


def test_feature_and_target_names_are_unique():
    assert len(schema.AUX_FEATURE_NAMES) == len(set(schema.AUX_FEATURE_NAMES))
    assert len(schema.TARGET_NAMES) == len(set(schema.TARGET_NAMES))


def test_history_contract_is_10_frames_at_20_hz():
    assert schema.HISTORY_LENGTH == 10
    assert schema.CONTROL_RATE_HZ == 20.0


def test_canonical_lidar_contract_is_360_rays_over_179_degrees():
    assert schema.CANONICAL_LIDAR_RAYS == 360
    assert schema.CANONICAL_LIDAR_FOV_DEG == 179.0
    assert schema.SCHEMA_VERSION == 3


def test_raw_aux_contract_has_required_order():
    assert schema.AUX_FEATURE_NAMES == (
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


def test_policy_targets_are_steering_and_nonnegative_target_speed():
    assert schema.TARGET_NAMES == (
        "steering_tire_angle_rad",
        "target_longitudinal_speed_mps",
    )
