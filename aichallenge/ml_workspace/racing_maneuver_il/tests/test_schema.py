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
    assert schema.SCHEMA_VERSION == 1


def test_raw_aux_contract_has_required_order():
    assert schema.AUX_FEATURE_NAMES == (
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
