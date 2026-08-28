from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from racing_maneuver_controller.controller_core import (
    AUX_FEATURE_NAMES,
    ControllerCore,
    OpponentObservation,
    canonicalize_lidar,
    mode_is_permitted,
    pose_jump_exceeds,
    publication_decision,
    sources_are_fresh,
    summarize_opponents,
    torch_versions_compatible,
    validate_artifact_metadata,
)


class RecordingPolicy:
    def __init__(self, output=(0.0, 0.0)):
        self.output = output
        self.calls = []

    def __call__(self, lidar, auxiliary):
        self.calls.append((lidar.clone(), auxiliary.clone()))
        return torch.tensor([self.output], dtype=torch.float32)


def frame(value: float):
    return np.full(360, value, dtype=np.float32), np.full(11, value, dtype=np.float32)


def make_core(policy=None):
    return ControllerCore(
        policy or RecordingPolicy(),
        history_frames=10,
        lidar_rays=360,
        auxiliary_dim=11,
        steering_limit_rad=0.5,
        acceleration_limit_mps2=2.0,
        steering_rate_limit_rad_s=1.0,
        acceleration_rate_limit_mps3=2.0,
    )


def test_history_warms_up_chronologically_and_reset_clears_causal_state():
    policy = RecordingPolicy()
    core = make_core(policy)
    for i in range(9):
        core.append_frame(*frame(float(i)))
        assert not core.ready
    core.append_frame(*frame(9.0))
    assert core.ready
    core.infer_and_limit(0.05)
    lidar, aux = policy.calls[0]
    assert lidar.shape == (1, 10, 360)
    assert aux.shape == (1, 10, 11)
    np.testing.assert_array_equal(lidar[0, :, 0], np.arange(10))
    np.testing.assert_array_equal(aux[0, :, 0], np.arange(10))
    core.reset()
    assert not core.ready
    np.testing.assert_array_equal(core.previous_safe_output, [0.0, 0.0])


def test_invalid_frame_rejects_and_resets_existing_history():
    core = make_core()
    for _ in range(3):
        core.append_frame(*frame(1.0))
    bad = np.ones(360, dtype=np.float32)
    bad[4] = np.nan
    with pytest.raises(ValueError, match="finite"):
        core.append_frame(bad, np.ones(11, dtype=np.float32))
    assert core.history_size == 0


def test_policy_contract_clip_rate_limit_and_previous_safe_feedback():
    policy = RecordingPolicy((10.0, -10.0))
    core = make_core(policy)
    for _ in range(10):
        core.append_frame(*frame(1.0))
    first = core.infer_and_limit(0.1)
    np.testing.assert_allclose(first, [0.1, -0.2], atol=1e-6)
    np.testing.assert_allclose(core.previous_safe_output, first)
    second = core.infer_and_limit(0.1)
    np.testing.assert_allclose(second, [0.2, -0.4], atol=1e-6)


def test_nonfinite_or_wrong_policy_output_is_rejected_and_resets():
    for output in ((math.nan, 0.0), (0.0,)):
        core = make_core(RecordingPolicy(output))
        for _ in range(10):
            core.append_frame(*frame(1.0))
        with pytest.raises(RuntimeError, match=r"finite.*\[1, 2\]"):
            core.infer_and_limit(0.05)
        assert core.history_size == 0
        np.testing.assert_array_equal(core.previous_safe_output, [0.0, 0.0])


def test_canonical_lidar_default_accepts_the_deployed_179_degree_scan():
    result = canonicalize_lidar(
        np.linspace(1.0, 5.0, 750),
        angle_min=-1.5666074752807617,
        angle_increment=0.004188789986073971,
        range_min=0.0,
        range_max=25.0,
    )

    assert result.shape == (360,)
    assert np.all(np.isfinite(result))


def test_canonical_lidar_is_360_finite_rays_over_requested_fov():
    ranges = np.linspace(1.0, 5.0, 5)
    ranges[2] = np.inf
    result = canonicalize_lidar(
        ranges,
        angle_min=-math.pi / 2,
        angle_increment=math.pi / 4,
        range_min=0.1,
        range_max=8.0,
        target_angle_min=-math.pi / 2,
        target_angle_max=math.pi / 2,
    )
    assert result.shape == (360,)
    assert np.all(np.isfinite(result))
    assert result[0] == pytest.approx(1.0)
    assert result[-1] == pytest.approx(5.0)
    assert np.max(result) <= 8.0

    clipped = canonicalize_lidar(
        np.full(5, 6.0),
        angle_min=-math.pi / 2,
        angle_increment=math.pi / 4,
        range_min=0.1,
        range_max=8.0,
        target_angle_min=-math.pi / 2,
        target_angle_max=math.pi / 2,
        model_max_range=3.0,
    )
    assert np.max(clipped) <= 3.0


def test_pose_jump_detection_rejects_reset_discontinuity():
    assert not pose_jump_exceeds(None, (10.0, 20.0), 3.0)
    assert not pose_jump_exceeds((10.0, 20.0), (12.0, 20.0), 3.0)
    assert pose_jump_exceeds((10.0, 20.0), (14.0, 20.0), 3.0)
    with pytest.raises(ValueError, match="finite"):
        pose_jump_exceeds((10.0, 20.0), (math.nan, 20.0), 3.0)


def test_freshness_rejects_missing_future_and_stale_sources():
    now = 1_000_000_000
    limits = {"scan": 100, "ego": 100, "v2x": 250}
    assert sources_are_fresh(now, {"scan": now - 50_000_000, "ego": now, "v2x": now - 200_000_000}, limits)
    assert not sources_are_fresh(now, {"scan": now, "ego": now}, limits)
    assert not sources_are_fresh(now, {"scan": now + 1, "ego": now, "v2x": now}, limits)
    assert not sources_are_fresh(now, {"scan": now - 101_000_000, "ego": now, "v2x": now}, limits)


def test_opponent_summary_uses_nearest_and_body_frame_relative_kinematics():
    opponents = [
        OpponentObservation("far", 10.0, 0.0, 1.0, 0.0, 9.8),
        OpponentObservation("near", 2.0, 1.0, 3.0, 2.0, 9.9),
    ]
    summary = summarize_opponents(
        ego_x=0.0, ego_y=0.0, ego_yaw=math.pi / 2,
        ego_vx=1.0, ego_vy=0.0, now_seconds=10.0, opponents=opponents,
    )
    np.testing.assert_allclose(summary, [1.0, 1.0, -2.0, 2.0, -2.0, 0.1], atol=1e-6)
    assert AUX_FEATURE_NAMES[-6:] == (
        "opponent_present", "opponent_relative_x_body_m", "opponent_relative_y_body_m",
        "opponent_relative_vx_body_mps", "opponent_relative_vy_body_mps", "opponent_age_s",
    )


def test_no_opponent_summary_is_finite_and_marks_absence():
    result = summarize_opponents(0, 0, 0, 0, 0, 4.0, [])
    np.testing.assert_array_equal(result, np.zeros(6, dtype=np.float32))
    behind = [OpponentObservation("behind", -1.0, 0.0, 0.0, 0.0, 4.0)]
    np.testing.assert_array_equal(
        summarize_opponents(0, 0, 0, 0, 0, 4.0, behind),
        np.zeros(6, dtype=np.float32),
    )


def test_shadow_and_actuation_mode_gates_are_separate():
    assert mode_is_permitted(4, False, {1}, {1, 4})
    assert not mode_is_permitted(4, True, {1}, {1, 4})
    assert publication_decision(4, False, {1}, {1, 4}) == (True, False)
    assert publication_decision(1, True, {1}, {1, 4}) == (True, True)
    assert publication_decision(2, False, {1}, {1, 4}) == (False, False)


def test_torch_artifact_version_must_match_runtime_major_minor():
    assert torch_versions_compatible("1.8.0a0", "1.8.2+cpu")
    assert torch_versions_compatible("2.13.0+cu130", "2.13.1+cpu")
    assert not torch_versions_compatible("2.13.0", "1.8.0")
    assert not torch_versions_compatible("unknown", "1.8.0")


def test_artifact_metadata_must_match_runtime_feature_contract():
    metadata = {
        "schema_version": 1,
        "canonical_lidar_rays": 360,
        "canonical_lidar_fov_deg": 179.0,
        "model_max_range_m": 30.0,
        "opponent_forward_corridor_half_width_m": 5.0,
        "opponent_maximum_speed_mps": 50.0,
        "v2x_maximum_age_s": 0.25,
        "history_length": 10,
        "control_rate_hz": 20.0,
        "feature_ordering": ["canonical_lidar_ranges_m", *AUX_FEATURE_NAMES],
        "target_ordering": ["steering_tire_angle_rad", "longitudinal_acceleration_mps2"],
        "output_units": ["rad", "m/s^2"],
        "action_low": [-1.0, -3.0],
        "action_high": [1.0, 3.0],
    }
    validate_artifact_metadata(metadata, expected_rate_hz=20.0)
    invalid = dict(metadata, history_length=9)
    with pytest.raises(ValueError, match="history_length"):
        validate_artifact_metadata(invalid, expected_rate_hz=20.0)


def test_installed_node_shebang_starts_at_byte_zero():
    node = Path(__file__).parents[1] / "racing_maneuver_controller" / "node.py"
    assert node.read_bytes().startswith(b"#!/usr/bin/env python3\n")
