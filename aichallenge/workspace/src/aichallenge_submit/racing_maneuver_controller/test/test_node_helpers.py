from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from racing_maneuver_controller.node_helpers import (
    V2XTracker,
    assemble_auxiliary,
    quaternion_yaw,
)


class Point:
    def __init__(self, x, y):
        self.x, self.y = x, y


class Vehicle:
    def __init__(self, vehicle_id, x, y):
        self.vehicle_id = vehicle_id
        self.position = Point(x, y)


def test_auxiliary_order_includes_previous_safe_output_and_opponent_summary():
    result = assemble_auxiliary(
        speed=4.0,
        steering=0.2,
        acceleration=-0.3,
        previous_safe=np.asarray([0.1, -0.2]),
        opponent_summary=np.asarray([1.0, 3.0, 2.0, 0.5, -0.5, 0.1]),
    )
    np.testing.assert_allclose(result, [4, 0.2, -0.3, 0.1, -0.2, 1, 3, 2, 0.5, -0.5, 0.1])


def test_v2x_tracker_derives_velocity_causally_and_excludes_ego():
    tracker = V2XTracker(ego_vehicle_id="ego", maximum_speed_mps=50.0)
    tracker.update([Vehicle("ego", 9, 9), Vehicle("other", 1, 2)], 1.0)
    first = tracker.observations()
    assert len(first) == 1
    assert first[0].vx == 0.0
    tracker.update([Vehicle("other", 3, 5)], 2.0)
    second = tracker.observations()[0]
    assert (second.vx, second.vy, second.stamp_seconds) == (2.0, 3.0, 2.0)


def test_v2x_tracker_reset_drops_velocity_history():
    tracker = V2XTracker(ego_vehicle_id="ego", maximum_speed_mps=50.0)
    tracker.update([Vehicle("other", 0, 0)], 1.0)
    tracker.update([Vehicle("other", 2, 0)], 2.0)
    tracker.reset()
    assert tracker.observations() == []


def test_quaternion_yaw_is_finite():
    class Quaternion:
        x, y, z, w = 0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)

    assert quaternion_yaw(Quaternion()) == pytest.approx(np.pi / 2)


def test_deployment_defaults_are_shadow_only_and_launch_visible():
    root = Path(__file__).parents[1]
    config = (root / "config" / "racing_maneuver_controller.param.yaml").read_text()
    launch = (root / "launch" / "racing_maneuver.launch.xml").read_text()
    assert "publish_commands: false" in config
    assert '<arg name="publish_commands" default="false"/>' in launch
    assert "/racing_maneuver/debug/control_cmd" in config
    assert "/control/command/control_cmd" in config
