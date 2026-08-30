from types import SimpleNamespace as NS

import numpy as np

from racing_maneuver_il.extract_dataset import (
    ExtractionConfig,
    _clock_time_at_or_before,
    extract_aligned_streams,
)
from racing_maneuver_il.extraction import Stamped


def ns(**kwargs):
    return NS(**kwargs)


def scan(value):
    return ns(
        ranges=[value] * 1081,
        angle_min=-2.35619449,
        angle_increment=4.71238898 / 1080,
        range_min=0.1,
        range_max=30.0,
    )


def odom(x=0.0, speed=4.0):
    return ns(
        pose=ns(
            pose=ns(position=ns(x=x, y=0.0), orientation=ns(x=0.0, y=0.0, z=0.0, w=1.0))
        ),
        twist=ns(twist=ns(linear=ns(x=speed, y=0.0))),
    )


def command(steering, acceleration):
    return ns(
        lateral=ns(steering_tire_angle=steering),
        longitudinal=ns(acceleration=acceleration),
    )


def test_episode_marker_receive_time_maps_causally_to_simulation_clock():
    clock = [(100.0, 1.0), (101.0, 1.1), (102.0, 1.2)]
    assert _clock_time_at_or_before(clock, 101.8) == 1.1


def test_extract_aligned_streams_is_causal_manual_only_and_resets_prior_action():
    streams = {
        "control": [
            Stamped(1.00, command(0.1, 0.2)),
            Stamped(1.05, command(0.2, 0.3)),
        ],
        "scan": [Stamped(0.99, scan(5.0)), Stamped(1.01, scan(1.0))],
        "odom": [Stamped(0.99, odom()), Stamped(1.04, odom(x=0.2))],
        "acceleration": [Stamped(0.99, ns(accel=ns(accel=ns(linear=ns(x=0.4)))))],
        "steering": [Stamped(0.99, ns(steering_tire_angle=0.05))],
        "mode": [Stamped(0.99, ns(mode=4))],
        "v2x": [Stamped(0.99, ns(vehicles=[]))],
    }
    arrays, report = extract_aligned_streams(
        streams,
        recording_id="recording-a",
        scenario_id="scenario-a",
        maneuver_class="follow",
        ego_vehicle_id="ego",
        config=ExtractionConfig(sample_rate_hz=20.0, human_control_modes=(4,)),
    )

    assert report["accepted_rows"] == 2
    np.testing.assert_allclose(arrays["lidar"][0], 5.0)
    np.testing.assert_allclose(arrays["lidar"][1], 1.0)
    np.testing.assert_allclose(arrays["aux"][0, 3:5], [0.0, 0.0])
    np.testing.assert_allclose(arrays["aux"][1, 3:5], [0.1, 0.2])
    np.testing.assert_allclose(arrays["targets"], [[0.1, 0.2], [0.2, 0.3]])
    assert arrays["recording_ids"].tolist() == ["recording-a", "recording-a"]


def vehicle(vehicle_id, x, y=0.0):
    return ns(vehicle_id=vehicle_id, position=ns(x=x, y=y))


def test_initialpose_event_breaks_episode_and_resets_previous_action():
    streams = {
        "control": [Stamped(1.00, command(0.1, 0.2)), Stamped(1.05, command(0.2, 0.3))],
        "scan": [Stamped(0.99, scan(5.0)), Stamped(1.04, scan(5.0))],
        "odom": [Stamped(0.99, odom()), Stamped(1.04, odom())],
        "acceleration": [Stamped(0.99, ns(accel=ns(accel=ns(linear=ns(x=0.0)))))],
        "steering": [Stamped(0.99, ns(steering_tire_angle=0.0))],
        "mode": [Stamped(0.99, ns(mode=4))],
        "v2x": [Stamped(0.99, ns(vehicles=[]))],
        "reset": [Stamped(1.025, ns())],
    }
    arrays, _ = extract_aligned_streams(
        streams,
        recording_id="r",
        scenario_id="s",
        maneuver_class="recover",
        ego_vehicle_id="ego",
        config=ExtractionConfig(human_control_modes=(4,)),
    )
    np.testing.assert_allclose(arrays["aux"][:, 3:5], 0.0)
    assert arrays["episode_ids"].tolist() == ["s::0", "s::1"]


def test_lap_split_uses_latest_causal_status_and_resets_action_history():
    streams = {
        "control": [
            Stamped(1.00, command(0.1, 0.2)),
            Stamped(1.05, command(0.2, 0.3)),
            Stamped(1.10, command(0.3, 0.4)),
        ],
        "scan": [
            Stamped(0.99, scan(5.0)),
            Stamped(1.04, scan(5.0)),
            Stamped(1.09, scan(5.0)),
        ],
        "odom": [Stamped(0.99, odom()), Stamped(1.04, odom()), Stamped(1.09, odom())],
        "acceleration": [
            Stamped(0.99, ns(accel=ns(accel=ns(linear=ns(x=0.0))))),
            Stamped(1.09, ns(accel=ns(accel=ns(linear=ns(x=0.0))))),
        ],
        "steering": [
            Stamped(0.99, ns(steering_tire_angle=0.0)),
            Stamped(1.09, ns(steering_tire_angle=0.0)),
        ],
        "mode": [Stamped(0.99, ns(mode=4))],
        "v2x": [Stamped(0.99, ns(vehicles=[]))],
        # The future lap-2 status at 1.07 must not label the 1.05 command.
        "lap": [Stamped(0.99, ns(data=[0.0, 1.0])), Stamped(1.07, ns(data=[0.0, 2.0]))],
    }
    arrays, report = extract_aligned_streams(
        streams,
        recording_id="r",
        scenario_id="s",
        maneuver_class="follow",
        ego_vehicle_id="ego",
        config=ExtractionConfig(
            sample_rate_hz=20.0, human_control_modes=(4,), split_on_lap=True
        ),
    )

    assert arrays["episode_ids"].tolist() == ["s::lap-1", "s::lap-1", "s::lap-2"]
    np.testing.assert_allclose(arrays["aux"][:, 3:5], [[0.0, 0.0], [0.1, 0.2], [0.0, 0.0]])
    assert report["episodes"] == 2


def test_reused_v2x_report_preserves_velocity_from_prior_distinct_report():
    common = {
        "control": [Stamped(1.00, command(0.1, 0.2)), Stamped(1.05, command(0.2, 0.3))],
        "scan": [Stamped(0.99, scan(5.0)), Stamped(1.04, scan(5.0))],
        "odom": [Stamped(0.99, odom(speed=0.0)), Stamped(1.04, odom(speed=0.0))],
        "acceleration": [Stamped(0.99, ns(accel=ns(accel=ns(linear=ns(x=0.0)))))],
        "steering": [Stamped(0.99, ns(steering_tire_angle=0.0))],
        "mode": [Stamped(0.99, ns(mode=4))],
        "v2x": [
            Stamped(0.89, ns(vehicles=[vehicle("other", 1.0)])),
            Stamped(0.99, ns(vehicles=[vehicle("other", 2.0)])),
        ],
    }
    arrays, _ = extract_aligned_streams(
        common,
        recording_id="r",
        scenario_id="s",
        maneuver_class="follow",
        ego_vehicle_id="ego",
        config=ExtractionConfig(sample_rate_hz=20.0, human_control_modes=(4,)),
    )
    np.testing.assert_allclose(arrays["aux"][:, 8], [10.0, 10.0], atol=1e-5)


def test_extract_aligned_streams_rejects_non_manual_and_stale_rows():
    streams = {
        "control": [Stamped(1.0, command(0.1, 0.2)), Stamped(2.0, command(0.2, 0.3))],
        "scan": [Stamped(0.99, scan(5.0))],
        "odom": [Stamped(0.99, odom())],
        "acceleration": [Stamped(0.99, ns(accel=ns(accel=ns(linear=ns(x=0.4)))))],
        "steering": [Stamped(0.99, ns(steering_tire_angle=0.05))],
        "mode": [Stamped(0.99, ns(mode=1)), Stamped(1.99, ns(mode=4))],
        "v2x": [Stamped(0.99, ns(vehicles=[]))],
    }
    arrays, report = extract_aligned_streams(
        streams,
        recording_id="r",
        scenario_id="s",
        maneuver_class="follow",
        ego_vehicle_id="ego",
        config=ExtractionConfig(sample_rate_hz=20.0, human_control_modes=(4,)),
    )
    assert len(arrays["targets"]) == 0
    assert report["rejected_reasons"]["non_manual_mode"] == 1
    assert report["rejected_reasons"]["stale_or_missing_source"] == 1


def test_episode_markers_keep_closed_interval_and_discard_active_interval():
    streams = {
        "control": [
            Stamped(1.00, command(0.1, 0.2)),
            Stamped(1.05, command(0.2, 0.3)),
            Stamped(1.10, command(0.3, 0.4)),
            Stamped(1.15, command(0.4, 0.5)),
        ],
        "scan": [
            Stamped(0.99, scan(5.0)),
            Stamped(1.04, scan(5.0)),
            Stamped(1.09, scan(5.0)),
            Stamped(1.14, scan(5.0)),
        ],
        "odom": [
            Stamped(0.99, odom()),
            Stamped(1.04, odom()),
            Stamped(1.09, odom()),
            Stamped(1.14, odom()),
        ],
        "acceleration": [
            Stamped(0.99, ns(accel=ns(accel=ns(linear=ns(x=0.0))))),
            Stamped(1.09, ns(accel=ns(accel=ns(linear=ns(x=0.0))))),
        ],
        "steering": [
            Stamped(0.99, ns(steering_tire_angle=0.0)),
            Stamped(1.09, ns(steering_tire_angle=0.0)),
        ],
        "mode": [Stamped(0.99, ns(mode=4)), Stamped(1.09, ns(mode=4))],
        "v2x": [Stamped(0.99, ns(vehicles=[])), Stamped(1.09, ns(vehicles=[]))],
        "episode_control": [
            Stamped(1.01, ns(data="START")),
            Stamped(1.08, ns(data="STOP")),
            Stamped(1.09, ns(data="START")),
            Stamped(1.16, ns(data="DISCARD")),
        ],
    }
    arrays, report = extract_aligned_streams(
        streams,
        recording_id="r",
        scenario_id="s",
        maneuver_class="follow",
        ego_vehicle_id="ego",
        config=ExtractionConfig(
            sample_rate_hz=20.0, human_control_modes=(4,), require_episode_markers=True
        ),
    )
    np.testing.assert_allclose(arrays["targets"], [[0.2, 0.3]])
    assert arrays["episode_ids"].tolist() == ["s::episode-1"]
    assert report["rejected_reasons"]["outside_marked_episode"] == 1
    assert report["rejected_reasons"]["discarded_episode_rows"] == 2
    assert report["input_rows"] == report["accepted_rows"] + report["rejected_rows"]
