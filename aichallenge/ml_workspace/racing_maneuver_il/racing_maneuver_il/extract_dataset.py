"""Atomic writer/CLI for causally aligned frame datasets.

Bag-specific ROS deserialization is intentionally kept outside the generic core: callers
use :mod:`extraction` to align typed messages and pass the resulting arrays here.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .extraction import Stamped, latest_at_or_before
from .lidar import canonicalize_scan
from .opponents import OpponentObservation, summarize_opponents
from .recording_metadata import RecordingMetadata
from .schema import SCHEMA_VERSION, V2X_MAXIMUM_AGE_S

TOPICS = {
    "control": "/control/command/control_cmd",
    "scan": "/sensing/lidar/scan",
    "odom": "/localization/kinematic_state",
    "acceleration": "/localization/acceleration",
    "steering": "/vehicle/status/steering_status",
    "mode": "/vehicle/status/control_mode",
    "v2x": "/v2x/vehicle_positions",
}
OPTIONAL_TOPICS = {
    "reset": "/initialpose",
    "episode_control": "/racing_maneuver/episode_control",
    "lap": "/awsim/status",
}


@dataclass(frozen=True)
class ExtractionConfig:
    sample_rate_hz: float = 20.0
    maximum_gap_s: float = 0.15
    human_control_modes: tuple[int, ...] = (1,)
    scan_maximum_age_s: float = 0.10
    ego_maximum_age_s: float = 0.10
    acceleration_maximum_age_s: float = 0.10
    steering_maximum_age_s: float = 0.10
    mode_maximum_age_s: float = 0.50
    v2x_maximum_age_s: float = V2X_MAXIMUM_AGE_S
    require_episode_markers: bool = False
    split_on_lap: bool = False
    lap_maximum_age_s: float = 0.25

    def __post_init__(self) -> None:
        values = (
            self.sample_rate_hz,
            self.maximum_gap_s,
            self.scan_maximum_age_s,
            self.ego_maximum_age_s,
            self.acceleration_maximum_age_s,
            self.steering_maximum_age_s,
            self.mode_maximum_age_s,
            self.v2x_maximum_age_s,
            self.lap_maximum_age_s,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError(
                "rates, gaps, and source-age limits must be finite and positive"
            )
        if not self.human_control_modes:
            raise ValueError("human_control_modes must not be empty")


def _attribute(value: Any, path: str) -> Any:
    for component in path.split("."):
        value = getattr(value, component)
    return value


def _yaw(orientation: Any) -> float:
    x, y, z, w = (
        float(orientation.x),
        float(orientation.y),
        float(orientation.z),
        float(orientation.w),
    )
    if not np.all(np.isfinite((x, y, z, w))):
        raise ValueError("odometry orientation must be finite")
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _downsample_labels(
    labels: list[Stamped[Any]], rate_hz: float
) -> list[Stamped[Any]]:
    selected: list[Stamped[Any]] = []
    minimum_period = 1.0 / rate_hz
    for label in sorted(labels, key=lambda row: row.timestamp_s):
        if (
            not selected
            or label.timestamp_s - selected[-1].timestamp_s >= minimum_period - 1e-9
        ):
            selected.append(label)
    return selected


def _opponent_rows(
    message: Any, source_time_s: float, ego_vehicle_id: str
) -> list[OpponentObservation]:
    rows: list[OpponentObservation] = []
    for vehicle in message.vehicles:
        vehicle_id = str(vehicle.vehicle_id)
        if not vehicle_id or vehicle_id == ego_vehicle_id:
            continue
        x, y = float(vehicle.position.x), float(vehicle.position.y)
        if np.all(np.isfinite((x, y))):
            rows.append(OpponentObservation(vehicle_id, x, y, source_time_s))
    return rows


def _prior_opponent_map(
    v2x_stream: list[Stamped[Any]],
    current_timestamp_s: float,
    ego_vehicle_id: str,
) -> dict[str, OpponentObservation]:
    """Use the immediately prior distinct V2X report, matching the runtime tracker."""
    ordered = sorted(v2x_stream, key=lambda row: row.timestamp_s)
    index = bisect_left([row.timestamp_s for row in ordered], current_timestamp_s) - 1
    if index < 0:
        return {}
    prior = ordered[index]
    return {
        row.vehicle_id: row
        for row in _opponent_rows(prior.value, prior.timestamp_s, ego_vehicle_id)
    }


def extract_aligned_streams(
    streams: dict[str, list[Stamped[Any]]],
    *,
    recording_id: str,
    scenario_id: str,
    maneuver_class: str,
    ego_vehicle_id: str,
    config: ExtractionConfig | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build fixed-shape frames using only source data at-or-before each manual label."""
    config = config or ExtractionConfig()
    missing = sorted(set(TOPICS) - set(streams))
    if missing:
        raise ValueError(f"missing required streams: {missing}")
    if any(not streams[name] for name in TOPICS):
        empty = sorted(name for name in TOPICS if not streams[name])
        raise ValueError(f"required streams contain no messages: {empty}")

    if config.require_episode_markers and not streams.get("episode_control"):
        raise ValueError("episode markers are required but the marker stream is empty")
    if config.split_on_lap and not streams.get("lap"):
        raise ValueError("lap splitting is required but the /awsim/status stream is empty")

    labels = _downsample_labels(streams["control"], config.sample_rate_hz)
    accepted_lidar: list[np.ndarray] = []
    accepted_aux: list[np.ndarray] = []
    accepted_targets: list[np.ndarray] = []
    recording_ids: list[str] = []
    episode_ids: list[str] = []
    maneuver_classes: list[str] = []
    source_timestamps: list[np.ndarray] = []
    source_ages: list[np.ndarray] = []
    rejected: Counter[str] = Counter()
    marker_diagnostics: Counter[str] = Counter()
    previous_action = np.zeros(2, dtype=np.float32)
    episode = 0
    boundary_pending = True
    previous_label_time: float | None = None
    marker_events = iter(
        sorted(streams.get("episode_control", []), key=lambda row: row.timestamp_s)
    )
    next_marker = next(marker_events, None)
    marker_active = not config.require_episode_markers
    marker_episode = 0
    marker_start_index: int | None = None
    current_lap_id: int | None = None

    def discard_marker_episode() -> None:
        nonlocal marker_start_index
        if marker_start_index is None:
            return
        removed = len(accepted_targets) - marker_start_index
        if removed:
            rejected["discarded_episode_rows"] += removed
            for values in (
                accepted_lidar,
                accepted_aux,
                accepted_targets,
                recording_ids,
                episode_ids,
                maneuver_classes,
                source_timestamps,
                source_ages,
            ):
                del values[marker_start_index:]
        marker_start_index = None

    age_limits = {
        "scan": config.scan_maximum_age_s,
        "odom": config.ego_maximum_age_s,
        "acceleration": config.acceleration_maximum_age_s,
        "steering": config.steering_maximum_age_s,
        "mode": config.mode_maximum_age_s,
        "v2x": config.v2x_maximum_age_s,
    }
    source_order = tuple(age_limits)

    for label in labels:
        while next_marker is not None and next_marker.timestamp_s <= label.timestamp_s:
            event = str(getattr(next_marker.value, "data", "")).strip().upper()
            if event == "START":
                if marker_active:
                    discard_marker_episode()
                    marker_diagnostics["restarted_episode"] += 1
                marker_active = True
                marker_episode += 1
                marker_start_index = len(accepted_targets)
                boundary_pending = True
                previous_action = np.zeros(2, dtype=np.float32)
            elif event == "STOP":
                if marker_active:
                    marker_active = False
                    marker_start_index = None
                    boundary_pending = True
                    previous_action = np.zeros(2, dtype=np.float32)
                else:
                    marker_diagnostics["stop_without_active_episode"] += 1
            elif event == "DISCARD":
                if marker_active:
                    discard_marker_episode()
                    marker_active = False
                    boundary_pending = True
                    previous_action = np.zeros(2, dtype=np.float32)
                else:
                    marker_diagnostics["discard_without_active_episode"] += 1
            else:
                marker_diagnostics["invalid_episode_marker"] += 1
            next_marker = next(marker_events, None)
        if config.require_episode_markers and not marker_active:
            rejected["outside_marked_episode"] += 1
            previous_label_time = label.timestamp_s
            continue
        if previous_label_time is not None:
            if label.timestamp_s - previous_label_time > config.maximum_gap_s:
                boundary_pending = True
            if any(
                previous_label_time < event.timestamp_s <= label.timestamp_s
                for event in streams.get("reset", [])
            ):
                boundary_pending = True
        previous_label_time = label.timestamp_s
        try:
            selected: dict[str, Any] = {}
            ages: dict[str, float] = {}
            timestamps: dict[str, float] = {}
            for name in source_order:
                selected[name], ages[name] = latest_at_or_before(
                    streams[name], label.timestamp_s, max_age_s=age_limits[name]
                )
                timestamps[name] = label.timestamp_s - ages[name]
        except ValueError:
            rejected["stale_or_missing_source"] += 1
            boundary_pending = True
            continue

        lap_id: int | None = None
        if config.split_on_lap:
            try:
                lap_status, _ = latest_at_or_before(
                    streams["lap"],
                    label.timestamp_s,
                    max_age_s=config.lap_maximum_age_s,
                )
                lap_id = int(float(lap_status.data[1]))
                if lap_id < 1:
                    raise ValueError("lap id must be positive")
            except (AttributeError, IndexError, TypeError, ValueError):
                rejected["stale_or_invalid_lap_status"] += 1
                boundary_pending = True
                continue
            if current_lap_id != lap_id:
                current_lap_id = lap_id
                boundary_pending = True

        if int(selected["mode"].mode) not in config.human_control_modes:
            rejected["non_manual_mode"] += 1
            boundary_pending = True
            continue

        try:
            scan = selected["scan"]
            lidar = canonicalize_scan(
                scan.ranges,
                float(scan.angle_min),
                float(scan.angle_increment),
                float(scan.range_min),
                float(scan.range_max),
            )
            odometry = selected["odom"]
            position = odometry.pose.pose.position
            yaw = _yaw(odometry.pose.pose.orientation)
            speed = float(odometry.twist.twist.linear.x)
            lateral_speed = float(odometry.twist.twist.linear.y)
            cosine, sine = math.cos(yaw), math.sin(yaw)
            velocity_map = (
                cosine * speed - sine * lateral_speed,
                sine * speed + cosine * lateral_speed,
            )
            current_opponents = _opponent_rows(
                selected["v2x"], timestamps["v2x"], ego_vehicle_id
            )
            prior_opponents = _prior_opponent_map(
                streams["v2x"], timestamps["v2x"], ego_vehicle_id
            )
            opponent = summarize_opponents(
                (float(position.x), float(position.y)),
                yaw,
                velocity_map,
                current_opponents,
                label.timestamp_s,
                previous=prior_opponents,
                max_age_s=config.v2x_maximum_age_s,
            )
            target = np.asarray(
                [
                    float(label.value.lateral.steering_tire_angle),
                    float(label.value.longitudinal.acceleration),
                ],
                dtype=np.float32,
            )
            auxiliary = np.concatenate(
                (
                    np.asarray(
                        [
                            speed,
                            float(selected["steering"].steering_tire_angle),
                            float(
                                _attribute(
                                    selected["acceleration"], "accel.accel.linear.x"
                                )
                            ),
                        ],
                        dtype=np.float32,
                    ),
                    np.zeros(2, dtype=np.float32)
                    if boundary_pending
                    else previous_action,
                    opponent.as_array(),
                )
            ).astype(np.float32)
            if (
                auxiliary.shape != (11,)
                or not np.all(np.isfinite(auxiliary))
                or not np.all(np.isfinite(target))
            ):
                raise ValueError("non-finite or malformed frame")
        except (AttributeError, TypeError, ValueError) as error:
            rejected[f"invalid_message:{type(error).__name__}"] += 1
            boundary_pending = True
            continue

        if boundary_pending and accepted_targets:
            episode += 1
        accepted_lidar.append(lidar)
        accepted_aux.append(auxiliary)
        accepted_targets.append(target)
        recording_ids.append(recording_id)
        if config.split_on_lap:
            episode_ids.append(f"{scenario_id}::lap-{lap_id}")
        elif config.require_episode_markers:
            episode_ids.append(f"{scenario_id}::episode-{marker_episode}")
        else:
            episode_ids.append(f"{scenario_id}::{episode}")
        maneuver_classes.append(maneuver_class)
        source_timestamps.append(
            np.asarray([timestamps[name] for name in source_order], dtype=np.float64)
        )
        source_ages.append(
            np.asarray([ages[name] for name in source_order], dtype=np.float32)
        )
        previous_action = target.copy()
        boundary_pending = False

    while next_marker is not None:
        event = str(getattr(next_marker.value, "data", "")).strip().upper()
        if event == "STOP" and marker_active:
            marker_active = False
            marker_start_index = None
        elif event == "DISCARD" and marker_active:
            discard_marker_episode()
            marker_active = False
        elif event == "START":
            if marker_active:
                discard_marker_episode()
                marker_diagnostics["restarted_episode"] += 1
            marker_active = True
            marker_episode += 1
            marker_start_index = len(accepted_targets)
        elif event not in {"STOP", "DISCARD"}:
            marker_diagnostics["invalid_episode_marker"] += 1
        else:
            marker_diagnostics[f"{event.lower()}_without_active_episode"] += 1
        next_marker = next(marker_events, None)

    if config.require_episode_markers and marker_active:
        discard_marker_episode()
        rejected["unclosed_episode_rows"] += 1

    count = len(accepted_targets)
    arrays = {
        "lidar": np.stack(accepted_lidar).astype(np.float32)
        if count
        else np.empty((0, 360), dtype=np.float32),
        "aux": np.stack(accepted_aux).astype(np.float32)
        if count
        else np.empty((0, 11), dtype=np.float32),
        "targets": np.stack(accepted_targets).astype(np.float32)
        if count
        else np.empty((0, 2), dtype=np.float32),
        "recording_ids": np.asarray(recording_ids),
        "episode_ids": np.asarray(episode_ids),
        "maneuver_classes": np.asarray(maneuver_classes),
        "source_timestamps_s": np.stack(source_timestamps)
        if count
        else np.empty((0, len(source_order)), dtype=np.float64),
        "source_ages_s": np.stack(source_ages)
        if count
        else np.empty((0, len(source_order)), dtype=np.float32),
        "source_order": np.asarray(source_order),
    }
    report = {
        "input_rows": len(labels),
        "accepted_rows": count,
        "rejected_rows": int(sum(rejected.values())),
        "rejected_reasons": dict(sorted(rejected.items())),
        "episode_marker_diagnostics": dict(sorted(marker_diagnostics.items())),
        "source_order": list(source_order),
        "episodes": len(set(episode_ids)),
        "maneuver_counts": {maneuver_class: count} if count else {},
    }
    return arrays, report


def _stamp_seconds(message: Any) -> float | None:
    candidates = []
    for path in ("header.stamp", "stamp", "lateral.stamp", "longitudinal.stamp"):
        try:
            candidates.append(_attribute(message, path))
        except AttributeError:
            continue
    for stamp in candidates:
        try:
            seconds = int(stamp.sec) + int(stamp.nanosec) * 1e-9
        except (AttributeError, TypeError, ValueError):
            continue
        if math.isfinite(seconds) and seconds > 0.0:
            return seconds
    return None


def _clock_time_at_or_before(
    clock_rows: list[tuple[float, float]], receive_time_s: float
) -> float:
    """Map a headerless marker's bag time to the latest causal simulated clock time."""
    index = bisect_right([row[0] for row in clock_rows], receive_time_s) - 1
    if index < 0:
        raise ValueError("episode marker predates the first recorded simulation clock")
    return clock_rows[index][1]


def read_bag_streams(bag_path: str | Path) -> dict[str, list[Stamped[Any]]]:
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as error:
        raise RuntimeError("rosbags is required for ROS bag extraction") from error
    bag_path = Path(bag_path)
    topic_to_name = {
        **{topic: name for name, topic in TOPICS.items()},
        **{topic: name for name, topic in OPTIONAL_TOPICS.items()},
    }
    streams: dict[str, list[Stamped[Any]]] = {
        name: [] for name in (*TOPICS, *OPTIONAL_TOPICS)
    }
    with AnyReader([bag_path]) as reader:
        available = {connection.topic for connection in reader.connections}
        missing = sorted(set(TOPICS.values()) - available)
        if missing:
            raise ValueError(f"missing required bag topics: {missing}")
        clock_connections = [
            connection
            for connection in reader.connections
            if connection.topic == "/clock"
        ]
        clock_rows: list[tuple[float, float]] = []
        for connection, receive_time_ns, raw in reader.messages(
            connections=clock_connections
        ):
            message = reader.deserialize(raw, connection.msgtype)
            clock_rows.append(
                (
                    receive_time_ns * 1e-9,
                    int(message.clock.sec) + int(message.clock.nanosec) * 1e-9,
                )
            )
        clock_rows.sort(key=lambda row: row[0])
        connections = [
            connection
            for connection in reader.connections
            if connection.topic in topic_to_name
        ]
        for connection, receive_time_ns, raw in reader.messages(
            connections=connections
        ):
            message = reader.deserialize(raw, connection.msgtype)
            name = topic_to_name[connection.topic]
            source_time = _stamp_seconds(message)
            if source_time is None and name in {"episode_control", "lap"}:
                source_time = _clock_time_at_or_before(
                    clock_rows, receive_time_ns * 1e-9
                )
            if source_time is None:
                continue
            streams[name].append(Stamped(source_time, message))
    for rows in streams.values():
        rows.sort(key=lambda row: row.timestamp_s)
    return streams


def extract_bag(
    bag_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    config: ExtractionConfig | None = None,
) -> dict[str, Any]:
    config = config or ExtractionConfig()
    bag_path = Path(bag_path)
    metadata_path = bag_path / "recording.json"
    if not metadata_path.is_file():
        raise ValueError(f"recording metadata is missing: {metadata_path}")
    metadata_payload = json.loads(metadata_path.read_text())
    metadata = RecordingMetadata.from_mapping(metadata_payload)
    if bool(metadata.provenance.get("episode_markers_required", False)):
        config = replace(config, require_episode_markers=True)
    if metadata.expert_source == "mpc":
        config = replace(config, split_on_lap=True)
    streams = read_bag_streams(bag_path)
    arrays, alignment_report = extract_aligned_streams(
        streams,
        recording_id=metadata.recording_id,
        scenario_id=metadata.scenario_id,
        maneuver_class=metadata.maneuver_class,
        ego_vehicle_id=metadata.ego_vehicle_id,
        config=config,
    )
    if not alignment_report["accepted_rows"]:
        raise ValueError("zero confirmed manual synchronized rows survived extraction")
    report = write_processed_dataset(
        output_path,
        report_path,
        arrays,
        rejected_reasons=alignment_report["rejected_reasons"],
        input_rows=alignment_report["input_rows"],
    )
    report.update(
        {
            "bag": str(bag_path),
            "recording_metadata": metadata_payload,
            "source_order": alignment_report["source_order"],
            "episode_marker_diagnostics": alignment_report[
                "episode_marker_diagnostics"
            ],
            "timestamp_policy": (
                "ROS source stamps; latest-at-or-before causal joins; "
                "episode-control events map from rosbag receive time to the latest causal /clock"
            ),
        }
    )
    destination = Path(report_path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    return report


_REQUIRED = (
    "lidar",
    "aux",
    "targets",
    "recording_ids",
    "episode_ids",
    "maneuver_classes",
)


def write_processed_dataset(
    output_path: str | Path,
    report_path: str | Path,
    arrays: dict[str, np.ndarray],
    *,
    rejected_reasons: dict[str, int] | None = None,
    input_rows: int | None = None,
) -> dict:
    missing = [name for name in _REQUIRED if name not in arrays]
    if missing:
        raise ValueError(f"missing arrays: {missing}")
    values = {name: np.asarray(value) for name, value in arrays.items()}
    n = len(values["lidar"])
    if (
        values["lidar"].shape != (n, 360)
        or values["aux"].shape != (n, 11)
        or values["targets"].shape != (n, 2)
    ):
        raise ValueError("processed arrays violate schema shapes")
    if any(len(values[name]) != n for name in _REQUIRED):
        raise ValueError("processed arrays have unequal row counts")
    if not all(np.isfinite(values[name]).all() for name in ("lidar", "aux", "targets")):
        raise ValueError("processed numeric arrays must be finite")
    reasons = rejected_reasons or {}
    rejected = sum(int(value) for value in reasons.values())
    total = int(input_rows if input_rows is not None else n + rejected)
    if total != n + rejected:
        raise ValueError("input row accounting does not balance")
    classes = Counter(str(value) for value in values["maneuver_classes"])
    report = {
        "schema_version": SCHEMA_VERSION,
        "input_rows": total,
        "accepted_rows": n,
        "rejected_rows": rejected,
        "rejected_reasons": dict(sorted(reasons.items())),
        "episode_ids": sorted({str(x) for x in values["episode_ids"]}),
        "maneuver_counts": dict(sorted(classes.items())),
        "command_distributions": {
            "steering": {
                "min": float(values["targets"][:, 0].min()),
                "max": float(values["targets"][:, 0].max()),
            },
            "longitudinal_acceleration": {
                "min": float(values["targets"][:, 1].min()),
                "max": float(values["targets"][:, 1].max()),
            },
        },
    }
    if "source_ages_s" in values:
        report["source_age_distributions_s"] = {
            "max": np.max(values["source_ages_s"], axis=0).astype(float).tolist(),
            "mean": np.mean(values["source_ages_s"], axis=0).astype(float).tolist(),
        }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent, suffix=".npz", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(
            temporary,
            **values,
            schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int64),
        )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    report_temp = destination.with_suffix(destination.suffix + ".tmp")
    report_temp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    report_temp.replace(destination)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Extract a source-stamped manual maneuver ROS bag into a causal dataset"
    )
    parser.add_argument("bag")
    parser.add_argument("output_npz")
    parser.add_argument("report_json")
    parser.add_argument("--human-control-mode", type=int, action="append", default=None)
    args = parser.parse_args(argv)
    config = ExtractionConfig(human_control_modes=tuple(args.human_control_mode or [1]))
    print(
        json.dumps(
            extract_bag(args.bag, args.output_npz, args.report_json, config),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
