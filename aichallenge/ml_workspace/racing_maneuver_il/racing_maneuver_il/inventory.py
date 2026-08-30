"""Structural and provenance inventory for manual maneuver ROS bags."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from .recording_metadata import EXPECTED_COMMAND_OWNERS, RecordingMetadata

REQUIRED_TOPIC_TYPES = {
    "/clock": "rosgraph_msgs/msg/Clock",
    "/sensing/lidar/scan": "sensor_msgs/msg/LaserScan",
    "/localization/kinematic_state": "nav_msgs/msg/Odometry",
    "/localization/acceleration": "geometry_msgs/msg/AccelWithCovarianceStamped",
    "/sensing/imu/imu_raw": "sensor_msgs/msg/Imu",
    "/vehicle/status/velocity_status": "autoware_auto_vehicle_msgs/msg/VelocityReport",
    "/vehicle/status/steering_status": "autoware_auto_vehicle_msgs/msg/SteeringReport",
    "/vehicle/status/control_mode": "autoware_auto_vehicle_msgs/msg/ControlModeReport",
    "/control/command/control_cmd": "autoware_auto_control_msgs/msg/AckermannControlCommand",
    "/awsim/status": "std_msgs/msg/Float32MultiArray",
    "/v2x/vehicle_positions": "v2x_msgs/msg/V2XVehiclePositionArray",
}


def validate_with_rosbags(bag_path: Path, expected_counts: dict[str, int]) -> None:
    """Fully read the bag and reconcile its observed topic counts with metadata."""
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as exc:  # pragma: no cover - deployment dependency guard
        raise RuntimeError("rosbags is required for structural validation") from exc
    observed: dict[str, int] = {}
    with AnyReader([Path(bag_path)]) as reader:
        connections = list(reader.connections)
        if not connections:
            raise ValueError("bag has no readable connections")
        for connection, _, _ in reader.messages(connections=connections):
            observed[connection.topic] = observed.get(connection.topic, 0) + 1
    if not observed:
        raise ValueError("bag has no readable messages")
    mismatches = {
        topic: (expected, observed.get(topic, 0))
        for topic, expected in expected_counts.items()
        if observed.get(topic, 0) != expected
    }
    if mismatches:
        raise ValueError(f"metadata message counts disagree with bag: {mismatches}")


def _validate_expert_provenance(payload: dict[str, Any]) -> None:
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise TypeError("provenance must be an object")
    required = {
        "ros_domain_id",
        "command_topic",
        "command_owner",
        "command_snapshot_before",
        "command_snapshot_after",
        "v2x_required",
    }
    missing = sorted(required.difference(provenance))
    if missing:
        raise ValueError(f"provenance missing fields: {', '.join(missing)}")
    if provenance["command_topic"] != "/control/command/control_cmd":
        raise ValueError("provenance command_topic is not the label topic")
    owner = str(provenance["command_owner"]).strip()
    if not owner:
        raise ValueError("provenance command_owner is empty")
    expert_source = str(payload["expert_source"])
    expected_owner = EXPECTED_COMMAND_OWNERS[expert_source]
    if owner != expected_owner:
        raise ValueError(
            f"expert_source {expert_source} requires command_owner {expected_owner}"
        )
    if not isinstance(provenance["v2x_required"], bool):
        raise TypeError("provenance v2x_required must be boolean")
    for phase in ("before", "after"):
        snapshot = str(provenance[f"command_snapshot_{phase}"])
        if (
            "Publisher count: 1" not in snapshot
            or f"Node name: {owner}" not in snapshot
        ):
            raise ValueError(
                f"provenance command_snapshot_{phase} does not prove sole expected owner"
            )


def _topic_table(info: dict[str, Any]) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for entry in info.get("topics_with_message_count", []):
        metadata = entry.get("topic_metadata", {})
        name = str(metadata.get("name", ""))
        if name:
            table[name] = {
                "type": str(metadata.get("type", "")),
                "count": int(entry.get("message_count", 0)),
            }
    return table


def inspect_bag_directory(
    bag_path: Path,
    *,
    structural_validator: Callable[[Path], None] | None = validate_with_rosbags,
) -> dict[str, Any]:
    """Return a JSON-serializable accept/reject report for one bag directory."""
    bag_path = Path(bag_path)
    reasons: list[str] = []
    topics: dict[str, dict[str, Any]] = {}
    recording: dict[str, Any] | None = None

    metadata_path = bag_path / "metadata.yaml"
    if not metadata_path.is_file():
        reasons.append("metadata.yaml is missing")
        info: dict[str, Any] = {}
    else:
        try:
            raw = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
            info = raw.get("rosbag2_bagfile_information", {})
            topics = _topic_table(info)
        except Exception as exc:  # noqa: BLE001 - inventory must report every parser failure
            reasons.append(f"metadata.yaml is unreadable: {exc}")
            info = {}

    recording_path = bag_path / "recording.json"
    if not recording_path.is_file():
        reasons.append("recording.json is missing")
    else:
        try:
            parsed = json.loads(recording_path.read_text(encoding="utf-8"))
            recording_obj = RecordingMetadata.from_mapping(parsed)
            _validate_expert_provenance(parsed)
            recording = recording_obj.to_dict()
        except Exception as exc:  # noqa: BLE001 - inventory must report every parser failure
            reasons.append(f"recording.json is invalid: {exc}")

    if recording is not None and recording["provenance"].get(
        "episode_markers_required", False
    ):
        marker = topics.get("/racing_maneuver/episode_control")
        if marker is None:
            reasons.append("required episode marker topic is missing")
        elif marker["type"] != "std_msgs/msg/String":
            reasons.append("episode marker topic must be std_msgs/msg/String")
        elif marker["count"] <= 0:
            reasons.append("required episode marker topic is empty")

    for topic, expected_type in REQUIRED_TOPIC_TYPES.items():
        entry = topics.get(topic)
        if entry is None:
            reasons.append(f"required topic missing: {topic}")
            continue
        if entry["type"] != expected_type:
            reasons.append(
                f"required topic type mismatch: {topic}: "
                f"expected {expected_type}, got {entry['type']}"
            )
        if entry["count"] <= 0:
            reasons.append(f"required topic empty: {topic}")

    relative_files = [Path(value) for value in info.get("relative_file_paths", [])]
    if info.get("storage_identifier") != "mcap":
        reasons.append("storage_identifier must be mcap")
    if info.get("compression_format") != "zstd":
        reasons.append("compression_format must be zstd")
    if str(info.get("compression_mode", "")).lower() != "file":
        reasons.append("compression_mode must be file")
    if not relative_files and metadata_path.is_file():
        reasons.append("metadata contains no storage files")
    bag_root = bag_path.resolve()
    for relative in relative_files:
        if relative.is_absolute() or ".." in relative.parts:
            reasons.append(f"storage path escapes bag directory: {relative}")
            continue
        storage_file = bag_path / relative
        try:
            resolved = storage_file.resolve(strict=True)
        except FileNotFoundError:
            reasons.append(f"storage file is missing: {relative}")
            continue
        if not resolved.is_relative_to(bag_root) or storage_file.is_symlink():
            reasons.append(f"storage path escapes bag directory: {relative}")
        elif not str(resolved).endswith(".mcap.zstd") and resolved.suffix != ".mcap":
            reasons.append(
                f"storage file must use .mcap or .mcap.zstd suffix: {relative}"
            )
        elif resolved.stat().st_size <= 0:
            reasons.append(f"storage file is empty: {relative}")

    if recording is not None and recording["recording_id"] != bag_path.name:
        reasons.append("recording_id does not match bag directory name")

    if structural_validator is not None and not reasons:
        try:
            if structural_validator is validate_with_rosbags:
                validate_with_rosbags(
                    bag_path, {topic: entry["count"] for topic, entry in topics.items()}
                )
            else:
                structural_validator(bag_path)
        except Exception as exc:  # noqa: BLE001 - inventory must report every parser failure
            reasons.append(f"bag structural validation failed: {exc}")

    return {
        "bag_path": str(bag_path.resolve()),
        "accepted": not reasons,
        "reasons": reasons,
        "topics": topics,
        "recording": recording,
        "storage_identifier": info.get("storage_identifier"),
        "storage_files": [str(path) for path in relative_files],
    }
