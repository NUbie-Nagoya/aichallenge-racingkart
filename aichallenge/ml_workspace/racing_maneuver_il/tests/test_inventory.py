import json
from pathlib import Path

import pytest
import yaml

from racing_maneuver_il.inventory import REQUIRED_TOPIC_TYPES, inspect_bag_directory


def recording_metadata():
    return {
        "schema_version": 1,
        "recording_id": "bag-a",
        "expert_source": "manual",
        "ego_vehicle_id": "d1",
        "scenario_id": "scenario-a",
        "maneuver_class": "follow",
        "opponent_count": 1,
        "opponent_behavior": "constant_speed",
        "seed": 3,
        "notes": "",
        "provenance": {
            "ros_domain_id": 1,
            "command_topic": "/control/command/control_cmd",
            "command_owner": "teleop_manager_node",
            "command_snapshot_before": "Publisher count: 1\nNode name: teleop_manager_node",
            "command_snapshot_after": "Publisher count: 1\nNode name: teleop_manager_node",
            "v2x_required": True,
        },
    }


def ros_metadata(topics=None, files=None):
    if topics is None:
        topics = {
            topic: {"type": msg_type, "count": 10}
            for topic, msg_type in REQUIRED_TOPIC_TYPES.items()
        }
    if files is None:
        files = [{"path": "bag_0.mcap", "message_count": 100}]
    return {
        "rosbag2_bagfile_information": {
            "storage_identifier": "mcap",
            "compression_format": "zstd",
            "compression_mode": "file",
            "relative_file_paths": [entry["path"] for entry in files],
            "files": files,
            "topics_with_message_count": [
                {
                    "topic_metadata": {"name": topic, "type": entry["type"]},
                    "message_count": entry["count"],
                }
                for topic, entry in topics.items()
            ],
        }
    }


def make_bag(
    tmp_path: Path, *, topics=None, include_recording=True, storage_bytes=b"mcap"
):
    bag = tmp_path / "bag-a"
    bag.mkdir()
    (bag / "metadata.yaml").write_text(yaml.safe_dump(ros_metadata(topics)))
    (bag / "bag_0.mcap").write_bytes(storage_bytes)
    if include_recording:
        (bag / "recording.json").write_text(json.dumps(recording_metadata()))
    return bag


def test_accepts_complete_manual_multicar_bag(tmp_path):
    report = inspect_bag_directory(
        make_bag(tmp_path), structural_validator=lambda _: None
    )
    assert report["accepted"] is True
    assert report["reasons"] == []
    assert report["topics"]["/sensing/lidar/scan"]["count"] == 10


def test_accepts_mpc_bag_with_mpc_controller_owner(tmp_path):
    bag = make_bag(tmp_path)
    metadata = recording_metadata()
    metadata["expert_source"] = "mpc"
    metadata["provenance"]["command_owner"] = "mpc_controller"
    for phase in ("before", "after"):
        metadata["provenance"][f"command_snapshot_{phase}"] = (
            "Publisher count: 1\nNode name: mpc_controller"
        )
    (bag / "recording.json").write_text(json.dumps(metadata))

    report = inspect_bag_directory(bag, structural_validator=lambda _: None)

    assert report["accepted"] is True


def test_rejects_manual_metadata_with_mpc_owner(tmp_path):
    bag = make_bag(tmp_path)
    metadata = recording_metadata()
    metadata["provenance"]["command_owner"] = "mpc_controller"
    for phase in ("before", "after"):
        metadata["provenance"][f"command_snapshot_{phase}"] = (
            "Publisher count: 1\nNode name: mpc_controller"
        )
    (bag / "recording.json").write_text(json.dumps(metadata))

    report = inspect_bag_directory(bag, structural_validator=lambda _: None)

    assert report["accepted"] is False
    assert any("requires command_owner teleop_manager_node" in r for r in report["reasons"])


def test_rejects_missing_recording_metadata(tmp_path):
    report = inspect_bag_directory(
        make_bag(tmp_path, include_recording=False), structural_validator=lambda _: None
    )
    assert report["accepted"] is False
    assert "recording.json is missing" in report["reasons"]


def test_rejects_missing_manual_command_provenance(tmp_path):
    bag = make_bag(tmp_path)
    metadata = recording_metadata()
    metadata.pop("provenance")
    (bag / "recording.json").write_text(json.dumps(metadata))
    report = inspect_bag_directory(bag, structural_validator=lambda _: None)
    assert any("provenance" in reason for reason in report["reasons"])


@pytest.mark.parametrize("missing_topic", sorted(REQUIRED_TOPIC_TYPES))
def test_rejects_every_missing_required_topic(tmp_path, missing_topic):
    topics = {
        topic: {"type": msg_type, "count": 10}
        for topic, msg_type in REQUIRED_TOPIC_TYPES.items()
        if topic != missing_topic
    }
    report = inspect_bag_directory(
        make_bag(tmp_path, topics=topics), structural_validator=lambda _: None
    )
    assert report["accepted"] is False
    assert f"required topic missing: {missing_topic}" in report["reasons"]


def test_rejects_zero_count_and_wrong_type(tmp_path):
    topics = {
        topic: {"type": msg_type, "count": 10}
        for topic, msg_type in REQUIRED_TOPIC_TYPES.items()
    }
    topics["/sensing/lidar/scan"]["count"] = 0
    topics["/control/command/control_cmd"]["type"] = "wrong/msg/Type"
    report = inspect_bag_directory(
        make_bag(tmp_path, topics=topics), structural_validator=lambda _: None
    )
    assert "required topic empty: /sensing/lidar/scan" in report["reasons"]
    assert any(
        "type mismatch: /control/command/control_cmd" in r for r in report["reasons"]
    )


def test_accepts_rosbag2_uppercase_file_compression_and_mcap_zstd_suffix(tmp_path):
    bag = make_bag(tmp_path)
    metadata = ros_metadata(files=[{"path": "bag_0.mcap.zstd", "message_count": 100}])
    metadata["rosbag2_bagfile_information"]["compression_mode"] = "FILE"
    (bag / "metadata.yaml").write_text(yaml.safe_dump(metadata))
    (bag / "bag_0.mcap").unlink()
    (bag / "bag_0.mcap.zstd").write_bytes(b"compressed mcap")

    report = inspect_bag_directory(bag, structural_validator=lambda _: None)

    assert report["accepted"] is True
    assert report["storage_files"] == ["bag_0.mcap.zstd"]


def test_rejects_empty_storage_file(tmp_path):
    report = inspect_bag_directory(
        make_bag(tmp_path, storage_bytes=b""), structural_validator=lambda _: None
    )
    assert report["accepted"] is False
    assert "storage file is empty: bag_0.mcap" in report["reasons"]


def test_rejects_structural_validation_failure(tmp_path):
    def broken(_):
        raise ValueError("truncated mcap")

    report = inspect_bag_directory(make_bag(tmp_path), structural_validator=broken)
    assert report["accepted"] is False
    assert "bag structural validation failed: truncated mcap" in report["reasons"]


def test_rejects_wrong_storage_contract_and_escaping_storage_path(tmp_path):
    bag = make_bag(tmp_path)
    metadata = ros_metadata(files=[{"path": "../outside.mcap", "message_count": 100}])
    metadata["rosbag2_bagfile_information"]["storage_identifier"] = "sqlite3"
    metadata["rosbag2_bagfile_information"]["compression_format"] = "none"
    (bag / "metadata.yaml").write_text(yaml.safe_dump(metadata))
    report = inspect_bag_directory(bag, structural_validator=lambda _: None)
    assert report["accepted"] is False
    assert "storage_identifier must be mcap" in report["reasons"]
    assert "compression_format must be zstd" in report["reasons"]
    assert "storage path escapes bag directory: ../outside.mcap" in report["reasons"]


def test_rejects_recording_id_that_does_not_match_directory(tmp_path):
    bag = make_bag(tmp_path)
    metadata = recording_metadata()
    metadata["recording_id"] = "different-id"
    (bag / "recording.json").write_text(json.dumps(metadata))
    report = inspect_bag_directory(bag, structural_validator=lambda _: None)
    assert "recording_id does not match bag directory name" in report["reasons"]
