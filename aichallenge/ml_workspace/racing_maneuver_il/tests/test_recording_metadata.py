import json
from pathlib import Path

import pytest

from racing_maneuver_il.recording_metadata import (
    MANEUVER_CLASSES,
    RecordingMetadata,
    write_metadata_atomic,
)


def valid_metadata(**overrides):
    data = {
        "schema_version": 1,
        "recording_id": "20260817-210000",
        "expert_source": "manual",
        "ego_vehicle_id": "d1",
        "scenario_id": "straight-pass-left-01",
        "maneuver_class": "pass_left",
        "opponent_count": 1,
        "opponent_behavior": "constant_speed",
        "seed": 7,
        "notes": "clean pass",
    }
    data.update(overrides)
    return data


def test_accepts_complete_manual_metadata():
    metadata = RecordingMetadata.from_mapping(valid_metadata())
    assert metadata.expert_source == "manual"
    assert metadata.maneuver_class == "pass_left"


def test_accepts_mpc_expert_metadata():
    metadata = RecordingMetadata.from_mapping(valid_metadata(expert_source="mpc"))
    assert metadata.expert_source == "mpc"


@pytest.mark.parametrize("maneuver", sorted(MANEUVER_CLASSES))
def test_accepts_every_supported_maneuver(maneuver):
    assert (
        RecordingMetadata.from_mapping(
            valid_metadata(maneuver_class=maneuver)
        ).maneuver_class
        == maneuver
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("recording_id", ""),
        ("expert_source", "planner"),
        ("ego_vehicle_id", ""),
        ("scenario_id", ""),
        ("maneuver_class", "unknown"),
        ("opponent_count", -1),
        ("opponent_behavior", ""),
    ],
)
def test_rejects_invalid_metadata(field, value):
    with pytest.raises(ValueError, match=field):
        RecordingMetadata.from_mapping(valid_metadata(**{field: value}))


def test_atomic_write_round_trips_validated_json(tmp_path: Path):
    metadata = RecordingMetadata.from_mapping(valid_metadata())
    destination = tmp_path / "recording.json"
    write_metadata_atomic(destination, metadata)
    assert json.loads(destination.read_text()) == metadata.to_dict()


def test_atomic_write_does_not_touch_predictable_temp_path(tmp_path: Path):
    metadata = RecordingMetadata.from_mapping(valid_metadata())
    predictable = tmp_path / "recording.json.tmp"
    predictable.write_text("must remain untouched")
    write_metadata_atomic(tmp_path / "recording.json", metadata)
    assert predictable.read_text() == "must remain untouched"
