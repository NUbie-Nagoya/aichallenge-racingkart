"""Validated provenance metadata for manual maneuver recordings."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
EXPERT_SOURCES = frozenset({"manual", "mpc"})
EXPECTED_COMMAND_OWNERS = {
    "manual": "teleop_manager_node",
    "mpc": "mpc_controller",
}
MANEUVER_CLASSES = frozenset(
    {
        "follow",
        "pass_left",
        "pass_right",
        "side_by_side",
        "abort",
        "recover",
        "free_lap",
    }
)


@dataclass(frozen=True)
class RecordingMetadata:
    schema_version: int
    recording_id: str
    expert_source: str
    ego_vehicle_id: str
    scenario_id: str
    maneuver_class: str
    opponent_count: int
    opponent_behavior: str
    seed: int
    notes: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> RecordingMetadata:
        required = {
            "schema_version",
            "recording_id",
            "expert_source",
            "ego_vehicle_id",
            "scenario_id",
            "maneuver_class",
            "opponent_count",
            "opponent_behavior",
            "seed",
        }
        missing = sorted(required.difference(data))
        if missing:
            raise ValueError(f"missing metadata fields: {', '.join(missing)}")

        metadata = cls(
            schema_version=int(data["schema_version"]),
            recording_id=str(data["recording_id"]).strip(),
            expert_source=str(data["expert_source"]).strip(),
            ego_vehicle_id=str(data["ego_vehicle_id"]).strip(),
            scenario_id=str(data["scenario_id"]).strip(),
            maneuver_class=str(data["maneuver_class"]).strip(),
            opponent_count=int(data["opponent_count"]),
            opponent_behavior=str(data["opponent_behavior"]).strip(),
            seed=int(data["seed"]),
            notes=str(data.get("notes", "")).strip(),
            provenance=dict(data.get("provenance", {})),
        )
        metadata.validate()
        return metadata

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        for field_name in ("recording_id", "ego_vehicle_id", "scenario_id"):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must be non-empty")
        if self.expert_source not in EXPERT_SOURCES:
            raise ValueError(
                "expert_source must be one of " + ", ".join(sorted(EXPERT_SOURCES))
            )
        if self.maneuver_class not in MANEUVER_CLASSES:
            raise ValueError(
                "maneuver_class must be one of " + ", ".join(sorted(MANEUVER_CLASSES))
            )
        if self.opponent_count < 0:
            raise ValueError("opponent_count must be non-negative")
        if not self.opponent_behavior:
            raise ValueError("opponent_behavior must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_metadata_atomic(path: Path, metadata: RecordingMetadata) -> None:
    """Write validated metadata atomically without replacing raw bag content."""
    metadata.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(metadata.to_dict(), indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
