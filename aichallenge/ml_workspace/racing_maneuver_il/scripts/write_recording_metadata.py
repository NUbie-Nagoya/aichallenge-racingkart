#!/usr/bin/env python3
"""Create validated manual-expert recording metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

from racing_maneuver_il.recording_metadata import (
    MANEUVER_CLASSES,
    RecordingMetadata,
    write_metadata_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recording-id", required=True)
    parser.add_argument("--ego-vehicle-id", required=True)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument(
        "--maneuver-class", choices=sorted(MANEUVER_CLASSES), required=True
    )
    parser.add_argument("--opponent-count", type=int, required=True)
    parser.add_argument("--opponent-behavior", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--notes", default="")
    parser.add_argument("--ros-domain-id", type=int, required=True)
    parser.add_argument("--command-owner", required=True)
    parser.add_argument("--expert-source", choices=("manual", "mpc"), required=True)
    parser.add_argument("--command-snapshot-before", required=True)
    parser.add_argument("--command-snapshot-after", required=True)
    parser.add_argument("--v2x-required", action="store_true")
    parser.add_argument("--episode-markers-required", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = RecordingMetadata.from_mapping(
        {
            "schema_version": 1,
            "recording_id": args.recording_id,
            "expert_source": args.expert_source,
            "ego_vehicle_id": args.ego_vehicle_id,
            "scenario_id": args.scenario_id,
            "maneuver_class": args.maneuver_class,
            "opponent_count": args.opponent_count,
            "opponent_behavior": args.opponent_behavior,
            "seed": args.seed,
            "notes": args.notes,
            "provenance": {
                "ros_domain_id": args.ros_domain_id,
                "command_topic": "/control/command/control_cmd",
                "command_owner": args.command_owner,
                "command_snapshot_before": args.command_snapshot_before,
                "command_snapshot_after": args.command_snapshot_after,
                "v2x_required": args.v2x_required,
                "episode_markers_required": args.episode_markers_required,
                "episode_control_topic": "/racing_maneuver/episode_control",
            },
        }
    )
    write_metadata_atomic(args.output, metadata)
    print(f"Wrote validated recording metadata: {args.output}")


if __name__ == "__main__":
    main()
