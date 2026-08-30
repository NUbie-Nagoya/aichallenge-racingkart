#!/usr/bin/env python3
"""Inventory manual maneuver bags and emit auditable JSON reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from racing_maneuver_il.inventory import inspect_bag_directory


def discover_bags(root: Path) -> list[Path]:
    """Discover complete and incomplete candidate recording directories."""
    if not root.is_dir():
        return []
    candidates = {root} if any(
        (root / name).exists() for name in ("metadata.yaml", "recording.json")
    ) else set()
    candidates.update(path.parent for path in root.rglob("metadata.yaml"))
    candidates.update(path.parent for path in root.rglob("recording.json"))
    candidates.update(path.parent for path in root.rglob("*.mcap"))
    return sorted(candidates)


def mark_duplicate_recording_ids(reports: list[dict]) -> None:
    """Reject every inventory entry whose recording identifier is ambiguous."""
    by_id: dict[str, list[dict]] = {}
    for report in reports:
        recording = report.get("recording")
        if recording is not None:
            by_id.setdefault(str(recording["recording_id"]), []).append(report)
    for recording_id, duplicates in by_id.items():
        if len(duplicates) > 1:
            for report in duplicates:
                report["accepted"] = False
                report["reasons"].append(
                    f"duplicate recording_id across inventory: {recording_id}"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bags", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    bags = discover_bags(args.bags.expanduser().resolve())
    reports = [inspect_bag_directory(path) for path in bags]
    mark_duplicate_recording_ids(reports)
    result = {
        "bags": reports,
        "accepted": sum(report["accepted"] for report in reports),
        "rejected": sum(not report["accepted"] for report in reports),
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if result["rejected"] or not reports:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
