"""Merge independently extracted recording datasets without mixing group identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .extract_dataset import write_processed_dataset
from .schema import SCHEMA_VERSION

REQUIRED_ARRAYS = (
    "lidar",
    "aux",
    "targets",
    "recording_ids",
    "episode_ids",
    "maneuver_classes",
)
OPTIONAL_ARRAYS = ("source_timestamps_s", "source_ages_s")


def merge_processed_datasets(
    sources: list[str | Path], output_path: str | Path, report_path: str | Path
) -> dict[str, Any]:
    if not sources:
        raise ValueError("at least one processed dataset is required")
    source_paths = [Path(source) for source in sources]
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in REQUIRED_ARRAYS}
    optional_chunks: dict[str, list[np.ndarray]] = {
        name: [] for name in OPTIONAL_ARRAYS
    }
    source_orders: list[np.ndarray] = []
    seen_recordings: set[str] = set()
    total_rows = 0

    for source in source_paths:
        with np.load(source, allow_pickle=False) as archive:
            version = int(np.asarray(archive["schema_version"]).item())
            if version != SCHEMA_VERSION:
                raise ValueError(f"{source}: unsupported schema_version {version}")
            missing = [name for name in REQUIRED_ARRAYS if name not in archive]
            if missing:
                raise ValueError(f"{source}: missing arrays {missing}")
            row_count = len(archive["targets"])
            if row_count <= 0 or any(
                len(archive[name]) != row_count for name in REQUIRED_ARRAYS
            ):
                raise ValueError(f"{source}: inconsistent or empty row arrays")
            recordings = {str(value) for value in archive["recording_ids"]}
            overlap = seen_recordings.intersection(recordings)
            if overlap:
                raise ValueError(
                    f"duplicate recording ids across inputs: {sorted(overlap)}"
                )
            seen_recordings.update(recordings)
            for name in REQUIRED_ARRAYS:
                chunks[name].append(np.asarray(archive[name]))
            for name in OPTIONAL_ARRAYS:
                if name in archive:
                    optional_chunks[name].append(np.asarray(archive[name]))
            if "source_order" in archive:
                source_orders.append(np.asarray(archive["source_order"]))
            total_rows += row_count

    merged = {name: np.concatenate(values, axis=0) for name, values in chunks.items()}
    for name, values in optional_chunks.items():
        if values:
            if len(values) != len(source_paths):
                raise ValueError(
                    f"optional audit array {name} is absent from some inputs"
                )
            merged[name] = np.concatenate(values, axis=0)
    if source_orders:
        if len(source_orders) != len(source_paths) or any(
            not np.array_equal(order, source_orders[0]) for order in source_orders[1:]
        ):
            raise ValueError("source_order differs or is absent across inputs")
        merged["source_order"] = source_orders[0]

    report = write_processed_dataset(
        output_path,
        report_path,
        merged,
        rejected_reasons={},
        input_rows=total_rows,
    )
    report["merged_sources"] = [str(source) for source in source_paths]
    destination = Path(report_path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)
    report = merge_processed_datasets(args.sources, args.output, args.report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
