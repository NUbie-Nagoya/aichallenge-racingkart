"""Teacher-forced and sequential autoregressive offline evaluation CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_checkpoint
from .dataset import FrameData, TemporalDataset
from .metrics import compute_metrics
from .model import ExportPolicy
from .normalization import Normalizer


def _metrics(
    predictions: list[np.ndarray],
    targets: list[np.ndarray],
    maneuvers: list[str],
    present: list[float],
    closing: list[float],
) -> dict:
    return compute_metrics(
        np.asarray(predictions),
        np.asarray(targets),
        maneuver_classes=np.asarray(maneuvers),
        opponent_present=np.asarray(present),
        closing_speed_mps=np.asarray(closing),
    )


def _teacher_forced(wrapper: ExportPolicy, dataset: TemporalDataset) -> dict:
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    maneuvers: list[str] = []
    present: list[float] = []
    closing: list[float] = []
    wrapper.eval()
    with torch.no_grad():
        for lidar, auxiliary, target, metadata in dataset:
            action = wrapper(lidar.unsqueeze(0), auxiliary.unsqueeze(0))[0].cpu()
            predictions.append(action.numpy())
            targets.append(target.numpy())
            maneuvers.append(metadata["maneuver_class"])
            present.append(float(auxiliary[-1, 5]))
            closing.append(float(-auxiliary[-1, 8]))
    return _metrics(predictions, targets, maneuvers, present, closing)


def _autoregressive(
    wrapper: ExportPolicy,
    frames: FrameData,
    test_indices: np.ndarray,
    history_length: int = 10,
) -> dict:
    """Replay each held-out group in order and feed every prior prediction back."""
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    maneuvers: list[str] = []
    present: list[float] = []
    closing: list[float] = []
    wrapper.eval()

    groups = frames.groups
    for group in sorted({str(groups[index]) for index in test_indices}):
        group_indices = sorted(
            int(index) for index in test_indices if str(groups[index]) == group
        )
        if not group_indices:
            continue
        generated_auxiliary: dict[int, np.ndarray] = {}
        previous_safe = np.zeros(2, dtype=np.float32)
        previous_index: int | None = None
        with torch.no_grad():
            for index in group_indices:
                if previous_index is not None and index != previous_index + 1:
                    generated_auxiliary.clear()
                    previous_safe.fill(0.0)
                row = frames.aux[index].astype(np.float32, copy=True)
                row[3:5] = previous_safe
                generated_auxiliary[index] = row
                previous_index = index

                start = index - history_length + 1
                history_indices = list(range(start, index + 1))
                if (
                    start < 0
                    or any(
                        history_index not in generated_auxiliary
                        for history_index in history_indices
                    )
                    or len(
                        {
                            str(groups[history_index])
                            for history_index in history_indices
                        }
                    )
                    != 1
                ):
                    continue
                lidar = torch.as_tensor(
                    frames.lidar[history_indices], dtype=torch.float32
                )
                auxiliary = torch.as_tensor(
                    np.stack([generated_auxiliary[item] for item in history_indices]),
                    dtype=torch.float32,
                )
                action = wrapper(lidar.unsqueeze(0), auxiliary.unsqueeze(0))[0].cpu()
                previous_safe = action.numpy().astype(np.float32, copy=True)
                predictions.append(previous_safe.copy())
                targets.append(frames.targets[index].astype(np.float32, copy=True))
                maneuvers.append(str(frames.maneuver_classes[index]))
                present.append(float(row[5]))
                closing.append(float(-row[8]))

    if not predictions:
        raise ValueError("test split has no complete autoregressive history")
    return _metrics(predictions, targets, maneuvers, present, closing)


def _verify_dataset_provenance(dataset_path: Path, metadata: dict) -> None:
    expected = metadata.get("dataset_provenance", {}).get("sha256")
    if not expected:
        raise ValueError("checkpoint has no dataset provenance hash")
    actual = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(
            "dataset provenance mismatch: evaluation dataset differs from the checkpoint split manifest"
        )


def run_evaluation(
    checkpoint: str | Path,
    dataset_path: str | Path,
    output_path: str | Path,
) -> dict:
    model, payload = load_checkpoint(checkpoint)
    metadata = payload["metadata"]
    dataset_path = Path(dataset_path)
    _verify_dataset_provenance(dataset_path, metadata)
    normalizer = Normalizer.from_dict(metadata["normalizer"])
    limits = metadata["action_limits"]
    wrapper = ExportPolicy.from_normalizer(
        model.eval(), normalizer, tuple(limits["low"]), tuple(limits["high"])
    )
    frames = FrameData.from_npz(dataset_path)
    indices = np.asarray(
        metadata["split_manifest"]["test"]["frame_indices"], dtype=np.int64
    )
    teacher_dataset = TemporalDataset(frames, indices)
    if not teacher_dataset:
        raise ValueError("test split has no complete history")
    report = {
        "teacher_forced": _teacher_forced(wrapper, teacher_dataset),
        "autoregressive": _autoregressive(wrapper, frames, indices),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("dataset")
    parser.add_argument("output")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run_evaluation(args.checkpoint, args.dataset, args.output),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
