"""Leakage-resistant grouped temporal datasets."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class FrameData:
    lidar: np.ndarray
    aux: np.ndarray
    targets: np.ndarray
    recording_ids: np.ndarray
    episode_ids: np.ndarray
    maneuver_classes: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.lidar)
        if any(
            len(value) != n
            for value in (
                self.aux,
                self.targets,
                self.recording_ids,
                self.episode_ids,
                self.maneuver_classes,
            )
        ):
            raise ValueError("all frame fields must have equal length")
        if (
            self.lidar.shape[1:] != (360,)
            or self.aux.shape[1:] != (11,)
            or self.targets.shape[1:] != (2,)
        ):
            raise ValueError("frame tensor shapes violate schema")

    @property
    def groups(self) -> np.ndarray:
        return np.asarray(
            [
                f"{recording}::{episode}"
                for recording, episode in zip(self.recording_ids, self.episode_ids)
            ]
        )

    @classmethod
    def from_npz(cls, path: str | Path) -> FrameData:
        with np.load(path, allow_pickle=False) as data:
            return cls(
                *(
                    data[name]
                    for name in (
                        "lidar",
                        "aux",
                        "targets",
                        "recording_ids",
                        "episode_ids",
                        "maneuver_classes",
                    )
                )
            )


class TemporalDataset(Dataset):
    def __init__(
        self,
        frames: FrameData,
        allowed_frame_indices: np.ndarray,
        history_length: int = 10,
    ) -> None:
        self.frames, self.history_length = frames, history_length
        allowed = {int(i) for i in allowed_frame_indices}
        groups = frames.groups
        self.end_indices = []
        for end in sorted(allowed):
            start = end - history_length + 1
            window = range(start, end + 1)
            if (
                start >= 0
                and all(i in allowed for i in window)
                and len(set(groups[list(window)])) == 1
            ):
                self.end_indices.append(end)

    def __len__(self) -> int:
        return len(self.end_indices)

    def __getitem__(self, item: int):
        end = self.end_indices[item]
        start = end - self.history_length + 1
        sl = slice(start, end + 1)
        meta = {
            "group": str(self.frames.groups[end]),
            "maneuver_class": str(self.frames.maneuver_classes[end]),
            "frame_index": end,
        }
        return (
            torch.as_tensor(self.frames.lidar[sl], dtype=torch.float32),
            torch.as_tensor(self.frames.aux[sl], dtype=torch.float32),
            torch.as_tensor(self.frames.targets[end], dtype=torch.float32),
            meta,
        )


def grouped_split(
    frames: FrameData,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> dict[str, np.ndarray]:
    if (
        len(ratios) != 3
        or any(r < 0 for r in ratios)
        or not np.isclose(sum(ratios), 1.0)
    ):
        raise ValueError("split ratios must be three nonnegative values summing to one")
    groups = np.unique(frames.groups)
    rng = np.random.default_rng(seed)
    groups = groups[rng.permutation(len(groups))]
    n_train = round(len(groups) * ratios[0])
    n_validation = round(len(groups) * ratios[1])
    if len(groups) >= 3 and all(r > 0 for r in ratios):
        n_train = min(max(n_train, 1), len(groups) - 2)
        n_validation = min(max(n_validation, 1), len(groups) - n_train - 1)
    group_splits = {
        "train": groups[:n_train],
        "validation": groups[n_train : n_train + n_validation],
        "test": groups[n_train + n_validation :],
    }
    return {
        name: np.flatnonzero(np.isin(frames.groups, selected)).astype(np.int64)
        for name, selected in group_splits.items()
    }


def save_split_manifest(
    path: str | Path,
    frames: FrameData,
    split: dict[str, np.ndarray],
    *,
    dataset_provenance: dict | None = None,
) -> dict:
    manifest: dict = {"version": 1, "dataset_provenance": dataset_provenance or {}}
    for name, indices in split.items():
        classes = Counter(str(x) for x in frames.maneuver_classes[indices])
        manifest[name] = {
            "frame_indices": indices.tolist(),
            "groups": sorted(set(frames.groups[indices].tolist())),
            "maneuver_counts": dict(sorted(classes.items())),
        }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    return manifest
