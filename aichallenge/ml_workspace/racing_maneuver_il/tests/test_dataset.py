import json

import numpy as np
import pytest

from racing_maneuver_il.dataset import (
    FrameData,
    TemporalDataset,
    grouped_split,
    save_split_manifest,
)


def frames():
    n = 36
    return FrameData(
        lidar=np.arange(n, dtype=np.float32)[:, None] * np.ones((n, 360), np.float32),
        aux=np.arange(n, dtype=np.float32)[:, None] * np.ones((n, 9), np.float32),
        targets=np.zeros((n, 2), np.float32),
        recording_ids=np.array(["r1"] * 12 + ["r2"] * 12 + ["r3"] * 12),
        episode_ids=np.array(["e1"] * 10 + ["e2"] * 2 + ["e1"] * 12 + ["e1"] * 12),
        maneuver_classes=np.array(
            ["follow"] * 12 + ["pass_left"] * 12 + ["free_lap"] * 12
        ),
    )


def test_histories_are_oldest_to_newest_and_never_cross_groups():
    ds = TemporalDataset(frames(), np.arange(36), history_length=10)
    lidar, aux, _target, meta = ds[0]
    np.testing.assert_array_equal(lidar[:, 0], np.arange(10))
    assert (
        meta["group"] == "r1::e1" and lidar.shape == (10, 360) and aux.shape == (10, 9)
    )
    assert len(ds) == 7  # r1/e1=1, r2/e1=3, r3/e1=3; r1/e2 is too short


def test_grouped_splits_have_no_group_or_frame_overlap():
    split = grouped_split(frames(), (1 / 3, 1 / 3, 1 / 3), seed=4)
    sets = [set(split[name].tolist()) for name in ("train", "validation", "test")]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    data = frames()
    groups = [
        {f"{data.recording_ids[i]}::{data.episode_ids[i]}" for i in indices}
        for indices in sets
    ]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])


def test_split_ratios_must_be_valid():
    with pytest.raises(ValueError):
        grouped_split(frames(), (0.8, 0.2, 0.2))


def test_manifest_identifies_maneuver_counts(tmp_path):
    data = frames()
    split = grouped_split(data, (1 / 3, 1 / 3, 1 / 3), seed=1)
    path = tmp_path / "split.json"
    manifest = save_split_manifest(
        path, data, split, dataset_provenance={"sha256": "abc"}
    )
    assert json.loads(path.read_text()) == manifest
    assert all(
        "maneuver_counts" in manifest[name] for name in ("train", "validation", "test")
    )
