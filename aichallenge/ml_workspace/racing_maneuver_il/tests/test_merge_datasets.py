from pathlib import Path

import numpy as np

from racing_maneuver_il.merge_datasets import merge_processed_datasets


def write_dataset(path: Path, recording: str, offset: float) -> None:
    np.savez_compressed(
        path,
        schema_version=np.asarray(3, dtype=np.int64),
        lidar=np.full((2, 360), offset, dtype=np.float32),
        aux=np.full((2, 9), offset, dtype=np.float32),
        targets=np.full((2, 2), offset, dtype=np.float32),
        recording_ids=np.asarray([recording, recording]),
        episode_ids=np.asarray(["scenario::0", "scenario::0"]),
        maneuver_classes=np.asarray(["follow", "follow"]),
    )


def test_merge_processed_datasets_concatenates_complete_recordings_atomically(tmp_path):
    first, second = tmp_path / "a.npz", tmp_path / "b.npz"
    write_dataset(first, "a", 1.0)
    write_dataset(second, "b", 2.0)
    output, report = tmp_path / "merged.npz", tmp_path / "merged.json"
    result = merge_processed_datasets([first, second], output, report)
    with np.load(output, allow_pickle=False) as merged:
        assert merged["lidar"].shape == (4, 360)
        assert merged["recording_ids"].tolist() == ["a", "a", "b", "b"]
    assert result["accepted_rows"] == 4
    assert result["merged_sources"] == [str(first), str(second)]
    assert report.is_file()


def test_merge_rejects_duplicate_recording_ids(tmp_path):
    first, second = tmp_path / "a.npz", tmp_path / "b.npz"
    write_dataset(first, "same", 1.0)
    write_dataset(second, "same", 2.0)
    try:
        merge_processed_datasets(
            [first, second], tmp_path / "out.npz", tmp_path / "out.json"
        )
    except ValueError as error:
        assert "duplicate recording" in str(error)
    else:
        raise AssertionError("duplicate recording groups were accepted")
