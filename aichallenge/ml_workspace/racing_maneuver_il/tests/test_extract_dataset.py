import json

import numpy as np

from racing_maneuver_il.extract_dataset import ExtractionConfig, write_processed_dataset


def test_default_human_control_mode_matches_verified_teleop_mode():
    assert ExtractionConfig().human_control_modes == (1,)


def test_processed_dataset_and_audit_report_are_written_atomically(tmp_path):
    n = 12
    arrays = {
        "lidar": np.ones((n, 360), np.float32),
        "aux": np.ones((n, 9), np.float32),
        "targets": np.ones((n, 2), np.float32),
        "recording_ids": np.array(["r"] * n),
        "episode_ids": np.array(["e"] * n),
        "maneuver_classes": np.array(["follow"] * n),
        "source_timestamps_s": np.ones((n, 3)),
        "source_ages_s": np.zeros((n, 3)),
    }
    output = tmp_path / "data.npz"
    report_path = tmp_path / "report.json"
    report = write_processed_dataset(
        output, report_path, arrays, rejected_reasons={"stale_scan": 2}, input_rows=14
    )
    with np.load(output) as stored:
        assert stored["lidar"].shape == (n, 360) and stored["schema_version"] == 3
    assert json.loads(report_path.read_text()) == report
    assert report["accepted_rows"] == 12 and report["rejected_rows"] == 2
    assert "target_speed" in report["command_distributions"]


def test_processed_dataset_rejects_nonfinite_or_wrong_shapes(tmp_path):
    arrays = {
        "lidar": np.ones((2, 359)),
        "aux": np.ones((2, 9)),
        "targets": np.ones((2, 2)),
        "recording_ids": np.array(["r"] * 2),
        "episode_ids": np.array(["e"] * 2),
        "maneuver_classes": np.array(["follow"] * 2),
    }
    try:
        write_processed_dataset(tmp_path / "x.npz", tmp_path / "x.json", arrays)
    except ValueError:
        pass
    else:
        raise AssertionError("bad shape accepted")
