import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from racing_maneuver_il.evaluate import run_evaluation
from racing_maneuver_il.train import _resolve_device, run_training
from racing_maneuver_il.train import main as train_main


def make_dataset(path: Path):
    rng = np.random.default_rng(3)
    n = 48
    aux = rng.normal(size=(n, 11)).astype(np.float32)
    lidar = rng.uniform(0.2, 30, size=(n, 360)).astype(np.float32)
    targets = np.stack(
        (np.tanh(aux[:, 1]) * 0.4, np.tanh(aux[:, 2]) * 2), axis=1
    ).astype(np.float32)
    np.savez_compressed(
        path,
        lidar=lidar,
        aux=aux,
        targets=targets,
        recording_ids=np.repeat(np.array(["r1", "r2", "r3", "r4"]), 12),
        episode_ids=np.array(["e1"] * n),
        maneuver_classes=np.repeat(
            np.array(["follow", "pass_left", "recover", "free_lap"]), 12
        ),
    )


def config(dataset, output):
    return {
        "dataset": str(dataset),
        "output_dir": str(output),
        "seed": 2,
        "epochs": 1,
        "batch_size": 4,
        "learning_rate": 0.001,
        "split_ratios": [0.5, 0.25, 0.25],
        "model": {"hidden_size": 8, "lidar_embedding": 8, "aux_embedding": 4},
        "action_limits": {"low": [-0.5, -3.0], "high": [0.5, 2.0]},
        "loss_weights": {"steering": 1.0, "longitudinal": 1.0, "action_change": 0.05},
    }


def test_cuda_device_requires_an_available_cuda_runtime(monkeypatch):
    monkeypatch.setattr("racing_maneuver_il.train.torch.cuda.is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA is unavailable"):
        _resolve_device("cuda")


def test_training_creates_distinct_run_directories(tmp_path):
    dataset = tmp_path / "data.npz"
    output_root = tmp_path / "runs"
    make_dataset(dataset)

    first = run_training(config(dataset, output_root))
    second = run_training(config(dataset, output_root))

    assert first["output_dir"] != second["output_dir"]
    assert Path(first["output_dir"]).name == "run-1"
    assert Path(second["output_dir"]).name == "run-2"
    assert Path(first["output_dir"]).parent == output_root
    assert Path(second["output_dir"]).parent == output_root
    assert (Path(first["output_dir"]) / "best.pt").is_file()
    assert (Path(second["output_dir"]) / "best.pt").is_file()


def test_deterministic_synthetic_smoke_training_and_evaluation(tmp_path):
    dataset = tmp_path / "data.npz"
    output = tmp_path / "run"
    make_dataset(dataset)
    result = run_training(config(dataset, output))
    required = {
        "best.pt",
        "last.pt",
        "resolved_config.yaml",
        "normalizer.json",
        "split_manifest.json",
        "curves.json",
        "validation_metrics.json",
    }
    run_output = Path(result["output_dir"])
    assert required.issubset({p.name for p in run_output.iterdir()})
    assert result["best_epoch"] == 0
    assert result["device"] == "cpu"
    report = run_evaluation(
        run_output / "best.pt", dataset, run_output / "evaluation.json"
    )
    assert set(report) == {"teacher_forced", "autoregressive"}
    assert json.loads((run_output / "evaluation.json").read_text()) == report

    different_dataset = tmp_path / "different.npz"
    make_dataset(different_dataset)
    with np.load(different_dataset, allow_pickle=False) as source:
        changed = {name: source[name] for name in source.files}
    changed["targets"] = changed["targets"].copy()
    changed["targets"][0, 0] += 0.01
    np.savez_compressed(different_dataset, **changed)
    with pytest.raises(ValueError, match="provenance"):
        run_evaluation(run_output / "best.pt", different_dataset, run_output / "invalid.json")


def test_training_reports_epoch_progress_with_tqdm(monkeypatch, tmp_path):
    dataset = tmp_path / "data.npz"
    output = tmp_path / "run"
    make_dataset(dataset)
    calls = []

    class Progress:
        def __init__(self, iterable, **kwargs):
            calls.append(kwargs)
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, **kwargs):
            calls[-1]["postfix"] = kwargs

    monkeypatch.setattr("racing_maneuver_il.train.tqdm", Progress)
    run_training(config(dataset, output))

    assert calls[0]["desc"] == "Epoch 1/1"
    assert calls[0]["total"] > 0
    assert "loss" in calls[0]["postfix"]


def test_train_cli_reads_yaml(tmp_path):
    dataset = tmp_path / "data.npz"
    output = tmp_path / "run"
    make_dataset(dataset)
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(config(dataset, output)))
    train_main([str(config_path)])
    assert len(list(output.glob("run-*/last.pt"))) == 1
