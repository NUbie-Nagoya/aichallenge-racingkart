"""Deterministic behavioral-cloning training CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint
from .dataset import FrameData, TemporalDataset, grouped_split, save_split_manifest
from .metrics import compute_metrics
from .model import TemporalPolicy
from .normalization import Normalizer


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _physical(
    normalized: torch.Tensor, low: torch.Tensor, high: torch.Tensor
) -> torch.Tensor:
    return low + (normalized + 1.0) * 0.5 * (high - low)


def grouped_action_change_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    groups,
    frame_indices: torch.Tensor,
) -> torch.Tensor:
    """Action-delta loss only for adjacent frames in the same episode group."""
    if len(prediction) < 2:
        return prediction.sum() * 0.0
    valid = torch.as_tensor(
        [
            str(groups[index]) == str(groups[index - 1])
            and int(frame_indices[index]) == int(frame_indices[index - 1]) + 1
            for index in range(1, len(prediction))
        ],
        dtype=torch.bool,
        device=prediction.device,
    )
    if not bool(valid.any()):
        return prediction.sum() * 0.0
    prediction_delta = torch.diff(prediction, dim=0)[valid]
    target_delta = torch.diff(target, dim=0)[valid]
    return F.smooth_l1_loss(prediction_delta, target_delta)


def _required(config: dict, key: str):
    if key not in config:
        raise ValueError(f"missing config key: {key}")
    return config[key]


def run_training(config: dict) -> dict:
    config = dict(config)
    seed = int(config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(config.get("torch_threads", 1)))
    dataset_path = Path(_required(config, "dataset"))
    output = Path(_required(config, "output_dir"))
    output.mkdir(parents=True, exist_ok=True)
    frames = FrameData.from_npz(dataset_path)
    limits = _required(config, "action_limits")
    low = torch.tensor(limits["low"], dtype=torch.float32)
    high = torch.tensor(limits["high"], dtype=torch.float32)
    if (
        low.shape != (2,)
        or high.shape != (2,)
        or not torch.all(torch.isfinite(low))
        or not torch.all(torch.isfinite(high))
        or not torch.all(low < high)
    ):
        raise ValueError(
            "action_limits must contain two finite ordered low/high values"
        )
    targets = torch.as_tensor(frames.targets)
    outside = torch.any((targets < low) | (targets > high), dim=1)
    if torch.any(outside):
        raise ValueError(
            f"{int(torch.sum(outside))} target rows exceed configured action_limits"
        )
    split = grouped_split(
        frames, tuple(config.get("split_ratios", (0.7, 0.15, 0.15))), seed
    )
    provenance = {
        "path": str(dataset_path.resolve()),
        "sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
    }
    manifest = save_split_manifest(
        output / "split_manifest.json", frames, split, dataset_provenance=provenance
    )
    normalizer = Normalizer.fit(frames.lidar, frames.aux, split["train"])
    _write_json(output / "normalizer.json", normalizer.to_dict())
    lidar, aux = normalizer.transform(frames.lidar, frames.aux)
    normalized_frames = FrameData(
        lidar,
        aux,
        frames.targets,
        frames.recording_ids,
        frames.episode_ids,
        frames.maneuver_classes,
    )
    train_data = TemporalDataset(normalized_frames, split["train"])
    validation_data = TemporalDataset(normalized_frames, split["validation"])
    if not train_data or not validation_data:
        raise ValueError(
            "each train/validation split needs at least one 10-frame history"
        )
    batch_size = int(config.get("batch_size", 32))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    validation_loader = DataLoader(
        validation_data, batch_size=batch_size, shuffle=False
    )
    model_config = dict(config.get("model", {}))
    model = TemporalPolicy(**model_config)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config.get("learning_rate", 1e-3))
    )
    weights = config.get("loss_weights", {})
    ws = float(weights.get("steering", 1))
    wl = float(weights.get("longitudinal", 1))
    wc = float(weights.get("action_change", 0.05))
    resolved = dict(config)
    resolved.update(
        {"seed": seed, "dataset": str(dataset_path), "output_dir": str(output)}
    )
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=True)
    )
    metadata = {
        "normalizer": normalizer.to_dict(),
        "split_manifest": manifest,
        "dataset_provenance": provenance,
        "action_limits": limits,
        "resolved_config": resolved,
        "code_provenance": {"package_version": "0.1.0"},
    }
    curves = []
    best = float("inf")
    best_epoch = -1
    epochs = int(config.get("epochs", 10))
    for epoch in range(epochs):
        model.train()
        losses = []
        for lidar_batch, aux_batch, target, meta in train_loader:
            optimizer.zero_grad()
            normalized, _ = model(lidar_batch, aux_batch)
            prediction = _physical(normalized, low, high)
            loss = ws * F.smooth_l1_loss(
                prediction[:, 0], target[:, 0]
            ) + wl * F.smooth_l1_loss(prediction[:, 1], target[:, 1])
            loss = loss + wc * grouped_action_change_loss(
                prediction, target, meta["group"], meta["frame_index"]
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        predictions = []
        targets = []
        maneuvers = []
        with torch.no_grad():
            for lidar_batch, aux_batch, target, meta in validation_loader:
                normalized, _ = model(lidar_batch, aux_batch)
                predictions.append(_physical(normalized, low, high).numpy())
                targets.append(target.numpy())
                maneuvers.extend(meta["maneuver_class"])
        prediction_array = np.concatenate(predictions)
        target_array = np.concatenate(targets)
        validation = compute_metrics(
            prediction_array, target_array, maneuver_classes=np.asarray(maneuvers)
        )
        score = float(validation["steering_mae_rad"]) + float(
            validation["longitudinal_acceleration_mae_mps2"]
        )
        curve = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_score": score,
        }
        curves.append(curve)
        save_checkpoint(
            output / "last.pt",
            model,
            optimizer,
            epoch=epoch,
            metrics=curve,
            metadata=metadata,
            model_config=model_config,
        )
        if score < best:
            best = score
            best_epoch = epoch
            save_checkpoint(
                output / "best.pt",
                model,
                optimizer,
                epoch=epoch,
                metrics=curve,
                metadata=metadata,
                model_config=model_config,
            )
            _write_json(output / "validation_metrics.json", validation)
    _write_json(output / "curves.json", {"epochs": curves})
    return {
        "best_epoch": best_epoch,
        "best_validation_score": best,
        "output_dir": str(output),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train temporal racing maneuver policy"
    )
    parser.add_argument("config")
    args = parser.parse_args(argv)
    with Path(args.config).open() as stream:
        config = yaml.safe_load(stream)
    print(json.dumps(run_training(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
