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
from tqdm.auto import tqdm

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


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; install a CUDA-enabled PyTorch build or use device: cpu")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be one of: auto, cpu, cuda")


def _device_batches(
    *,
    lidar: torch.Tensor,
    aux: torch.Tensor,
    targets: torch.Tensor,
    end_indices: torch.Tensor,
    batch_size: int,
    history_length: int,
):
    """Yield temporal batches assembled entirely on the selected device."""
    offsets = torch.arange(
        1 - history_length, 1, device=end_indices.device, dtype=torch.long
    )
    for start in range(0, len(end_indices), batch_size):
        ends = end_indices[start : start + batch_size]
        indices = ends.unsqueeze(1) + offsets.unsqueeze(0)
        yield lidar[indices], aux[indices], targets[ends], ends


def _create_run_directory(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    existing = [
        int(path.name.removeprefix("run-"))
        for path in output_root.iterdir()
        if path.is_dir() and path.name.removeprefix("run-").isdigit()
    ]
    run_number = max(existing, default=0) + 1
    while True:
        destination = output_root / f"run-{run_number}"
        try:
            destination.mkdir()
            return destination
        except FileExistsError:
            run_number += 1


def run_training(config: dict) -> dict:
    config = dict(config)
    seed = int(config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(config.get("torch_threads", 1)))
    device = _resolve_device(str(config.get("device", "cpu")))
    dataset_path = Path(_required(config, "dataset"))
    output = _create_run_directory(Path(_required(config, "output_dir")))
    frames = FrameData.from_npz(dataset_path)
    limits = _required(config, "action_limits")
    low = torch.tensor(limits["low"], dtype=torch.float32, device=device)
    high = torch.tensor(limits["high"], dtype=torch.float32, device=device)
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
    target_tensor = torch.as_tensor(
        frames.targets, dtype=torch.float32, device=device
    )
    outside = torch.any((target_tensor < low) | (target_tensor > high), dim=1)
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
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    # Keep source frames and temporal indexing on-device. This removes Python
    # DataLoader collation and host-to-device copies from the training hot path.
    lidar_device = torch.as_tensor(lidar, dtype=torch.float32, device=device)
    aux_device = torch.as_tensor(aux, dtype=torch.float32, device=device)
    train_end_indices = torch.as_tensor(
        train_data.end_indices, dtype=torch.long, device=device
    )
    validation_end_indices = torch.as_tensor(
        validation_data.end_indices, dtype=torch.long, device=device
    )
    groups = normalized_frames.groups
    adjacent = np.zeros(len(groups), dtype=np.bool_)
    adjacent[1:] = groups[1:] == groups[:-1]
    adjacent_device = torch.as_tensor(adjacent, dtype=torch.bool, device=device)
    model_config = dict(config.get("model", {}))
    model = TemporalPolicy(**model_config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config.get("learning_rate", 1e-3))
    )
    weights = config.get("loss_weights", {})
    ws = float(weights.get("steering", 1))
    wl = float(weights.get("longitudinal", 1))
    wc = float(weights.get("action_change", 0.05))
    resolved = dict(config)
    resolved.update(
        {
            "seed": seed,
            "dataset": str(dataset_path),
            "output_dir": str(output),
            "device": str(device),
        }
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
        total_loss = torch.zeros((), dtype=torch.float32, device=device)
        batch_count = 0
        progress = tqdm(
            _device_batches(
                lidar=lidar_device,
                aux=aux_device,
                targets=target_tensor,
                end_indices=train_end_indices,
                batch_size=batch_size,
                history_length=train_data.history_length,
            ),
            desc=f"Epoch {epoch + 1}/{epochs}",
            total=(len(train_end_indices) + batch_size - 1) // batch_size,
            unit="batch",
        )
        for lidar_batch, aux_batch, target, ends in progress:
            optimizer.zero_grad()
            normalized, _ = model(lidar_batch, aux_batch)
            prediction = _physical(normalized, low, high)
            loss = ws * F.smooth_l1_loss(
                prediction[:, 0], target[:, 0]
            ) + wl * F.smooth_l1_loss(prediction[:, 1], target[:, 1])
            valid = adjacent_device[ends[1:]]
            delta_error = F.smooth_l1_loss(
                torch.diff(prediction, dim=0),
                torch.diff(target, dim=0),
                reduction="none",
            ).mean(dim=1)
            change_loss = (delta_error * valid.float()).sum() / valid.sum().clamp_min(1)
            loss = loss + wc * change_loss
            loss.backward()
            optimizer.step()
            total_loss = total_loss + loss.detach()
            batch_count += 1
            progress.set_postfix(loss=f"{loss.detach().item():.4f}")
        model.eval()
        predictions = []
        validation_targets = []
        maneuvers = []
        with torch.no_grad():
            for lidar_batch, aux_batch, target, ends in _device_batches(
                lidar=lidar_device,
                aux=aux_device,
                targets=target_tensor,
                end_indices=validation_end_indices,
                batch_size=batch_size,
                history_length=validation_data.history_length,
            ):
                normalized, _ = model(lidar_batch, aux_batch)
                predictions.append(_physical(normalized, low, high).cpu().numpy())
                validation_targets.append(target.cpu().numpy())
                maneuvers.extend(normalized_frames.maneuver_classes[ends.cpu().numpy()])
        prediction_array = np.concatenate(predictions)
        target_array = np.concatenate(validation_targets)
        validation = compute_metrics(
            prediction_array, target_array, maneuver_classes=np.asarray(maneuvers)
        )
        score = float(validation["steering_mae_rad"]) + float(
            validation["longitudinal_acceleration_mae_mps2"]
        )
        curve = {
            "epoch": epoch,
            "train_loss": float((total_loss / batch_count).item()),
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
        "device": str(device),
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
