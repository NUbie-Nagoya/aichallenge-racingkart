"""Portable checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .model import TemporalPolicy


def save_checkpoint(
    path: str | Path,
    model: TemporalPolicy,
    optimizer: torch.optim.Optimizer | None,
    *,
    epoch: int,
    metrics: dict[str, float],
    metadata: dict[str, Any],
    model_config: dict[str, Any],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "epoch": epoch,
        "metrics": metrics,
        "metadata": metadata,
        "model_config": model_config,
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[TemporalPolicy, dict[str, Any]]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # PyTorch 1.8 in the ROS 2 Humble runtime predates ``weights_only``.
        payload = torch.load(path, map_location=map_location)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    model = TemporalPolicy(**payload["model_config"])
    model.load_state_dict(payload["model_state"])
    return model, payload
