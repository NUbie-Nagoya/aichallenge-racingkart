"""Verified TorchScript export CLI and API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .checkpoint import load_checkpoint
from .model import ExportPolicy
from .normalization import Normalizer
from .schema import (
    AUX_FEATURE_NAMES,
    CANONICAL_LIDAR_FOV_DEG,
    CANONICAL_LIDAR_RAYS,
    CONTROL_RATE_HZ,
    HISTORY_LENGTH,
    MODEL_MAX_RANGE_M,

    SCHEMA_VERSION,
    TARGET_NAMES,

)


def _contract(wrapper: ExportPolicy, metadata: dict) -> dict:
    reserved = {
        "schema_version",
        "canonical_lidar_rays",
        "canonical_lidar_fov_deg",
        "model_max_range_m",

        "history_length",
        "control_rate_hz",
        "feature_ordering",
        "target_ordering",
        "output_units",
        "action_low",
        "action_high",
        "normalizer",
        "torch_version",
    }
    overlap = reserved.intersection(metadata)
    if overlap:
        raise ValueError(f"metadata cannot override contract fields: {sorted(overlap)}")
    return {
        "schema_version": SCHEMA_VERSION,
        "canonical_lidar_rays": CANONICAL_LIDAR_RAYS,
        "canonical_lidar_fov_deg": CANONICAL_LIDAR_FOV_DEG,
        "model_max_range_m": MODEL_MAX_RANGE_M,

        "history_length": HISTORY_LENGTH,
        "control_rate_hz": CONTROL_RATE_HZ,
        "feature_ordering": ["canonical_lidar_ranges_m", *AUX_FEATURE_NAMES],
        "target_ordering": list(TARGET_NAMES),
        "output_units": ["rad", "m/s"],
        "action_low": wrapper.action_low.detach().cpu().tolist(),
        "action_high": wrapper.action_high.detach().cpu().tolist(),
        "normalizer": {
            "lidar_mean": wrapper.lidar_mean.detach().cpu().tolist(),
            "lidar_std": wrapper.lidar_std.detach().cpu().tolist(),
            "aux_mean": wrapper.aux_mean.detach().cpu().tolist(),
            "aux_std": wrapper.aux_std.detach().cpu().tolist(),
        },
        "torch_version": torch.__version__,
        **metadata,
    }


def export_torchscript(
    wrapper: ExportPolicy, path: str | Path, metadata: dict | None = None
) -> dict:
    wrapper = wrapper.cpu().eval()
    scripted = torch.jit.script(wrapper)
    contract = _contract(wrapper, metadata or {})
    extra = {"metadata.json": json.dumps(contract, sort_keys=True)}
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.jit.save(scripted, str(temporary), _extra_files=extra)
    loaded = torch.jit.load(str(temporary))
    loaded(torch.zeros(1, HISTORY_LENGTH, 360), torch.zeros(1, HISTORY_LENGTH, len(AUX_FEATURE_NAMES)))
    temporary.replace(destination)
    sidecar = destination.with_suffix(destination.suffix + ".metadata.json")
    sidecar_temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
    sidecar_temporary.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    sidecar_temporary.replace(sidecar)
    return contract


def load_metadata(path: str | Path) -> dict:
    extra = {"metadata.json": ""}
    torch.jit.load(str(path), _extra_files=extra)
    value = extra["metadata.json"]
    value = value.decode() if isinstance(value, bytes) else value
    return json.loads(value)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    args = parser.parse_args(argv)
    model, payload = load_checkpoint(args.checkpoint)
    metadata = payload["metadata"]
    normalizer = Normalizer.from_dict(metadata["normalizer"])
    limits = metadata["action_limits"]
    wrapper = ExportPolicy.from_normalizer(
        model.eval(), normalizer, tuple(limits["low"]), tuple(limits["high"])
    )
    export_torchscript(
        wrapper,
        args.output,
        {
            "dataset_provenance": metadata.get("dataset_provenance", {}),
            "split_provenance": metadata.get("split_manifest", {}),
        },
    )


if __name__ == "__main__":
    main()
