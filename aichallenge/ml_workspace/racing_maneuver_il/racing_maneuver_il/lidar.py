"""Angle-aware LiDAR canonicalization.

The output grid is ordered left-to-right from -135 to +135 degrees relative
to the vehicle's forward (+x) axis, including both endpoints. Values remain
in metres; normalization belongs to the exported policy wrapper.
"""

from __future__ import annotations

import numpy as np

from .schema import CANONICAL_LIDAR_FOV_DEG, MODEL_MAX_RANGE_M


def canonicalize_scan(
    ranges,
    angle_min_rad: float,
    angle_increment_rad: float,
    range_min_m: float,
    range_max_m: float,
    *,
    target_fov_deg: float = CANONICAL_LIDAR_FOV_DEG,
    target_rays: int = 360,
    model_max_range_m: float = MODEL_MAX_RANGE_M,
) -> np.ndarray:
    source = np.asarray(ranges, dtype=np.float64)
    if (
        source.ndim != 1
        or source.size < 2
        or not np.isfinite(angle_increment_rad)
        or angle_increment_rad <= 0
    ):
        raise ValueError("invalid source scan geometry")
    if not (0 <= range_min_m < range_max_m and model_max_range_m > range_min_m):
        raise ValueError("invalid range bounds")
    half_fov = np.deg2rad(target_fov_deg) / 2.0
    source_angles = angle_min_rad + np.arange(source.size) * angle_increment_rad
    if source_angles[0] > -half_fov + 1e-7 or source_angles[-1] < half_fov - 1e-7:
        raise ValueError("source scan has insufficient angular coverage")
    invalid_fill = min(float(range_max_m), float(model_max_range_m))
    source = np.where(np.isfinite(source), source, invalid_fill)
    source = np.clip(source, range_min_m, invalid_fill)
    targets = np.linspace(-half_fov, half_fov, target_rays)
    return np.interp(targets, source_angles, source).astype(np.float32)
