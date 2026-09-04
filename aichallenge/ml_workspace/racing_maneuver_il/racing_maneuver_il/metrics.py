"""Physical-unit action metrics with maneuver and interaction slices."""

from __future__ import annotations

import numpy as np


def _base(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    error = prediction - target
    steering = error[:, 0]
    longitudinal = error[:, 1]
    change = (
        float(np.mean(np.abs(np.diff(prediction, axis=0) - np.diff(target, axis=0))))
        if len(prediction) > 1
        else 0.0
    )
    steering_mae = float(np.mean(np.abs(steering)))
    steering_rmse = float(np.sqrt(np.mean(steering**2)))
    return {
        "count": len(prediction),
        "steering_mae_rad": steering_mae,
        "steering_rmse_rad": steering_rmse,
        "steering_mae_deg": float(np.rad2deg(steering_mae)),
        "steering_rmse_deg": float(np.rad2deg(steering_rmse)),
        "target_speed_mae_mps": float(np.mean(np.abs(longitudinal))),
        "target_speed_rmse_mps": float(np.sqrt(np.mean(longitudinal**2))),
        "action_change_mae": change,
    }


def _slices(
    prediction: np.ndarray, target: np.ndarray, labels: np.ndarray
) -> dict[str, dict]:
    return {
        str(label): _base(prediction[labels == label], target[labels == label])
        for label in np.unique(labels)
        if np.any(labels == label)
    }


def compute_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    maneuver_classes: np.ndarray | None = None,
    opponent_present: np.ndarray | None = None,
    closing_speed_mps: np.ndarray | None = None,
) -> dict:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if (
        prediction.shape != target.shape
        or prediction.ndim != 2
        or prediction.shape[1] != 2
        or len(prediction) == 0
    ):
        raise ValueError("prediction and target must be nonempty matching [N,2] arrays")
    report = dict(_base(prediction, target))
    if maneuver_classes is not None:
        labels = np.asarray(maneuver_classes)
        report["by_maneuver"] = _slices(prediction, target, labels)
    if opponent_present is not None:
        labels = np.where(
            np.asarray(opponent_present).astype(bool), "present", "absent"
        )
        report["by_opponent_presence"] = _slices(prediction, target, labels)
    if closing_speed_mps is not None:
        speed = np.asarray(closing_speed_mps)
        labels = np.select(
            (speed < -0.5, speed < 0.5, speed < 3),
            ("receding", "steady", "closing"),
            default="fast_closing",
        )
        report["by_closing_speed_bin"] = _slices(prediction, target, labels)
    return report


def normalized_selection_score(
    metrics: dict,
    *,
    steering_tolerance_rad: float,
    target_speed_tolerance_mps: float,
) -> float:
    """Unitless validation objective from explicit physical error tolerances."""
    tolerances = (steering_tolerance_rad, target_speed_tolerance_mps)
    if not all(np.isfinite(value) and value > 0.0 for value in tolerances):
        raise ValueError("selection tolerances must be finite and positive")
    return float(
        float(metrics["steering_mae_rad"]) / steering_tolerance_rad
        + float(metrics["target_speed_mae_mps"])
        / target_speed_tolerance_mps
    )
