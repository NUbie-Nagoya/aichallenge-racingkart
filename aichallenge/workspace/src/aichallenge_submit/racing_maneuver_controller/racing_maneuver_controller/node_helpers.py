"""ROS-message duck-typed adapters kept importable without ROS."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from .controller_core import AUX_FEATURE_NAMES, OpponentObservation


def quaternion_yaw(quaternion) -> float:
    values = np.asarray(
        [quaternion.x, quaternion.y, quaternion.z, quaternion.w], dtype=np.float64
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("orientation quaternion must be finite")
    yaw = math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )
    if not math.isfinite(yaw):
        raise ValueError("orientation yaw must be finite")
    return yaw


def assemble_auxiliary(
    *,
    speed: float,
    steering: float,
    acceleration: float,
    previous_safe: np.ndarray,
    opponent_summary: np.ndarray,
) -> np.ndarray:
    result = np.concatenate(
        (
            np.asarray([speed, steering, acceleration], dtype=np.float32),
            np.asarray(previous_safe, dtype=np.float32),
            np.asarray(opponent_summary, dtype=np.float32),
        )
    ).astype(np.float32)
    if result.shape != (len(AUX_FEATURE_NAMES),) or not np.all(np.isfinite(result)):
        raise ValueError("auxiliary feature vector must be finite and 11-dimensional")
    return result


class V2XTracker:
    """Causal finite-difference tracks for position-only V2X messages."""

    def __init__(self, ego_vehicle_id: str, maximum_speed_mps: float) -> None:
        if not ego_vehicle_id:
            raise ValueError("ego_vehicle_id must not be empty")
        if not math.isfinite(maximum_speed_mps) or maximum_speed_mps <= 0.0:
            raise ValueError("maximum_speed_mps must be finite and positive")
        self.ego_vehicle_id = ego_vehicle_id
        self.maximum_speed_mps = maximum_speed_mps
        self._previous: dict[str, tuple[float, float, float]] = {}
        self._observations: dict[str, OpponentObservation] = {}

    def reset(self) -> None:
        self._previous.clear()
        self._observations.clear()

    def update(self, vehicles: Iterable[object], stamp_seconds: float) -> None:
        if not math.isfinite(stamp_seconds) or stamp_seconds <= 0.0:
            raise ValueError("V2X stamp must be finite and positive")
        next_observations: dict[str, OpponentObservation] = {}
        for vehicle in vehicles:
            vehicle_id = str(vehicle.vehicle_id)
            if not vehicle_id or vehicle_id == self.ego_vehicle_id:
                continue
            x, y = float(vehicle.position.x), float(vehicle.position.y)
            if not math.isfinite(x) or not math.isfinite(y):
                continue
            vx = vy = 0.0
            previous = self._previous.get(vehicle_id)
            if previous is not None:
                old_stamp, old_x, old_y = previous
                dt = stamp_seconds - old_stamp
                if dt > 0.0:
                    candidate_vx = (x - old_x) / dt
                    candidate_vy = (y - old_y) / dt
                    if math.hypot(candidate_vx, candidate_vy) <= self.maximum_speed_mps:
                        vx, vy = candidate_vx, candidate_vy
            self._previous[vehicle_id] = (stamp_seconds, x, y)
            next_observations[vehicle_id] = OpponentObservation(
                vehicle_id, x, y, vx, vy, stamp_seconds
            )
        self._observations = next_observations

    def observations(self) -> list[OpponentObservation]:
        return list(self._observations.values())
