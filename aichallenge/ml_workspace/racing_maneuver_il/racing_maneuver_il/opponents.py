"""Causal body-frame nearest-opponent summarization."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .schema import (
    OPPONENT_FORWARD_CORRIDOR_HALF_WIDTH_M,
    OPPONENT_MAXIMUM_SPEED_MPS,
    V2X_MAXIMUM_AGE_S,
)


@dataclass(frozen=True)
class OpponentObservation:
    vehicle_id: str
    x_map_m: float
    y_map_m: float
    timestamp_s: float


@dataclass(frozen=True)
class OpponentSummary:
    present: float = 0.0
    relative_x_body_m: float = 0.0
    relative_y_body_m: float = 0.0
    relative_vx_body_mps: float = 0.0
    relative_vy_body_mps: float = 0.0
    age_s: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.asarray(
            (
                self.present,
                self.relative_x_body_m,
                self.relative_y_body_m,
                self.relative_vx_body_mps,
                self.relative_vy_body_mps,
                self.age_s,
            ),
            dtype=np.float32,
        )


def _body(vector: tuple[float, float], yaw: float) -> tuple[float, float]:
    c, s = math.cos(yaw), math.sin(yaw)
    return c * vector[0] + s * vector[1], -s * vector[0] + c * vector[1]


def summarize_opponents(
    ego_position_map_m: tuple[float, float],
    ego_yaw_rad: float,
    ego_velocity_map_mps: tuple[float, float],
    observations: Sequence[OpponentObservation],
    label_time_s: float,
    *,
    previous: Mapping[str, OpponentObservation] | None = None,
    max_age_s: float = V2X_MAXIMUM_AGE_S,
    corridor_half_width_m: float = OPPONENT_FORWARD_CORRIDOR_HALF_WIDTH_M,
    max_velocity_dt_s: float = 2.0,
    max_opponent_speed_mps: float = OPPONENT_MAXIMUM_SPEED_MPS,
) -> OpponentSummary:
    candidates = []
    for observation in observations:
        age = label_time_s - observation.timestamp_s
        if age < 0 or age > max_age_s:
            continue
        rel = (
            observation.x_map_m - ego_position_map_m[0],
            observation.y_map_m - ego_position_map_m[1],
        )
        bx, by = _body(rel, ego_yaw_rad)
        if bx > 0 and abs(by) <= corridor_half_width_m:
            candidates.append((math.hypot(bx, by), observation, bx, by, age))
    if not candidates:
        return OpponentSummary()
    _, current, bx, by, age = min(candidates, key=lambda item: item[0])
    opponent_velocity = (0.0, 0.0)
    prior = (previous or {}).get(current.vehicle_id)
    if prior is not None:
        dt = current.timestamp_s - prior.timestamp_s
        if 0 < dt <= max_velocity_dt_s:
            candidate_velocity = (
                (current.x_map_m - prior.x_map_m) / dt,
                (current.y_map_m - prior.y_map_m) / dt,
            )
            if math.hypot(*candidate_velocity) <= max_opponent_speed_mps:
                opponent_velocity = candidate_velocity
    relative_velocity = (
        opponent_velocity[0] - ego_velocity_map_mps[0],
        opponent_velocity[1] - ego_velocity_map_mps[1],
    )
    bvx, bvy = _body(relative_velocity, ego_yaw_rad)
    return OpponentSummary(1.0, bx, by, bvx, bvy, age)
