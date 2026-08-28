"""Pure, ROS-independent safety seam for the racing maneuver policy."""

from __future__ import annotations

import math
import re
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import torch

HISTORY_FRAMES = 10
LIDAR_RAYS = 360
LIDAR_FOV_DEG = 179.0
AUX_FEATURE_NAMES = (
    "longitudinal_velocity_mps",
    "measured_steering_rad",
    "longitudinal_acceleration_mps2",
    "previous_safe_steering_command_rad",
    "previous_safe_longitudinal_command",
    "opponent_present",
    "opponent_relative_x_body_m",
    "opponent_relative_y_body_m",
    "opponent_relative_vx_body_mps",
    "opponent_relative_vy_body_mps",
    "opponent_age_s",
)


def torch_versions_compatible(artifact_version: str, runtime_version: str) -> bool:
    """TorchScript only promises compatibility within the producing runtime line."""
    pattern = re.compile(r"^(\d+)\.(\d+)")
    artifact = pattern.match(str(artifact_version))
    runtime = pattern.match(str(runtime_version))
    return bool(artifact and runtime and artifact.groups() == runtime.groups())


def validate_artifact_metadata(
    metadata: Mapping[str, object],
    expected_rate_hz: float,
    expected_corridor_half_width_m: float = 5.0,
    expected_opponent_maximum_speed_mps: float = 50.0,
    expected_v2x_maximum_age_s: float = 0.25,
    expected_steering_limit_rad: float = 1.0,
    expected_acceleration_limit_mps2: float = 3.0,
) -> None:
    """Reject a model whose embedded preprocessing/action contract differs."""
    expected = {
        "schema_version": 1,
        "canonical_lidar_rays": LIDAR_RAYS,
        "canonical_lidar_fov_deg": LIDAR_FOV_DEG,
        "model_max_range_m": 30.0,
        "opponent_forward_corridor_half_width_m": expected_corridor_half_width_m,
        "opponent_maximum_speed_mps": expected_opponent_maximum_speed_mps,
        "v2x_maximum_age_s": expected_v2x_maximum_age_s,
        "history_length": HISTORY_FRAMES,
        "control_rate_hz": float(expected_rate_hz),
        "feature_ordering": ["canonical_lidar_ranges_m", *AUX_FEATURE_NAMES],
        "target_ordering": [
            "steering_tire_angle_rad",
            "longitudinal_acceleration_mps2",
        ],
        "output_units": ["rad", "m/s^2"],
        "action_low": [
            -float(expected_steering_limit_rad),
            -float(expected_acceleration_limit_mps2),
        ],
        "action_high": [
            float(expected_steering_limit_rad),
            float(expected_acceleration_limit_mps2),
        ],
    }
    for key, expected_value in expected.items():
        if key not in metadata:
            raise ValueError(f"artifact metadata is missing {key}")
        actual = metadata[key]
        if isinstance(expected_value, float):
            try:
                matches = math.isclose(float(actual), expected_value, rel_tol=0.0, abs_tol=1e-6)
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual == expected_value
        if not matches:
            raise ValueError(
                f"artifact {key} mismatch: expected {expected_value!r}, got {actual!r}"
            )


@dataclass(frozen=True)
class OpponentObservation:
    vehicle_id: str
    x: float
    y: float
    vx: float
    vy: float
    stamp_seconds: float


def mode_is_permitted(
    mode: int,
    publish_commands: bool,
    actuation_modes: set[int],
    shadow_modes: set[int],
) -> bool:
    """Use the narrow actuation allow-list whenever real output is enabled."""
    return mode in (actuation_modes if publish_commands else shadow_modes)


def publication_decision(
    mode: int,
    publish_commands: bool,
    actuation_modes: set[int],
    shadow_modes: set[int],
) -> tuple[bool, bool]:
    """Return ``(publish_debug, publish_real)`` without conflating mode gates."""
    if not mode_is_permitted(mode, publish_commands, actuation_modes, shadow_modes):
        return False, False
    return True, bool(publish_commands and mode in actuation_modes)


def pose_jump_exceeds(
    previous_xy: tuple[float, float] | None,
    current_xy: tuple[float, float],
    maximum_jump_m: float,
) -> bool:
    values = np.asarray(
        [*(previous_xy or current_xy), *current_xy, maximum_jump_m], dtype=np.float64
    )
    if not np.all(np.isfinite(values)) or maximum_jump_m <= 0.0:
        raise ValueError("pose and maximum jump must be finite and valid")
    if previous_xy is None:
        return False
    return math.hypot(current_xy[0] - previous_xy[0], current_xy[1] - previous_xy[1]) > maximum_jump_m


def sources_are_fresh(
    now_ns: int,
    stamps_ns: Mapping[str, int],
    maximum_age_ms: Mapping[str, float],
) -> bool:
    """Require every named source to be present, causal, and within its age limit."""
    for name, age_ms in maximum_age_ms.items():
        stamp = stamps_ns.get(name)
        if stamp is None or stamp <= 0 or stamp > now_ns:
            return False
        if now_ns - stamp > float(age_ms) * 1_000_000.0:
            return False
    return True


def canonicalize_lidar(
    ranges: Iterable[float],
    *,
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    target_angle_min: float = -math.radians(LIDAR_FOV_DEG / 2.0),
    target_angle_max: float = math.radians(LIDAR_FOV_DEG / 2.0),
    model_max_range: float = 30.0,
    rays: int = LIDAR_RAYS,
) -> np.ndarray:
    """Interpolate a LaserScan onto chronological angular rays, left to right."""
    values = np.asarray(tuple(ranges), dtype=np.float64)
    scalars = (
        angle_min,
        angle_increment,
        range_min,
        range_max,
        target_angle_min,
        target_angle_max,
        model_max_range,
    )
    if values.size < 2 or rays < 2 or not np.all(np.isfinite(scalars)):
        raise ValueError("LiDAR geometry must be finite and contain at least two rays")
    if angle_increment <= 0.0 or range_min < 0.0 or range_max <= range_min or model_max_range <= range_min:
        raise ValueError("invalid LiDAR angular or range limits")
    source_angles = angle_min + np.arange(values.size, dtype=np.float64) * angle_increment
    physical_maximum = min(range_max, model_max_range)
    valid = np.isfinite(values) & (values >= range_min) & (values <= range_max)
    cleaned = np.where(valid, values, physical_maximum)
    cleaned = np.clip(cleaned, range_min, physical_maximum)
    targets = np.linspace(target_angle_min, target_angle_max, rays, dtype=np.float64)
    if targets[0] < source_angles[0] or targets[-1] > source_angles[-1]:
        raise ValueError("LaserScan does not cover configured canonical field of view")
    result = np.interp(targets, source_angles, cleaned).astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError("canonical LiDAR must be finite")
    return result


def summarize_opponents(
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    ego_vx: float,
    ego_vy: float,
    now_seconds: float,
    opponents: Iterable[OpponentObservation],
    corridor_half_width_m: float = 5.0,
) -> np.ndarray:
    """Summarize the nearest opponent in the ego body frame."""
    ego = np.asarray([ego_x, ego_y, ego_yaw, ego_vx, ego_vy, now_seconds], dtype=np.float64)
    candidates = tuple(opponents)
    if not np.all(np.isfinite(ego)):
        raise ValueError("ego state and time must be finite")
    if not candidates:
        return np.zeros(6, dtype=np.float32)
    finite_candidates: list[tuple[float, OpponentObservation, float, float]] = []
    cosine, sine = math.cos(ego_yaw), math.sin(ego_yaw)
    for opponent in candidates:
        row = np.asarray(
            [opponent.x, opponent.y, opponent.vx, opponent.vy, opponent.stamp_seconds],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(row)) or opponent.stamp_seconds > now_seconds:
            continue
        dx, dy = opponent.x - ego_x, opponent.y - ego_y
        body_x = cosine * dx + sine * dy
        body_y = -sine * dx + cosine * dy
        if body_x > 0.0 and abs(body_y) <= corridor_half_width_m:
            finite_candidates.append((math.hypot(body_x, body_y), opponent, body_x, body_y))
    if not finite_candidates:
        return np.zeros(6, dtype=np.float32)
    _, nearest, body_x, body_y = min(finite_candidates, key=lambda item: item[0])
    dvx, dvy = nearest.vx - ego_vx, nearest.vy - ego_vy
    return np.asarray(
        [
            1.0,
            body_x,
            body_y,
            cosine * dvx + sine * dvy,
            -sine * dvx + cosine * dvy,
            max(0.0, now_seconds - nearest.stamp_seconds),
        ],
        dtype=np.float32,
    )


class ControllerCore:
    """Causal history, untrusted inference, and final output safety limits."""

    def __init__(
        self,
        policy: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        history_frames: int = HISTORY_FRAMES,
        lidar_rays: int = LIDAR_RAYS,
        auxiliary_dim: int = len(AUX_FEATURE_NAMES),
        steering_limit_rad: float,
        acceleration_limit_mps2: float,
        steering_rate_limit_rad_s: float,
        acceleration_rate_limit_mps3: float,
    ) -> None:
        dimensions = (history_frames, lidar_rays, auxiliary_dim)
        limits = (
            steering_limit_rad,
            acceleration_limit_mps2,
            steering_rate_limit_rad_s,
            acceleration_rate_limit_mps3,
        )
        if min(dimensions) < 1 or not np.all(np.isfinite(limits)) or min(limits) <= 0:
            raise ValueError("dimensions and finite safety limits must be positive")
        self.policy = policy
        self.history_frames = history_frames
        self.lidar_rays = lidar_rays
        self.auxiliary_dim = auxiliary_dim
        self._absolute_limits = np.asarray(limits[:2], dtype=np.float32)
        self._rate_limits = np.asarray(limits[2:], dtype=np.float32)
        self._lidar_history: deque[np.ndarray] = deque(maxlen=history_frames)
        self._auxiliary_history: deque[np.ndarray] = deque(maxlen=history_frames)
        self.previous_safe_output = np.zeros(2, dtype=np.float32)

    @property
    def history_size(self) -> int:
        return len(self._lidar_history)

    @property
    def ready(self) -> bool:
        return self.history_size == self.history_frames

    def reset(self) -> None:
        self._lidar_history.clear()
        self._auxiliary_history.clear()
        self.previous_safe_output.fill(0.0)

    def append_frame(self, lidar: np.ndarray, auxiliary: np.ndarray) -> None:
        lidar_row = np.asarray(lidar, dtype=np.float32)
        auxiliary_row = np.asarray(auxiliary, dtype=np.float32)
        if lidar_row.shape != (self.lidar_rays,) or auxiliary_row.shape != (self.auxiliary_dim,):
            self.reset()
            raise ValueError(
                f"expected lidar [{self.lidar_rays}] and auxiliary [{self.auxiliary_dim}]"
            )
        if not np.all(np.isfinite(lidar_row)) or not np.all(np.isfinite(auxiliary_row)):
            self.reset()
            raise ValueError("input frame must contain only finite values")
        self._lidar_history.append(lidar_row.copy())
        self._auxiliary_history.append(auxiliary_row.copy())

    def history_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.ready:
            raise RuntimeError("history is not warm")
        return (
            np.stack(tuple(self._lidar_history), axis=0),
            np.stack(tuple(self._auxiliary_history), axis=0),
        )

    def infer_and_limit(self, dt_seconds: float) -> np.ndarray:
        if not np.isfinite(dt_seconds) or dt_seconds <= 0.0:
            self.reset()
            raise ValueError("dt_seconds must be finite and positive")
        lidar, auxiliary = self.history_arrays()
        # ROS 2 Humble images may carry PyTorch versions predating
        # ``inference_mode``; ``no_grad`` provides the required inference seam.
        with torch.no_grad():
            raw = self.policy(
                torch.from_numpy(lidar).unsqueeze(0),
                torch.from_numpy(auxiliary).unsqueeze(0),
            )
        if isinstance(raw, (tuple, list)) and len(raw) == 1:
            raw = raw[0]
        try:
            action = np.asarray(raw.detach().cpu().numpy(), dtype=np.float32)
        except (AttributeError, TypeError, ValueError) as error:
            self.reset()
            raise RuntimeError("policy must return a finite [1, 2] tensor") from error
        if action.shape != (1, 2) or not np.all(np.isfinite(action)):
            self.reset()
            raise RuntimeError("policy must return a finite [1, 2] tensor")
        desired = np.clip(action[0], -self._absolute_limits, self._absolute_limits)
        maximum_delta = self._rate_limits * float(dt_seconds)
        safe = self.previous_safe_output + np.clip(
            desired - self.previous_safe_output, -maximum_delta, maximum_delta
        )
        safe = np.clip(safe, -self._absolute_limits, self._absolute_limits).astype(np.float32)
        if not np.all(np.isfinite(safe)):
            self.reset()
            raise RuntimeError("limited policy output is not finite")
        self.previous_safe_output = safe
        return safe.copy()
