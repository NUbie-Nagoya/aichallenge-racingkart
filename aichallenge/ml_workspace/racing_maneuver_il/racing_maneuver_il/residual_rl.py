"""Residual-RL building blocks for racing-focused BC fine-tuning.

This module deliberately keeps the learned residual bounded around the frozen
behavioral-cloning policy.  It rewards racing progress and safe vehicle motion,
not matching an MPC reference-speed table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import time

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from .schema import AUX_FEATURE_NAMES, CANONICAL_LIDAR_RAYS, HISTORY_LENGTH


@dataclass(frozen=True)
class ResidualActionConfig:
    max_steering_delta_rad: float = 0.10
    max_target_speed_delta_mps: float = 0.75
    steering_limit_rad: float = 1.0
    target_speed_limit_mps: float = 15.0
    steering_rate_limit_rad_s: float = 1.5
    target_speed_rate_limit_mps2: float = 4.0
    control_rate_hz: float = 20.0

    def __post_init__(self) -> None:
        values = np.asarray(
            (
                self.max_steering_delta_rad,
                self.max_target_speed_delta_mps,
                self.steering_limit_rad,
                self.target_speed_limit_mps,
                self.steering_rate_limit_rad_s,
                self.target_speed_rate_limit_mps2,
                self.control_rate_hz,
            ),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("residual action limits must be positive and finite")


@dataclass(frozen=True)
class WarmupConfig:
    """Straight-line reset warm-up that moves BC into its training envelope."""

    minimum_duration_s: float = 1.5
    handoff_speed_mps: float = 4.5
    maximum_duration_s: float = 12.0
    target_speed_mps: float = 7.5

    def __post_init__(self) -> None:
        values = np.asarray(
            (
                self.minimum_duration_s,
                self.handoff_speed_mps,
                self.maximum_duration_s,
                self.target_speed_mps,
            ),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("warm-up values must be positive and finite")
        if self.minimum_duration_s > self.maximum_duration_s:
            raise ValueError("warm-up minimum_duration_s cannot exceed maximum_duration_s")
        if self.handoff_speed_mps > self.target_speed_mps:
            raise ValueError("warm-up handoff_speed_mps cannot exceed target_speed_mps")


@dataclass(frozen=True)
class CrashCheckerConfig:
    """Physics-independent crash/stall termination thresholds."""

    velocity_drop_threshold_mps: float = 2.0
    maximum_drop_interval_s: float = 0.25
    minimum_speed_before_drop_mps: float = 1.0
    low_speed_threshold_mps: float = 0.1
    low_speed_duration_s: float = 10.0

    def __post_init__(self) -> None:
        values = np.asarray(tuple(self.__dict__.values()), dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("crash-checker thresholds must be positive and finite")


class CrashChecker:
    """Detect an impact-like speed loss or a sustained stationary failure."""

    def __init__(self, config: CrashCheckerConfig | None = None) -> None:
        self.config = config or CrashCheckerConfig()
        self.reset()

    def reset(self) -> None:
        self._previous_speed: float | None = None
        self._previous_timestamp: float | None = None
        self._low_speed_since: float | None = None

    def update(self, *, speed_mps: float, timestamp_s: float) -> str | None:
        if not np.isfinite(speed_mps) or not np.isfinite(timestamp_s) or speed_mps < 0.0:
            raise ValueError("crash-checker speed/timestamp must be finite and nonnegative")
        if self._previous_timestamp is not None and timestamp_s < self._previous_timestamp:
            raise ValueError("crash-checker timestamps must be monotonic")
        reason = None
        if self._previous_speed is not None and self._previous_timestamp is not None:
            elapsed = timestamp_s - self._previous_timestamp
            if (
                elapsed <= self.config.maximum_drop_interval_s
                and self._previous_speed >= self.config.minimum_speed_before_drop_mps
                and self._previous_speed - speed_mps >= self.config.velocity_drop_threshold_mps
            ):
                reason = "velocity_drop"
        if speed_mps < self.config.low_speed_threshold_mps:
            self._low_speed_since = timestamp_s if self._low_speed_since is None else self._low_speed_since
            if timestamp_s - self._low_speed_since >= self.config.low_speed_duration_s:
                reason = reason or "prolonged_low_speed"
        else:
            self._low_speed_since = None
        self._previous_speed = speed_mps
        self._previous_timestamp = timestamp_s
        return reason


@dataclass(frozen=True)
class RacingRewardConfig:
    progress_weight: float = 4.0
    speed_weight: float = 0.08
    collision_penalty: float = 100.0
    off_track_penalty: float = 50.0
    danger_clearance_m: float = 2.5
    proximity_penalty_weight: float = 3.0
    hazard_braking_weight: float = 1.5
    hazard_steering_weight: float = 0.20
    residual_change_weight: float = 0.03


def combine_residual_action(
    *,
    base_action: np.ndarray,
    residual_action: np.ndarray,
    config: ResidualActionConfig,
) -> np.ndarray:
    """Add normalized residuals to BC control and enforce physical limits."""
    base = np.asarray(base_action, dtype=np.float32)
    residual = np.asarray(residual_action, dtype=np.float32)
    if base.shape != (2,) or residual.shape != (2,):
        raise ValueError("base and residual actions must have shape [2]")
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(residual)):
        raise ValueError("base and residual actions must be finite")
    clipped_residual = np.clip(residual, -1.0, 1.0)
    command = base + clipped_residual * np.asarray(
        (config.max_steering_delta_rad, config.max_target_speed_delta_mps),
        dtype=np.float32,
    )
    return np.clip(
        command,
        (-config.steering_limit_rad, 0.0),
        (config.steering_limit_rad, config.target_speed_limit_mps),
    ).astype(np.float32)


def rate_limit_command(
    *, previous_command: np.ndarray, desired_command: np.ndarray, config: ResidualActionConfig
) -> np.ndarray:
    """Apply standalone-equivalent final-command slew limits at the control rate."""
    previous = np.asarray(previous_command, dtype=np.float32)
    desired = np.asarray(desired_command, dtype=np.float32)
    if previous.shape != (2,) or desired.shape != (2,):
        raise ValueError("previous and desired commands must have shape [2]")
    if not np.all(np.isfinite(np.concatenate((previous, desired)))):
        raise ValueError("previous and desired commands must be finite")
    maximum_delta = np.asarray(
        (
            config.steering_rate_limit_rad_s / config.control_rate_hz,
            config.target_speed_rate_limit_mps2 / config.control_rate_hz,
        ),
        dtype=np.float32,
    )
    limited = previous + np.clip(desired - previous, -maximum_delta, maximum_delta)
    return np.clip(
        limited,
        (-config.steering_limit_rad, 0.0),
        (config.steering_limit_rad, config.target_speed_limit_mps),
    ).astype(np.float32)


def racing_reward(
    *,
    progress_m: float,
    speed_mps: float,
    collision: bool,
    off_track: bool,
    nearest_clearance_m: float,
    command: np.ndarray,
    measured_speed_mps: float,
    residual_action: np.ndarray,
    previous_residual: np.ndarray,
    config: RacingRewardConfig,
) -> float:
    """Reward track progress/speed and hazard response; never uses ref_vel."""
    scalars = np.asarray(
        (progress_m, speed_mps, nearest_clearance_m, measured_speed_mps), dtype=np.float64
    )
    action = np.asarray(command, dtype=np.float64)
    residual = np.asarray(residual_action, dtype=np.float64)
    previous = np.asarray(previous_residual, dtype=np.float64)
    if (
        not np.all(np.isfinite(scalars))
        or action.shape != (2,)
        or residual.shape != (2,)
        or previous.shape != (2,)
        or not np.all(np.isfinite(np.concatenate((action, residual, previous))))
    ):
        raise ValueError("racing reward inputs must be finite scalar/action values")

    reward = config.progress_weight * max(0.0, progress_m)
    reward += config.speed_weight * max(0.0, speed_mps)
    # Any ray may represent another kart or a track barrier. Use the closest
    # valid LiDAR return rather than rewarding only a front-sector reaction.
    hazard = float(np.clip((config.danger_clearance_m - nearest_clearance_m) / config.danger_clearance_m, 0.0, 1.0))
    # A smooth quadratic penalty creates a learning signal before contact while
    # avoiding a sharp threshold at the danger-clearance boundary.
    reward -= config.proximity_penalty_weight * hazard**2
    braking = max(0.0, measured_speed_mps - float(action[1]))
    reward += hazard * (
        config.hazard_braking_weight * braking
        + config.hazard_steering_weight * abs(float(action[0]))
    )
    reward -= config.residual_change_weight * float(np.square(residual - previous).sum())
    if off_track:
        reward -= config.off_track_penalty
    if collision:
        reward -= config.collision_penalty
    return float(reward)


@dataclass(frozen=True)
class ResidualTransition:
    lidar: np.ndarray
    aux: np.ndarray
    progress_m: float
    speed_mps: float
    collision: bool
    off_track: bool
    timestamp_s: float | None = None

    def validate(self) -> None:
        lidar = np.asarray(self.lidar, dtype=np.float32)
        aux = np.asarray(self.aux, dtype=np.float32)
        if lidar.shape != (CANONICAL_LIDAR_RAYS,) or aux.shape != (len(AUX_FEATURE_NAMES),):
            raise ValueError("transition observation violates the racing-maneuver schema")
        if not np.all(np.isfinite(lidar)) or not np.all(np.isfinite(aux)):
            raise ValueError("transition observation must be finite")
        if not np.all(np.isfinite((self.progress_m, self.speed_mps))):
            raise ValueError("transition progress and speed must be finite")
        if self.timestamp_s is not None and not np.isfinite(self.timestamp_s):
            raise ValueError("transition timestamp must be finite when provided")


class ResidualRacingBackend(Protocol):
    """AWSIM adapter seam; implementations own reset, ticking, and publishing."""

    def reset(self) -> ResidualTransition: ...

    def observe(self) -> ResidualTransition: ...

    def step(self, command: np.ndarray) -> ResidualTransition: ...

    def close(self) -> None: ...


class ResidualRacingEnv(gym.Env):
    """Gym environment whose action is a bounded residual around frozen BC."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        backend: ResidualRacingBackend,
        bc_policy: torch.nn.Module,
        action_config: ResidualActionConfig | None = None,
        warmup_config: WarmupConfig | None = None,
        reward_config: RacingRewardConfig | None = None,
        crash_config: CrashCheckerConfig | None = None,
        max_steps: int = 2_400,
    ) -> None:
        super().__init__()
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.backend = backend
        self.bc_policy = bc_policy.eval()
        for parameter in self.bc_policy.parameters():
            parameter.requires_grad_(False)
        self.action_config = action_config or ResidualActionConfig()
        self.warmup_config = warmup_config or WarmupConfig()
        self.reward_config = reward_config or RacingRewardConfig()
        self.crash_checker = CrashChecker(crash_config)
        self.max_steps = max_steps
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "lidar": spaces.Box(
                    low=0.0, high=30.0, shape=(HISTORY_LENGTH, CANONICAL_LIDAR_RAYS), dtype=np.float32
                ),
                "aux": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(HISTORY_LENGTH, len(AUX_FEATURE_NAMES)), dtype=np.float32
                ),
            }
        )
        self._lidar_history: list[np.ndarray] = []
        self._aux_history: list[np.ndarray] = []
        self._previous_residual = np.zeros(2, dtype=np.float32)
        # aux[7:9] (previous commanded action) is owned by this environment:
        # the backend reports zeros there and the env overwrites them with the
        # actual command issued on the previous step. Feeding zeros every frame
        # (the historical bug) pushes the frozen BC far out of distribution
        # (~5 deg steering, ~4 m/s target-speed shift on real checkpoints).
        self._last_command = np.zeros(2, dtype=np.float32)
        self._step_count = 0

    def _with_action_history(self, transition: ResidualTransition) -> ResidualTransition:
        """Stamp the previous commanded action into aux[7:9] before appending."""
        aux = np.asarray(transition.aux, dtype=np.float32).copy()
        aux[7] = float(self._last_command[0])
        aux[8] = float(self._last_command[1])
        return ResidualTransition(
            lidar=transition.lidar,
            aux=aux,
            progress_m=transition.progress_m,
            speed_mps=transition.speed_mps,
            collision=transition.collision,
            off_track=transition.off_track,
            timestamp_s=transition.timestamp_s,
        )

    def _append(self, transition: ResidualTransition) -> None:
        transition.validate()
        self._lidar_history.append(np.asarray(transition.lidar, dtype=np.float32).copy())
        self._aux_history.append(np.asarray(transition.aux, dtype=np.float32).copy())
        self._lidar_history = self._lidar_history[-HISTORY_LENGTH:]
        self._aux_history = self._aux_history[-HISTORY_LENGTH:]

    def _observation(self) -> dict[str, np.ndarray]:
        if not self._lidar_history:
            raise RuntimeError("environment has not been reset")
        lidar = np.stack(self._lidar_history)
        aux = np.stack(self._aux_history)
        while len(lidar) < HISTORY_LENGTH:
            lidar = np.concatenate((lidar[:1], lidar), axis=0)
            aux = np.concatenate((aux[:1], aux), axis=0)
        return {"lidar": lidar.astype(np.float32), "aux": aux.astype(np.float32)}

    def _base_action(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        with torch.no_grad():
            result = self.bc_policy(
                torch.from_numpy(observation["lidar"]).unsqueeze(0),
                torch.from_numpy(observation["aux"]).unsqueeze(0),
            )
        if isinstance(result, tuple):
            result = result[0]
        action = np.asarray(result.detach().cpu().numpy(), dtype=np.float32)
        if action.shape != (1, 2) or not np.all(np.isfinite(action)):
            raise RuntimeError("BC policy must return a finite [1,2] physical action")
        return action[0]

    @staticmethod
    def _nearest_clearance(lidar: np.ndarray) -> float:
        scan = np.asarray(lidar, dtype=np.float32)
        if scan.shape != (CANONICAL_LIDAR_RAYS,) or not np.all(np.isfinite(scan)):
            raise ValueError("LiDAR clearance scan must be a finite canonical scan")
        return float(np.min(scan))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        transition = self.backend.reset()
        transition.validate()
        # Episode start mirrors the extraction boundary semantics: no previous
        # command exists yet, so aux[7:9] is zero in the first frame.
        self._last_command = np.zeros(2, dtype=np.float32)
        self._lidar_history = []
        self._aux_history = []
        self._append(self._with_action_history(transition))
        # The checkpoint was trained only on moving MPC trajectories (minimum
        # measured speed 4.27 m/s). Keep one command owner and bring the kart
        # into that regime with straight target-speed control before BC/PPO
        # inference. The existing backend feedback loop supplies the bounded
        # +3 m/s² acceleration from rest until the 5.5 m/s handoff for this
        # 7.5 m/s target.
        warmup_command = np.asarray((0.0, self.warmup_config.target_speed_mps), dtype=np.float32)
        started_at = time.monotonic() if transition.timestamp_s is None else transition.timestamp_s
        warmup_steps = 0
        handoff_reason = "maximum_duration"
        while True:
            self._last_command = warmup_command.copy()
            transition = self.backend.step(warmup_command)
            transition.validate()
            self._append(self._with_action_history(transition))
            warmup_steps += 1
            # Production ROS transitions carry monotonic timestamps. The
            # deterministic fallback preserves real 20 Hz semantics for a
            # backend that cannot provide one, rather than busy-looping.
            elapsed = (
                warmup_steps / self.action_config.control_rate_hz
                if transition.timestamp_s is None
                else max(0.0, transition.timestamp_s - started_at)
            )
            speed_ready = transition.speed_mps >= self.warmup_config.handoff_speed_mps
            history_ready = len(self._lidar_history) >= HISTORY_LENGTH
            if history_ready and elapsed >= self.warmup_config.minimum_duration_s and speed_ready:
                handoff_reason = "speed_ready"
                break
            if history_ready and elapsed >= self.warmup_config.maximum_duration_s:
                break
        self._previous_residual.fill(0.0)
        self._step_count = 0
        self.crash_checker.reset()
        self.crash_checker.update(
            speed_mps=transition.speed_mps,
            timestamp_s=time.monotonic() if transition.timestamp_s is None else transition.timestamp_s,
        )
        return self._observation(), {
            "speed_mps": transition.speed_mps,
            "warmup_elapsed_s": elapsed,
            "warmup_handoff_reason": handoff_reason,
        }

    def step(self, action: np.ndarray):
        residual = np.asarray(action, dtype=np.float32)
        if residual.shape != (2,) or not np.all(np.isfinite(residual)):
            raise ValueError("residual action must be finite shape [2]")
        observation = self._observation()
        base = self._base_action(observation)
        desired_command = combine_residual_action(
            base_action=base, residual_action=residual, config=self.action_config
        )
        # Apply the same final-output slew limits as standalone BC. This is
        # deliberately after residual combination: PPO must not bypass the
        # safety seam or turn the reset command immediately to full magnitude.
        command = rate_limit_command(
            previous_command=self._last_command,
            desired_command=desired_command,
            config=self.action_config,
        )
        transition = self.backend.step(command)
        transition.validate()
        crash_reason = self.crash_checker.update(
            speed_mps=transition.speed_mps,
            timestamp_s=time.monotonic() if transition.timestamp_s is None else transition.timestamp_s,
        )
        crash = crash_reason is not None
        # Record the command actually issued so the next observation window's
        # aux[7:9] carries it, matching how the training data was extracted.
        self._last_command = np.asarray(command, dtype=np.float32).copy()
        self._append(self._with_action_history(transition))
        reward = racing_reward(
            progress_m=transition.progress_m,
            speed_mps=transition.speed_mps,
            collision=bool(transition.collision or crash),
            off_track=transition.off_track,
            nearest_clearance_m=self._nearest_clearance(transition.lidar),
            command=command,
            measured_speed_mps=transition.speed_mps,
            residual_action=np.clip(residual, -1.0, 1.0),
            previous_residual=self._previous_residual,
            config=self.reward_config,
        )
        self._previous_residual = np.clip(residual, -1.0, 1.0).copy()
        self._step_count += 1
        terminated = bool(transition.collision or transition.off_track or crash)
        truncated = bool(self._step_count >= self.max_steps)
        info = {
            "base_action": base.copy(),
            "command": command.copy(),
            "speed_mps": transition.speed_mps,
            "progress_m": transition.progress_m,
            "collision": bool(transition.collision or crash),
            "off_track": transition.off_track,
            "termination_reason": (
                "collision" if transition.collision else "off_track" if transition.off_track else crash_reason
            ),
        }
        publish_debug = getattr(self.backend, "publish_debug", None)
        if callable(publish_debug):
            publish_debug(
                base_action=base.copy(),
                residual_action=np.clip(residual, -1.0, 1.0).copy(),
                command=command.copy(),
                step_progress_m=transition.progress_m,
                speed_mps=transition.speed_mps,
                termination_reason=info["termination_reason"],
            )
        return self._observation(), reward, terminated, truncated, info

    def close(self) -> None:
        self.backend.close()
