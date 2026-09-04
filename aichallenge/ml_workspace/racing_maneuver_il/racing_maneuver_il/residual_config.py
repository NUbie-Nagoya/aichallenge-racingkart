"""Versioned YAML configuration for residual-PPO training."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULTS: dict[str, dict[str, Any]] = {
    "training": {
        "timesteps": 200_000,
        "seed": 0,
        "rollout_steps": 2_048,
        "batch_size": 64,
        "device": "auto",
        # Per-episode control-step time limit (truncation). At the default
        # 20 Hz backend rate this is ~120 s of sim time.
        "max_steps": 2_400,
    },
    "residual_action": {
        "max_steering_delta_rad": 0.10,
        "max_target_speed_delta_mps": 0.75,
        # Mirror racing_maneuver_controller's final safety seam.
        "steering_rate_limit_rad_s": 1.5,
        "target_speed_rate_limit_mps2": 4.0,
        "control_rate_hz": 20.0,
    },
    "warmup": {
        "minimum_duration_s": 1.5,
        "handoff_speed_mps": 4.5,
        "maximum_duration_s": 12.0,
        # 7.5 with kp=1.5 gives the bounded +3 m/s² acceleration until 4.5 m/s.
        "target_speed_mps": 7.5,
    },
    "ppo": {
        "policy": "MultiInputPolicy",
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.20,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "n_epochs": 10,
        "target_kl": None,
    },
    "reward": {
        "progress_weight": 4.0,
        "speed_weight": 0.08,
        "collision_penalty": 100.0,
        "off_track_penalty": 50.0,
        "danger_clearance_m": 2.5,
        "proximity_penalty_weight": 0.5,
        "hazard_braking_weight": 1.5,
        "hazard_steering_weight": 0.20,
        "residual_change_weight": 0.03,
    },
    "crash": {
        "velocity_drop_threshold_mps": 2.0,
        "maximum_drop_interval_s": 0.25,
        "minimum_speed_before_drop_mps": 1.0,
        "low_speed_threshold_mps": 0.1,
        "low_speed_duration_s": 10.0,
    },
    "backend": {
        "control_topic": "/control/command/control_cmd",
        "reset_topic": "/awsim/reset",
        "debug_topic": "/racing_maneuver/residual_rl/debug",
        "control_rate_hz": 20.0,
        "speed_control_kp": 1.5,
        "acceleration_limit_mps2": 3.0,
        "step_timeout_s": 1.0,
        "initial_pose_service": "/set_initial_pose",
        "initial_pose_timeout_s": 2.0,
        "control_mode_request_topic": "/awsim/control_mode_request_topic",
    },
}

_REQUIRED_TOP_LEVEL = frozenset({"version", "bc_checkpoint", "backend_factory", "output_root"})
_ALLOWED_TOP_LEVEL = _REQUIRED_TOP_LEVEL | frozenset(DEFAULTS)


def _merge_section(name: str, supplied: Any) -> dict[str, Any]:
    if supplied is None:
        supplied = {}
    if not isinstance(supplied, dict):
        raise ValueError(f"residual-PPO config {name} must be a mapping")
    unknown = set(supplied) - set(DEFAULTS[name])
    if unknown:
        raise ValueError(f"unknown residual-PPO {name} keys: {sorted(unknown)}")
    result = deepcopy(DEFAULTS[name])
    result.update(supplied)
    return result


def load_residual_training_config(path: str | Path) -> dict[str, Any]:
    """Load, validate, and resolve a version-1 residual-PPO YAML config."""
    source = Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("residual-PPO config must be a mapping")
    unknown = set(raw) - _ALLOWED_TOP_LEVEL
    if unknown:
        raise ValueError(f"unknown residual-PPO config keys: {sorted(unknown)}")
    missing = _REQUIRED_TOP_LEVEL - set(raw)
    if missing:
        raise ValueError(f"missing residual-PPO config keys: {sorted(missing)}")
    if raw["version"] != 1:
        raise ValueError("only residual-PPO config version 1 is supported")
    for key in ("bc_checkpoint", "backend_factory", "output_root"):
        if not isinstance(raw[key], str) or not raw[key]:
            raise ValueError(f"residual-PPO config {key} must be a nonempty string")

    resolved: dict[str, Any] = {key: raw[key] for key in _REQUIRED_TOP_LEVEL}
    for name in DEFAULTS:
        resolved[name] = _merge_section(name, raw.get(name))

    training = resolved["training"]
    if int(training["timesteps"]) < 1:
        raise ValueError("residual-PPO training.timesteps must be positive")
    if int(training["rollout_steps"]) < 2:
        raise ValueError("residual-PPO training.rollout_steps must be at least 2")
    if not 1 <= int(training["batch_size"]) <= int(training["rollout_steps"]):
        raise ValueError("residual-PPO training.batch_size must be in [1, rollout_steps]")
    if str(training["device"]) not in {"auto", "cpu", "cuda"}:
        raise ValueError("residual-PPO training.device must be auto, cpu, or cuda")
    if int(training["max_steps"]) < 1:
        raise ValueError("residual-PPO training.max_steps must be positive")
    ppo = resolved["ppo"]
    if ppo["policy"] != "MultiInputPolicy":
        raise ValueError("residual-PPO ppo.policy must be MultiInputPolicy")
    positive = ("learning_rate", "gamma", "gae_lambda", "clip_range", "vf_coef", "max_grad_norm")
    if any(float(ppo[key]) <= 0.0 for key in positive):
        raise ValueError("residual-PPO PPO positive hyperparameters must be positive")
    if not 0.0 <= float(ppo["ent_coef"]):
        raise ValueError("residual-PPO ppo.ent_coef must be nonnegative")
    if int(ppo["n_epochs"]) < 1:
        raise ValueError("residual-PPO ppo.n_epochs must be positive")
    if ppo["target_kl"] is not None and float(ppo["target_kl"]) <= 0.0:
        raise ValueError("residual-PPO ppo.target_kl must be positive or null")
    return resolved
