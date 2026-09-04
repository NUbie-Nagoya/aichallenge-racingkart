from pathlib import Path

import pytest

from racing_maneuver_il.residual_ros_backend import all_sensor_sequences_advanced
from racing_maneuver_il.residual_config import load_residual_training_config
from train_residual_ppo import EpisodeStatistics, ppo_hyperparameters


def test_loads_versioned_yaml_and_applies_safe_defaults(tmp_path: Path):
    config_path = tmp_path / "residual-ppo-v1.yaml"
    config_path.write_text(
        """
version: 1
bc_checkpoint: runs/bc/best.pt
backend_factory: racing_maneuver_il.residual_ros_backend:create_training_backend
output_root: runs/residual-rl
training:
  timesteps: 20000
  seed: 7
  rollout_steps: 256
  batch_size: 64
residual_action:
  max_steering_delta_rad: 0.10
  max_target_speed_delta_mps: 0.75
"""
    )

    config = load_residual_training_config(config_path)

    assert config["version"] == 1
    assert config["training"]["timesteps"] == 20_000
    assert config["training"]["device"] == "auto"
    # Episode time limit is resolved from YAML with a safe default.
    assert config["training"]["max_steps"] == 2400
    assert config["reward"]["collision_penalty"] == 100.0
    assert config["crash"]["low_speed_duration_s"] == 10.0
    assert config["backend"]["control_topic"] == "/control/command/control_cmd"
    assert config["backend"]["initial_pose_service"] == "/set_initial_pose"
    assert config["backend"]["initial_pose_timeout_s"] == 2.0
    assert config["warmup"] == {
        "minimum_duration_s": 1.5,
        "handoff_speed_mps": 4.5,
        "maximum_duration_s": 12.0,
        "target_speed_mps": 7.5,
    }


def test_episode_statistics_reports_completed_and_cumulative_episodes_by_reason():
    stats = EpisodeStatistics()
    stats.record(
        dones=[False, True, True],
        infos=[{}, {"termination_reason": "velocity_drop"}, {"TimeLimit.truncated": True}],
    )

    assert stats.snapshot_and_clear_rollout() == {
        "episodes_completed": 2,
        "episodes_total": 2,
        "velocity_drop_total": 1,
        "time_limit_total": 1,
    }
    assert stats.snapshot_and_clear_rollout()["episodes_completed"] == 0


def test_ppo_hyperparameters_are_resolved_from_yaml():
    config = {
        "training": {"seed": 3, "device": "cuda"},
        "ppo": {
            "policy": "MultiInputPolicy",
            "learning_rate": 0.0001,
            "gamma": 0.98,
            "gae_lambda": 0.90,
            "clip_range": 0.15,
            "ent_coef": 0.02,
            "vf_coef": 0.4,
            "max_grad_norm": 0.7,
            "n_epochs": 5,
            "target_kl": 0.03,
        },
    }

    assert ppo_hyperparameters(config) == {
        "policy": "MultiInputPolicy",
        "seed": 3,
        "device": "cuda",
        "learning_rate": 0.0001,
        "gamma": 0.98,
        "gae_lambda": 0.90,
        "clip_range": 0.15,
        "ent_coef": 0.02,
        "vf_coef": 0.4,
        "max_grad_norm": 0.7,
        "n_epochs": 5,
        "target_kl": 0.03,
    }


def test_reset_requires_all_sensor_streams_to_advance_before_observation_is_accepted():
    baseline = (100, 200, 300, 400)

    assert not all_sensor_sequences_advanced((101, 200, 301, 401), baseline)
    assert not all_sensor_sequences_advanced((101, 201, 300, 401), baseline)
    assert all_sensor_sequences_advanced((101, 201, 301, 401), baseline)


def test_rejects_obsolete_sleep_based_backend_reset_setting(tmp_path: Path):
    config_path = tmp_path / "obsolete-reset-delay.yaml"
    config_path.write_text(
        """
version: 1
bc_checkpoint: model.pt
backend_factory: pkg:factory
output_root: runs/test
backend:
  reset_settle_s: 3.0
"""
    )

    with pytest.raises(ValueError, match="unknown residual-PPO backend keys"):
        load_residual_training_config(config_path)


def test_rejects_unknown_or_invalid_residual_training_config(tmp_path: Path):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        """
version: 1
bc_checkpoint: model.pt
backend_factory: pkg:factory
output_root: runs/test
training:
  timesteps: 0
  rollout_steps: 32
  batch_size: 64
residual_action:
  max_steering_delta_rad: 0.10
  max_target_speed_delta_mps: 0.75
unknown: true
"""
    )

    with pytest.raises(ValueError, match="unknown residual-PPO config keys"):
        load_residual_training_config(config_path)
