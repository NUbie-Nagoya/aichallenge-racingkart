#!/usr/bin/env python3
"""Train bounded PPO residuals around a frozen racing-maneuver BC policy.

The backend factory isolates AWSIM/ROS ownership from the learning loop.  It must
return a ResidualRacingBackend and may only publish real controls in a dedicated
simulator training route with no competing command publisher.
"""

from __future__ import annotations

import argparse
import importlib
from collections import Counter
from pathlib import Path

import yaml

from racing_maneuver_il.checkpoint import load_checkpoint
from racing_maneuver_il.model import ExportPolicy
from racing_maneuver_il.normalization import Normalizer
from racing_maneuver_il.residual_config import load_residual_training_config
from racing_maneuver_il.residual_rl import (
    CrashCheckerConfig,
    RacingRewardConfig,
    ResidualActionConfig,
    ResidualRacingEnv,
    WarmupConfig,
)


def load_bc_policy(checkpoint: Path) -> ExportPolicy:
    model, payload = load_checkpoint(checkpoint)
    metadata = payload["metadata"]
    limits = metadata["action_limits"]
    return ExportPolicy.from_normalizer(
        model.eval(),
        Normalizer.from_dict(metadata["normalizer"]),
        tuple(limits["low"]),
        tuple(limits["high"]),
    ).eval()


def load_factory(spec: str):
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("backend factory must be MODULE:CALLABLE")
    return getattr(importlib.import_module(module_name), attribute)


class EpisodeStatistics:
    """Cumulative and per-rollout episode accounting for live PPO logs."""

    def __init__(self) -> None:
        self.episodes_total = 0
        self._episodes_since_snapshot = 0
        self._reasons: Counter[str] = Counter()

    def record(self, *, dones, infos) -> None:
        for done, info in zip(dones, infos):
            if not bool(done):
                continue
            self.episodes_total += 1
            self._episodes_since_snapshot += 1
            if bool(info.get("TimeLimit.truncated", False)):
                reason = "time_limit"
            else:
                reason = str(info.get("termination_reason") or "unspecified")
            self._reasons[reason] += 1

    def snapshot_and_clear_rollout(self) -> dict[str, int]:
        result = {
            "episodes_completed": self._episodes_since_snapshot,
            "episodes_total": self.episodes_total,
            **{f"{reason}_total": count for reason, count in sorted(self._reasons.items())},
        }
        self._episodes_since_snapshot = 0
        return result


class EpisodeMetricsCallback:
    """Adapter defined without SB3 inheritance so its accounting is unit-testable."""

    def __init__(self) -> None:
        self.statistics = EpisodeStatistics()

    def record_step(self, *, dones, infos) -> None:
        self.statistics.record(dones=dones, infos=infos)


def _create_run_directory(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    existing = [
        int(path.name.removeprefix("run-"))
        for path in output_root.iterdir()
        if path.is_dir() and path.name.removeprefix("run-").isdigit()
    ]
    run_number = max(existing, default=0) + 1
    destination = output_root / f"run-{run_number}"
    destination.mkdir()
    return destination


def ppo_hyperparameters(config: dict) -> dict:
    """Convert the validated YAML PPO/training sections to SB3 constructor args."""
    training = config["training"]
    ppo = config["ppo"]
    return {
        "policy": ppo["policy"],
        "seed": int(training["seed"]),
        "device": str(training["device"]),
        "learning_rate": float(ppo["learning_rate"]),
        "gamma": float(ppo["gamma"]),
        "gae_lambda": float(ppo["gae_lambda"]),
        "clip_range": float(ppo["clip_range"]),
        "ent_coef": float(ppo["ent_coef"]),
        "vf_coef": float(ppo["vf_coef"]),
        "max_grad_norm": float(ppo["max_grad_norm"]),
        "n_epochs": int(ppo["n_epochs"]),
        "target_kl": None if ppo["target_kl"] is None else float(ppo["target_kl"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="Versioned residual-PPO YAML config")
    args = parser.parse_args()
    config = load_residual_training_config(args.config)
    training = config["training"]

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as error:
        raise RuntimeError("stable-baselines3 is required for residual PPO training") from error

    class Sb3EpisodeMetricsCallback(BaseCallback):
        """Write explicit episode counts into SB3's console and TensorBoard logs."""

        def __init__(self) -> None:
            super().__init__()
            self.metrics = EpisodeMetricsCallback()

        def _on_step(self) -> bool:
            self.metrics.record_step(dones=self.locals["dones"], infos=self.locals["infos"])
            return True

        def _on_rollout_end(self) -> None:
            for name, value in self.metrics.statistics.snapshot_and_clear_rollout().items():
                self.logger.record(f"rollout/{name}", value)

    factory = load_factory(config["backend_factory"])
    backend = factory(config["backend"])
    action_config = ResidualActionConfig(
        **config["residual_action"],
    )
    output_dir = _create_run_directory(Path(config["output_root"]))
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
    )
    env = ResidualRacingEnv(
        backend=backend,
        bc_policy=load_bc_policy(Path(config["bc_checkpoint"])),
        action_config=action_config,
        warmup_config=WarmupConfig(**config["warmup"]),
        reward_config=RacingRewardConfig(**config["reward"]),
        crash_config=CrashCheckerConfig(**config["crash"]),
        max_steps=int(training["max_steps"]),
    )
    model = PPO(
        env=env,
        verbose=1,
        n_steps=int(training["rollout_steps"]),
        batch_size=int(training["batch_size"]),
        tensorboard_log=str(output_dir / "tensorboard"),
        **ppo_hyperparameters(config),
    )
    try:
        model.learn(
            total_timesteps=int(training["timesteps"]),
            callback=Sb3EpisodeMetricsCallback(),
        )
        model.save(str(output_dir / "residual_ppo"))
    finally:
        env.close()


if __name__ == "__main__":
    main()
