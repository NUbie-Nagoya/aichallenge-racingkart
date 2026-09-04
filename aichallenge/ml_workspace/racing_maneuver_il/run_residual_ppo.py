#!/usr/bin/env python3
"""Run a trained PPO residual around a frozen racing-maneuver BC policy.

This runner is for the dedicated AWSIM ``rl_train`` control route only.  It
requires RESIDUAL_RL_ALLOW_CONTROL=1 and no competing publisher on
/control/command/control_cmd.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Protocol

from racing_maneuver_il.residual_rl import ResidualActionConfig, ResidualRacingEnv
from train_residual_ppo import load_bc_policy, load_factory


class PredictPolicy(Protocol):
    def predict(self, observation: Any, deterministic: bool = True) -> tuple[Any, Any]: ...


class InferenceEnv(Protocol):
    def reset(self) -> tuple[Any, dict[str, Any]]: ...

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]: ...


def run_deterministic_inference(
    env: InferenceEnv,
    policy: PredictPolicy,
    *,
    max_steps: int,
    max_episodes: int = 0,
) -> tuple[int, int]:
    """Run deterministic PPO actions and reset after every terminal episode.

    A zero ``max_episodes`` means no episode-count limit. ``max_steps`` must be
    positive and is intentionally required so live commands have an explicit
    stopping bound.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if max_episodes < 0:
        raise ValueError("max_episodes must be nonnegative")

    observation, _ = env.reset()
    steps = 0
    episodes = 0
    while steps < max_steps:
        action, _ = policy.predict(observation, deterministic=True)
        observation, _, terminated, truncated, _ = env.step(action)
        steps += 1
        if terminated or truncated:
            episodes += 1
            if max_episodes and episodes >= max_episodes:
                break
            observation, _ = env.reset()
    return steps, episodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bc_checkpoint", type=Path)
    parser.add_argument("ppo_model", type=Path)
    parser.add_argument(
        "--backend-factory",
        default="racing_maneuver_il.residual_ros_backend:create_training_backend",
        help="MODULE:CALLABLE; must use the isolated rl_train command route.",
    )
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--max-steering-residual-rad", type=float, default=0.10)
    parser.add_argument("--max-speed-residual-mps", type=float, default=0.75)
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
    except ImportError as error:
        raise RuntimeError("stable-baselines3 is required for residual PPO inference") from error

    backend = load_factory(args.backend_factory)()
    env = ResidualRacingEnv(
        backend=backend,
        bc_policy=load_bc_policy(args.bc_checkpoint),
        action_config=ResidualActionConfig(
            max_steering_delta_rad=args.max_steering_residual_rad,
            max_target_speed_delta_mps=args.max_speed_residual_mps,
        ),
    )
    try:
        policy = PPO.load(args.ppo_model, device="auto")
        steps, episodes = run_deterministic_inference(
            env,
            policy,
            max_steps=args.max_steps,
            max_episodes=args.max_episodes,
        )
        print(f"Residual PPO inference completed: steps={steps} terminal_episodes={episodes}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
