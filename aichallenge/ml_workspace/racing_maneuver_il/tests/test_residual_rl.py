import numpy as np
import pytest
import torch

from racing_maneuver_il.model import ExportPolicy, TemporalPolicy
from run_residual_ppo import run_deterministic_inference
from racing_maneuver_il.normalization import Normalizer
from racing_maneuver_il.residual_rl import (
    CrashChecker,
    CrashCheckerConfig,
    RacingRewardConfig,
    ResidualActionConfig,
    ResidualRacingEnv,
    ResidualTransition,
    WarmupConfig,
    combine_residual_action,
    racing_reward,
)


class FakeBackend:
    def __init__(self):
        self.commands = []
        self.debug = []
        self._frame = ResidualTransition(
            lidar=np.full(360, 10.0, dtype=np.float32),
            aux=np.zeros(9, dtype=np.float32),
            progress_m=0.0,
            speed_mps=4.0,
            collision=False,
            off_track=False,
        )

    def reset(self):
        return self._frame

    def observe(self):
        return self._frame

    def step(self, command):
        self.commands.append(command.copy())
        return ResidualTransition(
            lidar=np.full(360, 8.0, dtype=np.float32),
            aux=np.zeros(9, dtype=np.float32),
            progress_m=0.25,
            speed_mps=5.0,
            collision=False,
            off_track=False,
        )
    def publish_debug(self, **values):
        self.debug.append(values)

    def close(self):
        pass


def export_policy():
    model = TemporalPolicy(lidar_embedding=8, aux_embedding=4, hidden_size=8)
    normalizer = Normalizer(
        lidar_mean=np.zeros(360, dtype=np.float32),
        lidar_std=np.ones(360, dtype=np.float32),
        aux_mean=np.zeros(9, dtype=np.float32),
        aux_std=np.ones(9, dtype=np.float32),
    )
    return ExportPolicy.from_normalizer(model, normalizer, (-1.0, 0.0), (1.0, 15.0)).eval()


def test_residual_action_is_bounded_around_bc_command():
    command = combine_residual_action(
        base_action=np.array([0.95, 14.8], dtype=np.float32),
        residual_action=np.array([1.0, 1.0], dtype=np.float32),
        config=ResidualActionConfig(max_steering_delta_rad=0.1, max_target_speed_delta_mps=0.75),
    )
    np.testing.assert_allclose(command, [1.0, 15.0])


def test_racing_reward_prioritizes_progress_speed_and_collision_avoidance_without_ref_speed():
    config = RacingRewardConfig()
    safe_braking = racing_reward(
        progress_m=0.25,
        speed_mps=8.0,
        collision=False,
        off_track=False,
        nearest_clearance_m=1.0,
        command=np.array([0.15, 3.0], dtype=np.float32),
        measured_speed_mps=8.0,
        residual_action=np.array([0.0, -1.0], dtype=np.float32),
        previous_residual=np.zeros(2, dtype=np.float32),
        config=config,
    )
    collision = racing_reward(
        progress_m=0.25,
        speed_mps=8.0,
        collision=True,
        off_track=False,
        nearest_clearance_m=1.0,
        command=np.array([0.15, 3.0], dtype=np.float32),
        measured_speed_mps=8.0,
        residual_action=np.array([0.0, -1.0], dtype=np.float32),
        previous_residual=np.zeros(2, dtype=np.float32),
        config=config,
    )
    assert safe_braking > 0.0
    assert collision < -50.0


def test_racing_reward_penalizes_proximity_even_without_a_collision():
    config = RacingRewardConfig()
    common = {
        "progress_m": 0.0,
        "speed_mps": 0.0,
        "collision": False,
        "off_track": False,
        "command": np.array([0.0, 0.0], dtype=np.float32),
        "measured_speed_mps": 0.0,
        "residual_action": np.zeros(2, dtype=np.float32),
        "previous_residual": np.zeros(2, dtype=np.float32),
        "config": config,
    }

    clear = racing_reward(nearest_clearance_m=10.0, **common)
    dangerously_close = racing_reward(nearest_clearance_m=0.1, **common)

    assert dangerously_close < clear


class FreshHistoryBackend:
    """Backend that exposes a distinct coherent sensor frame per observation."""

    def __init__(self):
        self.observe_calls = 0
        self._sequence = 0
        self._reset_count = 0

    def _transition(self):
        value = float(self._sequence)
        return ResidualTransition(
            lidar=np.full(360, value, dtype=np.float32),
            aux=np.array([value, 0.0, 0.0, value, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            progress_m=0.0,
            speed_mps=value,
            collision=False,
            off_track=False,
        )

    def reset(self):
        self._sequence = 100 * self._reset_count
        self._reset_count += 1
        return self._transition()

    def observe(self):
        self.observe_calls += 1
        self._sequence += 1
        return self._transition()

    def step(self, command):
        self._sequence += 1
        return self._transition()

    def close(self):
        pass


class WarmupBackend:
    """Deterministic backend that accelerates only when reset warm-up commands it."""

    def __init__(self):
        self.commands = []
        self._step = 0

    def _transition(self, speed):
        return ResidualTransition(
            lidar=np.full(360, float(self._step), dtype=np.float32),
            aux=np.array([speed, 0.0, 0.0, float(self._step), 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            progress_m=0.0,
            speed_mps=speed,
            collision=False,
            off_track=False,
            timestamp_s=self._step * 0.05,
        )

    def reset(self):
        self._step = 0
        return self._transition(0.0)

    def observe(self):
        self._step += 1
        return self._transition(0.0)

    def step(self, command):
        self.commands.append(np.asarray(command, dtype=np.float32).copy())
        self._step += 1
        return self._transition(min(6.0, 0.6 * self._step))

    def close(self):
        pass


def test_reset_warms_up_with_zero_steering_before_bc_rl_handoff():
    backend = WarmupBackend()
    env = ResidualRacingEnv(
        backend=backend,
        bc_policy=export_policy(),
        warmup_config=WarmupConfig(
            minimum_duration_s=0.10,
            handoff_speed_mps=0.50,
            maximum_duration_s=1.0,
            target_speed_mps=6.0,
        ),
    )

    observation, info = env.reset()

    assert len(backend.commands) == 9  # full fresh ten-frame history, including reset frame
    np.testing.assert_allclose(backend.commands, np.tile([0.0, 6.0], (9, 1)))
    np.testing.assert_allclose(observation["lidar"][:, 0], np.arange(10, dtype=np.float32))
    np.testing.assert_allclose(observation["aux"][-1, 7:9], [0.0, 6.0])
    assert info["warmup_elapsed_s"] == pytest.approx(0.45)
    assert info["warmup_handoff_reason"] == "speed_ready"


def test_environment_reset_warms_bc_with_ten_fresh_sensor_frames():
    backend = FreshHistoryBackend()
    env = ResidualRacingEnv(backend=backend, bc_policy=export_policy())

    observation, _ = env.reset()

    assert backend.observe_calls == 0  # warm-up publishes one straight command per fresh frame
    np.testing.assert_allclose(observation["lidar"][:, 0], np.arange(21, 31, dtype=np.float32))
    np.testing.assert_allclose(observation["aux"][:, 0], np.arange(21, 31, dtype=np.float32))


def test_environment_reset_discards_all_previous_episode_bc_input_history():
    backend = FreshHistoryBackend()
    env = ResidualRacingEnv(backend=backend, bc_policy=export_policy())

    first, _ = env.reset()
    env.step(np.zeros(2, dtype=np.float32))  # add an episode-1 command/history frame
    second, _ = env.reset()

    np.testing.assert_allclose(first["lidar"][:, 0], np.arange(21, 31, dtype=np.float32))
    np.testing.assert_allclose(second["lidar"][:, 0], np.arange(121, 131, dtype=np.float32))
    np.testing.assert_allclose(second["aux"][:, 7], np.zeros(10, dtype=np.float32))
    np.testing.assert_allclose(second["aux"][:, 8], np.full(10, 7.5, dtype=np.float32))


class FixedActionBc(torch.nn.Module):
    """Physical-action BC stub for rate-limit behavior checks."""

    def forward(self, lidar, aux):
        return torch.tensor([[0.50, 10.0]], dtype=torch.float32), None


def test_environment_rate_limits_first_post_reset_combined_command():
    backend = FakeBackend()
    env = ResidualRacingEnv(
        backend=backend,
        bc_policy=FixedActionBc().eval(),
        action_config=ResidualActionConfig(
            max_steering_delta_rad=0.10,
            max_target_speed_delta_mps=0.75,
            steering_rate_limit_rad_s=1.5,
            target_speed_rate_limit_mps2=4.0,
            control_rate_hz=20.0,
        ),
    )
    env.reset()

    env.step(np.zeros(2, dtype=np.float32))

    # Steering ramps from zero; speed ramps from the warm-up's 7.5 m/s target.
    np.testing.assert_allclose(backend.commands[-1], [0.075, 7.7], atol=1e-6)


def test_environment_applies_residual_to_bc_command_and_returns_temporal_observation():
    backend = FakeBackend()
    env = ResidualRacingEnv(backend=backend, bc_policy=export_policy())
    observation, _ = env.reset()
    assert observation["lidar"].shape == (10, 360)
    assert observation["aux"].shape == (10, 9)

    _, _, terminated, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert not terminated
    assert backend.commands[-1].shape == (2,)
    assert 0.0 <= backend.commands[-1][1] <= 15.0
    assert "base_action" in info and "command" in info
    assert len(backend.debug) == 1
    assert set(backend.debug[-1]) == {
        "base_action",
        "residual_action",
        "command",
        "step_progress_m",
        "speed_mps",
        "termination_reason",
    }


def test_nearest_lidar_clearance_detects_a_close_object_outside_the_front_sector():
    scan = np.full(360, 10.0, dtype=np.float32)
    scan[5] = 0.40  # Side/rear-side object; not in the previous front-only sector.

    assert ResidualRacingEnv._nearest_clearance(scan) == pytest.approx(0.40)


def test_crash_checker_terminates_on_large_velocity_drop_within_one_control_period():
    checker = CrashChecker(
        CrashCheckerConfig(
            velocity_drop_threshold_mps=2.0,
            maximum_drop_interval_s=0.25,
            minimum_speed_before_drop_mps=1.0,
        )
    )
    assert checker.update(speed_mps=6.0, timestamp_s=10.0) is None
    assert checker.update(speed_mps=3.5, timestamp_s=10.1) == "velocity_drop"


def test_crash_checker_terminates_only_after_ten_seconds_below_point_one_mps():
    checker = CrashChecker(CrashCheckerConfig(low_speed_duration_s=10.0))
    assert checker.update(speed_mps=0.05, timestamp_s=0.0) is None
    assert checker.update(speed_mps=0.05, timestamp_s=9.99) is None
    assert checker.update(speed_mps=0.05, timestamp_s=10.0) == "prolonged_low_speed"


class InferencePolicy:
    def __init__(self):
        self.deterministic_flags = []

    def predict(self, observation, deterministic):
        self.deterministic_flags.append(deterministic)
        return np.array([0.25, -0.5], dtype=np.float32), None


class InferenceEnv:
    def __init__(self):
        self.reset_count = 0
        self.actions = []
        self._episode_steps = 0

    def reset(self):
        self.reset_count += 1
        self._episode_steps = 0
        return {"frame": self.reset_count}, {}

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self._episode_steps += 1
        terminated = self._episode_steps == 2
        return {"frame": len(self.actions)}, 1.0, terminated, False, {}


def test_deterministic_inference_runs_requested_steps_and_resets_after_termination():
    env = InferenceEnv()
    policy = InferencePolicy()

    steps, episodes = run_deterministic_inference(env, policy, max_steps=3)

    assert (steps, episodes) == (3, 1)
    assert env.reset_count == 2
    assert policy.deterministic_flags == [True, True, True]
    np.testing.assert_allclose(env.actions, [[0.25, -0.5]] * 3)


class CommandTrackingBackend:
    """Backend that records the commanded action it was given each step."""

    def __init__(self):
        self.commands = []
        self._frame = ResidualTransition(
            lidar=np.full(360, 10.0, dtype=np.float32),
            aux=np.array([4.0, 0.1, 0.5, 100.0, 50.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            progress_m=0.1,
            speed_mps=4.0,
            collision=False,
            off_track=False,
        )

    def reset(self):
        return self._frame

    def observe(self):
        return self._frame

    def step(self, command):
        self.commands.append(np.asarray(command, dtype=np.float32).copy())
        return self._frame

    def close(self):
        pass


def test_environment_feeds_previous_command_into_action_history_aux_features():
    """The BC policy was trained with aux[7:9] = previous commanded action.

    Feeding zeros there (the historical bug) shifts the frozen BC output by
    ~5 deg steering and ~4 m/s target speed, so each observation's newest frame
    must carry the command actually issued on the previous step.
    """
    backend = CommandTrackingBackend()
    env = ResidualRacingEnv(backend=backend, bc_policy=export_policy())

    # After straight warm-up, BC sees the actual last warm-up target, not a
    # zero prior-speed command from the stationary reset state.
    env.reset()
    np.testing.assert_allclose(env._last_command, [0.0, 7.5])

    # The command issued on a step must appear in the NEXT observation's newest
    # frame (training data carries target[i-1] in aux[7:9] of frame i).
    obs1, _, _, _, _ = env.step(np.zeros(2, dtype=np.float32))
    cmd1 = backend.commands[-1].copy()
    np.testing.assert_allclose(obs1["aux"][-1, 7], float(cmd1[0]), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(obs1["aux"][-1, 8], float(cmd1[1]), rtol=0.0, atol=1e-6)

    obs2, _, _, _, _ = env.step(np.zeros(2, dtype=np.float32))
    cmd2 = backend.commands[-1].copy()
    np.testing.assert_allclose(obs2["aux"][-1, 7], float(cmd2[0]), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(obs2["aux"][-1, 8], float(cmd2[1]), rtol=0.0, atol=1e-6)
