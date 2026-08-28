import numpy as np

from racing_maneuver_il.opponents import OpponentObservation, summarize_opponents


def test_nearest_forward_corridor_opponent_transforms_to_body_frame():
    obs = [
        OpponentObservation("far", 10, 10, 2.0),
        OpponentObservation("near", 5, 5, 2.0),
    ]
    s = summarize_opponents((0, 0), np.pi / 2, (0, 0), obs, label_time_s=2.1)
    assert s.present == 1.0
    np.testing.assert_allclose(
        [s.relative_x_body_m, s.relative_y_body_m], [5, -5], atol=1e-6
    )


def test_selects_only_forward_body_corridor():
    obs = [
        OpponentObservation("behind", -2, 0, 1),
        OpponentObservation("wide", 3, 9, 1),
    ]
    s = summarize_opponents(
        (0, 0), 0, (0, 0), obs, label_time_s=1.1, corridor_half_width_m=3
    )
    assert s.as_array().tolist() == [0.0] * 6


def test_velocity_uses_only_previous_v2x_sample():
    now = [OpponentObservation("a", 6, 0, 2)]
    previous = {"a": OpponentObservation("a", 4, 0, 1)}
    s = summarize_opponents((0, 0), 0, (1, 0), now, 2.1, previous=previous)
    assert s.relative_vx_body_mps == 1.0
    assert s.relative_vy_body_mps == 0.0


def test_unbounded_or_future_velocity_history_is_not_used():
    now = [OpponentObservation("a", 6, 0, 2)]
    future = {"a": OpponentObservation("a", 4, 0, 3)}
    s = summarize_opponents((0, 0), 0, (1, 0), now, 2.1, previous=future)
    assert s.relative_vx_body_mps == -1.0


def test_stale_observations_return_absent():
    s = summarize_opponents(
        (0, 0), 0, (0, 0), [OpponentObservation("a", 2, 0, 0)], 2, max_age_s=0.2
    )
    assert s.present == 0.0
