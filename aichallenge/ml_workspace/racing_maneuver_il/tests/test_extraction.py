import pytest

from racing_maneuver_il.extraction import (
    PriorActionHistory,
    Stamped,
    align_causal,
    latest_at_or_before,
)


def test_uses_latest_observation_at_or_before_label_time():
    rows = [Stamped(1.0, "old"), Stamped(2.0, "latest"), Stamped(3.0, "future")]
    got, age = latest_at_or_before(rows, 2.5, max_age_s=1.0)
    assert got == "latest" and age == pytest.approx(0.5)


def test_does_not_use_future_observation():
    with pytest.raises(ValueError, match="no causal"):
        latest_at_or_before([Stamped(2.0, 1)], 1.0, max_age_s=2)


def test_rejects_stale_source():
    with pytest.raises(ValueError, match="stale"):
        latest_at_or_before([Stamped(1.0, 1)], 2.0, max_age_s=0.2)


def test_generic_alignment_rejects_stale_v2x():
    sources = {"scan": [Stamped(1.0, "s")], "v2x": [Stamped(0.5, "v")]}
    with pytest.raises(ValueError, match="v2x"):
        align_causal(Stamped(1.1, "final-control"), sources, {"scan": 0.2, "v2x": 0.2})


def test_alignment_preserves_source_timestamps_and_ages():
    aligned = align_causal(
        Stamped(1.1, "label"), {"scan": [Stamped(1.0, "s")]}, {"scan": 0.2}
    )
    assert aligned.values == {"scan": "s"}
    assert aligned.source_timestamps == {"scan": 1.0}
    assert aligned.ages_s["scan"] == pytest.approx(0.1)
    assert aligned.label == "label"


def test_episode_break_resets_previous_safe_action_history():
    h = PriorActionHistory()
    assert h.previous == (0.0, 0.0)
    h.update((0.2, 1.0))
    assert h.previous == (0.2, 1.0)
    h.reset()
    assert h.previous == (0.0, 0.0)


def test_prior_action_uses_safe_runtime_command_not_future_label():
    h = PriorActionHistory()
    h.update((0.1, 0.5))
    assert h.features_for_label((0.9, 9.0)) == (0.1, 0.5)
