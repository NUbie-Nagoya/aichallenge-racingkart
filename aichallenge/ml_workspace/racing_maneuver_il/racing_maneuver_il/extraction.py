"""Generic causal synchronization primitives independent of ROS message types."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Stamped(Generic[T]):
    timestamp_s: float
    value: T


@dataclass(frozen=True)
class AlignedSample:
    label: Any
    label_timestamp_s: float
    values: dict[str, Any]
    source_timestamps: dict[str, float]
    ages_s: dict[str, float]


def _selected(rows: Sequence[Stamped[T]], label_time_s: float) -> Stamped[T]:
    ordered = sorted(rows, key=lambda row: row.timestamp_s)
    index = bisect_right([row.timestamp_s for row in ordered], label_time_s) - 1
    if index < 0:
        raise ValueError("no causal observation at or before label time")
    return ordered[index]


def latest_at_or_before(
    rows: Sequence[Stamped[T]], label_time_s: float, *, max_age_s: float
) -> tuple[T, float]:
    row = _selected(rows, label_time_s)
    age = label_time_s - row.timestamp_s
    if age > max_age_s:
        raise ValueError(f"stale source observation ({age:.6f}s > {max_age_s:.6f}s)")
    return row.value, age


def align_causal(
    label: Stamped[Any],
    sources: Mapping[str, Sequence[Stamped[Any]]],
    max_ages_s: Mapping[str, float],
) -> AlignedSample:
    values, timestamps, ages = {}, {}, {}
    for name, rows in sources.items():
        if name not in max_ages_s:
            raise ValueError(f"missing maximum age for {name}")
        try:
            row = _selected(rows, label.timestamp_s)
            value, age = latest_at_or_before(
                rows, label.timestamp_s, max_age_s=max_ages_s[name]
            )
        except ValueError as exc:
            raise ValueError(f"{name}: {exc}") from exc
        values[name], timestamps[name], ages[name] = value, row.timestamp_s, age
    return AlignedSample(label.value, label.timestamp_s, values, timestamps, ages)


class PriorActionHistory:
    """Tracks the prior independently safety-limited runtime command."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.previous = (0.0, 0.0)

    def update(self, safe_action: tuple[float, float]) -> None:
        self.previous = (float(safe_action[0]), float(safe_action[1]))

    def features_for_label(
        self, _future_label: tuple[float, float]
    ) -> tuple[float, float]:
        return self.previous
