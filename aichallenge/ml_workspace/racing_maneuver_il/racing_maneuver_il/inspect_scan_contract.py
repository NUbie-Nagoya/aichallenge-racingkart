"""Offline helper for documenting a captured LaserScan contract."""

from __future__ import annotations

import argparse
import json

import numpy as np


def summarize_scan_contract(
    ranges,
    angle_min_rad: float,
    angle_increment_rad: float,
    range_min_m: float,
    range_max_m: float,
    *,
    rate_hz: float | None = None,
) -> dict:
    values = np.asarray(ranges, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or angle_increment_rad <= 0:
        raise ValueError("invalid scan")
    report = {
        "ray_count": int(values.size),
        "angle_min_rad": float(angle_min_rad),
        "angle_max_rad": float(angle_min_rad + (values.size - 1) * angle_increment_rad),
        "angle_increment_rad": float(angle_increment_rad),
        "range_min_m": float(range_min_m),
        "range_max_m": float(range_max_m),
        "finite_count": int(np.isfinite(values).sum()),
        "positive_infinity_count": int(np.isposinf(values).sum()),
        "negative_infinity_count": int(np.isneginf(values).sum()),
        "nan_count": int(np.isnan(values).sum()),
    }
    if rate_hz is not None:
        report["rate_hz"] = float(rate_hz)
    finite = values[np.isfinite(values)]
    if finite.size:
        report["finite_return_min_m"], report["finite_return_max_m"] = (
            float(finite.min()),
            float(finite.max()),
        )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fixture", help="NPZ containing ranges and scalar scan geometry"
    )
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    with np.load(args.fixture, allow_pickle=False) as data:
        kwargs = {
            name: float(data[name])
            for name in (
                "angle_min_rad",
                "angle_increment_rad",
                "range_min_m",
                "range_max_m",
            )
        }
        if "rate_hz" in data:
            kwargs["rate_hz"] = float(data["rate_hz"])
        report = summarize_scan_contract(data["ranges"], **kwargs)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        with open(args.output, "w") as stream:
            stream.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
