import numpy as np
import pytest

from racing_maneuver_il.lidar import canonicalize_scan


def full_scan(n=1081):
    angles = np.linspace(-np.pi, np.pi, n)
    return angles, (angles + np.pi).astype(np.float32)


def test_canonicalizer_returns_360_float32_raw_meter_rays():
    angles, ranges = full_scan()
    out = canonicalize_scan(ranges, angles[0], angles[1] - angles[0], 0.1, 50.0)
    assert out.shape == (360,) and out.dtype == np.float32
    assert out.max() > 1.0


def test_canonicalizer_uses_physical_angles_not_array_position():
    angles, ranges = full_scan(721)
    out = canonicalize_scan(ranges, angles[0], angles[1] - angles[0], 0.1, 50.0)
    target = np.deg2rad(np.linspace(-89.5, 89.5, 360)) + np.pi
    np.testing.assert_allclose(out, target, atol=2e-3)


def test_invalid_returns_have_deterministic_policy():
    ranges = np.ones(721, dtype=np.float32)
    ranges[360] = np.inf
    out = canonicalize_scan(
        ranges, -np.pi, 2 * np.pi / 720, 0.1, 40.0, model_max_range_m=30.0
    )
    assert out[179] > 1.0 and out[180] > 1.0
    ranges[:] = np.nan
    out = canonicalize_scan(
        ranges, -np.pi, 2 * np.pi / 720, 0.1, 40.0, model_max_range_m=30.0
    )
    assert np.all(out == 30.0)


def test_clips_below_min_and_above_model_max():
    ranges = np.linspace(-1, 100, 721)
    out = canonicalize_scan(
        ranges, -np.pi, 2 * np.pi / 720, 0.2, 80.0, model_max_range_m=30.0
    )
    assert out.min() >= 0.2 and out.max() <= 30.0


def test_rejects_source_scan_with_insufficient_angular_coverage():
    with pytest.raises(ValueError, match="coverage"):
        canonicalize_scan(np.ones(100), -0.5, 0.01, 0.1, 30.0)


def test_rejects_bad_geometry():
    with pytest.raises(ValueError):
        canonicalize_scan([], 0, 0.1, 0.1, 30)
    with pytest.raises(ValueError):
        canonicalize_scan([1, 2], 0, 0, 0.1, 30)
