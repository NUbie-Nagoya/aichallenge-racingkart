import numpy as np

from racing_maneuver_il.inspect_scan_contract import summarize_scan_contract


def test_scan_contract_summary_reports_geometry_and_invalid_counts():
    report = summarize_scan_contract(
        np.array([1.0, np.inf, np.nan, 4.0]), -0.2, 0.1, 0.1, 30.0, rate_hz=20.0
    )
    assert report["ray_count"] == 4 and report["finite_count"] == 2
    assert report["positive_infinity_count"] == 1 and report["nan_count"] == 1
    assert np.isclose(report["angle_max_rad"], 0.1) and report["rate_hz"] == 20.0
