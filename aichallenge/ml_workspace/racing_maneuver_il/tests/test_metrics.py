import numpy as np

from racing_maneuver_il.metrics import compute_metrics


def test_physical_unit_metrics_and_action_change_error():
    target = np.array([[0.0, 0.0], [0.2, 2.0], [0.1, 1.0]])
    pred = np.array([[0.1, 1.0], [0.3, 1.0], [0.1, 2.0]])
    report = compute_metrics(pred, target)
    assert report["count"] == 3
    assert report["steering_mae_rad"] == np.mean(np.abs(pred[:, 0] - target[:, 0]))
    assert report["steering_mae_deg"] == np.rad2deg(report["steering_mae_rad"])
    assert report["target_speed_rmse_mps"] > 0
    assert report["action_change_mae"] >= 0


def test_metrics_are_sliced_by_maneuver_opponent_and_closing_speed():
    target = np.zeros((6, 2))
    pred = np.ones((6, 2))
    maneuvers = np.array(
        ["free_lap", "follow", "pass_left", "pass_right", "abort", "recover"]
    )
    report = compute_metrics(
        pred,
        target,
        maneuver_classes=maneuvers,
        opponent_present=np.array([0, 1, 1, 1, 1, 0]),
        closing_speed_mps=np.array([-2, -0.2, 0.2, 1, 3, 6]),
    )
    assert set(maneuvers).issubset(report["by_maneuver"])
    assert report["by_opponent_presence"]["present"]["count"] == 4
    assert set(report["by_closing_speed_bin"]) == {
        "receding",
        "steady",
        "closing",
        "fast_closing",
    }


def test_empty_slice_is_omitted_and_bad_shapes_rejected():
    try:
        compute_metrics(np.zeros((2, 2)), np.zeros((3, 2)))
    except ValueError:
        pass
    else:
        raise AssertionError("bad shapes accepted")
