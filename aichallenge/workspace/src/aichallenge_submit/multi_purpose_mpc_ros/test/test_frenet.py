import math, numpy as np, pytest
from multi_purpose_mpc_ros.core.reference_path import (
    ReferencePath,
    has_collision_in_line,
)
import csv
from multi_purpose_mpc_ros.core.map import Map, Obstacle
from multi_purpose_mpc_ros.core.frenet_converter import FrenetConverter
from multi_purpose_mpc_ros.core.frenet_planner import FrenetCfg, FrenetPlanner, State
from multi_purpose_mpc_ros.core.quintic_polynomial import quintic_coeff, Quintic
try:
    from geometry_msgs.msg import Pose2D
except ModuleNotFoundError:  # host without ROS – minimal stub for unit test
    from dataclasses import dataclass

    @dataclass
    class Pose2D:
        x: float = 0.0
        y: float = 0.0
        theta: float = 0.0

from pathlib import Path

PKG_DIR = Path(__file__).resolve().parents[1]
YAML = str(PKG_DIR / "env/final_ver3/occupancy_grid_map.yaml")
CSV = str(PKG_DIR / "env/final_ver3/traj_mincurv.csv")


def _load_track_anchor():
    with open(CSV) as f:
        r = next(csv.DictReader(f))
        return float(r["x_m"]), float(r["y_m"]), float(r["psi_rad"])


OX, OY, PSI0 = _load_track_anchor()


def _to_track(xs, ys):
    # rotate local (xs along track s, ys lateral d) by PSI0 then translate
    import math

    ox, oy, psi0 = OX, OY, PSI0
    out_x, out_y = [], []
    for x, y in zip(xs, ys):
        rx = x * math.cos(psi0) - y * math.sin(psi0)
        ry = x * math.sin(psi0) + y * math.cos(psi0)
        out_x.append(ox + rx)
        out_y.append(oy + ry)
    return out_x, out_y


def make_straight_ref():
    # minimal map stub: skip occupancy, use real OGM if available
    map = Map(YAML)
    wp_x_raw = list(np.linspace(0, 100, 50))
    wp_y_raw = [0] * 50
    wp_x, wp_y = _to_track(wp_x_raw, wp_y_raw)
    ref = ReferencePath(
        map,
        wp_x,
        wp_y,
        resolution=0.6,
        smoothing_distance=2,
        max_width=6.0,
        circular=False,
    )
    ref.compute_speed_profile(
        {"a_min": -1.6, "a_max": 0.7, "v_min": 0, "v_max": 8, "ay_max": 12}
    )
    return ref


def test_roundtrip():
    ref = make_straight_ref()
    c = FrenetConverter(ref)
    for s, d in [(10, 1.0), (30, -1.2), (50, 0.5)]:
        x, y, _, _ = c.frenet_to_cartesian(s, d)
        s2, d2, _ = c.cartesian_to_frenet(x, y)
        assert abs(s - s2) < 0.05 and abs(d - d2) < 0.05


def test_quintic_bc():
    c = quintic_coeff(0, 0, 0, 1, 0, 0, 1.0)
    q = Quintic(c, 1.0)
    assert abs(q.eval(0) - 0) < 1e-9 and abs(q.eval(1) - 1) < 1e-9
    assert abs(q.eval_d(0)) < 1e-9 and abs(q.eval_d(1)) < 1e-9


def make_ref(wp_x=None, wp_y=None, circular=False):
    m = Map(YAML)
    if wp_x is None:
        # use real track corridor from final_ver3 CSV – guarantees free width
        with open(CSV) as f:
            rows = list(csv.DictReader(f))[:80]
            wp_x = [float(r["x_m"]) for r in rows]
            wp_y = [float(r["y_m"]) for r in rows]
        # keep linear segment (not full loop) to match previous test lengths
        circular = False
    ref = ReferencePath(
        m,
        wp_x,
        wp_y,
        resolution=0.6,
        smoothing_distance=2,
        max_width=6.0,
        circular=circular,
    )
    ref.compute_speed_profile(
        {"a_min": -1.6, "a_max": 0.7, "v_min": 0, "v_max": 8, "ay_max": 12}
    )
    return m, ref


def pose_at(s, d, conv):
    x, y, psi, _ = conv.frenet_to_cartesian(s, d)
    p = Pose2D()
    p.x = x
    p.y = y
    p.theta = psi
    return p


class DummyCfg:
    ay_max = 8.0


def test_converter_roundtrip_curve():
    # curve to catch psi slerp bug
    xs_raw = [0, 20, 40, 60]
    ys_raw = [0, 0, 10, 10]
    xs, ys = _to_track(xs_raw, ys_raw)
    m, ref = make_ref(xs, ys)
    c = FrenetConverter(ref)
    for s, d in [(5, 0.8), (25, -1.0), (45, 0.0)]:
        x, y, _, _ = c.frenet_to_cartesian(s, d)
        s2, d2, _ = c.cartesian_to_frenet(x, y)
        assert abs(s - s2) < 0.3, f"s {s}->{s2}"
        assert abs(d - d2) < 0.25


def test_planner_overtake_feasible():
    m, ref = make_ref()
    conv = FrenetConverter(ref)
    # obstacle 18m ahead on centerline -> must swerve ±1.2
    obs = Obstacle(*conv.frenet_to_cartesian(30, 0.0)[:2], radius=0.5)
    pose = pose_at(12, 0.0, conv)
    planner = FrenetPlanner(conv, m, DummyCfg(), FrenetCfg())
    cand = planner.plan_overtake(12, 0.0, 5.0, 30, 0.0)
    assert cand is not None, "overtake should find corridor"
    assert len(cand.xs) == 20
    assert (
        max(
            abs(d)
            for _, d, _ in [
                conv.cartesian_to_frenet(x, y) for x, y in zip(cand.xs, cand.ys)
            ]
        )
        > 0.8
    )
    for i in range(19):
        assert not has_collision_in_line(
            m, (cand.xs[i], cand.ys[i]), (cand.xs[i + 1], cand.ys[i + 1])
        )


def test_planner_braking_gap():
    m, ref = make_ref()
    conv = FrenetConverter(ref)
    planner = FrenetPlanner(conv, m, DummyCfg(), FrenetCfg(gap0=2.0, gap_t=0.8))
    # v0=5 -> gap 6.0, s_obs 25 -> s_target 19
    cand = planner.plan_braking(10, 0.0, 5.0, 25, 2.0)
    assert cand is not None
    s_last, _, _ = conv.cartesian_to_frenet(cand.xs[-1], cand.ys[-1])
    assert abs(s_last - 19) < 2.0
    assert cand.vs[-1] <= 2.5


def test_planner_return_to_zero():
    m, ref = make_ref()
    conv = FrenetConverter(ref)
    planner = FrenetPlanner(conv, m, DummyCfg())
    cand = planner.plan_return(20, 1.4, 5.0)
    assert cand is not None
    _, d_last, _ = conv.cartesian_to_frenet(cand.xs[-1], cand.ys[-1])
    assert abs(d_last) < 0.15


def test_state_overtake_then_return():
    m, ref = make_ref()
    conv = FrenetConverter(ref)
    obs = Obstacle(*conv.frenet_to_cartesian(25, 0.0)[:2], radius=0.5)
    planner = FrenetPlanner(
        conv, m, DummyCfg(), FrenetCfg(ttc_thr=10.0)
    )  # force trigger
    pose = pose_at(10, 0.0, conv)
    cand = planner.update(pose, 6.0, [obs], car=None)
    assert planner.state == State.OVERTAKE and cand is not None
    # simulate passing: ego now ahead
    pose2 = pose_at(27, cand.ys[10] and 1.2 or 1.2, conv)  # ~27m ahead, d~1.2
    # need fresh s for return check: directly call update
    cand2 = planner.update(pose2, 6.0, [obs], car=None)
    assert planner.state == State.RETURN
