import math
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum

from multi_purpose_mpc_ros.core.map import Map, Obstacle
from multi_purpose_mpc_ros.core.reference_path import ReferencePath, Waypoint
from multi_purpose_mpc_ros.core.frenet_converter import FrenetConverter
from multi_purpose_mpc_ros.core.quintic_polynomial import (
    lateral_quintic,
    longitudinal_quintic,
)
from multi_purpose_mpc_ros.core.reference_path import has_collision_in_line


@dataclass
class FrenetCfg:
    lane_margin: float = 1.0
    lateral_offsets: List[float] = None  # [1.2,1.5]
    T_overtake: List[float] = None  # [1.5,2.0,2.5,3.0]
    T_return: List[float] = None  # [1.5,2.0,2.5]
    T_brake: List[float] = None
    clearance: float = 1.5
    ttc_thr: float = 1.5
    gap0: float = 2.0
    gap_t: float = 0.8
    decel_lim: float = 12.0
    w_jerk: float = 1.0
    w_curv: float = 5.0
    w_clear: float = 2.0
    w_time: float = 0.5

    def __post_init__(self):
        if self.lateral_offsets is None:
            self.lateral_offsets = [1.2, 1.5]
        if self.T_overtake is None:
            self.T_overtake = [1.5, 2.0, 2.5, 3.0]
        if self.T_return is None:
            self.T_return = [1.5, 2.0, 2.5]
        if self.T_brake is None:
            self.T_brake = [1.5, 2.0, 2.5]


class State(Enum):
    IDLE = 0
    OVERTAKE = 1
    RETURN = 2
    BRAKING = 3


@dataclass
class LocalPath:
    xs: List[float]
    ys: List[float]
    psis: List[float]
    kappas: List[float]
    vs: List[float]

    def to_reference_path(self, map_obj: Map, resolution=0.6, circular=False):
        # build lightweight ReferencePath without recomputing width fully — reuse global widths if needed
        # simplest: construct via ReferencePath.__new__ and fill waypoints
        rp = ReferencePath.__new__(ReferencePath)
        rp.map = map_obj
        rp.resolution = resolution
        rp.circular = False
        rp.smoothing_distance = 0
        rp.eps = 1e-12
        rp.waypoints = [
            Waypoint(x, y, psi, k)
            for x, y, psi, k in zip(self.xs, self.ys, self.psis, self.kappas)
        ]
        for w, v in zip(rp.waypoints, self.vs):
            w.v_ref = v
        rp.n_waypoints = len(rp.waypoints)
        # straight length for MPC s tracking
        rp.length, rp.segment_lengths = rp._compute_length()
        # copy width from nearest global if available else compute
        rp._compute_width(max_width=6.0)  # ~2ms for 20 pts
        rp.path_constraints = None
        from multi_purpose_mpc_ros.core.reference_path import BorderCells

        rp.border_cells = BorderCells()
        rp.COUNT = 0
        return rp


class FrenetPlanner:
    def __init__(
        self, conv: FrenetConverter, map_obj: Map, mpc_cfg, frenet_cfg: FrenetCfg = None
    ):
        self.conv = conv
        self.map = map_obj
        self.mpc_cfg = mpc_cfg
        self.cfg = frenet_cfg or FrenetCfg()
        self.state = State.IDLE
        self.lead_id = None
        self.last_lead_s: Optional[float] = None

    def update(
        self, pose, v_ego: float, obstacles: List[Obstacle], car
    ) -> Optional[LocalPath]:
        s_ego, d_ego, _ = self.conv.cartesian_to_frenet(pose.x, pose.y, pose.theta)
        # find lead in corridor
        lead = None
        best_ds = float("inf")
        v_lead = 0.0
        for ob in obstacles:
            s_o, d_o, _ = self.conv.cartesian_to_frenet(ob.cx, ob.cy)
            if (
                s_o > s_ego
                and s_o - s_ego < 40
                and abs(d_o - d_ego) < self.cfg.lane_margin
            ):
                ds = s_o - s_ego
                if ds < best_ds:
                    best_ds = ds
                    lead = ob
                    lead_s = s_o
                    lead_d = d_o
        if lead is not None:
            self.last_lead_s = lead_s
            # estimate v_lead via tracker if available else 0
            ttc = (lead_s - s_ego) / max(v_ego - 2.0, 0.5)  # fallback 2m/s if unknown
            if self.state == State.IDLE and (ttc < self.cfg.ttc_thr):
                cand = self.plan_overtake(s_ego, d_ego, v_ego, lead_s, lead_d)
                if cand:
                    self.state = State.OVERTAKE
                    return cand
                else:  # no corridor -> braking
                    self.state = State.BRAKING
                    return self.plan_braking(s_ego, d_ego, v_ego, lead_s, 2.0)
            if self.state == State.OVERTAKE and s_ego > lead_s + self.cfg.clearance:
                self.state = State.RETURN
                return self.plan_return(s_ego, d_ego, v_ego)
            if self.state == State.RETURN and abs(d_ego) < 0.3:
                self.state = State.IDLE
                return None
            if self.state == State.BRAKING:
                # if corridor opens, overtake
                cand = self.plan_overtake(s_ego, d_ego, v_ego, lead_s, lead_d)
                if cand:
                    self.state = State.OVERTAKE
                    return cand
                return self.plan_braking(s_ego, d_ego, v_ego, lead_s, 2.0)
        else:
            # no lead currently visible – allow OVERTAKE->RETURN via remembered lead
            if self.state == State.OVERTAKE and self.last_lead_s is not None and s_ego > self.last_lead_s + self.cfg.clearance:
                self.state = State.RETURN
                return self.plan_return(s_ego, d_ego, v_ego)
            if self.state == State.RETURN and abs(d_ego) < 0.3:
                self.state = State.IDLE
            elif self.state == State.BRAKING:
                self.state = State.IDLE
        return None

    def plan_overtake(self, s0, d0, v0, s_obs, d_obs) -> Optional[LocalPath]:
        best = None
        best_cost = float("inf")
        # lateral velocity estimate
        d_dot0 = 0.0
        for T in self.cfg.T_overtake:
            s_target = s0 + np.clip(v0, 2.0, 8.0) * T
            v_target = self.conv.get_raceline_v(s_target)
            for off in self.cfg.lateral_offsets:
                for sign in (-1, 1):
                    d_target = d_obs + sign * off
                    ub, lb = self.conv.get_corridor(s_target)
                    if not (lb + 0.4 < d_target < ub - 0.4):
                        continue
                    try:
                        lat = lateral_quintic(d0, d_dot0, d_target, T)
                        lon = longitudinal_quintic(s0, v0, s_target, v_target, T)
                    except:
                        continue
                    xs, ys, psis, kappas = [], [], [], []
                    ts = np.linspace(0, T, 20)
                    feasible = True
                    for t in ts:
                        s = lon.eval(t)
                        d = lat.eval(t)
                        x, y, psi, k = self.conv.frenet_to_cartesian(s, d)
                        xs.append(x)
                        ys.append(y)
                        psis.append(psi)
                        kappas.append(k)
                    # track boundary + obstacle
                    for i in range(len(xs) - 1):
                        if has_collision_in_line(
                            self.map, (xs[i], ys[i]), (xs[i + 1], ys[i + 1])
                        ):
                            feasible = False
                            break
                        # width check
                        s_c, _, _ = self.conv.cartesian_to_frenet(xs[i], ys[i])
                        ub, lb = self.conv.get_corridor(s_c)
                        _, d_c, _ = self.conv.cartesian_to_frenet(xs[i], ys[i])
                        if not (lb + 0.3 < d_c < ub - 0.3):
                            feasible = False
                            break
                    if not feasible:
                        continue
                    # clearance to lead
                    lead_x, lead_y, _, _ = self.conv.frenet_to_cartesian(s_obs, d_obs)
                    dists = [math.hypot(x - lead_x, y - lead_y) for x, y in zip(xs, ys)]
                    min_d = min(dists)
                    # costs frenet_framework.md:17
                    jerk = np.mean([lat.jerk(t) ** 2 + lon.jerk(t) ** 2 for t in ts])
                    curv = np.sum([abs(k) for k in kappas])
                    cost = (
                        self.cfg.w_jerk * jerk
                        + self.cfg.w_curv * curv
                        + self.cfg.w_clear * (1 / (min_d + 0.5))
                        + self.cfg.w_time * T
                    )
                    if cost < best_cost:
                        best_cost = cost
                        # velocity cap sqrt(ay/v) :25
                        ay = self.mpc_cfg.ay_max
                        vs = [
                            min(
                                self.conv.get_raceline_v(lon.eval(t)),
                                math.sqrt(ay / (abs(k) + 1e-9)),
                            )
                            for t, k in zip(ts, kappas)
                        ]
                        best = LocalPath(xs, ys, psis, kappas, vs)
        return best

    def plan_return(self, s0, d0, v0) -> Optional[LocalPath]:
        for T in self.cfg.T_return:
            s_target = s0 + np.clip(v0, 2, 8) * T
            v_target = self.conv.get_raceline_v(s_target)
            try:
                lat = lateral_quintic(d0, 0.0, 0.0, T)
                lon = longitudinal_quintic(s0, v0, s_target, v_target, T)
            except:
                continue
            ts = np.linspace(0, T, 20)
            xs, ys, psis, kappas = [], [], [], []
            for t in ts:
                x, y, psi, k = self.conv.frenet_to_cartesian(lon.eval(t), lat.eval(t))
                xs.append(x)
                ys.append(y)
                psis.append(psi)
                kappas.append(k)
            vs = [
                min(v_target, math.sqrt(self.mpc_cfg.ay_max / (abs(k) + 1e-9)))
                for k in kappas
            ]
            return LocalPath(xs, ys, psis, kappas, vs)
        return None

    def plan_braking(self, s0, d0, v0, s_obs, v_obs) -> Optional[LocalPath]:
        gap = self.cfg.gap0 + self.cfg.gap_t * v0  # :49
        s_target = s_obs - gap
        if s_target <= s0:
            s_target = s0 + 1.0
        v_target = min(v_obs, self.conv.get_raceline_v(s_target))  # :51
        for T in self.cfg.T_brake:
            a_req = 2 * ((s_target - s0) - v0 * T) / T**2
            if abs(a_req) > self.cfg.decel_lim:
                continue  # :58 fallback handled by caller
            try:
                lon = longitudinal_quintic(s0, v0, s_target, v_target, T)
            except:
                continue
            ts = np.linspace(0, T, 20)
            xs, ys, psis, kappas = [], [], [], []
            for t in ts:
                x, y, psi, k = self.conv.frenet_to_cartesian(lon.eval(t), d0)
                xs.append(x)
                ys.append(y)
                psis.append(psi)
                kappas.append(k)
            vs = [lon.eval_d(t) for t in ts]
            return LocalPath(xs, ys, psis, kappas, vs)
        # threshold braking
        return None
