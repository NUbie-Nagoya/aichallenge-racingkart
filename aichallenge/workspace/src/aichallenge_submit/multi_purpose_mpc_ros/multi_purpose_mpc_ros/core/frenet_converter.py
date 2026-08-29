# This file is for converting between (x, y, psi) MPC <-> (s,d) frenet
import numpy as np
import math
from multi_purpose_mpc_ros.core.reference_path import ReferencePath


class FrenetConverter:
    def __init__(self, ref_path: ReferencePath):
        self.ref_path = ref_path

        # Compute cumulative path length
        self.length_cum = np.concatenate([[0], np.cumsum(ref_path.segment_lengths)])
        self.total_length = self.length_cum[-1]

        # get waypoint info
        self.wp_xy = np.array([(w.x, w.y) for w in ref_path.waypoints])
        self.wp_psi = np.array([w.psi for w in ref_path.waypoints])
        self.wp_kappa = np.array([w.kappa for w in ref_path.waypoints])
        self.wp_lb = np.array([w.lb for w in ref_path.waypoints])
        self.wp_ub = np.array([w.ub for w in ref_path.waypoints])

        self.wp_len = len(self.wp_xy)

        # from circular config (from ReferencePath)
        self.circular = ref_path.circular

        if self.wp_len >= 2:                                                                                          
                 self.seg_vec = self.wp_xy[1:] - self.wp_xy[:-1]  # (n-1,2)                                           
                 self.seg_len = np.linalg.norm(self.seg_vec, axis=1) + 1e-12                                          
        else:                                                                                                    
            self.seg_vec = np.zeros((0,2))                                                                       
            self.seg_len = np.zeros((0,))

    # x,y, yaw -> tuple(s,d, wp_id)
    def cartesian_to_frenet(self, x, y, yaw=None, window_size=3):
        if self.wp_len == 0:
            return 0.0, 0.0, 0
        closest_waypoint = self.get_closest_waypoint(x, y)

        wp_len = self.wp_len

        search_ids = []
        for i in range(-window_size, window_size + 1):
            target_idx = closest_waypoint + i
            if self.circular:
                # handle circular loop
                if target_idx < 0:
                    search_ids.append(wp_len + target_idx)
                elif target_idx >= wp_len:
                    search_ids.append(target_idx - wp_len)
                else:
                    search_ids.append(target_idx)
            else:
                if target_idx < 0 or target_idx >= wp_len - 1:
                    continue
                search_ids.append(target_idx)

        best_s, best_d, best_idx, best_err = 0.0, 0.0, closest_waypoint, float("inf")
        p = np.array([x, y], dtype=np.float64)
        for idx in search_ids:
            next_idx = (
                idx + 1
            ) % wp_len  # for non-circular: checked already that next_idx is < wp_len
            p0 = self.wp_xy[idx]
            seg = self.wp_xy[next_idx] - p0

            # project target xy on segment
            L2 = float(np.dot(seg, seg) + 1e-12)
            t = float(np.dot(p - p0, seg) / L2)

            # clamp t
            t = max(0.0, min(1.0, t))

            # longitudinal offset
            proj = p0 + t * seg

            # lateral offset -> cross(seg, p-p0)/|seg| (left is +)
            cross = seg[0] * (p[1] - p0[1]) - seg[1] * (p[0] - p0[0])
            d = cross / math.sqrt(L2)
            s = self.length_cum[idx] + t * math.sqrt(L2)
            err = np.sum((p - proj) ** 2)
            if err < best_err:
                best_err = err
                best_s = s
                best_d = d
                best_idx = idx

        if self.circular:
            best_s %= self.total_length
        return best_s, best_d, best_idx

    def frenet_to_cartesian(self, s, d):
        """-> (x,y,psi,kappa_eff)  frenet_framework.md:24  inverse of s2t:156"""
        if self.wp_len == 0:
            return 0.0, 0.0, 0.0, 0.0
        if self.circular:
            s = s % self.total_length
        else:
            s = max(0.0, min(s, self.total_length - 1e-9))

        idx = int(np.searchsorted(self.length_cum, s, side="right") - 1)
        idx = max(0, min(idx, self.wp_len - 2))
        t = (
            (s - self.length_cum[idx]) / self.seg_len[idx]
            if self.seg_len[idx] > 1e-9
            else 0.0
        )
        t = max(0.0, min(1.0, t))
        p0 = self.wp_xy[idx]
        p1 = self.wp_xy[(idx + 1) % self.wp_len] if self.circular else self.wp_xy[idx + 1]
        x0 = float(p0[0] + t * (p1[0] - p0[0]))
        y0 = float(p0[1] + t * (p1[1] - p0[1]))
        # yaw slerp
        psi0 = float(self.wp_psi[idx])
        psi1 = float(
            self.wp_psi[(idx + 1) % self.wp_len] if self.circular else self.wp_psi[idx + 1]
        )
        # wrap diff
        dpsi = math.atan2(math.sin(psi1 - psi0), math.cos(psi1 - psi0))
        psi = psi0 + t * dpsi
        psi = math.atan2(math.sin(psi), math.cos(psi))
        kappa = float(self.wp_kappa[idx])
        # offset curvature
        kappa_eff = kappa / (1.0 - d * kappa + 1e-9)
        x = x0 - d * math.sin(psi)  # s2t:166
        y = y0 + d * math.cos(psi)  # s2t:168
        return x, y, psi, kappa_eff

    def get_corridor(self, s):
        s = (
            s % self.total_length
            if self.circular
            else max(0, min(s, self.total_length))
        )
        idx = int(np.searchsorted(self.length_cum, s, side="right") - 1)
        idx = max(0, min(idx, self.wp_len - 1))
        return float(self.wp_ub[idx]), float(self.wp_lb[idx])

    def get_raceline_v(self, s):
        s = (
            s % self.total_length
            if self.circular
            else max(0, min(s, self.total_length))
        )
        idx = int(np.searchsorted(self.length_cum, s, side="right") - 1)
        idx = max(0, min(idx, self.wp_len - 1))
        v = self.ref_path.waypoints[idx].v_ref
        return float(v) if v is not None else 0.0

    def batch_cartesian_to_frenet(self, xs, ys):
        return [self.cartesian_to_frenet(float(x), float(y)) for x, y in zip(xs, ys)]

    def get_closest_waypoint(self, x, y):
        # Compute distances from the point to all waypoints
        distances = np.sqrt(
            (np.array([wp.x for wp in self.ref_path.waypoints]) - x) ** 2
            + (np.array([wp.y for wp in self.ref_path.waypoints]) - y) ** 2
        )

        # Get the index of the closest waypoint
        closest_wp_id = np.argmin(distances)
        return closest_wp_id


if __name__ == "__main__":
    frenet_obj = FrenetConverter
    print(frenet_obj.cartesian_to_frenet(1, 2))
