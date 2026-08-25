from scipy.interpolate import splprep, splev
import cv2
import yaml
import argparse
import numpy as np
import pandas as pd
from math import dist, ceil, atan2, sqrt
from casadi import Opti, norm_2, exp
from typing import Tuple, List
from tabulate import tabulate

class StandaloneMapManager:
    def __init__(self, map_yaml_file):
        self.map_yaml_file = map_yaml_file
        with open(map_yaml_file, "r", encoding='utf-8') as stream:
            self.map_yaml_obj = yaml.safe_load(stream)
        
        # Load image relative to the yaml file
        import os
        map_dir_path = os.path.dirname(os.path.abspath(map_yaml_file))
        img_path = os.path.join(map_dir_path, self.map_yaml_obj["image"])
        self.map_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        self.map_height, self.map_width = self.map_img.shape[:2]

    def pixel2metric_point(self, p_point) -> list:
        m_point = [
            self.map_yaml_obj["origin"][0]
            + self.map_yaml_obj["resolution"] / 2
            + p_point[0] * self.map_yaml_obj["resolution"],
            self.map_yaml_obj["origin"][1]
            + self.map_yaml_obj["resolution"] / 2
            + (self.map_height - p_point[1] - 1) * self.map_yaml_obj["resolution"]
        ]
        return m_point

    def pixel2metric_line(self, p_line) -> list:
        m_line = []
        for p_point in p_line:
            m_point = self.pixel2metric_point(p_point)
            m_line.append(m_point)
        return m_line

    def pixel2metric_lines(self, p_lines) -> np.ndarray:
        m_lines = []
        for p_line in p_lines:
            m_line = self.pixel2metric_line(p_line)
            m_lines.append(m_line)
        return np.array(m_lines)

    def metric2pixel_point(self, m_point) -> list:
        p_point = [
            (m_point[0] - self.map_yaml_obj["origin"][0] - self.map_yaml_obj["resolution"] / 2) / self.map_yaml_obj["resolution"],
            (self.map_height - 1 - (m_point[1] - self.map_yaml_obj["origin"][1] - self.map_yaml_obj["resolution"] / 2) / self.map_yaml_obj["resolution"])
        ]
        return p_point

    def metric2pixel_line(self, m_line) -> list:
        p_line = []
        for m_point in m_line:
            p_point = self.metric2pixel_point(m_point)
            p_line.append(p_point)
        return p_line

    def metric2pixel_lines(self, m_lines) -> np.ndarray:
        p_lines = []
        for m_line in m_lines:
            p_line = self.metric2pixel_line(m_line)
            p_lines.append(p_line)
        return np.array(p_lines)

    def metric2pixel_length(self, m_length) -> float:
        p_length = m_length / self.map_yaml_obj["resolution"]
        return p_length

# IMPORTANT: ContourDetector now inherits from StandaloneMapManager
class ContourDetector(StandaloneMapManager):
    def __init__(self, map_yaml_file) -> None:
        super().__init__(map_yaml_file)

    def get_angle(self, vector1, vector2) -> float:
        angle = atan2(vector1[1], vector1[0]) - atan2(vector2[1], vector2[0])
        if angle <= -np.pi:
            angle += 2 * np.pi
        elif angle > np.pi:
            angle -= 2 * np.pi
        return angle

    def get_course_contours(self, metric_crop_length) -> list:
        p_crop_length = self.metric2pixel_length(metric_crop_length)
        map_img_gray = cv2.cvtColor(self.map_img, cv2.COLOR_BGR2GRAY)
        map_img_bin = cv2.threshold(map_img_gray, 250, 255, cv2.THRESH_BINARY)[1]
        map_img_eroded = cv2.erode(map_img_bin, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=int(p_crop_length))
        
        contours, hierarchy = cv2.findContours(map_img_eroded, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)

        hierarchy = hierarchy.reshape(len(hierarchy[0]), 4)
        hierarchy_w_area = []
        for idx, (contour, data) in enumerate(zip(contours, hierarchy)):
            data_w_area = [idx, cv2.contourArea(contour), data]
            hierarchy_w_area.append(data_w_area)

        hierarchy_w_area.sort(key=lambda x: x[1], reverse=True)

        outer_idx, inner_idx = None, None
        for (idx, area, data) in hierarchy_w_area:
            if (cv2.contourArea(contours[idx]) < (self.map_height-p_crop_length-2)*(self.map_width-p_crop_length-2)
                and data[2] >= 0):
                outer_idx = idx
                inner_idx = data[2]

        pixel_contours = [
            contours[outer_idx], 
            contours[inner_idx],
        ]
        pixel_contours = [
            contour.reshape(len(contour), 2) for contour in pixel_contours
        ]

        metric_contours = []
        for contour in pixel_contours:
            metric_contour = self.pixel2metric_line(contour)
            metric_contours.append(np.array(metric_contour))

        return metric_contours

class LateralLineCreator(ContourDetector):
    def __init__(self, map_yaml_file) -> None:
        super().__init__(map_yaml_file)

    def get_nearest_index(self, ref_point, points_list, prev_nearest_idx=None, search_length=None) -> int:
        min_idx = 0
        if (prev_nearest_idx is not None and search_length is not None): 
            start_idx = (prev_nearest_idx - search_length) % len(points_list)
            end_idx = (prev_nearest_idx + search_length) % len(points_list)
            if start_idx > end_idx:
                start_idx -= len(points_list)
            clamp_list = []
            for i in range(start_idx, end_idx + 1):
                clamp_list.append(points_list[i])
            points_list = clamp_list
            min_idx += start_idx
        min_idx += min(enumerate(points_list), key=lambda x: dist(ref_point, x[1]))[0]
        return min_idx

    def get_point2line_position(self, point, line) -> int:
        dot_start = (point[0] - line[0][0]) * (line[1][0] - line[0][0]) + (point[1] - line[0][1]) * (line[1][1] - line[0][1])
        dot_end = (point[0] - line[1][0]) * (line[0][0] - line[1][0]) + (point[1] - line[1][1]) * (line[0][1] - line[1][1])
        if dot_start <= 0: return -1
        elif dot_end <= 0: return 1
        else: return 0

    def get_point2line_distance(self, point, line) -> float:
        return abs((line[1][0] - line[0][0]) * (line[0][1] - point[1]) - (line[0][0] - point[0]) * (line[1][1] - line[0][1])) / dist(line[0], line[1])

    def get_line_data_list(self, ref_point, line_list) -> Tuple[List[float], List[float]]:
        dist_list = [] 
        proj_list = [] 
        for idx, line in enumerate(line_list):
            pos = self.get_point2line_position(ref_point, line)
            if pos == 0:
                line_dist = self.get_point2line_distance(ref_point, line)
                dist_list.append(line_dist)
                ref_dist = dist(line[0], ref_point)
                if ((ref_dist * ref_dist - line_dist * line_dist) < 0):
                    proj = 0
                else:
                    proj = sqrt(ref_dist * ref_dist - line_dist * line_dist) / dist(line[0], line[1])
                proj_list.append(proj)
            elif pos == -1:
                dist_list.append(dist(line[0], ref_point))
                proj_list.append(0)
            elif pos == 1:
                dist_list.append(dist(line[1], ref_point))
                proj_list.append(1)
        return dist_list, proj_list

    def get_lateral_lines(self, contour_list) -> list:
        if len(contour_list) != 2:
            raise ValueError(f"Invalid numbers of contours: {len(contour_list)}. (Expected: 2)")

        lateral_line_list = []
        for i in range(2): 
            ref_contour = contour_list[i-1]
            srch_contour = contour_list[i]
            for ref_idx, ref_point in enumerate(ref_contour):
                line_list = []
                for idx in range(len(srch_contour)):
                    line_list.append([srch_contour[idx-1], srch_contour[idx]])
                dist_list, proj_list = self.get_line_data_list(ref_point, line_list) 
                line_idx, (line, line_dist, proj) = min(enumerate(zip(line_list, dist_list, proj_list)), key=lambda x: x[1][1]) 
                nearest_point = line[0] + (line[1] - line[0]) * proj 
                if i == 0:
                    lateral_line_list.append([nearest_point, ref_point])
                else: 
                    lateral_line_list.append([ref_point, nearest_point])
        return lateral_line_list

    def center_point(self, line) -> np.ndarray:
        return np.array([(line[0][0] + line[1][0]) / 2, (line[0][1] + line[1][1]) / 2])

    def sort_lines(self, line_list, contour_list, is_clockwise=True):
        contour_line_list = []
        for idx in range(len(contour_list[0])):
            contour_line_list.append([contour_list[0][idx-1], contour_list[0][idx]])

        extended_list = []
        for line in line_list:
            extended_list.append([line, self.center_point(line), 0])

        for idx in reversed(range(len(extended_list))):
            not_detected = True
            for contour_line in contour_line_list:
                not_detected = not_detected and not self.detect_overlaps(extended_list[idx][0], contour_line)
            if not_detected:
                extended_list.pop(idx)

        sorted_list = []
        overlap_list = []
        for i, contour_line in enumerate(contour_line_list):
            for j in reversed(range(len(extended_list))):
                if self.detect_overlaps(extended_list[j][0], contour_line):
                    overlap_list.append(extended_list.pop(j))
            if len(overlap_list) == 1:
                sorted_list.append(overlap_list[0])
            elif len(overlap_list) > 1:
                for k, data in enumerate(overlap_list):
                    overlap_list[k][2] = dist(contour_line_list[i+1][0], data[1])
                overlap_list.sort(key=lambda x: x[2], reverse=True)
                while len(overlap_list) > 0:
                    sorted_list.append(overlap_list.pop(0))
            if len(sorted_list) > 0:
                break

        for contour_line in contour_line_list:
            overlap_list = []
            for idx in reversed(range(len(extended_list))):
                if self.detect_overlaps(extended_list[idx][0], contour_line):
                    overlap_list.append(extended_list.pop(idx))
            if len(overlap_list) == 1:
                sorted_list.append(overlap_list[0])
            elif len(overlap_list) > 1:
                for idx, data in enumerate(overlap_list):
                    overlap_list[idx][2] = dist(sorted_list[-1][1], data[1])
                overlap_list.sort(key=lambda x: x[2])
                while len(overlap_list) > 0:
                    sorted_list.append(overlap_list.pop(0))

        sorted_line_list = [data[0] for data in sorted_list]

        angle_sum = 0
        for idx, line in enumerate(sorted_line_list):
            vector1 = sorted_line_list[idx][1] - sorted_line_list[idx-1][1]
            vector2 = sorted_line_list[idx-1][1] - sorted_line_list[idx-2][1]
            angle_sum += self.get_angle(vector1, vector2)
        if is_clockwise and angle_sum > 0:
            sorted_line_list = sorted_line_list[::-1]
        elif not is_clockwise and angle_sum < 0:
            sorted_line_list = sorted_line_list[::-1]

        return sorted_line_list

    def detect_overlaps(self, line1, line2) -> bool:
        ax, ay = line1[0][0], line1[0][1]
        bx, by = line1[1][0], line1[1][1]
        cx, cy = line2[0][0], line2[0][1]
        dx, dy = line2[1][0], line2[1][1]
        s = (ax - bx) * (cy - ay) - (ay - by) * (cx - ax)
        t = (ax - bx) * (dy - ay) - (ay - by) * (dx - ax)
        if s*t > 0: return False
        elif s*t == 0:
            if t != 0:
                if ((cx < ax and cx < bx) or (cx > ax and cx > bx) or (cy < ay and cy < by) or (cy > ay and cy > by)): return False
            if s != 0:
                if ((dx < ax and dx < bx) or (dx > ax and dx > bx) or (dy < ay and dy < by) or (dy > ay and dy > by)): return False
            if ((cx < ax and cx < bx and dx < ax and dx < bx) or (cx > ax and cx > bx and dx > ax and dx > bx) or (cy < ay and cy < by and dy < ay and dy < by) or (cy > ay and cy > by and dy > ay and dy > by)): return False

        s = (cx - dx) * (ay - cy) - (cy - dy) * (ax - cx)
        t = (cx - dx) * (by - cy) - (cy - dy) * (bx - cx)
        if s*t > 0: return False
        elif s*t == 0:
            if t != 0:
                if ((ax < cx and ax < dx) or (ax > cx and ax > dx) or (ay < cy and ay < dy) or (ay > cy and ay > dy)): return False
            if s != 0:
                if ((bx < cx and bx < dx) or (bx > cx and bx > dx) or (by < cy and by < dy) or (by > cy and by > dy)): return False
            if ((ax < cx and ax < dx and bx < cx and bx < dx) or (ax > cx and ax > dx and bx > cx and bx > dx) or (ay < cy and ay < dy and by < cy and by < dy) or (ay > cy and ay > dy and by > cy and by > dy)): return False
        return True

    def detect_mid_overlaps(self, line1, line2) -> bool:
        ax, ay = line1[0][0], line1[0][1]
        bx, by = line1[1][0], line1[1][1]
        cx, cy = line2[0][0], line2[0][1]
        dx, dy = line2[1][0], line2[1][1]
        s = (ax - bx) * (cy - ay) - (ay - by) * (cx - ax)
        t = (ax - bx) * (dy - ay) - (ay - by) * (dx - ax)
        if s*t > 0: return False
        elif s*t == 0:
            if s != 0 or t != 0: return False
            if ((cx <= ax and cx <= bx and dx <= ax and dx <= bx) or (cx >= ax and cx >= bx and dx >= ax and dx >= bx)): return False

        s = (cx - dx) * (ay - cy) - (cy - dy) * (ax - cx)
        t = (cx - dx) * (by - cy) - (cy - dy) * (bx - cx)
        if s*t > 0: return False
        elif s*t == 0:
            if s != 0 or t != 0: return False
            if ((ax <= cx and ax <= dx and bx <= cx and bx <= dx) or (ax >= cx and ax >= dx and bx >= cx and bx >= dx) or (ay <= cy and ay <= dy and by <= cy and by <= dy) or (ay >= cy and ay >= dy and by >= cy and by >= dy)): return False
        return True

    def remove_self_overlaps(self, line_list: list) -> np.ndarray:
        overlap_array = np.zeros((len(line_list),len(line_list)), dtype=np.uint8)
        for i in range(len(line_list)):
            for j in range(i + 1, len(line_list)):
                if self.detect_mid_overlaps(line_list[i], line_list[j]):
                    overlap_array[i][j] += 1
        overlap_array = overlap_array + overlap_array.T
        mask = np.full(len(line_list), True, dtype=bool)
        try:
            while True:
                overlap_count = sum(overlap_array)
                if sum(overlap_count) == 0: break
                max_idx = np.argsort(overlap_count)[-1]
                mask[max_idx] = False
                overlap_array[:, max_idx] = 0
                overlap_array[max_idx, :] = 0
        except KeyboardInterrupt:
            pass
        line_list = np.array(line_list)[mask]
        return line_list

    def remove_contour_overlaps(self, line_list, contour_list) -> list:
        filtered_line_list = []
        for line in line_list:
            delete_flag = False
            for contour in contour_list:
                overlap_count = 0
                for idx, point in enumerate(contour):
                    contour_line = [contour[idx], contour[idx-1]]
                    if self.detect_overlaps(line, contour_line):
                        overlap_count += 1
                        if overlap_count >= 3:
                            delete_flag = True
                            break
                if delete_flag: break
            if not delete_flag: filtered_line_list.append(line)
        return filtered_line_list

    def fill_gaps(self, line_list, max_gap) -> list:
        filled_line_list = []
        for idx, line in enumerate(line_list):
            filled_line_list.append(line_list[idx - 1])
            start_dist = dist(line[0], line_list[idx - 1][0])
            end_dist = dist(line[1], line_list[idx - 1][1])
            if start_dist > max_gap or end_dist > max_gap:
                gap_ratio = max(ceil(start_dist / max_gap), ceil(end_dist / max_gap))
                for i in range(1, gap_ratio):
                    filled_line_list.append([
                        line_list[idx - 1][0] + (line[0] - line_list[idx - 1][0]) * i / gap_ratio,
                        line_list[idx - 1][1] + (line[1] - line_list[idx - 1][1]) * i / gap_ratio,
                    ])
        return filled_line_list

    def thin_out(self, line_list, min_gap) -> list:
        thinned_line_list = [line_list[0]]
        for idx, line in enumerate(line_list):
            d_1 = self.get_point2line_distance(line[0], thinned_line_list[-1])
            d_2 = self.get_point2line_distance(line[1], thinned_line_list[-1])
            if ((d_1 >= min_gap or d_2 >= min_gap) and d_1 > 0 and d_2 > 0):
                thinned_line_list.append(line)
        return thinned_line_list

    def crop_edges(self, line_list, crop_length) -> list:
        cropped_line_list = []
        for line in line_list:
            cropped_line_list.append([
                line[0] + (line[1] - line[0]) * crop_length / dist(line[0], line[1]),
                line[1] + (line[0] - line[1]) * crop_length / dist(line[0], line[1]),
            ])
        return cropped_line_list

class PathOptimizer:
    def __init__(self, lateral_line_list, max_iter=10000, tolerance=0.1) -> None:
        self.line_list = lateral_line_list
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.optimization_failed = False

    def set_params(
        self, max_vel, lon_acc_max, lon_acc_min, lat_acc_abs_max,
        use_min_turn_r, min_turn_r, n_curv_calc, use_dir_lock,
        use_dec, use_dyn_lat, lat_acc_slope_center, lat_acc_slope_inclination,
        lat_acc_max_divisor, use_dyn_dec, dec_slope_center, dec_slope_inclination, dec_max_divisor
    ) -> None:
        self.max_vel = max_vel
        self.lon_acc_max = lon_acc_max
        self.lon_acc_min = lon_acc_min
        self.lat_acc_abs_max = lat_acc_abs_max
        self.use_min_turn_r = use_min_turn_r
        self.min_turn_r = min_turn_r
        self.n_curv_calc = n_curv_calc
        self.use_dir_lock = use_dir_lock
        self.use_dec = use_dec
        self.use_dyn_lat = use_dyn_lat
        self.lat_acc_slope_center = lat_acc_slope_center
        self.lat_acc_slope_inclination = lat_acc_slope_inclination
        self.lat_acc_max_divisor = lat_acc_max_divisor
        self.use_dyn_dec = use_dyn_dec
        self.dec_slope_center = dec_slope_center
        self.dec_slope_inclination = dec_slope_inclination
        self.dec_max_divisor = dec_max_divisor

    def get_curvature(self, point_s, point_m, point_e) -> float:
        numerator = (point_s[0] - point_m[0]) * (point_s[1] - point_e[1]) - (point_s[1] - point_m[1]) * (point_s[0] - point_e[0])
        divisor = norm_2(point_s - point_m) * norm_2(point_m - point_e) * norm_2(point_s - point_e)
        return 2 * numerator / divisor

    def denormalize(self, norm_pos_list) -> list:
        denorm_pos_list = []
        for idx, line in enumerate(self.line_list):
            denorm_pos_list.append(line[0] * (1 - norm_pos_list[idx]) + line[1] * norm_pos_list[idx])
        return denorm_pos_list
            
    def optimize_lap_time(self, optimizer: Opti):
        N = len(self.line_list)
        self.norm_pos_list = optimizer.variable(N)
        self.vel_list = optimizer.variable(N)
        self.curv_list = optimizer.variable(N)
        self.long_acc_list = optimizer.variable(N)
        self.lat_acc_list = optimizer.variable(N)

        optimizer.set_initial(self.norm_pos_list, 0.5)
        optimizer.set_initial(self.vel_list, 1.0)

        lap_time = 0
        denorm_pos_list = self.denormalize(self.norm_pos_list)
        for i in range(N):
            lap_time += ((norm_2(denorm_pos_list[(i + 1) % N] - denorm_pos_list[i]) + norm_2(denorm_pos_list[i] - denorm_pos_list[i - 1])) / 2) / self.vel_list[i]

            optimizer.subject_to(self.norm_pos_list[i] >= 0)
            optimizer.subject_to(self.norm_pos_list[i] <= 1)
            optimizer.subject_to(self.vel_list[i] > 0)
            optimizer.subject_to(self.vel_list[i] <= self.max_vel)
            optimizer.subject_to(self.long_acc_list[i] <= self.lon_acc_max)
            optimizer.subject_to(self.long_acc_list[i] >= self.lon_acc_min)
            
            optimizer.subject_to(pow(self.lat_acc_list[i], 2.0) <= pow(self.lat_acc_abs_max, 2.0))
            optimizer.subject_to(self.long_acc_list[i] >= self.lon_acc_min)
            optimizer.subject_to(self.long_acc_list[i] == (pow(self.vel_list[i],2.0) - pow(self.vel_list[i - 1],2.0)) / (2 * norm_2(denorm_pos_list[i] - denorm_pos_list[i - 1])))

            if self.use_dir_lock:
                vector_1 = denorm_pos_list[i] - denorm_pos_list[i - 1]
                vector_2 = denorm_pos_list[i - 1] - denorm_pos_list[i - 2]
                dot = vector_1[0] * vector_2[0] + vector_1[1] * vector_2[1]
                optimizer.subject_to(dot > 0.0)

            optimizer.subject_to(self.curv_list[i] == self.get_curvature(denorm_pos_list[(i + self.n_curv_calc + 1) % N], denorm_pos_list[ i], denorm_pos_list[ i - self.n_curv_calc - 1]))
            optimizer.subject_to(self.lat_acc_list[i] == pow(self.vel_list[i], 2.0) * self.curv_list[i])
            optimizer.subject_to(pow(self.long_acc_list[i] / (self.lon_acc_max), 2.0) + pow(self.lat_acc_list[i] / (self.lat_acc_abs_max), 2.0) <= 1.0)
            
            if self.use_min_turn_r:
                optimizer.subject_to(pow(self.curv_list[i],2.0) <= pow(1 / (self.min_turn_r* (1.0 - pow(self.vel_list[i]/self.max_vel, 2.0))), 2.0))

        optimizer.minimize(lap_time)

        try:
            self.solution = optimizer.solve()
        except RuntimeError:
            print("Optimization failed")
            self.solution = optimizer.debug
            self.optimization_failed = True

    def solve(self) -> None:
        optimizer = Opti()
        optimizer.solver(
            "ipopt", 
            {"print_time": True}, 
            {"max_iter": self.max_iter, "print_level": 5, "sb": "yes", "tol": self.tolerance, "print_frequency_time": 1},
        )
        self.optimize_lap_time(optimizer)

    def get_results(self):
        norm_pos_list = self.solution.value(self.norm_pos_list)
        optimized_path = []
        for idx, (norm_pos, line) in enumerate(zip(norm_pos_list, self.line_list)):
            optimized_path.append(line[0] * (1 - norm_pos) + line[1] * norm_pos)
        return optimized_path

def box_print(text) -> None:
    print(f"\n{tabulate(text, tablefmt='grid')}\n")

def run_standalone_optimization(map_yaml, output_csv, ay_max=12.0, ax_max=0.7, ax_min=-1.6, vmax_kmh=35.0):
    print("Loading map and detecting contours...")
    m_crop_length = 0.7
    m_max_gap = 2.0
    m_min_gap = 1.0
    is_clockwise = True
    
    lat_line_creator = LateralLineCreator(map_yaml)
    contour_list = lat_line_creator.get_course_contours(m_crop_length)
    
    print("Generating lateral lines...")
    line_list_raw = lat_line_creator.get_lateral_lines(contour_list)
    line_list_no_self_overlap = lat_line_creator.remove_self_overlaps(line_list_raw)
    line_list_no_overlap = lat_line_creator.remove_contour_overlaps(line_list_no_self_overlap, contour_list)
    line_list_sorted = lat_line_creator.sort_lines(line_list_no_overlap, contour_list, is_clockwise)
    line_list_dense = lat_line_creator.fill_gaps(line_list_sorted, m_max_gap)
    line_list_cropped = lat_line_creator.crop_edges(line_list_dense, 1e-2)
    line_list_thinned = lat_line_creator.thin_out(line_list_cropped, m_min_gap)
    
    print("Running CasADi Optimizer for Track Geometry...")
    path_optimizer = PathOptimizer(line_list_thinned)
    path_optimizer.set_params(
        max_vel=8.33, lon_acc_max=0.7, lon_acc_min=-1.6, lat_acc_abs_max=12.0,
        use_min_turn_r=False, min_turn_r=1.0, n_curv_calc=2, use_dir_lock=True,
        use_dec=False, use_dyn_lat=False, lat_acc_slope_center=0, lat_acc_slope_inclination=0,
        lat_acc_max_divisor=1, use_dyn_dec=False, dec_slope_center=0, dec_slope_inclination=0, dec_max_divisor=1
    )
    
    path_optimizer.solve()
    opt_path = path_optimizer.get_results()
    wp_x = [p[0] for p in opt_path]
    wp_y = [p[1] for p in opt_path]
    
    print("Applying kinematics and calculating velocity profiles...")
    vmax = vmax_kmh / 3.6 
    x = wp_x + [wp_x[0]]
    y = wp_y + [wp_y[0]]
    
    tck, u = splprep([x, y], s=0, per=True)
    u_new = np.linspace(0, 1.0, 400)
    out = splev(u_new, tck)
    sx, sy = out[0], out[1]
    
    dx, dy = splev(u_new, tck, der=1)
    ddx, ddy = splev(u_new, tck, der=2)
    
    psi = np.arctan2(dy, dx)
    kappa = (dx * ddy - dy * ddx) / ((dx**2 + dy**2)**1.5 + 1e-10)
    
    ds = np.sqrt(dx**2 + dy**2)
    du = u_new[1] - u_new[0]
    s_m = np.cumsum(ds * du)
    s_m -= s_m[0]
    
    v_lim = np.zeros_like(kappa)
    for i, k in enumerate(kappa):
        if abs(k) > 1e-5:
            v_lim[i] = min(np.sqrt(ay_max / abs(k)), vmax)
        else:
            v_lim[i] = vmax
            
    v_fwd = np.copy(v_lim)
    for i in range(1, len(v_fwd)):
        v_fwd[i] = min(v_fwd[i], np.sqrt(v_fwd[i-1]**2 + 2 * ax_max * (s_m[i] - s_m[i-1])))
        
    v_bwd = np.copy(v_fwd)
    for i in range(len(v_bwd)-2, -1, -1):
        v_bwd[i] = min(v_bwd[i], np.sqrt(v_bwd[i+1]**2 + 2 * abs(ax_min) * (s_m[i+1] - s_m[i])))
        
    vx_mps = v_bwd
    ax_mps2 = np.zeros_like(vx_mps)
    for i in range(1, len(vx_mps)-1):
        ax_mps2[i] = (vx_mps[i+1]**2 - vx_mps[i-1]**2) / (2 * (s_m[i+1] - s_m[i-1]))
        
    df_out = pd.DataFrame({
        's_m': s_m, 'x_m': sx, 'y_m': sy, 'psi_rad': psi, 
        'kappa_radpm': kappa, 'vx_mps': vx_mps, 'ax_mps2': ax_mps2
    })
    
    df_out.to_csv(output_csv, index=False)
    print(f"✅ Final trajectory with velocity profiles saved to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_yaml', type=str, required=True, help="Path to occupancy grid yaml")
    parser.add_argument('--output', type=str, default="traj0.csv")
    args = parser.parse_args()
    run_standalone_optimization(args.map_yaml, args.output) # ay_max=12.0, ax_max=5.0, ax_min=-12.0, vmax_kmh=0.0)