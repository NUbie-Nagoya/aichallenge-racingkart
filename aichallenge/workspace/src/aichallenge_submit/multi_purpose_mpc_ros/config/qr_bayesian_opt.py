from datetime import datetime
from logging import config

import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import subprocess
import time
import yaml
import os

import math
import numpy as np
from pathlib import Path


from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores

try:
    from rosbags.typesys import get_typestore
    typestore = get_typestore(Stores.ROS2_HUMBLE)
except ImportError:
    from rosbags.typesys import get_typesystem
    typestore = get_typesystem(Stores.ROS2_HUMBLE)


########################
# EXTRACT REF WAYPOINTS
########################

def load_reference_waypoints(csv_path: str) -> np.ndarray:
    """
    Loads CSV and extracts [x_m, y_m, psi_rad, vx_mps].

    ../env/final_ver3/traj_mincurv.csv has the following columns;
    0: t_s, 1: x_m, 2: y_m, 3: psi_rad, 4: kappa_1pm, 5: vx_mps, 6: ax_mps2, 7: ay_mps2
    
    """
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    return data[:, [1, 2, 3, 5]]

###########################
# CONVERT QUATERNION TO YAW
###########################

def quaternion_to_yaw(z: float, w: float) -> float:
    """Extract yaw angle (heading) from 2D planar quaternion."""
    return 2.0 * math.atan2(z, w)

####################
# TAKE ERRORS FROM ROSBAG
####################

def calculate_jbo_from_rosbag(mcap_path: str, ref_points: np.ndarray) -> float:
    """
    Computes tracking error and smoothness against reference waypoints.
    ref_points: Nx4 array [x_ref, y_ref, psi_ref, vx_ref]
    """
    bag_file = Path(mcap_path)
    if not bag_file.exists():
        return 10000.0

    total_ey = 0.0
    total_epsi = 0.0
    total_ev = 0.0
    total_steer_rate = 0.0
    sample_count = 0

    with Reader(bag_file) as reader:
        topics = {'/localization/kinematic_state', '/control/command/control_cmd'}
        connections = [c for c in reader.connections if c.topic in topics]
        
        if not connections:
            return 10000.0

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            # Use typestore.deserialize_cdr instead:
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            
            # 1. State Tracking Errors
            if connection.topic == '/localization/kinematic_state':
                x = msg.pose.pose.position.x
                y = msg.pose.pose.position.y
                qz = msg.pose.pose.orientation.z
                qw = msg.pose.pose.orientation.w
                v_act = msg.twist.twist.linear.x
                
                yaw_act = quaternion_to_yaw(qz, qw)

                # Nearest-neighbor matching
                dists = np.hypot(ref_points[:, 0] - x, ref_points[:, 1] - y)
                idx = np.argmin(dists)
                
                # Cross-track error (ey)
                ey = dists[idx]
                
                # Heading error (epsi): wrap to [-pi, pi]
                yaw_ref = ref_points[idx, 2]
                epsi = abs(math.atan2(math.sin(yaw_act - yaw_ref), math.cos(yaw_act - yaw_ref)))
                
                # Velocity error (ev) against dynamic target velocity
                v_ref = ref_points[idx, 3]
                ev = abs(v_act - v_ref)

                total_ey += ey
                total_epsi += epsi
                total_ev += ev
                sample_count += 1

            # 2. Control Smoothness Penalty
            elif connection.topic == '/control/command/control_cmd':
                total_steer_rate += abs(msg.lateral.steering_tire_rotation_rate)

    if sample_count == 0:
        return 10000.0

    # BO Objective Value
    J_BO = (
        1.0 * (total_ey / sample_count) +
        0.5 * (total_epsi / sample_count) +
        0.1 * (total_ev / sample_count) +
        0.05 * (total_steer_rate / max(1, sample_count))
    )

    return float(J_BO)

# --- Example Usage in your BO Loop ---
# latest_bag_path = "/path/to/aichallenge-racingkart/output/latest/d1/rosbag2_autoware.mcap"
# cost = calculate_jbo_from_rosbag(latest_bag_path)
# print(f"Run completed with cost: {cost}")

######################################
# ACTUAL BAYESIAN OPTIMIZATION SETUP
######################################

# DEFINITION: Search space for the 7 MPC weights (q_y would be anchored to 1.0)
# space = [
#     Real(1e-3, 1e3, prior='log-uniform', name='q_psi'),
#     Real(1e-3, 1e3, prior='log-uniform', name='q_v'),
#     Real(1e-3, 1e3, prior='log-uniform', name='qN_y'),
#     Real(1e-3, 1e3, prior='log-uniform', name='qN_psi'),
#     Real(1e-3, 1e3, prior='log-uniform', name='qN_v'),
#     Real(1e-3, 1e3, prior='log-uniform', name='r_v'),
#     Real(1e-3, 1e3, prior='log-uniform', name='r_delta')
# ]

# Second try
# space = [
#     Real(5e5, 1e7, prior='log-uniform', name='q_y'),
#     Real(1e7, 5e8, prior='log-uniform', name='q_psi'),
#     Real(5e4, 5e6, prior='log-uniform', name='q_v'),
#     Real(1e-2, 1e2, prior='log-uniform', name='r_delta')
# ]

# Third try

R_a = 1.0
space = [
    Real(1, 300, prior='log-uniform', name='q_y'),
    Real(10, 1000, prior='log-uniform', name='q_psi'),
    Real(0.1, 50, prior='log-uniform', name='q_v'),
    Real(0.01, 10, prior='log-uniform', name='r_delta'),
    Real(1, 20, prior='uniform', name='factor_qn')
]

# FUNCTION: Update YAML with newMPC weights
def update_mpc_yaml(q_y, q_psi, q_v, r_delta, factor_qn): # qN_y, qN_psi, qN_v, r_v
    yaml_path = os.path.expanduser("~/aichallenge-racingkart/aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros/config/config.yaml") # Make sure this is the absolute path to your Autoware config
    
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
        
    # Second try # Directly assign to the dictionary keys
    # config['mpc']['Q'] = [float(q_y), float(q_psi), float(q_v)]
    # config['mpc']['R'] = [100000.0, float(r_delta)]
    # config['mpc']['QN'] = [1000000.0, 1000.0, 10000.0] # [float(qN_y), float(qN_psi), float(qN_v)]

    config['mpc']['Q'] = [float(q_y), float(q_psi), float(q_v)]
    config['mpc']['R'] = [float(R_a), float(r_delta)]
    config['mpc']['QN'] = [float(q_y)*float(factor_qn), float(q_psi)*float(factor_qn), float(q_v)*float(factor_qn)]

    with open(yaml_path, 'w') as file:
        yaml.dump(config, file)

# MAIN EVALUATION FUNCTION: Evaluate MPC maneuver with given weights
# The @use_named_args decorator maps the 7 variables directly into the function
# @use_named_args(space)
# def evaluate_mpc_maneuver(q_psi, q_v, qN_y, qN_psi, qN_v, r_v, r_delta):
    
#     # YAML UPDATE
#     update_mpc_yaml(q_psi, q_v, qN_y, qN_psi, qN_v, r_v, r_delta)
    
#     # SIMULATION LAUNCH:Launch the simulation using subprocess (make dev) in the correct directory
#     print(f"Launching simulation with weights: R=[{r_v:.3f}, {r_delta:.3f}]...")
#     repo_dir = os.path.expanduser("~/aichallenge-racingkart")
#     sim_process = subprocess.Popen(["make", "dev"], cwd=repo_dir)
    
#     # DEFINE PATHS: Stalled flag and latest rosbag
#     stalled_flag_path = Path(repo_dir) / "output" / "stalled.flag"
#     bag_path = Path(repo_dir) / "output" / "latest" / "d1" / "rosbag2_autoware.mcap"
    
#     # CLEAR STALLED FLAG
#     if stalled_flag_path.exists():
#         stalled_flag_path.unlink()
        
#     # WATCHDOG LOOP: Monitor for stalled flag or simulation completion, default finished (sim ends normally)
#     status = "finished"
#     timeout_sec = 60.0  # Max seconds allowed for a single lap
#     start_time = time.time()

#     while True:
#         # Check if the watchdog created the stall flag
#         if stalled_flag_path.exists():
#             print("Detected stalled.flag! Killing simulation...")
#             status = "stalled"
#             break

#         # Check if simulation completed normally on its own
#         if sim_process.poll() is not None:
#             break

#         # Hard timeout fallback (in case the car stops but watchdog misses)
#         if time.time() - start_time > timeout_sec:
#             print("Maneuver timed out! Killing simulation...")
#             status = "stalled"
#             break

#         time.sleep(1.0)

#     # CLEANING & SIMULATION KILL
#     sim_process.terminate()
#     subprocess.run(["docker", "compose", "down"], cwd=repo_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) # Better than pkill for Docker
#     time.sleep(3)

#     # EPISODE SCORING
#     if status == "stalled":
#         print("Watchdog tripped! Applying death penalty.")
#         return 10000.0
        
#     else:
#         # NOTE: You need to load your actual track reference points here!
#         # For now, this is a placeholder array so the code runs.
#         # Shape must be Nx4: [x, y, yaw, target_velocity]

#         ref_points = load_reference_waypoints("aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros/env/final_ver3/traj_mincurv.csv")  # Update this path

#         # Call the parser you built at the top of the script
#         J_BO = calculate_jbo_from_rosbag(str(bag_path), ref_points)
#         print(f"Run finished successfully. J_BO Cost: {J_BO:.3f}")

#         return J_BO

@use_named_args(space)
def evaluate_mpc_maneuver(q_y, q_psi, q_v, r_delta, factor_qn):
    # 1. Update config parameters
    update_mpc_yaml(q_y, q_psi, q_v, r_delta, factor_qn)
    
    repo_dir = Path(os.path.expanduser("~/aichallenge-racingkart"))
    output_dir = repo_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[BO] Starting run: Q=[{q_y:.1f}, {q_psi:.1f}, {q_v:.1f}], R=[{r_delta:.3f}]")
    
    # 2. Record launch time and start simulation
    launch_time = time.time()
    env = os.environ.copy()
    env["ROSBAG"] = "true"
    subprocess.run(["make", "dev"], cwd=repo_dir, env=env, stdout=subprocess.DEVNULL)
    
    # 3. Poll for the newly created run directory and its flags
    status = "timeout"
    timeout_sec = 130.0
    start_time = time.time()
    active_run_dir = None
    
    while time.time() - start_time < timeout_sec:
        # Dynamically locate the directory created for this run
        if active_run_dir is None:
            candidates = [
                d for d in output_dir.iterdir()
                if d.is_dir() and d.name.startswith("202") and d.stat().st_mtime >= (launch_time - 2.0)
            ]
            if candidates:
                # Pick the latest matching timestamp folder
                active_run_dir = max(candidates, key=lambda d: d.stat().st_mtime)

        # Check for status flags inside the active run's d1 folder
        if active_run_dir is not None:
            stalled_flag = active_run_dir / "d1" / "stalled.flag"
            finished_flag = active_run_dir / "d1" / "finished.flag"
            
            if stalled_flag.exists():
                status = "stalled"
                break
            if finished_flag.exists():
                status = "finished"
                break
                
        time.sleep(1.0)
        
    print(f"[BO] Run ended with status: {status} ({time.time() - start_time:.1f}s)")
    
    # 4. Teardown containers
    subprocess.run(["docker", "compose", "down"], cwd=repo_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2.0)
    
    # 5. Compute objective score
    if status == "finished" and active_run_dir is not None:
        bag_path = active_run_dir / "d1" / "rosbag2_autoware.mcap"
        time.sleep(1.0)  # Ensure MCAP write finishes
        return calculate_jbo_from_rosbag(str(bag_path), REFERENCE_WAYPOINTS)
    else:
        return 10000.0
    
# 4. Run the Bayesian Optimization Loop
if __name__ == "__main__":
    print("Starting Bayesian Optimization...")
    
    res = gp_minimize(
        func=evaluate_mpc_maneuver,   # the function to minimize
        dimensions=space,             # the 7D bounds
        acq_func="EI",                # Expected Improvement acquisition function
        n_calls=150,                   # Total number of simulation runs
        n_initial_points=10,          # Random exploration points before GP takes over
        random_state=42               # For reproducible results
    )
    
    print("\nOptimization Finished!")
    print(f"Best BO Score: {res.fun}")
    print(f"Optimal Weights [q_psi, q_v, qN_y, qN_psi, qN_v, r_v, r_delta]:")
    print(np.round(res.x, 3))