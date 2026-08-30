# Frenet Framework

## Overtaking Maneuver (Offensive – when opportunity to overtake is presented)

### Part 1: Drawing Alongside and Passing
- Obstacle Detection and Triggering:
    - Convert current ego pose and obstacle bounding boxes from Cartesian (x, y) to Frenet coordinates (s, d) using your track reference line.
    - Check if the obstacle lies within your lane corridor (cross-track distance between obstacle and raceline is under a safety margin, e.g., 1.0 m) and is ahead of you along the track.
    - If an obstacle is detected in your path, trigger the local planner avoidance mode.
- Local Trajectory Generation:
    - Define the planning horizon by time (T between 1.5 and 3.0 seconds) rather than a fixed waypoint count, so the maneuver scales naturally with vehicle speed (target track distance = current s + speed * T).
    - Define terminal lateral boundary conditions: target lateral offset = d_target, with target lateral velocity and acceleration set to 0 so the kart finishes parallel to the track centerline.
    - Sample candidate lateral offsets around the obstacle (e.g., d_target = obstacle d +/- 1.2 m) across different horizon times T.
    Generate smooth quintic polynomials in (s, d) for each sampled candidate.
- Feasibility Filtering and Cost Evaluation:
    - Hard Discards: Immediately reject any candidate path that intersects the obstacle's padded bounding box or crosses the physical track boundaries and asphalt limits.
    - Cost Ranking: Score surviving trajectories using a weighted sum of:
        - Total lateral and longitudinal jerk (smoothness)
        - Path curvature (lower curvature allows higher achievable cornering speed)
        - Obstacle clearance distance (penalize getting too close to the competitor)
        - Time horizon T (penalize overly sluggish transitions)
    - Select the single candidate with the lowest aggregate cost.
- Velocity Profiling and Cartesian Conversion:
    -  Convert the winning polynomial from (s, d) back to Cartesian coordinates (x, y, heading yaw, curvature).
    -  Recalculate target velocity along the path: cap target speed using the new path curvature and lateral grip limits (sqrt(max_lateral_accel / curvature)) so the kart does not understeer or slide out wide.
-  MPC Execution:
    - Extract the first 20 points of the selected local path and pass them as the tracking reference to the MPC.
    - Execute the immediate control output and repeat the planning loop.


### Part 2: Longitudinal Check and Return Maneuver
- Pass Completion Trigger:
    - Check when the rear bounding box coordinate of the ego vehicle is ahead of the front bounding box coordinate of the obstacle plus a clearance margin (approximately 1.5 to 2.0 meters along track distance s).
- Smooth Return Planning:
    - Do not snap abruptly to the nearest original raceline point.
    - Instead, generate a new return quintic polynomial:
        - Start State: Current ego lateral deviation, lateral velocity, and lateral acceleration.
        - Target State: Optimal raceline (target lateral offset = 0, target lateral velocity = 0, target lateral acceleration = 0) over a horizon of 1.5 to 2.5 seconds.
- Feed the first 20 points of this return path to the MPC until the kart is realigned on the main racing line, then resume baseline tracking.


## Braking Maneuver (Defensive – when someone is overtaking us)
### Part 1: Yielding and Following (Speed Reduction Under Overtake/Blocking)
- Threat Detection and Triggering:
    - Detect a vehicle or obstacle ahead in your corridor where an immediate lateral pass is unfeasible (e.g., track width is too narrow, multiple vehicles block all candidate lateral corridors, or a competitor alongside has corner apex priority).
    - Calculate the time-to-collision (TTC) based on relative longitudinal distance and closing speed: TTC = (s_obstacle - s_ego) / (v_ego - v_obstacle).
    - Trigger the defensive/braking maneuver if TTC drops below a safety threshold (e.g., under 1.5 to 2.0 seconds) or if all candidate lateral overtaking paths fail collision checks.
- Target Following Distance and Speed Calculation:
    - Define a safe dynamic following gap: target_gap = d_standstill + time_gap * v_ego (e.g., 2.0 m buffer + 0.8 s * v_ego).
    - Set the target terminal longitudinal distance: target_s = s_obstacle - target_gap.
    - Set the target terminal velocity: target_v = min(v_obstacle, v_raceline).
- Deceleration Trajectory Generation (Longitudinal Quintic/Quartic Polynomial):
    - Generate a smooth longitudinal profile s(t) over a braking horizon T (typically 1.5 to 2.5 seconds):
        - Start State: Current ego longitudinal position, velocity, and acceleration along the track.
        - Target State: target_s, with velocity matching the lead vehicle (target_v) and acceleration set to 0.
    - Lateral Profile: Hold the current lateral lane or raceline path (target lateral offset = current d or raceline d, with target lateral velocity and acceleration set to 0).
- Feasibility and Comfort Filtering:
    - Check that the required longitudinal deceleration does not exceed maximum braking capacity (e.g., |a_lon| <= 10 to 12 m/s^2) or cause tire lockup based on current steering angle.
    - If the demanded deceleration exceeds safe limits, apply maximum threshold braking immediately rather than a gentle polynomial profile.
- MPC:
    - Discretize the generated deceleration path into the MPC horizon (first 20 points).
    - Send the speed-reduced reference trajectory to the MPC tracker so it commands regenerative/mechanical braking smoothly.
    - Re-plan to continuously match lead vehicle speed fluctuations.

### Part 2: Resuming Race Pace / Transitioning to Attack

- Clearance / Opportunity Trigger:
    - Monitor the road ahead and identify when either:
        1. The lead vehicle accelerates away and opens a gap larger than the safety margin.
        2. A lateral passing corridor opens up with sufficient track width.
- Smooth Re-Acceleration Planning:
    - **If track is clear**: Switch the longitudinal target velocity back to the full raceline velocity profile (v_target = v_raceline) and generate a smooth acceleration quintic polynomial up to top speed.
    - **If an overtaking corridor opens**: Hand control back directly to the Overtaking Planner (Part 1 from the previous step) to initiate the lateral swing and pass.
     


