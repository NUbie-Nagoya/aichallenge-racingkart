# Frenet Flow

This module adds a local Frenet-based planner on top of the existing MPC stack.
It is designed to generate short candidate paths in track coordinates `(s, d)`
and hand the selected candidate back to the MPC as a temporary reference path.

## Data Flow

1. The controller keeps a global `ReferencePath` for the track.
2. `FrenetConverter` maps between Cartesian pose `(x, y, psi)` and Frenet pose `(s, d)`.
3. `FrenetPlanner` evaluates local maneuvers using quintic polynomials.
4. The best candidate is converted back to a short `ReferencePath`.
5. MPC tracks the first part of that local path until the planner returns to the global line.

## Converter

`core/frenet_converter.py` is responsible for geometry conversion.
It computes the closest segment on the reference path and returns:

- `cartesian_to_frenet(x, y, yaw)` -> `(s, d, waypoint_id)`
- `frenet_to_cartesian(s, d)` -> `(x, y, psi, kappa_eff)`

Important details:

- Circular tracks wrap around using the total path length.
- Short paths are handled safely so single-waypoint or short local paths do not crash.
- `kappa_eff` includes the lateral offset curvature correction used by the MPC path projection.

## Planner

`core/frenet_planner.py` generates three kinds of local maneuvers:

- `OVERTAKE`: move laterally around a lead obstacle and continue forward.
- `RETURN`: smoothly return to the center line after passing.
- `BRAKING`: hold lane position and slow down when no safe lateral path exists.

Planning uses these constraints from `config.yaml`:

- `lane_margin`
- `lateral_offsets`
- `T_overtake`
- `T_return`
- `T_brake`
- `clearance`
- `ttc_thr`
- `gap0`
- `gap_t`
- `decel_lim`

Candidate scoring uses:

- jerk cost
- curvature cost
- obstacle clearance cost
- time cost

The planner works in track coordinates, then converts the winning path back to Cartesian coordinates for MPC.

## Quintic Polynomials

`core/quintic_polynomial.py` provides the trajectory primitives:

- `quintic_coeff(...)`
- `lateral_quintic(d0, d_dot0, d_target, T)`
- `longitudinal_quintic(s0, v0, s_target, v_target, T)`

Each candidate path is sampled over time and checked for:

- corridor feasibility
- obstacle collision
- boundary clearance

## MPC Handoff

The controller keeps the Frenet path as a temporary override.
When a candidate is accepted:

- it is converted to a `ReferencePath`
- a speed profile is assigned to the new local path
- the MPC tracks that local path immediately

For Frenet-only operation (`frenet.enabled: true`, obstacle avoidance disabled):

- the controller also builds simple corridor constraints for the local path
- once the vehicle is realigned, the controller restores the original global path

## Safety and MPC Ownership

Frenet generates temporary reference paths; it does not replace the MPC controller. MPC continues to compute and publish the control command for both the global path and any accepted local path.

The local maneuver is sampled with 20 points. Its path constraints must match the MPC horizon. Changing the sample count or horizon independently can cause constraint dimension mismatches.

If MPC rejects a Frenet local path, the controller logs `Frenet local path rejected by MPC; reverting to global path`, clears the local override, resets the planner to `IDLE`, restores the global path, and retries MPC in the same control cycle. This prevents a bad corner candidate from stopping the controller.

## Controller Notes

`mpc_controller.py` now keeps the Frenet converter aligned with the active global reference path.
If the global reference path is refreshed from a trajectory topic, the Frenet stack is rebuilt with the new path.

The controller also publishes the local candidate for visualization on:

- `/mpc/frenet_candidate`

## Practical Tips

- Keep `reprojection_rate_hz` below or equal to the MPC control rate.
- Use the global reference path for Frenet coordinate conversion, not the temporary local path.
- For circular tracks, obstacle detection and overtake completion use wrapped `s` distances.
- If the planner returns no candidate, the controller falls back to the global MPC path or braking behavior.
- For circular tracks, lead detection, passing completion, and braking gaps use wrapped forward `s` distance.
- After changing Frenet code, rebuild the package with `make autoware-build` before running Docker.
- Check each per-domain log for `START!`, MPC tracebacks, and repeated `No control signal computed!` messages.
