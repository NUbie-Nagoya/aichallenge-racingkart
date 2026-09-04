"""ROS 2 / AWSIM backend for residual racing RL.

Real command publication is disabled unless RESIDUAL_RL_ALLOW_CONTROL=1 is set.
The training route must have no baseline/MPC publisher on control_cmd.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from threading import Lock

import numpy as np

from .residual_rl import ResidualTransition


def all_sensor_sequences_advanced(
    current: tuple[int, int, int, int], baseline: tuple[int, int, int, int]
) -> bool:
    """Return true only when LiDAR, odometry, acceleration, and steering advanced.

    A single new LiDAR scan is not a valid BC input if the remaining state was
    cached before a simulator reset or previous control action.
    """
    return all(now > before for now, before in zip(current, baseline))


@dataclass(frozen=True)
class RosBackendConfig:
    control_topic: str = "/control/command/control_cmd"
    reset_topic: str = "/awsim/reset"
    debug_topic: str = "/racing_maneuver/residual_rl/debug"
    control_rate_hz: float = 20.0
    speed_control_kp: float = 1.5
    acceleration_limit_mps2: float = 3.0
    step_timeout_s: float = 1.0
    initial_pose_service: str = "/set_initial_pose"
    initial_pose_timeout_s: float = 2.0
    control_mode_request_topic: str = "/awsim/control_mode_request_topic"


class RosResidualRacingBackend:
    """Owns one explicit AWSIM simulator-training command route."""

    def __init__(self, config: RosBackendConfig | None = None) -> None:
        if os.environ.get("RESIDUAL_RL_ALLOW_CONTROL") != "1":
            raise RuntimeError(
                "refusing live RL control; set RESIDUAL_RL_ALLOW_CONTROL=1 only in a dedicated AWSIM training route"
            )
        from autoware_auto_control_msgs.msg import AckermannControlCommand
        from autoware_auto_vehicle_msgs.msg import SteeringReport
        from geometry_msgs.msg import AccelWithCovarianceStamped
        from nav_msgs.msg import Odometry
        from rclpy.node import Node
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import LaserScan
        from std_msgs.msg import Bool, Empty, Float32MultiArray
        from std_srvs.srv import Trigger
        import rclpy

        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init()
        self.config = config or RosBackendConfig()
        self._node = Node("racing_maneuver_residual_rl_backend")
        self._AckermannControlCommand = AckermannControlCommand
        self._Bool = Bool
        self._Empty = Empty
        self._Float32MultiArray = Float32MultiArray
        self._Trigger = Trigger
        self._lock = Lock()
        self._scan = None
        self._odom = None
        self._acceleration = None
        self._steering = None
        self._status = None

        self._scan_sequence = 0
        self._odom_sequence = 0
        self._acceleration_sequence = 0
        self._steering_sequence = 0
        self._previous_step_time = None
        self._episode_progress_m = 0.0
        self._publisher = self._node.create_publisher(AckermannControlCommand, self.config.control_topic, 1)
        self._reset_publisher = self._node.create_publisher(Empty, self.config.reset_topic, 1)
        self._initial_pose_client = self._node.create_client(
            Trigger, self.config.initial_pose_service
        )
        self._control_mode_publisher = self._node.create_publisher(
            Bool, self.config.control_mode_request_topic, 1
        )
        self._debug_publisher = self._node.create_publisher(Float32MultiArray, self.config.debug_topic, 10)
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self._node.create_subscription(LaserScan, "/sensing/lidar/scan", self._on_scan, sensor_qos)
        self._node.create_subscription(Odometry, "/localization/kinematic_state", self._on_odom, 1)
        self._node.create_subscription(AccelWithCovarianceStamped, "/localization/acceleration", self._on_acceleration, 1)
        self._node.create_subscription(SteeringReport, "/vehicle/status/steering_status", self._on_steering, 1)
        self._node.create_subscription(Float32MultiArray, "/awsim/status", self._on_status, 1)


    def _on_scan(self, message) -> None:
        with self._lock:
            self._scan = message
            self._scan_sequence += 1

    def _on_odom(self, message) -> None:
        with self._lock:
            self._odom = message
            self._odom_sequence += 1

    def _on_acceleration(self, message) -> None:
        with self._lock:
            self._acceleration = message
            self._acceleration_sequence += 1

    def _on_steering(self, message) -> None:
        with self._lock:
            self._steering = message
            self._steering_sequence += 1

    def _on_status(self, message) -> None:
        with self._lock:
            self._status = message


    def _sensor_sequences(self) -> tuple[int, int, int, int]:
        """Snapshot callback sequences in BC feature-vector order."""
        with self._lock:
            return (
                self._scan_sequence,
                self._odom_sequence,
                self._acceleration_sequence,
                self._steering_sequence,
            )

    def _spin_until_frame(self, minimum_sequences: tuple[int, int, int, int]) -> None:
        """Wait for a coherent all-new sensor set, never a mixed cached state."""
        deadline = time.monotonic() + self.config.step_timeout_s
        while time.monotonic() < deadline:
            self._rclpy.spin_once(self._node, timeout_sec=0.02)
            if all_sensor_sequences_advanced(self._sensor_sequences(), minimum_sequences):
                return
        raise TimeoutError("timed out waiting for fresh residual-RL sensor set")

    @staticmethod
    def _yaw(q) -> float:
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def _transition(self) -> ResidualTransition:
        from .lidar import canonicalize_scan

        with self._lock:
            scan, odom, acceleration, steering = self._scan, self._odom, self._acceleration, self._steering

        lidar = canonicalize_scan(scan.ranges, float(scan.angle_min), float(scan.angle_increment), float(scan.range_min), float(scan.range_max))
        yaw = self._yaw(odom.pose.pose.orientation)
        # AWSIM/odometry can report a tiny signed numerical velocity while the
        # kart is stationary. Manual crash/stall detection is based on speed
        # magnitude, not travel direction.
        speed = abs(float(odom.twist.twist.linear.x))
        position = odom.pose.pose.position
        accel = float(acceleration.accel.accel.linear.x)
        steer = float(steering.steering_tire_angle)
        now = time.monotonic()
        progress = 0.0 if self._previous_step_time is None else max(0.0, speed) * (now - self._previous_step_time)
        self._previous_step_time = now
        return ResidualTransition(
            lidar=lidar,
            aux=np.asarray([speed, steer, accel, float(position.x), float(position.y), math.sin(yaw), math.cos(yaw), 0.0, 0.0], dtype=np.float32),
            progress_m=progress,
            speed_mps=speed,
            # AWSIM does not currently expose a verified per-vehicle collision
            # report on this route. CrashChecker performs the intentional,
            # measured-speed-based manual detection in ResidualRacingEnv.
            collision=False,
            off_track=False,
            timestamp_s=now,
        )

    def _publish(self, command: np.ndarray) -> None:
        target_speed, measured = float(command[1]), 0.0
        with self._lock:
            if self._odom is not None:
                measured = float(self._odom.twist.twist.linear.x)
        acceleration = float(np.clip(self.config.speed_control_kp * (target_speed - measured), -self.config.acceleration_limit_mps2, self.config.acceleration_limit_mps2))
        message = self._AckermannControlCommand()
        stamp = self._node.get_clock().now().to_msg()
        message.stamp = stamp
        message.lateral.stamp = stamp
        message.lateral.steering_tire_angle = float(command[0])
        message.lateral.steering_tire_rotation_rate = 2.0
        message.longitudinal.stamp = stamp
        message.longitudinal.speed = target_speed
        message.longitudinal.acceleration = acceleration
        self._publisher.publish(message)

    def publish_debug(
        self,
        *,
        base_action: np.ndarray,
        residual_action: np.ndarray,
        command: np.ndarray,
        step_progress_m: float,
        speed_mps: float,
        termination_reason: str | None,
    ) -> None:
        """Publish numeric residual-policy telemetry for PlotJuggler/ros2 echo.

        data ordering: base steer/speed, residual steer/speed, final steer/speed,
        step progress, episode progress, measured speed, termination code.
        """
        base = np.asarray(base_action, dtype=np.float32)
        residual = np.asarray(residual_action, dtype=np.float32)
        final = np.asarray(command, dtype=np.float32)
        if base.shape != (2,) or residual.shape != (2,) or final.shape != (2,):
            raise ValueError("residual debug actions must each have shape [2]")
        self._episode_progress_m += max(0.0, float(step_progress_m))
        termination_codes = {
            None: 0.0,
            "collision": 1.0,
            "off_track": 2.0,
            "velocity_drop": 3.0,
            "prolonged_low_speed": 4.0,
        }
        message = self._Float32MultiArray()
        message.data = [
            float(base[0]), float(base[1]),
            float(residual[0]), float(residual[1]),
            float(final[0]), float(final[1]),
            float(step_progress_m), float(self._episode_progress_m),
            float(speed_mps), termination_codes.get(termination_reason, -1.0),
        ]
        self._debug_publisher.publish(message)

    def _request_initial_pose(self) -> None:
        """Synchronously reinitialize the vehicle/localization pose on domain 1."""
        if not self._initial_pose_client.service_is_ready():
            raise RuntimeError(
                f"initial-pose service is unavailable: {self.config.initial_pose_service}"
            )
        future = self._initial_pose_client.call_async(self._Trigger.Request())
        deadline = time.monotonic() + self.config.initial_pose_timeout_s
        while not future.done() and time.monotonic() < deadline:
            self._rclpy.spin_once(self._node, timeout_sec=0.02)
        if not future.done():
            raise TimeoutError("timed out waiting for initial-pose reset service")
        response = future.result()
        if response is None or not bool(response.success):
            detail = "" if response is None else str(response.message)
            raise RuntimeError(f"initial-pose reset failed: {detail}")

    def reset(self) -> ResidualTransition:
        """Perform AWSIM reset, pose reset, then accept only post-pose sensor state."""
        sequences_before_reset = self._sensor_sequences()
        self._reset_publisher.publish(self._Empty())
        # This first barrier gives the admin reset bridge/AWSIM time to process
        # without a fixed sleep. No transition is exposed to BC yet.
        self._spin_until_frame(sequences_before_reset)
        self._request_initial_pose()
        # Reset/pose initialization can release the simulator's drive permit.
        # Reassert its documented request before accepting the post-pose frames.
        self._control_mode_publisher.publish(self._Bool(data=True))
        sequences_before_post_pose_frame = self._sensor_sequences()
        self._spin_until_frame(sequences_before_post_pose_frame)
        self._previous_step_time = None
        self._episode_progress_m = 0.0
        return self._transition()

    def observe(self) -> ResidualTransition:
        """Return the next all-new sensor set without publishing a command."""
        sequences = self._sensor_sequences()
        self._spin_until_frame(sequences)
        return self._transition()

    def step(self, command: np.ndarray) -> ResidualTransition:
        sequences = self._sensor_sequences()
        self._publish(command)
        self._spin_until_frame(sequences)
        return self._transition()

    def close(self) -> None:
        self._node.destroy_node()


def create_training_backend(config: dict | None = None) -> RosResidualRacingBackend:
    """Construct the backend from the versioned residual-PPO YAML section."""
    return RosResidualRacingBackend(RosBackendConfig(**(config or {})))
