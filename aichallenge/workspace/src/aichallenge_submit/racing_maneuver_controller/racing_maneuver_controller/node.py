#!/usr/bin/env python3
"""Shadow-first ROS 2 wrapper for the racing maneuver TorchScript policy."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
import torch
from autoware_auto_control_msgs.msg import AckermannControlCommand
from autoware_auto_vehicle_msgs.msg import ControlModeReport, SteeringReport
from geometry_msgs.msg import AccelWithCovarianceStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
from v2x_msgs.msg import V2XVehiclePositionArray

from .controller_core import (
    AUX_FEATURE_NAMES,
    ControllerCore,
    canonicalize_lidar,
    mode_is_permitted,
    pose_jump_exceeds,
    publication_decision,
    sources_are_fresh,
    summarize_opponents,
    torch_versions_compatible,
    validate_artifact_metadata,
)
from .node_helpers import (
    V2XTracker,
    assemble_auxiliary,
    quaternion_yaw,
)


def stamp_ns(message: Any) -> int | None:
    stamp = getattr(getattr(message, "header", None), "stamp", None)
    if stamp is None:
        stamp = getattr(message, "stamp", None)
    if stamp is None:
        return None
    value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return value if value > 0 else None


class RacingManeuverControllerNode(Node):
    """20 Hz policy observer; actuation requires explicit opt-in and strict gates."""

    def __init__(self) -> None:
        super().__init__("racing_maneuver_controller")
        self.declare_parameter("model_path", "")
        self.declare_parameter("ego_vehicle_id", "d1")
        self.declare_parameter("publish_commands", False)
        self.declare_parameter("control_cmd_topic", "/control/command/control_cmd")
        self.declare_parameter("debug_control_topic", "/racing_maneuver/debug/control_cmd")
        self.declare_parameter("debug_features_topic", "/racing_maneuver/debug/features")
        self.declare_parameter("debug_safety_topic", "/racing_maneuver/debug/safety_status")
        self.declare_parameter("allowed_control_modes", [1])
        self.declare_parameter("shadow_allowed_control_modes", [1, 4])
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("scan_maximum_age_ms", 100.0)
        self.declare_parameter("ego_maximum_age_ms", 100.0)
        self.declare_parameter("v2x_maximum_age_ms", 250.0)
        self.declare_parameter("control_mode_maximum_age_ms", 500.0)
        self.declare_parameter("canonical_angle_min_rad", -math.radians(89.5))
        self.declare_parameter("canonical_angle_max_rad", math.radians(89.5))
        self.declare_parameter("lidar_model_max_range_m", 30.0)
        self.declare_parameter("maximum_pose_jump_m", 3.0)
        self.declare_parameter("steering_limit_rad", 1.0)
        self.declare_parameter("acceleration_limit_mps2", 3.0)
        self.declare_parameter("steering_rate_limit_rad_s", 1.5)
        self.declare_parameter("acceleration_rate_limit_mps3", 4.0)
        self.declare_parameter("command_steering_rotation_rate_rad_s", 1.5)
        self.declare_parameter("v2x_maximum_speed_mps", 50.0)
        self.declare_parameter("opponent_forward_corridor_half_width_m", 5.0)

        self.publish_commands = bool(self.get_parameter("publish_commands").value)
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        if not math.isfinite(self.rate_hz) or self.rate_hz <= 0.0:
            raise ValueError("rate_hz must be finite and positive")
        self.allowed_modes = {int(x) for x in self.get_parameter("allowed_control_modes").value}
        self.shadow_modes = {
            int(x) for x in self.get_parameter("shadow_allowed_control_modes").value
        }
        if not self.allowed_modes or not self.shadow_modes:
            raise ValueError("control mode allow-lists must not be empty")
        model_path = Path(str(self.get_parameter("model_path").value))
        if not model_path.is_file():
            raise FileNotFoundError(f"TorchScript policy does not exist: {model_path}")
        sidecar_path = model_path.with_suffix(model_path.suffix + ".metadata.json")
        if not sidecar_path.is_file():
            raise FileNotFoundError(f"TorchScript metadata sidecar does not exist: {sidecar_path}")
        try:
            artifact_metadata = json.loads(sidecar_path.read_text())
        except (OSError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("TorchScript metadata sidecar is invalid") from error
        if not torch_versions_compatible(
            str(artifact_metadata.get("torch_version", "")), torch.__version__
        ):
            raise ValueError(
                "artifact/runtime PyTorch major.minor mismatch: "
                f"artifact={artifact_metadata.get('torch_version')!r}, "
                f"runtime={torch.__version__!r}"
            )
        extra_files = {"metadata.json": ""}
        policy = torch.jit.load(
            str(model_path), map_location="cpu", _extra_files=extra_files
        ).eval()
        encoded_metadata = extra_files["metadata.json"]
        if isinstance(encoded_metadata, bytes):
            encoded_metadata = encoded_metadata.decode("utf-8")
        try:
            embedded_metadata = json.loads(encoded_metadata)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("TorchScript artifact has no valid metadata.json") from error
        if embedded_metadata != artifact_metadata:
            raise ValueError("TorchScript embedded metadata differs from its sidecar")
        validate_artifact_metadata(
            artifact_metadata,
            self.rate_hz,
            expected_corridor_half_width_m=float(
                self.get_parameter("opponent_forward_corridor_half_width_m").value
            ),
            expected_opponent_maximum_speed_mps=float(
                self.get_parameter("v2x_maximum_speed_mps").value
            ),
            expected_v2x_maximum_age_s=float(
                self.get_parameter("v2x_maximum_age_ms").value
            )
            * 1e-3,
            expected_steering_limit_rad=float(
                self.get_parameter("steering_limit_rad").value
            ),
            expected_acceleration_limit_mps2=float(
                self.get_parameter("acceleration_limit_mps2").value
            ),
        )
        configured_preprocessing = (
            float(self.get_parameter("canonical_angle_min_rad").value),
            float(self.get_parameter("canonical_angle_max_rad").value),
            float(self.get_parameter("lidar_model_max_range_m").value),
        )
        expected_preprocessing = (-math.radians(89.5), math.radians(89.5), 30.0)
        if not np.allclose(configured_preprocessing, expected_preprocessing, atol=1e-6, rtol=0.0):
            raise ValueError(
                "configured LiDAR preprocessing differs from the embedded model contract"
            )
        self.core = ControllerCore(
            policy,
            steering_limit_rad=float(self.get_parameter("steering_limit_rad").value),
            acceleration_limit_mps2=float(
                self.get_parameter("acceleration_limit_mps2").value
            ),
            steering_rate_limit_rad_s=float(
                self.get_parameter("steering_rate_limit_rad_s").value
            ),
            acceleration_rate_limit_mps3=float(
                self.get_parameter("acceleration_rate_limit_mps3").value
            ),
        )
        self._tracker = V2XTracker(
            str(self.get_parameter("ego_vehicle_id").value),
            float(self.get_parameter("v2x_maximum_speed_mps").value),
        )
        self._messages: dict[str, Any] = {}
        self._last_mode: int | None = None
        self._last_position_xy: tuple[float, float] | None = None
        self._last_appended_scan_stamp: int | None = None

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(LaserScan, "/sensing/lidar/scan", self._cache("scan"), qos)
        self.create_subscription(
            Odometry, "/localization/kinematic_state", self._cache("odom"), qos
        )
        self.create_subscription(
            AccelWithCovarianceStamped,
            "/localization/acceleration",
            self._cache("acceleration"),
            qos,
        )
        self.create_subscription(
            PoseWithCovarianceStamped,
            "/initialpose",
            lambda _message: self._reset("initial pose/reset event"),
            1,
        )
        self.create_subscription(
            SteeringReport,
            "/vehicle/status/steering_status",
            self._cache("steering"),
            qos,
        )
        self.create_subscription(
            ControlModeReport,
            "/vehicle/status/control_mode",
            self._mode_callback,
            qos,
        )
        self.create_subscription(
            V2XVehiclePositionArray,
            "/v2x/vehicle_positions",
            self._v2x_callback,
            qos,
        )
        self.debug_control_pub = self.create_publisher(
            AckermannControlCommand,
            str(self.get_parameter("debug_control_topic").value),
            1,
        )
        self.features_pub = self.create_publisher(
            Float32MultiArray,
            str(self.get_parameter("debug_features_topic").value),
            1,
        )
        self.safety_pub = self.create_publisher(
            Float32MultiArray,
            str(self.get_parameter("debug_safety_topic").value),
            1,
        )
        # Creating the endpoint is not publication. Calls to publish are guarded
        # independently below by publish_commands and the actuation mode list.
        self.command_pub = self.create_publisher(
            AckermannControlCommand,
            str(self.get_parameter("control_cmd_topic").value),
            1,
        )
        self.create_timer(1.0 / self.rate_hz, self._tick)
        self.get_logger().info(
            f"loaded {model_path}; shadow-first publish_commands={self.publish_commands}; "
            f"auxiliary_features={len(AUX_FEATURE_NAMES)}"
        )

    def _cache(self, name: str):
        def callback(message: Any) -> None:
            self._messages[name] = message

        return callback

    def _v2x_callback(self, message: V2XVehiclePositionArray) -> None:
        source_stamp = stamp_ns(message)
        if source_stamp is not None:
            try:
                self._tracker.update(message.vehicles, source_stamp * 1e-9)
            except ValueError:
                self._reset("invalid V2X update")
        self._messages["v2x"] = message

    def _mode_callback(self, message: ControlModeReport) -> None:
        mode = int(message.mode)
        if self._last_mode is not None and mode != self._last_mode:
            self._reset("control mode transition")
        self._last_mode = mode
        self._messages["mode"] = message

    def _reset(self, reason: str) -> None:
        self.core.reset()
        self._tracker.reset()
        self._last_position_xy = None
        self._last_appended_scan_stamp = None
        self.get_logger().warn(f"history reset: {reason}", throttle_duration_sec=2.0)

    def _fresh(self, now_ns: int) -> bool:
        required = ("scan", "odom", "acceleration", "steering", "v2x", "mode")
        if any(name not in self._messages for name in required):
            return False
        stamps = {
            "scan": stamp_ns(self._messages["scan"]),
            "ego": min(
                stamp_ns(self._messages["odom"]) or 0,
                stamp_ns(self._messages["acceleration"]) or 0,
                stamp_ns(self._messages["steering"]) or 0,
            ),
            "v2x": stamp_ns(self._messages["v2x"]),
        }
        limits = {
            "scan": float(self.get_parameter("scan_maximum_age_ms").value),
            "ego": float(self.get_parameter("ego_maximum_age_ms").value),
            "v2x": float(self.get_parameter("v2x_maximum_age_ms").value),
        }
        if self.publish_commands:
            stamps["mode"] = stamp_ns(self._messages["mode"])
            limits["mode"] = float(
                self.get_parameter("control_mode_maximum_age_ms").value
            )
        return sources_are_fresh(now_ns, stamps, limits)

    def _status(self, fresh: bool, mode_ok: bool, real_ok: bool) -> None:
        self.safety_pub.publish(
            Float32MultiArray(
                data=[
                    float(fresh),
                    float(mode_ok),
                    float(self.core.ready),
                    float(self.core.history_size),
                    float(self.publish_commands),
                    float(real_ok),
                ]
            )
        )

    def _command(self, steering: float, acceleration: float, speed: float):
        stamp = self.get_clock().now().to_msg()
        command = AckermannControlCommand()
        command.stamp = stamp
        command.lateral.stamp = stamp
        command.lateral.steering_tire_angle = steering
        command.lateral.steering_tire_rotation_rate = float(
            self.get_parameter("command_steering_rotation_rate_rad_s").value
        )
        command.longitudinal.stamp = stamp
        command.longitudinal.speed = speed
        command.longitudinal.acceleration = acceleration
        return command

    def _tick(self) -> None:
        now_ns = self.get_clock().now().nanoseconds
        fresh = self._fresh(now_ns)
        mode = int(self._messages["mode"].mode) if "mode" in self._messages else -1
        mode_ok = mode_is_permitted(
            mode, self.publish_commands, self.allowed_modes, self.shadow_modes
        )
        debug_ok, real_ok = publication_decision(
            mode, self.publish_commands, self.allowed_modes, self.shadow_modes
        )
        self._status(fresh, mode_ok, real_ok and fresh)
        if not fresh or not mode_ok:
            self._reset("stale source or disallowed control mode")
            return
        scan: LaserScan = self._messages["scan"]
        scan_stamp = stamp_ns(scan)
        if scan_stamp is None or scan_stamp == self._last_appended_scan_stamp:
            return
        if self._last_appended_scan_stamp is not None and scan_stamp < self._last_appended_scan_stamp:
            self._reset("non-causal LiDAR timestamp")
            return
        odom: Odometry = self._messages["odom"]
        acceleration: AccelWithCovarianceStamped = self._messages["acceleration"]
        steering: SteeringReport = self._messages["steering"]
        try:
            lidar = canonicalize_lidar(
                scan.ranges,
                angle_min=float(scan.angle_min),
                angle_increment=float(scan.angle_increment),
                range_min=float(scan.range_min),
                range_max=float(scan.range_max),
                target_angle_min=float(
                    self.get_parameter("canonical_angle_min_rad").value
                ),
                target_angle_max=float(
                    self.get_parameter("canonical_angle_max_rad").value
                ),
                model_max_range=float(
                    self.get_parameter("lidar_model_max_range_m").value
                ),
            )
            position_xy = (
                float(odom.pose.pose.position.x),
                float(odom.pose.pose.position.y),
            )
            if pose_jump_exceeds(
                self._last_position_xy,
                position_xy,
                float(self.get_parameter("maximum_pose_jump_m").value),
            ):
                self._reset("pose discontinuity/reset")
                return
            yaw = quaternion_yaw(odom.pose.pose.orientation)
            speed = float(odom.twist.twist.linear.x)
            lateral_speed = float(odom.twist.twist.linear.y)
            cosine, sine = math.cos(yaw), math.sin(yaw)
            ego_vx = cosine * speed - sine * lateral_speed
            ego_vy = sine * speed + cosine * lateral_speed
            opponent = summarize_opponents(
                float(odom.pose.pose.position.x),
                float(odom.pose.pose.position.y),
                yaw,
                ego_vx,
                ego_vy,
                now_ns * 1e-9,
                self._tracker.observations(),
                corridor_half_width_m=float(
                    self.get_parameter("opponent_forward_corridor_half_width_m").value
                ),
            )
            auxiliary = assemble_auxiliary(
                speed=speed,
                steering=float(steering.steering_tire_angle),
                acceleration=float(acceleration.accel.accel.linear.x),
                previous_safe=self.core.previous_safe_output,
                opponent_summary=opponent,
            )
            self.core.append_frame(lidar, auxiliary)
            self._last_position_xy = position_xy
        except (ValueError, AttributeError) as error:
            self._reset(str(error))
            return
        self._last_appended_scan_stamp = scan_stamp
        self.features_pub.publish(
            Float32MultiArray(data=np.concatenate((lidar, auxiliary)).astype(float).tolist())
        )
        if not self.core.ready:
            return
        try:
            steering_cmd, acceleration_cmd = self.core.infer_and_limit(1.0 / self.rate_hz)
        except (RuntimeError, ValueError) as error:
            self._reset(str(error))
            return
        command = self._command(float(steering_cmd), float(acceleration_cmd), speed)
        if debug_ok:
            self.debug_control_pub.publish(command)
        if real_ok and fresh and self.publish_commands and mode in self.allowed_modes:
            self.command_pub.publish(command)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RacingManeuverControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
