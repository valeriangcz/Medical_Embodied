#!/usr/bin/env python3

import math
from pathlib import Path
from typing import Dict, Tuple

import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import Imu

try:
    from ament_index_python.packages import get_package_share_directory
except ImportError:
    get_package_share_directory = None


def matmul_3x3(
    a: Tuple[Tuple[float, float, float], ...], b: Tuple[Tuple[float, float, float], ...]
) -> Tuple[Tuple[float, float, float], ...]:
    return (
        (
            a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1],
            a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2],
        ),
        (
            a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1],
            a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2],
        ),
        (
            a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0],
            a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1],
            a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2],
        ),
    )


def rx(angle: float) -> Tuple[Tuple[float, float, float], ...]:
    c = math.cos(angle)
    s = math.sin(angle)
    return (
        (1.0, 0.0, 0.0),
        (0.0, c, -s),
        (0.0, s, c),
    )


def ry(angle: float) -> Tuple[Tuple[float, float, float], ...]:
    c = math.cos(angle)
    s = math.sin(angle)
    return (
        (c, 0.0, s),
        (0.0, 1.0, 0.0),
        (-s, 0.0, c),
    )


def rz(angle: float) -> Tuple[Tuple[float, float, float], ...]:
    c = math.cos(angle)
    s = math.sin(angle)
    return (
        (c, -s, 0.0),
        (s, c, 0.0),
        (0.0, 0.0, 1.0),
    )


def euler_to_rotation_matrix(
    roll: float, pitch: float, yaw: float, rotation_order: str = "zyx"
) -> Tuple[Tuple[float, float, float], ...]:
    """
    Build rotation matrix from order string, e.g.:
    - "zyx" => Rz(yaw) * Ry(pitch) * Rx(roll) (rslidar_sdk default)
    - "xyz" => Rx(roll) * Ry(pitch) * Rz(yaw)
    """
    order = rotation_order.strip().lower()
    if len(order) != 3 or any(ch not in ("x", "y", "z") for ch in order):
        raise ValueError(f"Invalid rotation_order={rotation_order}, expected one of xyz permutations")

    axis_to_rot = {"x": rx(roll), "y": ry(pitch), "z": rz(yaw)}
    # Left-to-right multiply to match textual order.
    result = axis_to_rot[order[0]]
    result = matmul_3x3(result, axis_to_rot[order[1]])
    result = matmul_3x3(result, axis_to_rot[order[2]])
    return result


def rotate_vector(
    rotation: Tuple[Tuple[float, float, float], ...], vector: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    return (
        rotation[0][0] * vector[0] + rotation[0][1] * vector[1] + rotation[0][2] * vector[2],
        rotation[1][0] * vector[0] + rotation[1][1] * vector[1] + rotation[1][2] * vector[2],
        rotation[2][0] * vector[0] + rotation[2][1] * vector[1] + rotation[2][2] * vector[2],
    )


def transpose_rotation(rotation: Tuple[Tuple[float, float, float], ...]) -> Tuple[Tuple[float, float, float], ...]:
    return (
        (rotation[0][0], rotation[1][0], rotation[2][0]),
        (rotation[0][1], rotation[1][1], rotation[2][1]),
        (rotation[0][2], rotation[1][2], rotation[2][2]),
    )


def identity_rotation() -> Tuple[Tuple[float, float, float], ...]:
    return (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )


def vec_norm(v: Tuple[float, float, float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def vec_normalize(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    n = vec_norm(v)
    if n < 1e-9:
        return (0.0, 0.0, 0.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def vec_dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vec_cross(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def rotation_from_axis_angle(axis: Tuple[float, float, float], angle: float) -> Tuple[Tuple[float, float, float], ...]:
    x, y, z = vec_normalize(axis)
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    return (
        (t * x * x + c, t * x * y - s * z, t * x * z + s * y),
        (t * x * y + s * z, t * y * y + c, t * y * z - s * x),
        (t * x * z - s * y, t * y * z + s * x, t * z * z + c),
    )


def rotation_between_vectors(
    src: Tuple[float, float, float], dst: Tuple[float, float, float]
) -> Tuple[Tuple[float, float, float], ...]:
    src_n = vec_normalize(src)
    dst_n = vec_normalize(dst)
    dot_v = max(-1.0, min(1.0, vec_dot(src_n, dst_n)))
    if dot_v > 1.0 - 1e-8:
        return identity_rotation()
    if dot_v < -1.0 + 1e-8:
        # 180-degree rotation: choose an axis orthogonal to src.
        trial = (1.0, 0.0, 0.0) if abs(src_n[0]) < 0.9 else (0.0, 1.0, 0.0)
        axis = vec_cross(src_n, trial)
        if vec_norm(axis) < 1e-8:
            axis = vec_cross(src_n, (0.0, 0.0, 1.0))
        return rotation_from_axis_angle(axis, math.pi)

    axis = vec_cross(src_n, dst_n)
    angle = math.acos(dot_v)
    return rotation_from_axis_angle(axis, angle)


class ImuAlignWithLidarNode(Node):
    def __init__(self) -> None:
        super().__init__("imu_align_with_lidar")

        self.declare_parameter("input_topic", "/rslidar_imu_data")
        self.declare_parameter("output_topic", "/rslidar_imu_data_aligned")
        self.declare_parameter("config_file", "")
        self.declare_parameter("use_inverse_rotation", False)
        self.declare_parameter("rotation_order", "zyx")
        self.declare_parameter("auto_static_calibration", False)
        self.declare_parameter("static_sample_count", 200)
        self.declare_parameter("target_gravity_axis", "-z")
        # Gravity-based static calibration has yaw ambiguity; default do not apply it to gyro.
        self.declare_parameter("apply_static_calib_to_gyro", False)

        input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        config_file = self.get_parameter("config_file").get_parameter_value().string_value.strip()
        use_inverse = self.get_parameter("use_inverse_rotation").get_parameter_value().bool_value
        rotation_order = self.get_parameter("rotation_order").get_parameter_value().string_value
        self.auto_static_calibration = self.get_parameter("auto_static_calibration").get_parameter_value().bool_value
        self.apply_static_calib_to_gyro = (
            self.get_parameter("apply_static_calib_to_gyro").get_parameter_value().bool_value
        )
        self.static_sample_count = max(
            20, self.get_parameter("static_sample_count").get_parameter_value().integer_value
        )
        target_gravity_axis = (
            self.get_parameter("target_gravity_axis").get_parameter_value().string_value.strip().lower()
        )

        cfg_path = self._resolve_config_path(config_file)
        roll, pitch, yaw = self._load_rpy_from_config(cfg_path)
        rotation = euler_to_rotation_matrix(roll, pitch, yaw, rotation_order=rotation_order)
        self.base_rotation = transpose_rotation(rotation) if use_inverse else rotation
        self.calib_rotation = identity_rotation()
        self.rotation = self.base_rotation

        self.gravity_target = self._parse_target_gravity_axis(target_gravity_axis)
        self._calib_sum = (0.0, 0.0, 0.0)
        self._calib_count = 0
        self._calib_done = not self.auto_static_calibration

        self.subscriber = self.create_subscription(Imu, input_topic, self.imu_callback, 100)
        self.publisher = self.create_publisher(Imu, output_topic, 100)

        self.get_logger().info(
            f"IMU align node started. input={input_topic}, output={output_topic}, "
            f"config={cfg_path}, roll={roll:.6f}, pitch={pitch:.6f}, yaw={yaw:.6f}, "
            f"rotation_order={rotation_order}, use_inverse_rotation={use_inverse}, "
            f"auto_static_calibration={self.auto_static_calibration}, "
            f"static_sample_count={self.static_sample_count}, target_gravity_axis={target_gravity_axis}, "
            f"apply_static_calib_to_gyro={self.apply_static_calib_to_gyro}"
        )
        if self.auto_static_calibration:
            self.get_logger().warn(
                "Static auto calibration enabled. Keep IMU/LiDAR still until calibration completes."
            )

    def _resolve_config_path(self, config_file: str) -> Path:
        if config_file:
            return Path(config_file).expanduser().resolve()

        if get_package_share_directory is not None:
            try:
                share_dir = Path(get_package_share_directory("rslidar_sdk"))
                return (share_dir / "config" / "config.yaml").resolve()
            except Exception:
                pass

        # Fallback to source tree path for local runs.
        return (Path(__file__).resolve().parents[1] / "config" / "config.yaml").resolve()

    def _load_rpy_from_config(self, config_path: Path) -> Tuple[float, float, float]:
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as f:
            config: Dict = yaml.safe_load(f)

        lidar_cfg = config.get("lidar", [])
        if not lidar_cfg or "driver" not in lidar_cfg[0]:
            raise ValueError(f"Invalid config format in {config_path}: missing lidar[0].driver")

        driver = lidar_cfg[0]["driver"]
        roll = float(driver.get("roll", 0.0))
        pitch = float(driver.get("pitch", 0.0))
        yaw = float(driver.get("yaw", 0.0))
        return roll, pitch, yaw

    def _parse_target_gravity_axis(self, axis_text: str) -> Tuple[float, float, float]:
        mapping = {
            "x": (1.0, 0.0, 0.0),
            "-x": (-1.0, 0.0, 0.0),
            "y": (0.0, 1.0, 0.0),
            "-y": (0.0, -1.0, 0.0),
            "z": (0.0, 0.0, 1.0),
            "-z": (0.0, 0.0, -1.0),
        }
        if axis_text not in mapping:
            raise ValueError(f"Invalid target_gravity_axis={axis_text}, expected one of {list(mapping.keys())}")
        return mapping[axis_text]

    def _update_static_calibration(self, la_base: Tuple[float, float, float]) -> None:
        if self._calib_done:
            return
        self._calib_sum = (
            self._calib_sum[0] + la_base[0],
            self._calib_sum[1] + la_base[1],
            self._calib_sum[2] + la_base[2],
        )
        self._calib_count += 1
        if self._calib_count < self.static_sample_count:
            return

        avg = (
            self._calib_sum[0] / self._calib_count,
            self._calib_sum[1] / self._calib_count,
            self._calib_sum[2] / self._calib_count,
        )
        self.calib_rotation = rotation_between_vectors(avg, self.gravity_target)
        self.rotation = matmul_3x3(self.calib_rotation, self.base_rotation)
        self._calib_done = True

        self.get_logger().warn(
            f"Static calibration complete. avg_acc=({avg[0]:.4f},{avg[1]:.4f},{avg[2]:.4f}), "
            f"|acc|={vec_norm(avg):.4f}"
        )

    def imu_callback(self, msg: Imu) -> None:
        out = Imu()
        out.header = msg.header
        out.orientation = msg.orientation
        out.orientation_covariance = msg.orientation_covariance
        out.angular_velocity_covariance = msg.angular_velocity_covariance
        out.linear_acceleration_covariance = msg.linear_acceleration_covariance

        av = (msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z)
        la = (msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z)

        la_base = rotate_vector(self.base_rotation, la)
        self._update_static_calibration(la_base)

        gyro_rotation = self.rotation if self.apply_static_calib_to_gyro else self.base_rotation
        av_rot = rotate_vector(gyro_rotation, av)
        la_rot = rotate_vector(self.rotation, la)

        out.angular_velocity.x = av_rot[0]
        out.angular_velocity.y = av_rot[1]
        out.angular_velocity.z = av_rot[2]
        out.linear_acceleration.x = la_rot[0]
        out.linear_acceleration.y = la_rot[1]
        out.linear_acceleration.z = la_rot[2]

        self.publisher.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ImuAlignWithLidarNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
