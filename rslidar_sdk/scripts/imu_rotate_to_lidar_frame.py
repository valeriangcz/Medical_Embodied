#!/usr/bin/env python3

import math
from typing import Tuple

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu


Matrix3 = Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]


def matmul_3x3(a: Matrix3, b: Matrix3) -> Matrix3:
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


def rx(roll: float) -> Matrix3:
    c = math.cos(roll)
    s = math.sin(roll)
    return ((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c))


def ry(pitch: float) -> Matrix3:
    c = math.cos(pitch)
    s = math.sin(pitch)
    return ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))


def rz(yaw: float) -> Matrix3:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


def build_rotation(roll: float, pitch: float, yaw: float, order: str) -> Matrix3:
    order = order.strip().lower()
    if len(order) != 3 or any(ch not in ("x", "y", "z") for ch in order):
        raise ValueError("rotation_order must be a 3-letter axis order, e.g. xyz / zyx")
    mats = {"x": rx(roll), "y": ry(pitch), "z": rz(yaw)}
    r = mats[order[0]]
    r = matmul_3x3(r, mats[order[1]])
    r = matmul_3x3(r, mats[order[2]])
    return r


def rotate_vec(r: Matrix3, v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (
        r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
        r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
        r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
    )


def quaternion_xyzw_to_matrix3(values) -> Matrix3:
    if len(values) != 4:
        raise ValueError("direct_rotation_quaternion_xyzw must contain exactly 4 values [x, y, z, w]")
    x, y, z, w = (float(v) for v in values)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        raise ValueError("Quaternion norm is zero")
    x /= norm
    y /= norm
    z /= norm
    w /= norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )


class ImuRotateToLidarFrameNode(Node):
    def __init__(self) -> None:
        super().__init__("imu_rotate_to_lidar_frame")

        self.declare_parameter("input_topic", "/rslidar_imu_data")
        self.declare_parameter("output_topic", "/rslidar_imu_data_rotated")
        self.declare_parameter("output_frame_id", "lidar_3d_frame_fastlio")

        self.declare_parameter("use_direct_quaternion_rotation", True)
        self.declare_parameter(
            "direct_rotation_quaternion_xyzw",
            [-0.016609, 0.800555, 0.0, 0.599030],
        )
        self.declare_parameter("direct_roll", 3.141593)
        self.declare_parameter("direct_pitch", 1.284203)
        self.declare_parameter("direct_yaw", 0.0)
        self.declare_parameter("direct_rotation_order", "xyz")

        self.declare_parameter("negate_gyro_before_rotation", True)
        self.declare_parameter("negate_gyro_after_rotation", False)
        self.declare_parameter("normalize_acceleration", True)
        self.declare_parameter("gravity_magnitude", 9.81)
        self.declare_parameter("acceleration_mode", "gravity")
        self.declare_parameter("clip_acceleration_to_unit", False)

        # 静止标定：由 raw=[-9.17458,-0.19034,-2.70066] 对齐到 [0,0,+9.81] 后残差≈0。
        self.declare_parameter(
            "linear_acceleration_bias",
            [0.0, 0.0, 0.0],
        )
        # 静止时采样 /rslidar_imu_data angular_velocity 后填入。
        self.declare_parameter("angular_velocity_bias", [0.0, 0.0, 0.0])

        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value
        self.output_frame_id = self.get_parameter("output_frame_id").value

        use_direct_quat = bool(self.get_parameter("use_direct_quaternion_rotation").value)
        direct_quat = self.get_parameter("direct_rotation_quaternion_xyzw").value
        direct_roll = float(self.get_parameter("direct_roll").value)
        direct_pitch = float(self.get_parameter("direct_pitch").value)
        direct_yaw = float(self.get_parameter("direct_yaw").value)
        direct_order = str(self.get_parameter("direct_rotation_order").value)

        self.negate_gyro_before_rotation = bool(
            self.get_parameter("negate_gyro_before_rotation").value
        )
        self.negate_gyro_after_rotation = bool(
            self.get_parameter("negate_gyro_after_rotation").value
        )
        self.normalize_acceleration = bool(self.get_parameter("normalize_acceleration").value)
        self.gravity_magnitude = float(self.get_parameter("gravity_magnitude").value)
        self.acceleration_mode = str(self.get_parameter("acceleration_mode").value).strip().lower()
        self.clip_acceleration_to_unit = bool(self.get_parameter("clip_acceleration_to_unit").value)
        if self.acceleration_mode not in ("gravity", "g_unit"):
            raise ValueError("acceleration_mode must be 'gravity' or 'g_unit'")

        self.linear_acceleration_bias = tuple(
            float(v) for v in self.get_parameter("linear_acceleration_bias").value
        )
        self.angular_velocity_bias = tuple(
            float(v) for v in self.get_parameter("angular_velocity_bias").value
        )
        if len(self.linear_acceleration_bias) != 3 or len(self.angular_velocity_bias) != 3:
            raise ValueError("linear_acceleration_bias and angular_velocity_bias must have 3 values")

        if use_direct_quat:
            self.rotation = quaternion_xyzw_to_matrix3(direct_quat)
        else:
            self.rotation = build_rotation(direct_roll, direct_pitch, direct_yaw, direct_order)

        self.sub = self.create_subscription(Imu, input_topic, self.on_imu, 100)
        self.pub = self.create_publisher(Imu, output_topic, 100)

        self.get_logger().info(
            f"imu_rotate_to_lidar_frame started. input={input_topic}, output={output_topic}, "
            f"out_frame={self.output_frame_id}, use_direct_quaternion_rotation={use_direct_quat}, "
            f"direct_quat_xyzw={direct_quat}, acc_bias={self.linear_acceleration_bias}, "
            f"gyro_bias={self.angular_velocity_bias}"
        )

    def on_imu(self, msg: Imu) -> None:
        out = Imu()
        out.header = msg.header
        out.header.frame_id = self.output_frame_id
        out.orientation = msg.orientation
        out.orientation_covariance = msg.orientation_covariance
        out.angular_velocity_covariance = msg.angular_velocity_covariance
        out.linear_acceleration_covariance = msg.linear_acceleration_covariance

        av_in = (msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z)
        if self.negate_gyro_before_rotation:
            av_in = (-av_in[0], -av_in[1], -av_in[2])
        la_in = (msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z)

        av_r = rotate_vec(self.rotation, av_in)
        la_r = rotate_vec(self.rotation, la_in)

        if self.normalize_acceleration:
            la_norm = math.sqrt(la_r[0] * la_r[0] + la_r[1] * la_r[1] + la_r[2] * la_r[2])
            if la_norm > 1e-9:
                target_norm = self.gravity_magnitude if self.acceleration_mode == "gravity" else 1.0
                scale = target_norm / la_norm
                la_r = (la_r[0] * scale, la_r[1] * scale, la_r[2] * scale)
        if self.acceleration_mode == "g_unit" and self.clip_acceleration_to_unit:
            la_r = (
                max(-1.0, min(1.0, la_r[0])),
                max(-1.0, min(1.0, la_r[1])),
                max(-1.0, min(1.0, la_r[2])),
            )

        la_r = (
            la_r[0] - self.linear_acceleration_bias[0],
            la_r[1] - self.linear_acceleration_bias[1],
            la_r[2] - self.linear_acceleration_bias[2],
        )
        av_r = (
            av_r[0] - self.angular_velocity_bias[0],
            av_r[1] - self.angular_velocity_bias[1],
            av_r[2] - self.angular_velocity_bias[2],
        )

        out.angular_velocity.x, out.angular_velocity.y, out.angular_velocity.z = av_r
        if self.negate_gyro_after_rotation:
            out.angular_velocity.x = -out.angular_velocity.x
            out.angular_velocity.y = -out.angular_velocity.y
            out.angular_velocity.z = -out.angular_velocity.z

        out.linear_acceleration.x, out.linear_acceleration.y, out.linear_acceleration.z = la_r
        self.pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ImuRotateToLidarFrameNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()