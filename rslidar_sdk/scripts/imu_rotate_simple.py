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


def transpose_3x3(m: Matrix3) -> Matrix3:
    return (
        (m[0][0], m[1][0], m[2][0]),
        (m[0][1], m[1][1], m[2][1]),
        (m[0][2], m[1][2], m[2][2]),
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
        raise ValueError("rotation_order must be a 3-letter axis order, e.g. zyx / xyz")

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


class ImuRotateSimpleNode(Node):
    def __init__(self) -> None:
        super().__init__("imu_rotate_simple")

        self.declare_parameter("input_topic", "/rslidar_imu_data")
        self.declare_parameter("output_topic", "/rslidar_imu_data_rotated")
        self.declare_parameter("roll",0.03895)
        self.declare_parameter("pitch",  -1.2782913)
        self.declare_parameter("yaw", 0.0)
        self.declare_parameter("rotation_order", "yxz")
        self.declare_parameter("use_inverse_rotation", False)
        self.declare_parameter("normalize_acceleration", True)
        self.declare_parameter("gravity_magnitude", 9.57)
        # acceleration_mode:
        # - "gravity": output in m/s^2, vector norm -> gravity_magnitude
        # - "g_unit": output in g, vector norm -> 1.0
        self.declare_parameter("acceleration_mode", "g_unit")
        self.declare_parameter("clip_acceleration_to_unit", False)

        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value
        roll = float(self.get_parameter("roll").value)
        pitch = float(self.get_parameter("pitch").value)
        yaw = float(self.get_parameter("yaw").value)
        rotation_order = str(self.get_parameter("rotation_order").value)
        use_inverse = bool(self.get_parameter("use_inverse_rotation").value)
        self.normalize_acceleration = bool(self.get_parameter("normalize_acceleration").value)
        self.gravity_magnitude = float(self.get_parameter("gravity_magnitude").value)
        self.acceleration_mode = str(self.get_parameter("acceleration_mode").value).strip().lower()
        self.clip_acceleration_to_unit = bool(self.get_parameter("clip_acceleration_to_unit").value)
        if self.acceleration_mode not in ("gravity", "g_unit"):
            raise ValueError("acceleration_mode must be 'gravity' or 'g_unit'")

        rotation = build_rotation(roll, pitch, yaw, rotation_order)
        self.rotation = transpose_3x3(rotation) if use_inverse else rotation

        self.sub = self.create_subscription(Imu, input_topic, self.on_imu, 100)
        self.pub = self.create_publisher(Imu, output_topic, 100)

        self.get_logger().info(
            f"imu_rotate_simple started. input={input_topic}, output={output_topic}, "
            f"roll={roll}, pitch={pitch}, yaw={yaw}, order={rotation_order}, inverse={use_inverse}"
        )
        self.get_logger().info(
            f"normalize_acceleration={self.normalize_acceleration}, gravity_magnitude={self.gravity_magnitude}, "
            f"acceleration_mode={self.acceleration_mode}, clip_acceleration_to_unit={self.clip_acceleration_to_unit}"
        )

    def on_imu(self, msg: Imu) -> None:
        out = Imu()
        out.header = msg.header
        out.header.frame_id = "imu_link"
        out.orientation = msg.orientation
        out.orientation_covariance = msg.orientation_covariance
        out.angular_velocity_covariance = msg.angular_velocity_covariance
        out.linear_acceleration_covariance = msg.linear_acceleration_covariance

        av = (-msg.angular_velocity.x, -msg.angular_velocity.y, -msg.angular_velocity.z)
        la = (msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z)
        av_r = rotate_vec(self.rotation, av)
        la_r = rotate_vec(self.rotation, la)
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

        out.angular_velocity.x, out.angular_velocity.y, out.angular_velocity.z = av_r
        out.linear_acceleration.x, out.linear_acceleration.y, out.linear_acceleration.z = la_r
        self.pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ImuRotateSimpleNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
