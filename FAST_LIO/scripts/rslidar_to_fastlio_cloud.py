#!/usr/bin/env python3

import copy
import math
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2


class RslidarToFastlioCloud(Node):
    def __init__(self) -> None:
        super().__init__("rslidar_to_fastlio_cloud")

        self.declare_parameter("input_topic", "/rslidar_points")
        self.declare_parameter("output_topic", "/rslidar_points_fastlio")
        self.declare_parameter("raw_output_topic", "/rslidar_points_fastlio_frame")
        self.declare_parameter("target_frame_id", "lidar_3d_frame_fastlio")

        input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        raw_output_topic = self.get_parameter("raw_output_topic").get_parameter_value().string_value
        self.target_frame_id = (
            self.get_parameter("target_frame_id").get_parameter_value().string_value
        )
        if not raw_output_topic:
            raw_output_topic = f"{output_topic}_frame"

        self.sub = self.create_subscription(
            PointCloud2, input_topic, self.cloud_callback, qos_profile_sensor_data
        )
        # Keep subscriber as sensor_data; publish with Reliable QoS for RViz/tools compatibility.
        reliable_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.pub = self.create_publisher(PointCloud2, output_topic, reliable_qos)
        self.raw_pub = self.create_publisher(PointCloud2, raw_output_topic, reliable_qos)

        self.expected_fields = ("x", "y", "z", "intensity", "ring", "timestamp")
        self.output_fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="ring", offset=16, datatype=PointField.UINT16, count=1),
            PointField(name="time", offset=20, datatype=PointField.FLOAT32, count=1),
        ]

        self.get_logger().info(
            f"rslidar->fastlio cloud converter started. input={input_topic}, output={output_topic}, "
            f"raw_frame_output={raw_output_topic}, frame_id={self.target_frame_id}"
        )

    def _has_required_fields(self, msg: PointCloud2) -> bool:
        names = {f.name for f in msg.fields}
        return all(name in names for name in self.expected_fields)

    def cloud_callback(self, msg: PointCloud2) -> None:
        # Always republish raw cloud with target frame, independent from FAST-LIO field requirements.
        stamp = self.get_clock().now().to_msg()
        raw_msg = copy.deepcopy(msg)
        raw_msg.header.frame_id = "lidar_3d_link"
        raw_msg.header.stamp = stamp
        self.raw_pub.publish(raw_msg)

        if not self._has_required_fields(msg):
            names = [f.name for f in msg.fields]
            self.get_logger().warn(
                f"Input cloud missing required fields {self.expected_fields}, got={names}"
            )
            return

        points_in = list(
            point_cloud2.read_points(
                msg,
                field_names=self.expected_fields,
                skip_nans=True,
            )
        )
        if not points_in:
            return

        ts_min = min(p[5] for p in points_in if math.isfinite(p[5]))
        points_out: List[Tuple[float, float, float, float, int, float]] = []
        for x, y, z, intensity, ring, timestamp in points_in:
            if not math.isfinite(timestamp):
                continue
            # FAST-LIO velodyne_handler: curvature = time_field / 1000 (ms).
            # So time_field must be in microseconds; IMU undistortion then uses curvature as ms.
            time_offset_us = float(timestamp - ts_min) * 1e6
            points_out.append(
                (float(x), float(y), float(z), float(intensity), int(ring), time_offset_us)
            )

        if not points_out:
            return

        out_msg = point_cloud2.create_cloud(msg.header, self.output_fields, points_out)
        out_msg.header.frame_id = self.target_frame_id
        out_msg.height = 1
        out_msg.width = len(points_out)
        out_msg.is_bigendian = False
        out_msg.is_dense = True
        # out_msg.header.stamp = stamp
        self.pub.publish(out_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RslidarToFastlioCloud()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        # ROS launch/signal handler may already shutdown context.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
