import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_name = "lidar_imu_init"

    li_node = Node(
        package=package_name,
        name="laser_mapping",
        executable="li_init",
        output="screen",
        parameters=[
            {
                "point_filter_num": 3,
                "max_iteration": 5,
                "cube_side_length": 2000.0,
            },
            os.path.join(get_package_share_directory(package_name), "config", "rse1.yaml"),
        ],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz",
        arguments=[
            "-d",
            PathJoinSubstitution([
                FindPackageShare(package_name),
                "rviz_cfg",
                "airy.rviz",
            ]),
        ],
    )

    return LaunchDescription([li_node, rviz_node])
