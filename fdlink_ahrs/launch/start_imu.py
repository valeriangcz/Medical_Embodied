import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('fdlink_ahrs')
    launch_dir = os.path.join(pkg_share, 'launch')

    serial_port = LaunchConfiguration('serial_port')
    serial_baud = LaunchConfiguration('serial_baud')
    imu_topic = LaunchConfiguration('imu_topic')
    imu_frame_id = LaunchConfiguration('imu_frame_id')
    imu_base_link_topic = LaunchConfiguration('imu_base_link_topic')
    imu_base_link_frame_id = LaunchConfiguration('imu_base_link_frame_id')
    use_calibrated_rotation = LaunchConfiguration('use_calibrated_rotation')

    ahrs_driver = Node(
        package='fdlink_ahrs',
        executable='ahrs_driver_node',
        parameters=[{
            'if_debug': False,
            'serial_port': serial_port,
            'serial_baud': serial_baud,
            'imu_topic': imu_topic,
            'imu_frame_id': imu_frame_id,
            'mag_pose_2d_topic': '/mag_pose_2d',
            'device_type': 1,
        }],
        output='screen',
    )


    rotate_imu = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_dir, 'rotate_imu.launch.py'),
        ),
        launch_arguments={
            'input_topic': imu_topic,
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument('serial_port', default_value='/dev/imu'),
        DeclareLaunchArgument('serial_baud', default_value='921600'),
        DeclareLaunchArgument('imu_topic', default_value='/imu'),
        DeclareLaunchArgument('imu_frame_id', default_value='imu_link'),
        DeclareLaunchArgument('imu_base_link_topic', default_value='/imu/base_link'),
        DeclareLaunchArgument('imu_base_link_frame_id', default_value='base_link'),
        DeclareLaunchArgument('use_calibrated_rotation', default_value='true'),
        ahrs_driver,
        rotate_imu,
    ])
