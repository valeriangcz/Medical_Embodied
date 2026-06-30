import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('fdlink_ahrs')
    default_config = os.path.join(pkg_share, 'config', 'imu_filter.yaml')

    config_file = LaunchConfiguration('config_file')
    input_topic = LaunchConfiguration('input_topic')
    filtered_topic = LaunchConfiguration('filtered_topic')
    output_topic = LaunchConfiguration('output_topic')

    imu_filter_node = Node(
        package='imu_filter_madgwick',
        executable='imu_filter_madgwick_node',
        name='imu_filter',
        output='screen',
        parameters=[config_file],
        remappings=[
            ('imu/data_raw', input_topic),
            ('imu/data', filtered_topic),
        ],
    )

    imu_throttle_node = Node(
        package='fdlink_ahrs',
        executable='imu_throttle_node',
        name='imu_throttle',
        output='screen',
        parameters=[
            config_file,
            {
                'input_topic': filtered_topic,
                'output_topic': output_topic,
            },
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config,
            description='Path to imu_filter and throttle parameter file',
        ),
        DeclareLaunchArgument(
            'input_topic',
            default_value='/imu/link1',
            description='Raw IMU topic published by fdlink_ahrs driver',
        ),
        DeclareLaunchArgument(
            'filtered_topic',
            default_value='/imu/data_filtered',
            description='Internal topic between imu_filter and imu_throttle',
        ),
        DeclareLaunchArgument(
            'output_topic',
            default_value='/imu/data',
            description='Throttled filtered IMU output topic',
        ),
        imu_filter_node,
        # imu_throttle_node,
    ])
