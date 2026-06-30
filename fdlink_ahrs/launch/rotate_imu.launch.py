from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    input_topic = LaunchConfiguration('input_topic')
    output_topic = LaunchConfiguration('output_topic')
    output_frame_id = LaunchConfiguration('output_frame_id')

    return LaunchDescription([
        DeclareLaunchArgument('input_topic', default_value='/imu'),
        DeclareLaunchArgument('output_topic', default_value='/imu/link1'),
        DeclareLaunchArgument('output_frame_id', default_value='lidar_3d_frame_fastlio'),
        Node(
            package='fdlink_ahrs',
            executable='imu_to_link1_node',
            parameters=[{
                'input_topic': input_topic,
                'output_topic': output_topic,
                'output_frame_id': output_frame_id,
                'rotation_matrix': [
                    -0.000000, -0.283172, 0.959069,
                    1.000000, -0.000000, 0.000000,
                    0.000000, 0.959069, 0.283172,
                ],
            }],
            output='screen',
        ),
    ])
