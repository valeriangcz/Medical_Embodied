from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='loadconfig',
            executable='loadconfig_server',
            name='loadconfig_server',
            output='screen',
        ),
        Node(
            package='behavior_tree',
            executable='ros_bt_runner',
            name='ros_bt_runner',
            output='screen',
            parameters=[{
                'max_ticks': 1000000,
                'config_id': 'default',
                'battery_low_threshold': 20.0,
            }],
        ),
        # Node(
        #     package='behavior_tree',
        #     executable='medical_bt_ros_test_driver',
        #     name='medical_bt_ros_test_driver',
        #     output='screen',
        #     parameters=[{
        #         # 'start_delay': 1.0,
        #         # 'ticks': 140,
        #         # 'tick_hz': 20,
        #     }],
        # ),
    ])
