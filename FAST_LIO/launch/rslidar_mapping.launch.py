import os.path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    package_path = get_package_share_directory("fast_lio")
    default_config_path = os.path.join(package_path, "config")
    default_rviz_config_path = os.path.join(package_path, "rviz", "fastlio.rviz")

    use_sim_time = LaunchConfiguration("use_sim_time")
    config_path = LaunchConfiguration("config_path")
    config_file = LaunchConfiguration("config_file")
    rviz_use = LaunchConfiguration("rviz")
    rviz_cfg = LaunchConfiguration("rviz_cfg")
    startup_delay = LaunchConfiguration("startup_delay")
    run_fastlio = LaunchConfiguration("run_fastlio")
    launch_imu_rotate = LaunchConfiguration("launch_imu_rotate")

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        "use_sim_time", default_value="false", description="Use simulation clock if true"
    )
    declare_config_path_cmd = DeclareLaunchArgument(
        "config_path", default_value=default_config_path, description="Yaml config file path"
    )
    declare_config_file_cmd = DeclareLaunchArgument(
        "config_file", default_value="e1r.yaml", description="Config file"
    )
    declare_rviz_cmd = DeclareLaunchArgument(
        "rviz", default_value="true", description="Use RViz when run_fastlio is true"
    )
    declare_rviz_config_path_cmd = DeclareLaunchArgument(
        "rviz_cfg", default_value=default_rviz_config_path, description="RViz config file path"
    )
    declare_startup_delay_cmd = DeclareLaunchArgument(
        "startup_delay",
        default_value="3.0",
        description="Delay (s) before starting delayed nodes",
    )
    declare_run_fastlio_cmd = DeclareLaunchArgument(
        "run_fastlio",
        default_value="false",
        description=(
            "true: FAST-LIO mapping + RViz; no cloud converter. "
            "false: cloud converter only; pair with xjrobot localization for FastLIO/IMU rotate."
        ),
    )
    declare_launch_imu_rotate_cmd = DeclareLaunchArgument(
        "launch_imu_rotate",
        default_value="false",
        description=(
            "When run_fastlio:=false, optionally start IMU rotate here. "
            "Keep false if robot.launch / localization.launch already runs imu_rotate_node."
        ),
    )

    rslidar_start_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("rslidar_sdk"), "launch", "start.py")
        )
    )

    fdlink_imu_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("fdlink_ahrs"), "launch", "start_imu.py")
        )
    )

    imu_rotate_node = Node(
        package="rslidar_sdk",
        executable="imu_rotate_to_lidar_frame.py",
        output="screen",
        condition=IfCondition(
            PythonExpression(
                [
                    "'",
                    run_fastlio,
                    "' == 'false' and '",
                    launch_imu_rotate,
                    "' == 'true'",
                ]
            )
        ),
        parameters=[
            {
                "input_topic": "/rslidar_imu_data",
                "output_topic": "/rslidar_imu_data_rotated",
                "output_frame_id": "lidar_3d_frame_fastlio",
                "use_direct_quaternion_rotation": True,
                "direct_rotation_quaternion_xyzw": [-0.016609, 0.800555, 0.0, 0.599030],
                "normalize_acceleration": True,
                "gravity_magnitude": 9.81,
                "acceleration_mode": "gravity",
                "negate_gyro_before_rotation": True,
                "linear_acceleration_bias": [0.0, 0.0, 0.0],
                "angular_velocity_bias": [0.0, 0.0, 0.0],
            }
        ],
    )

    converter_node = Node(
        package="fast_lio",
        executable="rslidar_to_fastlio_cloud.py",
        parameters=[
            {
                "input_topic": "/rslidar_points",
                "output_topic": "/rslidar_points_fastlio",
            }
        ],
        output="screen",
        condition=UnlessCondition(run_fastlio),
    )

    fast_lio_node = Node(
        package="fast_lio",
        executable="fastlio_mapping",
        parameters=[PathJoinSubstitution([config_path, config_file]), {"use_sim_time": use_sim_time}],
        output="screen",
        condition=IfCondition(run_fastlio),
    )

    delayed_nodes_group = TimerAction(
        period=startup_delay,
        actions=[imu_rotate_node, converter_node, fast_lio_node],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", rviz_cfg],
        condition=IfCondition(
            PythonExpression(["'", run_fastlio, "' == 'true' and '", rviz_use, "' == 'true'"])
        ),
    )

    ld = LaunchDescription()
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_config_path_cmd)
    ld.add_action(declare_config_file_cmd)
    ld.add_action(declare_rviz_cmd)
    ld.add_action(declare_rviz_config_path_cmd)
    ld.add_action(declare_startup_delay_cmd)
    ld.add_action(declare_run_fastlio_cmd)
    ld.add_action(declare_launch_imu_rotate_cmd)

    ld.add_action(rslidar_start_launch)
    ld.add_action(fdlink_imu_launch)
    ld.add_action(delayed_nodes_group)
    ld.add_action(rviz_node)
    return ld
