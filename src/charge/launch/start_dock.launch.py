import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory("charge"), "cfg", "docking_params.yaml"
    )
    apriltag_launch_file = os.path.join(
        get_package_share_directory("charge"), "launch", "tag_realsense_node.launch.py"
    )
    default_apriltag_params = os.path.join(
        get_package_share_directory("charge"), "cfg", "tags_36h11_node.yaml"
    )

    params_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params,
        description="Path to docking parameter file",
    )
    camera_frame_arg = DeclareLaunchArgument(
        "camera_frame",
        default_value="camera",
        description="Camera frame id",
    )
    base_frame_arg = DeclareLaunchArgument(
        "base_frame",
        default_value="base_link",
        description="Robot base frame id",
    )
    tag_frame_arg = DeclareLaunchArgument(
        "tag_frame",
        default_value="dock_frame",
        description="AprilTag child frame id from apriltag_ros",
    )
    start_apriltag_arg = DeclareLaunchArgument(
        "start_apriltag",
        default_value="true",
        description="Whether to launch apriltag detector node (default on: provides dock_frame TF)",
    )
    image_topic_arg = DeclareLaunchArgument(
        "image_topic",
        default_value="image_raw",
        description="Camera image topic suffix under camera_name",
    )
    camera_name_arg = DeclareLaunchArgument(
        "camera_name",
        default_value="/charge_cam",
        description="Camera namespace/prefix used by apriltag launch",
    )
    enable_apriltag_on_demand_arg = DeclareLaunchArgument(
        "enable_apriltag_on_demand",
        default_value="false",
        description="Start/stop apriltag detector only during docking "
                    "(disabled by default because apriltag is launched statically above)",
    )
    apriltag_launch_package_arg = DeclareLaunchArgument(
        "apriltag_launch_package",
        default_value="charge",
        description="Package that provides apriltag launch file",
    )
    apriltag_launch_file_arg = DeclareLaunchArgument(
        "apriltag_launch_file",
        default_value="tag_realsense_node.launch.py",
        description="Apriltag launch filename used by charge_services",
    )
    apriltag_startup_delay_sec_arg = DeclareLaunchArgument(
        "apriltag_startup_delay_sec",
        default_value="1.0",
        description="Delay after apriltag launch before docking starts",
    )
    apriltag_stop_timeout_sec_arg = DeclareLaunchArgument(
        "apriltag_stop_timeout_sec",
        default_value="3.0",
        description="Timeout when stopping apriltag detector process",
    )
    apriltag_start_retry_count_arg = DeclareLaunchArgument(
        "apriltag_start_retry_count",
        default_value="2",
        description="Retry count when apriltag process exits on startup",
    )
    apriltag_start_retry_delay_sec_arg = DeclareLaunchArgument(
        "apriltag_start_retry_delay_sec",
        default_value="0.8",
        description="Delay between apriltag startup retries",
    )
    docking_session_cooldown_sec_arg = DeclareLaunchArgument(
        "docking_session_cooldown_sec",
        default_value="5.0",
        description="Minimum seconds between docking stop and next start",
    )
    camera_x_arg = DeclareLaunchArgument("camera_x", default_value="-0.30")
    camera_y_arg = DeclareLaunchArgument("camera_y", default_value="-0.0")
    camera_z_arg = DeclareLaunchArgument("camera_z", default_value="-0.0")
    camera_roll_arg = DeclareLaunchArgument("camera_roll", default_value="1.5708")
    camera_pitch_arg = DeclareLaunchArgument("camera_pitch", default_value="0.0")
    camera_yaw_arg = DeclareLaunchArgument("camera_yaw", default_value="-1.5708")

    return LaunchDescription(
        [
            params_arg,
            camera_frame_arg,
            base_frame_arg,
            tag_frame_arg,
            start_apriltag_arg,
            image_topic_arg,
            camera_name_arg,
            enable_apriltag_on_demand_arg,
            apriltag_launch_package_arg,
            apriltag_launch_file_arg,
            apriltag_startup_delay_sec_arg,
            apriltag_stop_timeout_sec_arg,
            apriltag_start_retry_count_arg,
            apriltag_start_retry_delay_sec_arg,
            docking_session_cooldown_sec_arg,
            camera_x_arg,
            camera_y_arg,
            camera_z_arg,
            camera_roll_arg,
            camera_pitch_arg,
            camera_yaw_arg,
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(apriltag_launch_file),
                condition=IfCondition(LaunchConfiguration("start_apriltag")),
                launch_arguments={
                    "camera_name": LaunchConfiguration("camera_name"),
                    "image_topic": LaunchConfiguration("image_topic"),
                    "apriltag_params_file": default_apriltag_params,
                }.items(),
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="cam_to_base",
                arguments=[
                    LaunchConfiguration("camera_x"),
                    LaunchConfiguration("camera_y"),
                    LaunchConfiguration("camera_z"),
                    LaunchConfiguration("camera_roll"),
                    LaunchConfiguration("camera_pitch"),
                    LaunchConfiguration("camera_yaw"),
                    LaunchConfiguration("base_frame"),
                    LaunchConfiguration("camera_frame"),
                ],
            ),
            Node(
                package="charge",
                executable="qr_docking_node",
                name="qr_docking_node",
                output="screen",
                parameters=[
                    LaunchConfiguration("params_file"),
                    {
                        "base_frame": LaunchConfiguration("base_frame"),
                        "tag_frame": LaunchConfiguration("tag_frame"),
                    },
                ],
            ),
            Node(
                package="charge",
                executable="charge_services",
                name="charge_services",
                output="screen",
                parameters=[
                    LaunchConfiguration("params_file"),
                    {
                        "enable_apriltag_on_demand": LaunchConfiguration("enable_apriltag_on_demand"),
                        "apriltag_launch_package": LaunchConfiguration("apriltag_launch_package"),
                        "apriltag_launch_file": LaunchConfiguration("apriltag_launch_file"),
                        "apriltag_camera_name": LaunchConfiguration("camera_name"),
                        "apriltag_image_topic": LaunchConfiguration("image_topic"),
                        "apriltag_params_file": default_apriltag_params,
                        "apriltag_startup_delay_sec": LaunchConfiguration("apriltag_startup_delay_sec"),
                        "apriltag_stop_timeout_sec": LaunchConfiguration("apriltag_stop_timeout_sec"),
                        "apriltag_start_retry_count": LaunchConfiguration("apriltag_start_retry_count"),
                        "apriltag_start_retry_delay_sec": LaunchConfiguration(
                            "apriltag_start_retry_delay_sec"
                        ),
                        "docking_session_cooldown_sec": LaunchConfiguration(
                            "docking_session_cooldown_sec"
                        ),
                    }
                ],
            ),
        ]
    )
