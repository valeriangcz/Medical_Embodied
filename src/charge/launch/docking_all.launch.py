import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    launch_dir = os.path.join(get_package_share_directory("charge"), "launch")
    default_params = os.path.join(get_package_share_directory("charge"), "cfg", "docking_params.yaml")

    start_camera_arg = DeclareLaunchArgument(
        "start_camera",
        default_value="true",
        description="Whether to launch usb camera node",
    )
    start_docking_stack_arg = DeclareLaunchArgument(
        "start_docking_stack",
        default_value="true",
        description="Whether to launch docking controller and charge services",
    )
    params_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params,
        description="Path to docking parameter file",
    )
    start_apriltag_arg = DeclareLaunchArgument(
        "start_apriltag",
        default_value="true",
        description="Whether to launch apriltag detector (default on: provides dock_frame TF)",
    )
    enable_apriltag_on_demand_arg = DeclareLaunchArgument(
        "enable_apriltag_on_demand",
        default_value="false",
        description="Enable apriltag start/stop automatically with dock command "
                    "(disabled by default because apriltag is launched statically above)",
    )
    camera_name_arg = DeclareLaunchArgument(
        "camera_name",
        default_value="/charge_cam",
        description="Camera namespace/prefix for apriltag and charge services",
    )
    image_topic_arg = DeclareLaunchArgument(
        "image_topic",
        default_value="image_raw",
        description="Camera image topic suffix under camera_name",
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
    camera_frame_arg = DeclareLaunchArgument("camera_frame", default_value="camera")
    base_frame_arg = DeclareLaunchArgument("base_frame", default_value="base_link1")
    tag_frame_arg = DeclareLaunchArgument("tag_frame", default_value="dock_frame")
    camera_x_arg = DeclareLaunchArgument("camera_x", default_value="-0.30")
    camera_y_arg = DeclareLaunchArgument("camera_y", default_value="0.0")
    camera_z_arg = DeclareLaunchArgument("camera_z", default_value="-0.0")
    camera_roll_arg = DeclareLaunchArgument("camera_roll", default_value="1.5708")
    camera_pitch_arg = DeclareLaunchArgument("camera_pitch", default_value="0.0")
    camera_yaw_arg = DeclareLaunchArgument("camera_yaw", default_value="-1.5708")

    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(launch_dir, "camera.launch.py")),
        condition=IfCondition(LaunchConfiguration("start_camera")),
    )

    docking_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(launch_dir, "start_dock.launch.py")),
        condition=IfCondition(LaunchConfiguration("start_docking_stack")),
        launch_arguments={
            "params_file": LaunchConfiguration("params_file"),
            "start_apriltag": LaunchConfiguration("start_apriltag"),
            "enable_apriltag_on_demand": LaunchConfiguration("enable_apriltag_on_demand"),
            "camera_name": LaunchConfiguration("camera_name"),
            "image_topic": LaunchConfiguration("image_topic"),
            "apriltag_start_retry_count": LaunchConfiguration("apriltag_start_retry_count"),
            "apriltag_start_retry_delay_sec": LaunchConfiguration("apriltag_start_retry_delay_sec"),
            "camera_frame": LaunchConfiguration("camera_frame"),
            "base_frame": LaunchConfiguration("base_frame"),
            "tag_frame": LaunchConfiguration("tag_frame"),
            "camera_x": LaunchConfiguration("camera_x"),
            "camera_y": LaunchConfiguration("camera_y"),
            "camera_z": LaunchConfiguration("camera_z"),
            "camera_roll": LaunchConfiguration("camera_roll"),
            "camera_pitch": LaunchConfiguration("camera_pitch"),
            "camera_yaw": LaunchConfiguration("camera_yaw"),
        }.items(),
    )

    return LaunchDescription(
        [
            start_camera_arg,
            start_docking_stack_arg,
            params_arg,
            start_apriltag_arg,
            enable_apriltag_on_demand_arg,
            camera_name_arg,
            image_topic_arg,
            apriltag_start_retry_count_arg,
            apriltag_start_retry_delay_sec_arg,
            camera_frame_arg,
            base_frame_arg,
            tag_frame_arg,
            camera_x_arg,
            camera_y_arg,
            camera_z_arg,
            camera_roll_arg,
            camera_pitch_arg,
            camera_yaw_arg,
            camera_launch,
            docking_launch,
        ]
    )
