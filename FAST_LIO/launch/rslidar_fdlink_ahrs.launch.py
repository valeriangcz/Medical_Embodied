import os.path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    fast_lio_share = get_package_share_directory("fast_lio")
    default_config_path = os.path.join(fast_lio_share, "config")
    default_rviz_config_path = os.path.join(fast_lio_share, "rviz", "fastlio.rviz")

    use_sim_time = LaunchConfiguration("use_sim_time")
    config_path = LaunchConfiguration("config_path")
    config_file = LaunchConfiguration("config_file")
    rviz_use = LaunchConfiguration("rviz")
    rviz_cfg = LaunchConfiguration("rviz_cfg")
    enable_rslidar = LaunchConfiguration("enable_rslidar")
    enable_ahrs = LaunchConfiguration("enable_ahrs")
    enable_static_tf = LaunchConfiguration("enable_static_tf")
    serial_port = LaunchConfiguration("serial_port")

    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time", default_value="false", description="Use simulation clock if true"
    )
    declare_config_path = DeclareLaunchArgument(
        "config_path", default_value=default_config_path, description="YAML config directory"
    )
    declare_config_file = DeclareLaunchArgument(
        "config_file",
        default_value="rslidar_fdlink_ahrs.yaml",
        description="FAST-LIO config: /rslidar_points + /imu/base_link",
    )
    declare_rviz = DeclareLaunchArgument(
        "rviz", default_value="true", description="Launch RViz2"
    )
    declare_rviz_cfg = DeclareLaunchArgument(
        "rviz_cfg", default_value=default_rviz_config_path, description="RViz config file"
    )
    declare_enable_rslidar = DeclareLaunchArgument(
        "enable_rslidar",
        default_value="true",
        description="Start rslidar_sdk (publishes /rslidar_points)",
    )
    declare_enable_ahrs = DeclareLaunchArgument(
        "enable_ahrs",
        default_value="true",
        description="Start fdlink ahrs_driver + imu_to_base_link (/imu/base_link)",
    )
    declare_enable_static_tf = DeclareLaunchArgument(
        "enable_static_tf",
        default_value="true",
        description="Start fdlink static_transforms_3 (base_link/imu_link TF)",
    )
    declare_serial_port = DeclareLaunchArgument(
        "serial_port", default_value="/dev/ttyUSB1", description="fdlink AHRS serial port"
    )

    rslidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("rslidar_sdk"), "launch", "start.py")
        ),
        condition=IfCondition(enable_rslidar),
    )

    ahrs_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("fdlink_ahrs"), "launch", "ahrs_driver.launch.py")
        ),
        launch_arguments={
            "serial_port": serial_port,
            "imu_topic": "/imu",
            "imu_frame_id": "rslidar",
            "imu_base_link_topic": "/imu/base_link",
            "use_calibrated_rotation": "true",
        }.items(),
        condition=IfCondition(enable_ahrs),
    )

    static_tf_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("fdlink_ahrs"), "launch", "static_transforms_3.launch.py"
            )
        ),
        condition=IfCondition(enable_static_tf),
    )

    fast_lio_node = Node(
        package="fast_lio",
        executable="fastlio_mapping",
        parameters=[PathJoinSubstitution([config_path, config_file]), {"use_sim_time": use_sim_time}],
        output="screen",
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", rviz_cfg],
        condition=IfCondition(rviz_use),
    )

    return LaunchDescription(
        [
            declare_use_sim_time,
            declare_config_path,
            declare_config_file,
            declare_rviz,
            declare_rviz_cfg,
            declare_enable_rslidar,
            declare_enable_ahrs,
            declare_enable_static_tf,
            declare_serial_port,
            rslidar_launch,
            ahrs_launch,
            static_tf_launch,
            fast_lio_node,
            rviz_node,
        ]
    )
