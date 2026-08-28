from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


DEFAULT_MAP_PATH = "/home/medical/maps/map_0828_2.yaml"


def generate_launch_description():
    rslidar_mapping_launch = PathJoinSubstitution(
        [FindPackageShare("fast_lio"), "launch", "rslidar_mapping.launch.py"]
    )
    base_hdl_launch = PathJoinSubstitution(
        [FindPackageShare("xjrobot_base"), "launch", "base_core_hdl.launch.py"]
    )
    hdl_localization_launch = PathJoinSubstitution(
        [
            FindPackageShare("hdl_localization"),
            "launch",
            "hdl_localization_rslidar_stable.launch.py",
        ]
    )
    navigation_rpp_hdl_launch = PathJoinSubstitution(
        [FindPackageShare("xjrobot_navigation"), "launch", "nav_rpp_hdl.launch.py"]
    )
    bridge_launch = PathJoinSubstitution(
        [FindPackageShare("xjrobot_bridge"), "launch", "xjrobot_bridge.launch.py"]
    )
    default_waypoints_file = PathJoinSubstitution(
        [FindPackageShare("xjrobot_bridge"), "config", "hdl_waypoints.yaml"]
    )

    use_sim_time = LaunchConfiguration("use_sim_time")

    rslidar_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rslidar_mapping_launch),
        condition=IfCondition(LaunchConfiguration("launch_rslidar")),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "run_fastlio": "false",
            "rviz": "false",
            "launch_imu_rotate": LaunchConfiguration("rslidar_launch_imu_rotate"),
        }.items(),
    )

    base_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(base_hdl_launch),
        condition=IfCondition(LaunchConfiguration("launch_base")),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "launch_joy": LaunchConfiguration("launch_joy"),
            "joy_dev": LaunchConfiguration("joy_dev"),
            "launch_ground_segmentation": LaunchConfiguration("launch_ground_segmentation"),
            "ground_segmentation_pointcloud_topic": LaunchConfiguration(
                "ground_segmentation_pointcloud_topic"
            ),
            "ground_segmentation_imu_topic": LaunchConfiguration(
                "ground_segmentation_imu_topic"
            ),
        }.items(),
    )

    hdl_localization_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(hdl_localization_launch),
        condition=IfCondition(LaunchConfiguration("launch_localization")),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "imu_topic": LaunchConfiguration("hdl_imu_topic"),
            "odom_topic": LaunchConfiguration("hdl_odom_topic"),
            "use_imu": LaunchConfiguration("hdl_use_imu"),
            "use_global_localization": LaunchConfiguration("hdl_use_global_localization"),
            "invert_imu_acc": LaunchConfiguration("hdl_invert_imu_acc"),
            "invert_imu_gyro": LaunchConfiguration("hdl_invert_imu_gyro"),
            "log_output_frequency_hz": LaunchConfiguration("hdl_log_output_frequency_hz"),
        }.items(),
    )

    navigation_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(navigation_rpp_hdl_launch),
        condition=IfCondition(LaunchConfiguration("launch_navigation")),
        launch_arguments={
            "sim": use_sim_time,
            "rviz": LaunchConfiguration("navigation_rviz"),
            "map": LaunchConfiguration("map"),
        }.items(),
    )

    # 上层任务桥接：将业务层自定义 navigate action 转换为 Nav2 navigate_to_pose。
    # 依赖 Nav2 action server 已启动，因此在导航之后延迟拉起。
    bridge_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(bridge_launch),
        launch_arguments={
            "waypoints_file": LaunchConfiguration("waypoints_file"),
        }.items(),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description="Use simulation time for all included subsystems.",
            ),
            DeclareLaunchArgument(
                "map",
                default_value=DEFAULT_MAP_PATH,
                description="2D occupancy map yaml used by Nav2 map_server.",
            ),
            DeclareLaunchArgument(
                "waypoints_file",
                default_value=default_waypoints_file,
                description="xjrobot_bridge waypoint config file.",
            ),
            DeclareLaunchArgument(
                "launch_rslidar",
                default_value="true",
                description="Launch rslidar driver/IMU chain from fast_lio.",
            ),
            DeclareLaunchArgument(
                "rslidar_launch_imu_rotate",
                default_value="false",
                description="Pass launch_imu_rotate to rslidar_mapping.launch.py.",
            ),
            DeclareLaunchArgument(
                "launch_base",
                default_value="true",
                description="Launch HDL-safe chassis and ground segmentation bringup.",
            ),
            DeclareLaunchArgument(
                "launch_joy",
                default_value="true",
                description="Launch joy_node for manual/auto mode switching.",
            ),
            DeclareLaunchArgument(
                "joy_dev",
                default_value="/dev/input/js0",
                description="Joystick device path.",
            ),
            DeclareLaunchArgument(
                "launch_ground_segmentation",
                default_value="true",
                description="Launch ground segmentation for Nav2 /scan generation.",
            ),
            DeclareLaunchArgument(
                "ground_segmentation_pointcloud_topic",
                default_value="/rslidar_points",
                description="PointCloud2 topic consumed by ground segmentation.",
            ),
            DeclareLaunchArgument(
                "ground_segmentation_imu_topic",
                default_value="/imu/link1",
                description="IMU topic consumed by ground segmentation.",
            ),
            DeclareLaunchArgument(
                "launch_localization",
                default_value="true",
                description="Launch HDL localization map->rslidar and rslidar->base_link TF.",
            ),
            DeclareLaunchArgument(
                "hdl_imu_topic",
                default_value="/imu/link1",
                description="IMU topic passed to HDL localization.",
            ),
            DeclareLaunchArgument(
                "hdl_odom_topic",
                default_value="/hdl/odom",
                description="Odometry topic published by HDL localization.",
            ),
            DeclareLaunchArgument("hdl_use_imu", default_value="true"),
            DeclareLaunchArgument("hdl_use_global_localization", default_value="true"),
            DeclareLaunchArgument("hdl_invert_imu_acc", default_value="true"),
            DeclareLaunchArgument("hdl_invert_imu_gyro", default_value="false"),
            DeclareLaunchArgument(
                "hdl_log_output_frequency_hz",
                default_value="1.0",
                description="Throttle repeated hdl_localization logs to this frequency in Hz.",
            ),
            DeclareLaunchArgument(
                "launch_navigation",
                default_value="true",
                description="Launch Nav2 with nav_rpp_hdl.launch.py.",
            ),
            DeclareLaunchArgument(
                "launch_bridge",
                default_value="true",
                description="Launch the application-to-Nav2 bridge node.",
            ),
            DeclareLaunchArgument(
                "navigation_rviz",
                default_value="true",
                description="Launch RViz from the navigation package.",
            ),
            rslidar_include,
            base_include,
            hdl_localization_include,
            TimerAction(
                period=5.0,
                condition=IfCondition(LaunchConfiguration("launch_navigation")),
                actions=[navigation_include],
            ),
            TimerAction(
                period=8.0,
                condition=IfCondition(LaunchConfiguration("launch_bridge")),
                actions=[bridge_include],
            ),
        ]
    )
