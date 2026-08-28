from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.substitutions import FindPackageShare


# 实机默认地图：
# - DEFAULT_MAP_PATH：Nav2 使用的 2D 栅格地图 yaml
# - DEFAULT_PCD_MAP_PATH：FastLIO / 全局重定位使用的 3D PCD 地图
DEFAULT_MAP_PATH = "/home/medical/maps/map_0828_2.yaml"
DEFAULT_PCD_MAP_PATH = "/home/medical/maps/map_0828_2.pcd"


def generate_launch_description():
    # 本包只做系统级装配，不直接启动业务节点。
    # 各子系统的具体节点和参数仍由自己的 launch / config 维护，避免职责混杂。
    base_launch = PathJoinSubstitution(
        [FindPackageShare("xjrobot_base"), "launch", "base_core.launch.py"]
    )
    localization_launch = PathJoinSubstitution(
        [FindPackageShare("xjrobot_localization"), "launch", "localization.launch.py"]
    )
    localization_hdl_launch = PathJoinSubstitution(
        [FindPackageShare("xjrobot_localization"), "launch", "localization_hdl.launch.py"]
    )
    navigation_rpp_launch = PathJoinSubstitution(
        [FindPackageShare("xjrobot_navigation"), "launch", "navigation_rpp.launch.py"]
    )
    navigation_mppi_launch = PathJoinSubstitution(
        [FindPackageShare("xjrobot_navigation"), "launch", "navigation_mppi.launch.py"]
    )
    bridge_launch = PathJoinSubstitution(
        [FindPackageShare("xjrobot_bridge"), "launch", "xjrobot_bridge.launch.py"]
    )
    rslidar_mapping_launch = PathJoinSubstitution(
        [FindPackageShare("fast_lio"), "launch", "rslidar_mapping.launch.py"]
    )
    default_waypoints_file = PathJoinSubstitution(
        [FindPackageShare("xjrobot_bridge"), "config", "waypoints.yaml"]
    )

    use_sim_time = LaunchConfiguration("use_sim_time")
    controller = LaunchConfiguration("controller")
    localization_backend = LaunchConfiguration("localization_backend")

    # E1R 雷达与 IMU 驱动链路：
    # rslidar_sdk、fdlink IMU、点云转换由 fast_lio/rslidar_mapping.launch.py 维护。
    # run_fastlio 默认 false，FastLIO 本体仍由 localization.launch.py 启动。
    rslidar_mapping_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rslidar_mapping_launch),
        condition=IfCondition(LaunchConfiguration("launch_rslidar")),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "run_fastlio": LaunchConfiguration("rslidar_run_fastlio"),
            "rviz": LaunchConfiguration("rslidar_rviz"),
            "launch_imu_rotate": LaunchConfiguration("rslidar_launch_imu_rotate"),
        }.items(),
    )

    # 底盘与传感器链路：
    # CAN 底盘、手柄、E1R、点云地面分割、robot_state_publisher、EKF 都由 base.launch.py 维护。
    base_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(base_launch),
        condition=IfCondition(LaunchConfiguration("launch_base")),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "publish_joints": LaunchConfiguration("publish_joints"),
            "launch_joy": LaunchConfiguration("launch_joy"),
            "joy_dev": LaunchConfiguration("joy_dev"),
            "launch_ground_segmentation": LaunchConfiguration("launch_ground_segmentation"),
        }.items(),
    )

    # 3D 定位链路：
    # FastLIO、global localization、map->odom 融合和 PCD 地图发布都由 localization.launch.py 维护。
    # pcd_map 在这里作为总入口参数传入，覆盖 localization 配置文件里的默认 map_file_path。
    localization_fastlio_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(localization_launch),
        condition=IfCondition(
            PythonExpression(
                [
                    "'",
                    LaunchConfiguration("launch_localization"),
                    "' == 'true' and '",
                    localization_backend,
                    "' == 'fastlio'",
                ]
            )
        ),
        launch_arguments={
            "config_file": LaunchConfiguration("localization_config"),
            "pcd_map": LaunchConfiguration("pcd_map"),
            "use_sim_time": use_sim_time,
            "rviz": LaunchConfiguration("localization_rviz"),
            "publish_pcd_map": LaunchConfiguration("publish_pcd_map"),
            "launch_map_odom_tf": LaunchConfiguration("launch_map_odom_tf"),
            "publish_initial_pose": LaunchConfiguration("publish_initial_pose"),
            "initial_pose_x": LaunchConfiguration("initial_pose_x"),
            "initial_pose_y": LaunchConfiguration("initial_pose_y"),
            "initial_pose_yaw": LaunchConfiguration("initial_pose_yaw"),
            "initial_pose_delay": LaunchConfiguration("initial_pose_delay"),
        }.items(),
    )

    localization_hdl_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(localization_hdl_launch),
        condition=IfCondition(
            PythonExpression(
                [
                    "'",
                    LaunchConfiguration("launch_localization"),
                    "' == 'true' and '",
                    localization_backend,
                    "' == 'hdl'",
                ]
            )
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "pcd_map": LaunchConfiguration("pcd_map"),
            "rviz": LaunchConfiguration("localization_rviz"),
            "robot_odom_frame_id": LaunchConfiguration("robot_odom_frame_id"),
            "odom_child_frame_id": LaunchConfiguration("odom_child_frame_id"),
            "publish_initial_pose": LaunchConfiguration("publish_initial_pose"),
            "initial_pose_x": LaunchConfiguration("initial_pose_x"),
            "initial_pose_y": LaunchConfiguration("initial_pose_y"),
            "initial_pose_yaw": LaunchConfiguration("initial_pose_yaw"),
            "initial_pose_delay": LaunchConfiguration("initial_pose_delay"),
            "base_to_lidar_x": LaunchConfiguration("base_to_lidar_x"),
            "base_to_lidar_y": LaunchConfiguration("base_to_lidar_y"),
            "base_to_lidar_z": LaunchConfiguration("base_to_lidar_z"),
            "base_to_lidar_qx": LaunchConfiguration("base_to_lidar_qx"),
            "base_to_lidar_qy": LaunchConfiguration("base_to_lidar_qy"),
            "base_to_lidar_qz": LaunchConfiguration("base_to_lidar_qz"),
            "base_to_lidar_qw": LaunchConfiguration("base_to_lidar_qw"),
        }.items(),
    )

    # Nav2 导航链路：
    # RPP 与 MPPI 只差控制器参数和对应 launch，外部统一通过 controller:=rpp|mppi 选择。
    # navigation_rpp/mppi.launch.py 内部参数名仍叫 sim，这里统一把总入口的 use_sim_time 转发过去。
    rpp_navigation_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(navigation_rpp_launch),
        condition=IfCondition(PythonExpression(["'", controller, "' == 'rpp'"])),
        launch_arguments={
            "sim": use_sim_time,
            "rviz": LaunchConfiguration("navigation_rviz"),
            "map": LaunchConfiguration("map"),
            "initial_pose_x": LaunchConfiguration("initial_pose_x"),
            "initial_pose_y": LaunchConfiguration("initial_pose_y"),
            "initial_pose_yaw": LaunchConfiguration("initial_pose_yaw"),
        }.items(),
    )

    mppi_navigation_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(navigation_mppi_launch),
        condition=IfCondition(PythonExpression(["'", controller, "' == 'mppi'"])),
        launch_arguments={
            "sim": use_sim_time,
            "rviz": LaunchConfiguration("navigation_rviz"),
            "map": LaunchConfiguration("map"),
            "initial_pose_x": LaunchConfiguration("initial_pose_x"),
            "initial_pose_y": LaunchConfiguration("initial_pose_y"),
            "initial_pose_yaw": LaunchConfiguration("initial_pose_yaw"),
        }.items(),
    )

    # 上层任务桥接：
    # xjrobot_bridge 将业务层自定义导航 action 转换为 Nav2 action。
    # 它依赖 Nav2 action server 已经起来，因此在总 launch 末尾延迟启动。
    bridge_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(bridge_launch),
        launch_arguments={
            "waypoints_file": LaunchConfiguration("waypoints_file"),
        }.items(),
    )

    return LaunchDescription(
        [
            # 导航控制器选择。该参数只决定 include 哪个导航 launch，不改变其他子系统。
            DeclareLaunchArgument(
                "controller",
                default_value="rpp",
                choices=["rpp", "mppi"],
                description="Nav2 local controller profile to launch: rpp or mppi.",
            ),
            # 全系统统一时间源。实机默认 false，仿真或 rosbag 回放时再改 true。
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description="Use simulation clock for all included subsystems.",
            ),
            # 2D 地图给 Nav2 map_server 使用；3D 地图给 FastLIO / global localization 使用。
            # 两者通常同名不同后缀，但分开暴露，避免换图时隐式耦合。
            DeclareLaunchArgument(
                "map",
                default_value=DEFAULT_MAP_PATH,
                description="2D occupancy map yaml used by Nav2 map_server.",
            ),
            DeclareLaunchArgument(
                "pcd_map",
                default_value=DEFAULT_PCD_MAP_PATH,
                description="3D PCD map used by FastLIO localization and PCD map publisher.",
            ),
            # localization_config 仍作为定位参数总入口；pcd_map 只覆盖其中的地图路径。
            DeclareLaunchArgument(
                "localization_config",
                default_value="e1r.yaml",
                description="xjrobot_localization config file name under its config directory.",
            ),
            DeclareLaunchArgument(
                "localization_backend",
                default_value="fastlio",
                choices=["fastlio", "hdl"],
                description="Localization backend profile: fastlio or hdl.",
            ),
            DeclareLaunchArgument(
                "base_to_lidar_x",
                default_value="0.17262",
                description="For HDL backend: base_link->rslidar translation x (m).",
            ),
            DeclareLaunchArgument(
                "base_to_lidar_y",
                default_value="0.0",
                description="For HDL backend: base_link->rslidar translation y (m).",
            ),
            DeclareLaunchArgument(
                "base_to_lidar_z",
                default_value="0.0",
                description="For HDL backend: base_link->rslidar translation z (m).",
            ),
            DeclareLaunchArgument(
                "base_to_lidar_qx",
                default_value="0.0",
                description="For HDL backend: base_link->rslidar quaternion qx.",
            ),
            DeclareLaunchArgument(
                "base_to_lidar_qy",
                default_value="0.0",
                description="For HDL backend: base_link->rslidar quaternion qy.",
            ),
            DeclareLaunchArgument(
                "base_to_lidar_qz",
                default_value="0.0",
                description="For HDL backend: base_link->rslidar quaternion qz.",
            ),
            DeclareLaunchArgument(
                "base_to_lidar_qw",
                default_value="1.0",
                description="For HDL backend: base_link->rslidar quaternion qw.",
            ),
            DeclareLaunchArgument(
                "robot_odom_frame_id",
                default_value="odom",
                description="For HDL backend: odometry frame id.",
            ),
            DeclareLaunchArgument(
                "odom_child_frame_id",
                default_value="base_link",
                description="For HDL backend: odometry child frame id.",
            ),
            DeclareLaunchArgument(
                "waypoints_file",
                default_value=default_waypoints_file,
                description="xjrobot_bridge waypoint config file.",
            ),
            # 子系统开关用于实机调试和分段 bringup：
            # 例如只调定位时可 launch_navigation:=false launch_bridge:=false。
            DeclareLaunchArgument(
                "launch_rslidar",
                default_value="true",
                description="Launch E1R lidar driver, IMU and pointcloud converter via fast_lio/rslidar_mapping.launch.py.",
            ),
            DeclareLaunchArgument(
                "rslidar_run_fastlio",
                default_value="false",
                description=(
                    "Pass run_fastlio to rslidar_mapping.launch.py. "
                    "Keep false when localization.launch.py runs FastLIO."
                ),
            ),
            DeclareLaunchArgument(
                "rslidar_rviz",
                default_value="false",
                description="Pass rviz to rslidar_mapping.launch.py (only applies when rslidar_run_fastlio is true).",
            ),
            DeclareLaunchArgument(
                "rslidar_launch_imu_rotate",
                default_value="false",
                description=(
                    "Pass launch_imu_rotate to rslidar_mapping.launch.py. "
                    "Set true if no other launch already runs imu_rotate_node."
                ),
            ),
            DeclareLaunchArgument(
                "launch_base",
                default_value="true",
                description="Launch chassis, lidar, ground segmentation, robot state publisher and EKF.",
            ),
            DeclareLaunchArgument(
                "launch_localization",
                default_value="true",
                description="Launch FastLIO localization and map->odom fusion.",
            ),
            DeclareLaunchArgument(
                "launch_navigation",
                default_value="true",
                description="Launch Nav2 using the selected controller profile.",
            ),
            DeclareLaunchArgument(
                "launch_bridge",
                default_value="true",
                description="Launch the application-to-Nav2 bridge node.",
            ),
            # RViz 分开控制，避免同时打开多个 RViz 占资源。
            # 默认只开导航 RViz；定位 RViz 需要调试 FastLIO/PCD 地图时再打开。
            DeclareLaunchArgument(
                "navigation_rviz",
                default_value="true",
                description="Launch RViz from the navigation package.",
            ),
            DeclareLaunchArgument(
                "localization_rviz",
                default_value="false",
                description="Launch RViz from the localization package.",
            ),
            # base.launch.py 的硬件/传感器开关原样透出，便于实机现场按需裁剪。
            DeclareLaunchArgument(
                "publish_joints",
                default_value="false",
                description="Launch joint_state_publisher from base.launch.py.",
            ),
            DeclareLaunchArgument(
                "launch_joy",
                default_value="true",
                description="Launch joystick input from base.launch.py.",
            ),
            DeclareLaunchArgument(
                "joy_dev",
                default_value="/dev/input/js0",
                description="Joystick device path.",
            ),
            DeclareLaunchArgument(
                "launch_ground_segmentation",
                default_value="true",
                description="Launch ground segmentation from base.launch.py.",
            ),
            # 3D 地图点云和 map->odom 融合开关。
            DeclareLaunchArgument(
                "publish_pcd_map",
                default_value="true",
                description="Publish the PCD map pointcloud from localization.launch.py.",
            ),
            DeclareLaunchArgument(
                "launch_map_odom_tf",
                default_value="true",
                description="Launch map->odom TF fusion from localization.launch.py.",
            ),
            # 初始位姿用于 global localization 启动后自动给一次先验。
            DeclareLaunchArgument(
                "publish_initial_pose",
                default_value="true",
                description="Publish one initial pose for global localization after startup.",
            ),
            DeclareLaunchArgument("initial_pose_x", default_value="0.0"),
            DeclareLaunchArgument("initial_pose_y", default_value="0.0"),
            DeclareLaunchArgument("initial_pose_yaw", default_value="0.0"),
            DeclareLaunchArgument(
                "initial_pose_delay",
                default_value="2.0",
                description="Delay before publishing the initial pose, seconds.",
            ),
            rslidar_mapping_include,
            base_include,
            localization_fastlio_include,
            localization_hdl_include,
            # Nav2 依赖 /tf、/odom、/scan、map->odom 等链路。
            # 延迟启动能减少实机开机阶段的 TF / costmap 报警，让生命周期节点更稳。
            TimerAction(
                period=5.0,
                condition=IfCondition(LaunchConfiguration("launch_navigation")),
                actions=[
                    rpp_navigation_include,
                    mppi_navigation_include,
                ],
            ),
            # 桥接节点依赖 Nav2 action server，放在 Nav2 之后启动。
            TimerAction(
                period=8.0,
                condition=IfCondition(LaunchConfiguration("launch_bridge")),
                actions=[bridge_include],
            ),
        ]
    )
