from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


# 默认加载地图名（对应 xjrobot_localization/maps/<MAP_NAME>.yaml）
DEFAULT_MAP_PATH = "/home/medical/maps/map_0828_2.yaml"


def generate_launch_description():
    # Nav2 官方 bringup 入口
    nav2_launch_path = PathJoinSubstitution(
        [FindPackageShare("nav2_bringup"), "launch", "bringup_launch.py"]
    )
    # 将地面分割后的 3D 障碍点云先压成 2D LaserScan，再交给 Nav2 的 costmap
    pointcloud_to_scan_launch_path = PathJoinSubstitution(
        [FindPackageShare("xjrobot_navigation"), "launch", "pointcloud_to_laserscan.launch.py"]
    )

    # map->odom 发布器（由 FastLIO 链路推导）
    # map_odom_tf_launch_path = PathJoinSubstitution(
    #     [FindPackageShare("xjrobot_localization"), "launch", "map_odom_tf.launch.py"]
    # )

    # RViz 配置（延用现有 xjrobot_navigation 配置）
    rviz_config_path = PathJoinSubstitution(
        [FindPackageShare("xjrobot_navigation"), "rviz", "xjrobot_navigation.rviz"]
    )

    # 默认地图改为显式绝对路径，便于实机直接指定源地图文件
    default_map_path = DEFAULT_MAP_PATH

    # Nav2 参数总配置
    nav2_config_path = PathJoinSubstitution(
        [FindPackageShare("xjrobot_navigation"), "config", "RPP_hdl.yaml"]
    )

    # 仅启动 map_server（不启动 AMCL），地图由 3D 定位 + map->odom 维护
    map_server = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[
            {
                "yaml_filename": LaunchConfiguration("map"),
                "use_sim_time": LaunchConfiguration("sim"),
            },
        ],
    )

    lifecycle_manager_localization = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_localization",
        output="screen",
        parameters=[
            {"use_sim_time": LaunchConfiguration("sim")},
            {"autostart": True},
            {"node_names": ["map_server"]},
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                name="sim", default_value="false", description="是否启用仿真时间（use_sim_time）"
            ),
            DeclareLaunchArgument(name="rviz", default_value="true", description="是否启动 RViz2"),
            DeclareLaunchArgument(
                name="map", default_value=default_map_path, description="导航地图 yaml 的绝对路径"
            ),
            # 保留这三个参数，方便后续接不同定位器时复用同一启动接口
            DeclareLaunchArgument(name="initial_pose_x", default_value="0.5"),
            DeclareLaunchArgument(name="initial_pose_y", default_value="0.0"),
            DeclareLaunchArgument(name="initial_pose_yaw", default_value="0.0"),
            # 启动 FastLIO 对齐后的 map->odom 发布
            # IncludeLaunchDescription(
            #     PythonLaunchDescriptionSource(map_odom_tf_launch_path),
            #     launch_arguments={"use_sim_time": LaunchConfiguration("sim")}.items(),
            # ),
            # 这里单独放在导航 launch 中，而不是耦合进底盘 bringup：
            # 这样导航是否使用 2D scan 化处理，可以作为导航侧策略独立切换。
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(pointcloud_to_scan_launch_path),
                launch_arguments={
                    "pointcloud_topic": "/ground_segmentation/obstacle_points",
                    "scan_topic": "/scan",
                    "target_frame": "rslidar",
                }.items(),
            ),
            # 地图服务（仅 map_server）
            map_server,
            lifecycle_manager_localization,
            # 启动 Nav2 主栈，但关闭官方 localization（避免 AMCL 与 3D 定位冲突）
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(nav2_launch_path),
                launch_arguments={
                    "map": LaunchConfiguration("map"),
                    "use_sim_time": LaunchConfiguration("sim"),
                    "params_file": nav2_config_path,
                    "use_localization": "False",
                }.items(),
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", rviz_config_path],
                condition=IfCondition(LaunchConfiguration("rviz")),
                parameters=[{"use_sim_time": LaunchConfiguration("sim")}],
            ),
        ]
    )
