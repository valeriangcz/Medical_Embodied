from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

import os
import yaml


# 默认启动参数
DEFAULT_CONFIG_FILE = "e1r.yaml"
DEFAULT_USE_SIM_TIME = "false"
DEFAULT_RVIZ = "true"
DEFAULT_PUBLISH_PCD_MAP = "true"
DEFAULT_LAUNCH_MAP_ODOM_TF = "true"
DEFAULT_PUBLISH_INITIAL_POSE = "true"
DEFAULT_PCD_MAP = ""


def launch_setup(context, *args, **kwargs):
    package_path = get_package_share_directory("xjrobot_localization")
    # 主参数文件占位符，作为整个 localization launch 的统一配置入口
    config_file = LaunchConfiguration("config_file").perform(context)
    localization_config = os.path.join(package_path, "config", config_file)
    map_file_path = os.path.join("/home/medical/maps", "map_0828_2.pcd")

    try:
        with open(localization_config, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        map_file_path = (
            config.get("laser_mapping", {})
            .get("ros__parameters", {})
            .get("map_file_path", map_file_path)
        )
    except OSError:
        pass

    pcd_map_arg = LaunchConfiguration("pcd_map").perform(context).strip()
    if pcd_map_arg:
        map_file_path = pcd_map_arg

    rviz_config = PathJoinSubstitution(
        [FindPackageShare("xjrobot_localization"), "rviz", "fastlio_localization.rviz"]
    )

    map_odom_tf_launch = PathJoinSubstitution(
        [FindPackageShare("xjrobot_localization"), "launch", "map_odom_tf.launch.py"]
    )

    map_odom_tf_params = PathJoinSubstitution(
        [FindPackageShare("xjrobot_localization"), "config", "map_odom_tf.yaml"]
    )

    fast_lio_node = Node(
        package="xjrobot_localization",
        executable="fastlio_mapping",
        name="laser_mapping",
        output="screen",
        parameters=[
            localization_config,
            {
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                # 允许总 bringup 覆盖配置文件中的 laser_mapping.map_file_path
                "map_file_path": map_file_path,
            },
        ],
    )

    global_localization_node = Node(
        package="xjrobot_localization",
        executable="global_localization.py",
        name="global_localization",
        output="screen",
        parameters=[
            localization_config,
            {
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                # 统一复用 laser_mapping.map_file_path
                "pcd_map_path": map_file_path,
            },
        ],
    )

    manual_initial_pose_node = Node(
        package="xjrobot_localization",
        executable="manual_initial_pose_publisher.py",
        name="manual_initial_pose_publisher",
        output="screen",
        condition=IfCondition(LaunchConfiguration("publish_initial_pose")),
        parameters=[
            {
                "frame_id": "map",
                "x": LaunchConfiguration("initial_pose_x"),
                "y": LaunchConfiguration("initial_pose_y"),
                "yaw": LaunchConfiguration("initial_pose_yaw"),
                "publish_delay": LaunchConfiguration("initial_pose_delay"),
            }
        ],
    )

    # FastLIO 输出 /odom_fastlio 后，再由该节点重积分成适合 EKF 融合的平面相对里程计。
    # 由于它严格依赖 localization 链路，因此放在 xjrobot_localization 启动域内。
    fastlio_relative_odom_node = Node(
        package="xjrobot_localization",
        executable="fastlio_relative_odom_node",
        name="xjrobot_fastlio_relative_odom",
        output="screen",
        parameters=[
            localization_config,
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
    )

    transform_fusion_node = Node(
        package="xjrobot_localization",
        executable="transform_fusion.py",
        name="transform_fusion",
        output="screen",
        parameters=[
            localization_config,
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
    )

    pcd_publisher_node = Node(
        package="pcl_ros",
        executable="pcd_to_pointcloud",
        name="map_publisher",
        output="screen",
        condition=IfCondition(LaunchConfiguration("publish_pcd_map")),
        parameters=[
            localization_config,
            {
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                # 统一复用 laser_mapping.map_file_path
                "file_name": map_file_path,
            },
        ],
        remappings=[("cloud_pcd", "/pcd_map")],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
        condition=IfCondition(LaunchConfiguration("rviz")),
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
    )

    map_odom_tf_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(map_odom_tf_launch),
        launch_arguments={
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "params_file": map_odom_tf_params,
        }.items(),
        condition=IfCondition(LaunchConfiguration("launch_map_odom_tf")),
    )

    return [
        fast_lio_node,
        fastlio_relative_odom_node,
        global_localization_node,
        manual_initial_pose_node,
        transform_fusion_node,
        pcd_publisher_node,
        map_odom_tf_include,
        rviz_node,
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=DEFAULT_CONFIG_FILE,
                description="定位参数文件名，例如 sim.yaml 或 mid360.yaml",
            ),
            DeclareLaunchArgument(
                "pcd_map",
                default_value=DEFAULT_PCD_MAP,
                description="3D PCD 地图绝对路径；为空时使用 config_file 中的 laser_mapping.map_file_path",
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value=DEFAULT_USE_SIM_TIME,
                description="是否启用仿真时间（use_sim_time）",
            ),
            DeclareLaunchArgument(
                "rviz",
                default_value=DEFAULT_RVIZ,
                description="是否启动 RViz2",
            ),
            DeclareLaunchArgument(
                "publish_pcd_map",
                default_value=DEFAULT_PUBLISH_PCD_MAP,
                description="是否发布静态 PCD 地图到 PointCloud2 话题",
            ),
            DeclareLaunchArgument(
                "launch_map_odom_tf",
                default_value=DEFAULT_LAUNCH_MAP_ODOM_TF,
                description="是否启动 map->odom TF 发布器",
            ),
            DeclareLaunchArgument(
                "publish_initial_pose",
                default_value=DEFAULT_PUBLISH_INITIAL_POSE,
                description="是否在启动后自动发布一次手动设定的初始位姿",
            ),
            DeclareLaunchArgument(
                "initial_pose_x",
                default_value="0.0",
                description="手动初始位姿 x，单位 m",
            ),
            DeclareLaunchArgument(
                "initial_pose_y",
                default_value="0.0",
                description="手动初始位姿 y，单位 m",
            ),
            DeclareLaunchArgument(
                "initial_pose_yaw",
                default_value="0.0",
                description="手动初始位姿 yaw，单位 rad",
            ),
            DeclareLaunchArgument(
                "initial_pose_delay",
                default_value="2.0",
                description="启动后延时多久发布初始位姿，单位 s",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
