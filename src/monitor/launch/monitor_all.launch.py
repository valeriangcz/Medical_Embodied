from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import UnlessCondition
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """
    一键启动：L515相机 + 床位检测 + 人脸识别

    分步启动（推荐用于调试）：
      ros2 launch monitor l515_camera.launch.py
      ros2 launch monitor bed_detection.launch.py
      ros2 launch monitor face_identify.launch.py

    模拟相机测试：
      ros2 launch monitor monitor_all.launch.py use_mock_camera:=true camera_topic:=/camera/rgb/image_raw
    """

    monitor_launch_dir = os.path.join(
        get_package_share_directory('monitor'), 'launch'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_mock_camera',
            default_value='false',
            description='使用模拟相机替代真实相机（测试用）'
        ),
        DeclareLaunchArgument(
            'camera_topic',
            default_value='/camera/camera/color/image_raw',
            description='RGB图像话题名（mock相机需设为 /camera/rgb/image_raw）'
        ),
        DeclareLaunchArgument(
            'l515_preset',
            default_value='short_range',
            description='L515 visual preset: short_range | no_ambient | low_ambient | max_range | default'
        ),
        DeclareLaunchArgument(
            'waypoints_config_path',
            default_value='',
            description='导航waypoints.yaml路径（留空自动查找 xjrobot_bridge 包内 config/waypoints.yaml）'
        ),
        DeclareLaunchArgument(
            'vlm_model',
            default_value='qwen3.8-flash',
            description='VLM模型名（DashScope）'
        ),

        # ==================== 相机 (L515 via librealsense2 C++) ====================
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(monitor_launch_dir, 'l515_camera.launch.py')
            ),
            condition=UnlessCondition(
                LaunchConfiguration('use_mock_camera')
            ),
            launch_arguments={
                'l515_preset': LaunchConfiguration('l515_preset'),
            }.items(),
        ),

        # ==================== 床位检测 ====================
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(monitor_launch_dir, 'bed_detection.launch.py')
            ),
            launch_arguments={
                'use_mock_camera': LaunchConfiguration('use_mock_camera'),
                'camera_topic': LaunchConfiguration('camera_topic'),
                'waypoints_config_path': LaunchConfiguration('waypoints_config_path'),
                'vlm_model': LaunchConfiguration('vlm_model'),
            }.items(),
        ),

        # ==================== 人脸识别 ====================
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(monitor_launch_dir, 'face_identify.launch.py')
            ),
            launch_arguments={
                'use_mock_camera': LaunchConfiguration('use_mock_camera'),
                'camera_topic': LaunchConfiguration('camera_topic'),
            }.items(),
        ),
    ])
