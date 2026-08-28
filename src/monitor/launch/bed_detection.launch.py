import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _conda_prefix():
    """
    构造 conda monitor 环境激活 prefix。

    优先使用安装后的 share/monitor/scripts/conda_activate.sh，
    回退到源码目录 src/monitor/scripts/conda_activate.sh。
    返回 None 时表示找不到包装脚本（此时不使用 conda prefix）。

    注意: launch 会把 prefix 的 substitutions 拼接为单个字符串后再
    shlex.split 拆分, 因此路径与 '_' 占位符必须放在同一字符串中
    并用空格分隔（不能分开传 list 元素, 否则会拼接成 'xxx.sh_'）。
    """
    candidates = []
    try:
        share_dir = get_package_share_directory('monitor')
        candidates.append(os.path.join(share_dir, 'scripts', 'conda_activate.sh'))
    except Exception:
        pass
    # 源码目录回退: launch 文件所在目录的上级 scripts/
    current = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(os.path.dirname(current), 'scripts', 'conda_activate.sh'))

    for path in candidates:
        if os.path.exists(path):
            return [f'{path} _']
    return None


def generate_launch_description():
    """启动床位检测服务节点（相机节点需独立启动）"""

    conda_prefix = _conda_prefix()

    return LaunchDescription([
        # ==================== 启动参数 ====================
        DeclareLaunchArgument(
            'use_mock_camera',
            default_value='false',
            description='是否使用模拟相机（无真机时用于测试）'
        ),
        DeclareLaunchArgument(
            'camera_topic',
            default_value='/camera/camera/color/image_raw',
            description='RGB图像话题名（RealSense默认: /camera/camera/color/image_raw，模拟相机: /camera/rgb/image_raw）'
        ),
        DeclareLaunchArgument(
            'yolo_model_path',
            default_value='',
            description='YOLO模型路径（留空使用包内路径；Area模式已改用VLM，仅Bed模式CLIP回退使用）'
        ),
        DeclareLaunchArgument(
            'clip_model_path',
            default_value='',
            description='CLIP模型路径（留空使用包内路径；仅Bed模式VLM失败回退使用）'
        ),
        DeclareLaunchArgument(
            'max_beds',
            default_value='10',
            description='最大床位数'
        ),
        DeclareLaunchArgument(
            'yolo_conf_threshold',
            default_value='0.5',
            description='YOLO检测置信度阈值（Area模式已改用VLM，本参数不再生效）'
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
        DeclareLaunchArgument(
            'use_conda',
            default_value='true',
            description='是否通过 conda 包装脚本启动节点（conda monitor 环境）'
        ),

        # ==================== 模拟相机节点（仅测试用） ====================
        Node(
            package='monitor',
            executable='mock_camera',
            name='mock_camera_node',
            output='screen',
            emulate_tty=True,
            # 与床位检测节点同一 conda monitor 环境
            prefix=conda_prefix,
            condition=IfCondition(LaunchConfiguration('use_mock_camera'))
        ),

        # ==================== 床位检测服务节点 ====================
        Node(
            package='monitor',
            executable='bed_detection_server',
            name='bed_detection_server',
            output='screen',
            # 通过 conda 包装脚本激活 monitor 环境后再启动节点
            prefix=conda_prefix,
            parameters=[{
                'yolo_model_path': LaunchConfiguration('yolo_model_path'),
                'clip_model_path': LaunchConfiguration('clip_model_path'),
                'max_beds': LaunchConfiguration('max_beds'),
                'camera_topic': LaunchConfiguration('camera_topic'),
                'yolo_conf_threshold': LaunchConfiguration('yolo_conf_threshold'),
                'waypoints_config_path': LaunchConfiguration('waypoints_config_path'),
                'vlm_model': LaunchConfiguration('vlm_model'),
            }],
            emulate_tty=True
        ),
    ])
