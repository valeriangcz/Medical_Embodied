#!/usr/bin/env python3
# 这是一个用于ROS2 Foxy及更早版本的地图保存启动文件。
# 在建图完成后，运行此Launch文件，然后通过服务调用保存地图。
 
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
 
def generate_launch_description():
    # ************** 可配置参数部分 **************
    # 这里定义了一些变量，方便你根据实际情况修改
    use_sim_time = True  # 如果你使用Gazebo等仿真器，设为True；使用真实机器人则设为False
    map_saver_params_file = 'map_saver.yaml' # 参数文件名，我们稍后会创建它
    # *******************************************
 
    # 获取当前功能包的共享目录路径
    pkg_dir = get_package_share_directory('my_map_saver')
    # 拼接出参数文件的完整路径
    map_save_config_path = os.path.join(pkg_dir, 'config', map_saver_params_file)
 
    return LaunchDescription([
        # 声明启动参数：是否使用仿真时间
        DeclareLaunchArgument(
            'use_sim_time',
            default_value=str(use_sim_time),
            description='Use simulation (Gazebo) clock if true'
        ),
        # 声明启动参数：地图服务器参数文件路径
        DeclareLaunchArgument(
            'map_saver_params_file',
            default_value=map_save_config_path,
            description='Full path to the map saver parameters YAML file'
        ),
 
        # 启动地图保存服务器节点
        Node(
            package='nav2_map_server',
            executable='map_saver_server',
            output='screen',  # 将日志输出到屏幕，方便调试
            emulate_tty=True,  # 确保输出有颜色和格式
            parameters=[LaunchConfiguration('map_saver_params_file')] # 加载参数文件
        ),
 
        # 启动生命周期管理器节点，这是激活map_saver_server的关键！
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_map_saver', # 给管理器起个名字，避免与其他管理器冲突
            output='screen',
            emulate_tty=True,
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')},
                {'autostart': True}, # 设置为True，让管理器自动开始管理节点生命周期
                {'node_names': ['map_saver']} # 指定要管理的节点名称，必须与map_saver_server的节点名匹配
            ]
        )
    ])