from setuptools import setup
import os

package_name = 'monitor'

# 递归收集所有文件
def collect_data_files():
    data_files = []
    
    # 资源索引文件
    data_files.append(('share/ament_index/resource_index/packages', ['resource/' + package_name]))
    data_files.append(('share/' + package_name, ['package.xml']))
    
    # launch 文件
    launch_files = []
    if os.path.exists('launch'):
        for file in os.listdir('launch'):
            if file.endswith('.py'):
                launch_files.append(f'launch/{file}')
    data_files.append(('share/' + package_name + '/launch', launch_files))

    # scripts 辅助脚本（conda 环境包装等）
    scripts_files = []
    if os.path.exists('scripts'):
        for file in os.listdir('scripts'):
            if file.endswith('.sh'):
                scripts_files.append(f'scripts/{file}')
    if scripts_files:
        data_files.append(('share/' + package_name + '/scripts', scripts_files))
    
    # face_database 文件
    face_db_files = []
    if os.path.exists('face_database'):
        for file in os.listdir('face_database'):
            face_db_files.append(f'face_database/{file}')
    data_files.append(('share/' + package_name + '/face_database', face_db_files))
    
    # config 文件 (yaml + json)
    config_files = []
    if os.path.exists('config'):
        for file in os.listdir('config'):
            if file.endswith('.yaml') or file.endswith('.yml') or file.endswith('.json'):
                config_files.append(f'config/{file}')
    data_files.append(('share/' + package_name + '/config', config_files))
    
    # models 文件 (模型太大，只包含符号链接指向)
    model_files = []
    if os.path.exists('models'):
        for file in os.listdir('models'):
            # 只添加实际存在的文件（过滤符号链接）
            file_path = os.path.join('models', file)
            if not os.path.islink(file_path):
                model_files.append(f'models/{file}')
    if model_files:
        data_files.append(('share/' + package_name + '/models', model_files))
    
    # test_images 文件 (递归收集子文件夹)
    if os.path.exists('test_images'):
        for root, dirs, files in os.walk('test_images'):
            img_files = [os.path.join(root, f) for f in files
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
            if img_files:
                # root 如 'test_images/area_pic' -> install 到 'share/monitor/test_images/area_pic'
                rel_dir = root
                data_files.append(('share/' + package_name + '/' + rel_dir, img_files))
    
    return data_files

setup(
    name=package_name,
    version='0.0.2',  # 版本升级
    packages=[package_name],
    data_files=collect_data_files(),
    install_requires=[
        'setuptools',
        'opencv-python>=4.8.0',
        'numpy>=1.17.4,<2.0.0',  # 各模块通常 >=1.17.4；<2 与 ROS cv_bridge 一致
        'ultralytics>=8.0.0',        # 新增：YOLOv8
        'open-clip-torch>=2.0.0',    # 新增：CLIP
        'torch>=2.0.0',              # 新增：PyTorch
        'torchvision>=0.15.0',       # 新增：TorchVision
        'Pillow>=10.0.0',            # 新增：PIL
        'openai>=1.0.0',             # 新增：VLM API 调用
    ],
    zip_safe=True,
    maintainer='val',
    maintainer_email='val@todo.todo',
    description='Monitor publishers and detection services including bed detection (YOLOv8 + CLIP).',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'monitor_mock_pub = monitor.monitor_mock_pub:main',
            'anomaly_detect_server = monitor.anomaly_detect_server:main',
            'face_identify_server = monitor.face_identify_server:main',
            'bed_detection_server = monitor.bed_detection_server:main',      # 新增
            'mock_camera = monitor.mock_camera:main',                        # 新增
        ],
    },
)
