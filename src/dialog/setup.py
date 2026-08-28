from setuptools import setup

package_name = 'dialog'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/llm_mock.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='val',
    maintainer_email='val@todo.todo',
    description='Mock LLM interaction action server/client (ROS2).',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 注意: 实现文件是 bt_hci_interface.py (类 LLMMockServer)。
            # 历史提交 56482a4 把 llm_mock_server.py 重构为 bt_hci_interface.py 时
            # 漏改了这里的入口点, 导致启动报 ModuleNotFoundError: dialog.llm_mock_server。
            'llm_mock_server = dialog.bt_hci_interface:main',
            'llm_mock_client = dialog.llm_mock_client:main',
        ],
    },
)
