from setuptools import setup
import os
from glob import glob

package_name = 'lidar_obstacle_warning'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        # Install launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mahdi',
    maintainer_email='mahdi@example.com',
    description='Lidar obstacle warning node',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lidar = lidar_obstacle_warning.lidar_obstacle_warning_node:main',
            'robot_simulation = lidar_obstacle_warning.robot_simulation:main',
        ],
    },
)
