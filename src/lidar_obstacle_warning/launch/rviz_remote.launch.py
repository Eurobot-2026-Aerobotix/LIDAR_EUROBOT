#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    # Use the SAME RViz config as the working launch file
    rviz_config = os.path.join(
        os.environ['HOME'],
        'EUROBOT/LIDAR_EUROBOT/src/lidar_obstacle_warning/rviz/warning.rviz'
    )
    
    # RViz2 node for remote visualization
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config]  # Always use the config file
    )

    return LaunchDescription([
        rviz_node
    ])
