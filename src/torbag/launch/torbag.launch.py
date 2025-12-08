#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    urdf_path = os.path.join(
        os.environ['HOME'],
        'eurobot_bot/src/torbag/urdf/TORBAG.urdf'
    )


    # Robot state publisher
    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        arguments=[urdf_path]
    )


    return LaunchDescription([
        robot_state_pub,
    ])