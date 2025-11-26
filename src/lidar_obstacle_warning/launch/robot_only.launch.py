#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    urdf_path = os.path.join(
        os.environ['HOME'],
        'LIDAR_EUROBOT/src/robot_description/urdf/diffbot.urdf'
    )

    # LIDAR node
    ldlidar_node = Node(
      package='ldlidar_stl_ros2',
      executable='ldlidar_stl_ros2_node',
      name='LD19',
      output='screen',
      parameters=[
        {'product_name': 'LDLiDAR_LD19'},
        {'topic_name': 'scan'},
        {'frame_id': 'base_laser'},
        {'port_name': '/dev/ttyUSB0'},
        {'port_baudrate': 230400},
        {'laser_scan_dir': True},
        {'enable_angle_crop_func': False},
        {'angle_crop_min': 135.0},
        {'angle_crop_max': 225.0}
      ]
    )

    # Transform publishers
    ld19_to_laser_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='ld19_to_base_laser',
        arguments=['0','0','0','3.141592','0','0','ld19','base_laser']
    )

    # map_to_base_link = Node(
    #     package='tf2_ros',
    #     executable='static_transform_publisher',
    #     name='map_to_laser_link',
    #     arguments=['0','0','0','0','0','0','map','base_link']
    # )

    # # Robot pose broadcaster
    # robot_pose_tf = Node(
    #     package='lidar_obstacle_warning',
    #     executable='robot_pose_tf_broadcaster',
    #     name='robot_pose_tf_broadcaster',
    #     output='screen'
    # )

    # Robot state publisher
    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        arguments=[urdf_path]
    )

    # Obstacle detection node
    obstacle_detector = Node(
        package='lidar_obstacle_warning',
        executable='lidar_obstacle_warning_node',
        name='obstacle_detector',
        output='screen',
        parameters=[{
            'scan_topic': '/scan',
            'map_topic': '/map',
            'pose_topic': '/robot_pose_raw',
            'distance_threshold': 0.5,
            'lidar_offset': [0.0, 0.0, 0.0]
        }]
    )

    # Map server
    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{
            'yaml_filename': os.path.join(os.environ['HOME'], 'LIDAR_EUROBOT/src/euro_ros_map/map.yaml'),
            'use_sim_time': False
        }]
    )

    # Lifecycle manager for map server
    lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_map",
        output="screen",
        parameters=[{
            "use_sim_time": False,
            "autostart": True,
            "node_names": ["map_server"]
        }]
    )

    return LaunchDescription([
        ldlidar_node,
        ld19_to_laser_tf_node,
        # map_to_base_link,
        map_server,
        robot_state_pub,
        lifecycle_manager,
        # robot_pose_tf,
        obstacle_detector
    ])