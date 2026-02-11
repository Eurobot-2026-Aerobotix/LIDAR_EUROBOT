cmake_minimum_required(VERSION 3.8)
project(lidar_obstacle_warning)

find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  lidar_obstacle_warning/lidar_obstacle_warning_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

