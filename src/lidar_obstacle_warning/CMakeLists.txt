cmake_minimum_required(VERSION 3.8)
project(robot_description)

find_package(ament_cmake REQUIRED)

# Install URDF and meshes
install(
  DIRECTORY urdf meshes
  DESTINATION share/${PROJECT_NAME}
)

# Optionally install launch files
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

ament_package()
