cmake_minimum_required(VERSION 3.8)
project(robot_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)

# Install URDF, meshes, and launch files if they exist
install(
  DIRECTORY urdf launch meshes
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

# Optionally, you can install config or rviz files too:
# install(DIRECTORY config DESTINATION share/${PROJECT_NAME} OPTIONAL)

ament_package()
