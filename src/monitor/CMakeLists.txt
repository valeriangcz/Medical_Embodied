cmake_minimum_required(VERSION 3.22)
project(monitor)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------- find dependencies ----------------
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED)

# librealsense2 — custom prefix, set via CMAKE_PREFIX_PATH or realsense2_DIR
find_package(realsense2 REQUIRED)

# Copy linker search directories into install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# ---------------- C++ target: L515 camera node ----------------
add_executable(l515_camera_node src/l515_camera_node.cpp)

ament_target_dependencies(l515_camera_node
  rclcpp
  sensor_msgs
)

target_include_directories(l515_camera_node PRIVATE
  ${OpenCV_INCLUDE_DIRS}
)

if(REALSENSE2_LIB_DIR)
  target_link_directories(l515_camera_node PRIVATE "${REALSENSE2_LIB_DIR}")
endif()

target_link_libraries(l515_camera_node
  realsense2::realsense2
  ${OpenCV_LIBRARIES}
)

# ---------------- Python package (replaces setup.py) ----------------
ament_python_install_package(${PROJECT_NAME}
  PACKAGE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/monitor
  SCRIPTS_DESTINATION lib/${PROJECT_NAME}
)

# ---------------- install targets ----------------
install(TARGETS l515_camera_node
  DESTINATION lib/${PROJECT_NAME}
)

# Console-script wrappers
install(PROGRAMS
  scripts/anomaly_detect_server
  scripts/bed_detection_server
  scripts/face_identify_server
  scripts/mock_camera
  scripts/mock_patrol_detection_server
  scripts/monitor_mock_pub
  DESTINATION lib/${PROJECT_NAME}
)

# Data directories
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)
install(DIRECTORY models
  DESTINATION share/${PROJECT_NAME}
)
install(DIRECTORY face_database
  DESTINATION share/${PROJECT_NAME}
)

# 辅助脚本 (conda 环境包装等), 供 launch 文件通过 prefix 使用
# 注意: 必须保留可执行权限, 否则 launch prefix 无法运行
install(FILES scripts/conda_activate.sh
  DESTINATION share/${PROJECT_NAME}/scripts
  PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
)

# Package manifest
install(FILES package.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
