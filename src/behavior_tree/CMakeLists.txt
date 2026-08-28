cmake_minimum_required(VERSION 3.8)
project(behavior_tree)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(interfaces REQUIRED)
find_package(behaviortree_cpp REQUIRED)
find_package(yaml-cpp REQUIRED)



include_directories("include")

add_executable(root_tree src/root_tree.cpp)
ament_target_dependencies(root_tree behaviortree_cpp)

add_executable(reactivefallback src/reactivefallback.cpp)
ament_target_dependencies(reactivefallback behaviortree_cpp)

add_executable(medical_bt_test src/medical_bt_test.cpp)
ament_target_dependencies(medical_bt_test behaviortree_cpp)

add_executable(ros_bt_runner src/ros_bt_runner.cpp)
ament_target_dependencies(ros_bt_runner 
rclcpp rclcpp_action ament_index_cpp std_msgs 
interfaces behaviortree_cpp)
target_link_libraries(ros_bt_runner yaml-cpp)


install(TARGETS
  root_tree
  reactivefallback
  medical_bt_test
  ros_bt_runner
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/format_ros_test_log.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME format_ros_test_log
)
install(PROGRAMS
  scripts/medical_bt_ros_test_driver.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME medical_bt_ros_test_driver
)

install(DIRECTORY BH_xml assets launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
