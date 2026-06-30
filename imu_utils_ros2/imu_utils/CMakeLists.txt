cmake_minimum_required(VERSION 3.8)
project(imu_utils)

set(CMAKE_CXX_STANDARD 14)
add_compile_options(-Wall -Wextra -Wpedantic -Wno-sign-compare)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -fPIC")

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${CERES_INCLUDE_DIRS}
)

set(ACC_LIB_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/src/acc_lib/allan_acc.cpp
    ${PROJECT_SOURCE_DIR}/src/acc_lib/fitallan_acc.cpp
)
set(GYR_LIB_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/src/gyr_lib/allan_gyr.cpp
    ${PROJECT_SOURCE_DIR}/src/gyr_lib/fitallan_gyr.cpp
)

add_executable(imu_allan
    src/imu_allan.cpp
    ${GYR_LIB_SOURCE_FILES}
    ${ACC_LIB_SOURCE_FILES}
)
ament_target_dependencies(imu_allan rclcpp std_msgs sensor_msgs OpenCV)
target_link_libraries(imu_allan ${OpenCV_LIBS} ${CERES_LIBRARIES})

# Install launch files
install(
    DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}
)

# Install nodes
install(
    TARGETS imu_allan
    DESTINATION lib/${PROJECT_NAME}
)

ament_package()
