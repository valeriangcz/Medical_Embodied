#include <livox_to_pointcloud2/livox_to_pointcloud2_ros2.hpp>

#include <rclcpp_components/register_node_macro.hpp>

#ifdef LIVOX_ROS2_DRIVER
#include <livox_interfaces/msg/custom_msg.hpp>
#endif

#ifdef LIVOX_ROS_DRIVER2
#include <livox_ros_driver2/msg/custom_msg.hpp>
#endif

namespace livox_to_pointcloud2 {

LivoxToPointCloud2::LivoxToPointCloud2(const rclcpp::NodeOptions& options) : rclcpp::Node("livox_to_pointcloud2", options) {
  const auto output_topic = this->declare_parameter<std::string>("output_topic", "/livox/points");

#ifdef LIVOX_ROS2_DRIVER
  const auto input_topic = this->declare_parameter<std::string>("input_topic", "/livox/lidar");
  points_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>(output_topic, rclcpp::SensorDataQoS());
  // livox_ros2_driver
  livox_sub =
    this->create_subscription<livox_interfaces::msg::CustomMsg>(input_topic, rclcpp::SensorDataQoS(), [this](const livox_interfaces::msg::CustomMsg::ConstSharedPtr livox_msg) {
      const auto points_msg = converter.convert(*livox_msg);
      points_pub->publish(*points_msg);
    });
  RCLCPP_INFO(this->get_logger(), "Converting Livox CustomMsg from '%s' to PointCloud2 on '%s'", input_topic.c_str(), output_topic.c_str());
#endif

#ifdef LIVOX_ROS_DRIVER2
  const auto input_topic = this->declare_parameter<std::string>("input_topic", "/livox/lidar");
  points_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>(output_topic, rclcpp::SensorDataQoS());
  // livox_ros_driver2
  livox2_sub = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(
    input_topic,
    rclcpp::SensorDataQoS(),
    [this](const livox_ros_driver2::msg::CustomMsg::ConstSharedPtr livox_msg) {
      const auto points_msg = converter.convert(*livox_msg);
      points_pub->publish(*points_msg);
    });
  RCLCPP_INFO(this->get_logger(), "Converting Livox CustomMsg from '%s' to PointCloud2 on '%s'", input_topic.c_str(), output_topic.c_str());
#endif
}

LivoxToPointCloud2::~LivoxToPointCloud2() {}

}  // namespace livox_to_pointcloud2

RCLCPP_COMPONENTS_REGISTER_NODE(livox_to_pointcloud2::LivoxToPointCloud2);
