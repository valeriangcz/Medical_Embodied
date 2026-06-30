#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

class ImuThrottleNode : public rclcpp::Node
{
public:
  ImuThrottleNode() : Node("imu_throttle")
  {
    declare_parameter<std::string>("input_topic", "/imu/data_filtered");
    declare_parameter<std::string>("output_topic", "/imu/data");
    declare_parameter<double>("output_rate_hz", 200.0);

    const auto input_topic = get_parameter("input_topic").as_string();
    const auto output_topic = get_parameter("output_topic").as_string();
    output_rate_hz_ = get_parameter("output_rate_hz").as_double();

    if (output_rate_hz_ <= 0.0) {
      throw std::runtime_error("output_rate_hz must be greater than 0");
    }

    min_period_ = rclcpp::Duration::from_seconds(1.0 / output_rate_hz_);

    pub_ = create_publisher<sensor_msgs::msg::Imu>(output_topic, 10);
    sub_ = create_subscription<sensor_msgs::msg::Imu>(
      input_topic, 10, std::bind(&ImuThrottleNode::imuCallback, this, std::placeholders::_1));

    RCLCPP_INFO(
      get_logger(),
      "imu_throttle: %s -> %s at %.1f Hz",
      input_topic.c_str(),
      output_topic.c_str(),
      output_rate_hz_);
  }

private:
  void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    const auto now = this->now();
    if (last_publish_time_.nanoseconds() != 0 && (now - last_publish_time_) < min_period_) {
      return;
    }

    last_publish_time_ = now;
    pub_->publish(*msg);
  }

  double output_rate_hz_{200.0};
  rclcpp::Duration min_period_{0, 0};
  rclcpp::Time last_publish_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ImuThrottleNode>());
  rclcpp::shutdown();
  return 0;
}
