#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

namespace {

Eigen::Matrix3d rotationImuToLink1Default()
{
  Eigen::Matrix3d R;
  R << -0.000000, -0.283172, 0.959069, 1.000000, -0.000000, 0.000000, 0.000000, 0.959069,
    0.283172;
  return R;
}

Eigen::Matrix3d rotationFromRowMajor(const std::vector<double> & values)
{
  if (values.size() != 9) {
    throw std::runtime_error("rotation_matrix must contain exactly 9 row-major values");
  }
  Eigen::Matrix3d R;
  R << values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7],
    values[8];
  return R;
}

Eigen::Vector3d rotateVector(const Eigen::Matrix3d & R, const Eigen::Vector3d & v)
{
  return R * v;
}

void rotateCovariance(const Eigen::Matrix3d & R, std::array<double, 9> & cov)
{
  Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> C(cov.data());
  C = R * C * R.transpose();
}

}  // namespace

class ImuToLink1Node : public rclcpp::Node
{
public:
  ImuToLink1Node() : Node("imu_to_link1")
  {
    declare_parameter<std::string>("input_topic", "/imu");
    declare_parameter<std::string>("output_topic", "/imu/link1");
    declare_parameter<std::string>("output_frame_id", "lidar_3d_frame_fastlio");
    declare_parameter<std::vector<double>>(
      "rotation_matrix",
      std::vector<double>{
        -0.000000, -0.283172, 0.959069, 1.000000, -0.000000, 0.000000, 0.000000, 0.959069,
        0.283172});

    const auto input_topic = get_parameter("input_topic").as_string();
    const auto output_topic = get_parameter("output_topic").as_string();
    output_frame_id_ = get_parameter("output_frame_id").as_string();
    R_imu_to_link1_ = rotationFromRowMajor(get_parameter("rotation_matrix").as_double_array());

    pub_ = create_publisher<sensor_msgs::msg::Imu>(output_topic, 10);
    sub_ = create_subscription<sensor_msgs::msg::Imu>(
      input_topic, 10, std::bind(&ImuToLink1Node::imuCallback, this, std::placeholders::_1));

    RCLCPP_INFO(
      get_logger(),
      "imu_to_link1: %s -> %s, frame_id=%s",
      input_topic.c_str(),
      output_topic.c_str(),
      output_frame_id_.c_str());
    RCLCPP_INFO(
      get_logger(),
      "rotation_matrix (imu->Link1):\n%.6f %.6f %.6f\n%.6f %.6f %.6f\n%.6f %.6f %.6f",
      R_imu_to_link1_(0, 0),
      R_imu_to_link1_(0, 1),
      R_imu_to_link1_(0, 2),
      R_imu_to_link1_(1, 0),
      R_imu_to_link1_(1, 1),
      R_imu_to_link1_(1, 2),
      R_imu_to_link1_(2, 0),
      R_imu_to_link1_(2, 1),
      R_imu_to_link1_(2, 2));
  }

private:
  void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    sensor_msgs::msg::Imu out = *msg;
    out.header.frame_id = output_frame_id_;

    const Eigen::Vector3d omega(
      msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
    const Eigen::Vector3d accel(
      msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);

    const Eigen::Vector3d omega_link1 = rotateVector(R_imu_to_link1_, omega);
    const Eigen::Vector3d accel_link1 = rotateVector(R_imu_to_link1_, accel);

    out.angular_velocity.x = -omega_link1.x();
    out.angular_velocity.y = -omega_link1.y();
    out.angular_velocity.z = -omega_link1.z();
    out.linear_acceleration.x = accel_link1.x();
    out.linear_acceleration.y = accel_link1.y();
    out.linear_acceleration.z = accel_link1.z();

    const bool has_orientation =
      std::isfinite(msg->orientation.w) && std::isfinite(msg->orientation.x) &&
      std::isfinite(msg->orientation.y) && std::isfinite(msg->orientation.z) &&
      (std::abs(msg->orientation.w) + std::abs(msg->orientation.x) +
         std::abs(msg->orientation.y) + std::abs(msg->orientation.z) >
       1e-6);

    if (has_orientation) {
      Eigen::Quaterniond q_imu(
        msg->orientation.w, msg->orientation.x, msg->orientation.y, msg->orientation.z);
      q_imu.normalize();
      const Eigen::Quaterniond q_static(R_imu_to_link1_);
      const Eigen::Quaterniond q_link1 = q_imu * q_static.conjugate();
      out.orientation.w = q_link1.w();
      out.orientation.x = q_link1.x();
      out.orientation.y = q_link1.y();
      out.orientation.z = q_link1.z();
    }

    rotateCovariance(R_imu_to_link1_, out.angular_velocity_covariance);
    rotateCovariance(R_imu_to_link1_, out.linear_acceleration_covariance);
    if (has_orientation) {
      rotateCovariance(R_imu_to_link1_, out.orientation_covariance);
    }

    pub_->publish(out);
  }

  std::string output_frame_id_;
  Eigen::Matrix3d R_imu_to_link1_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ImuToLink1Node>());
  rclcpp::shutdown();
  return 0;
}
