#include <cmath>
#include <memory>
#include <string>

#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

namespace {

// Nominal rotation from static_transforms_3.launch.py TF1 (tf2 setRPY).
Eigen::Matrix3d rotationNominalImuToBaseLink()
{
  const double roll = 90.0 / 57.3;
  const double pitch = 180.0 / 57.3;
  const double yaw = 90.0 / 57.3;

  const Eigen::AngleAxisd rx(roll, Eigen::Vector3d::UnitX());
  const Eigen::AngleAxisd ry(pitch, Eigen::Vector3d::UnitY());
  const Eigen::AngleAxisd rz(yaw, Eigen::Vector3d::UnitZ());
  return (rz * ry * rx).matrix();
}

// Fine calibration from static rest sample (after nominal rotation):
// a_meas = [-2.6469, 1.9497, -9.2477] m/s^2, target gravity = [0, 0, -g] in base_link.
// R_calibrated = R_align * R_nominal, v_base = R_calibrated * v_imu.
Eigen::Matrix3d rotationCalibratedImuToBaseLink()
{
  Eigen::Matrix3d R;
  R << -0.0276322872, 0.2695843645, 0.9625802445, -0.9797317532, -0.1984246777,
    0.0274470223, 0.1983989628, -0.9423120065, 0.2696032898;
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

class ImuToBaseLinkNode : public rclcpp::Node
{
public:
  ImuToBaseLinkNode() : Node("imu_to_base_link")
  {
    declare_parameter<std::string>("input_topic", "/imu");
    declare_parameter<std::string>("output_topic", "/imu/base_link");
    declare_parameter<std::string>("output_frame_id", "base_link");
    declare_parameter<bool>("use_calibrated_rotation", true);

    const auto input_topic = get_parameter("input_topic").as_string();
    const auto output_topic = get_parameter("output_topic").as_string();
    output_frame_id_ = get_parameter("output_frame_id").as_string();
    const bool use_calibrated = get_parameter("use_calibrated_rotation").as_bool();

    R_imu_to_base_ =
      use_calibrated ? rotationCalibratedImuToBaseLink() : rotationNominalImuToBaseLink();

    pub_ = create_publisher<sensor_msgs::msg::Imu>(output_topic, 10);
    sub_ = create_subscription<sensor_msgs::msg::Imu>(
      input_topic, 10, std::bind(&ImuToBaseLinkNode::imuCallback, this, std::placeholders::_1));

    RCLCPP_INFO(
      get_logger(),
      "imu_to_base_link: %s -> %s, frame_id=%s, calibrated=%s",
      input_topic.c_str(),
      output_topic.c_str(),
      output_frame_id_.c_str(),
      use_calibrated ? "true" : "false");
  }

private:
  void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    sensor_msgs::msg::Imu out = *msg;
    out.header.stamp = now();
    out.header.frame_id = output_frame_id_;

    const Eigen::Vector3d omega(
      msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
    const Eigen::Vector3d accel(
      msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);

    const Eigen::Vector3d omega_base = -rotateVector(R_imu_to_base_, omega);
    const Eigen::Vector3d accel_base = -rotateVector(R_imu_to_base_, accel);

    out.angular_velocity.x = omega_base.x();
    out.angular_velocity.y = omega_base.y();
    out.angular_velocity.z = omega_base.z();
    out.linear_acceleration.x = accel_base.x();
    out.linear_acceleration.y = accel_base.y();
    out.linear_acceleration.z = accel_base.z();

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
      // R_world_base = R_world_imu * R_base_imu^T, with R_base_imu = R_imu_to_base
      const Eigen::Quaterniond q_static(R_imu_to_base_);
      const Eigen::Quaterniond q_base = q_imu * q_static.conjugate();
      out.orientation.w = q_base.w();
      out.orientation.x = q_base.x();
      out.orientation.y = q_base.y();
      out.orientation.z = q_base.z();
    }

    rotateCovariance(R_imu_to_base_, out.angular_velocity_covariance);
    rotateCovariance(R_imu_to_base_, out.linear_acceleration_covariance);
    if (has_orientation) {
      rotateCovariance(R_imu_to_base_, out.orientation_covariance);
    }

    pub_->publish(out);
  }

  std::string output_frame_id_;
  Eigen::Matrix3d R_imu_to_base_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ImuToBaseLinkNode>());
  rclcpp::shutdown();
  return 0;
}
