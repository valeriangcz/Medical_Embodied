#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/point_field.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"

class RslidarToFastlioCloud : public rclcpp::Node
{
public:
  RslidarToFastlioCloud() : Node("rslidar_to_fastlio_cloud")
  {
    this->declare_parameter<std::string>("input_topic", "/rslidar_points");
    this->declare_parameter<std::string>("output_topic", "/rslidar_points_fastlio");
    this->declare_parameter<std::string>("raw_output_topic", "/rslidar_points_fastlio_frame");
    this->declare_parameter<std::string>("target_frame_id", "lidar_3d_frame_fastlio");

    const auto input_topic = this->get_parameter("input_topic").as_string();
    const auto output_topic = this->get_parameter("output_topic").as_string();
    auto raw_output_topic = this->get_parameter("raw_output_topic").as_string();
    target_frame_id_ = this->get_parameter("target_frame_id").as_string();
    if (raw_output_topic.empty()) {
      raw_output_topic = output_topic + "_frame";
    }

    const auto reliable_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
    pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(output_topic, reliable_qos);
    raw_pub_ =
      this->create_publisher<sensor_msgs::msg::PointCloud2>(raw_output_topic, reliable_qos);
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic, rclcpp::SensorDataQoS(),
      std::bind(&RslidarToFastlioCloud::cloud_callback, this, std::placeholders::_1));

    RCLCPP_INFO(
      this->get_logger(),
      "rslidar->fastlio cloud converter started. input=%s, output=%s, raw_frame_output=%s, "
      "frame_id=%s",
      input_topic.c_str(), output_topic.c_str(), raw_output_topic.c_str(), target_frame_id_.c_str());
  }

private:
  const sensor_msgs::msg::PointField * find_field(
    const sensor_msgs::msg::PointCloud2 & msg, const std::string & name) const
  {
    const auto it =
      std::find_if(msg.fields.begin(), msg.fields.end(), [&name](const auto & f) { return f.name == name; });
    if (it == msg.fields.end()) {
      return nullptr;
    }
    return &(*it);
  }

  double read_as_double(const uint8_t * data, uint8_t datatype) const
  {
    switch (datatype) {
      case sensor_msgs::msg::PointField::INT8:
        return static_cast<double>(*reinterpret_cast<const int8_t *>(data));
      case sensor_msgs::msg::PointField::UINT8:
        return static_cast<double>(*reinterpret_cast<const uint8_t *>(data));
      case sensor_msgs::msg::PointField::INT16:
        return static_cast<double>(*reinterpret_cast<const int16_t *>(data));
      case sensor_msgs::msg::PointField::UINT16:
        return static_cast<double>(*reinterpret_cast<const uint16_t *>(data));
      case sensor_msgs::msg::PointField::INT32:
        return static_cast<double>(*reinterpret_cast<const int32_t *>(data));
      case sensor_msgs::msg::PointField::UINT32:
        return static_cast<double>(*reinterpret_cast<const uint32_t *>(data));
      case sensor_msgs::msg::PointField::FLOAT32:
        return static_cast<double>(*reinterpret_cast<const float *>(data));
      case sensor_msgs::msg::PointField::FLOAT64:
        return *reinterpret_cast<const double *>(data);
      default:
        return std::numeric_limits<double>::quiet_NaN();
    }
  }

  bool has_required_fields(const sensor_msgs::msg::PointCloud2 & msg) const
  {
    static constexpr const char * required[] = {"x", "y", "z", "intensity", "ring", "timestamp"};
    for (const auto * name : required) {
      const auto it = std::find_if(
        msg.fields.begin(), msg.fields.end(),
        [name](const sensor_msgs::msg::PointField & f) { return f.name == name; });
      if (it == msg.fields.end()) {
        return false;
      }
    }
    return true;
  }

  void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    sensor_msgs::msg::PointCloud2 raw_msg = *msg;
    raw_msg.header.frame_id = "lidar_3d_link";
    raw_msg.header.stamp = this->now();
    raw_pub_->publish(raw_msg);

    if (!has_required_fields(*msg)) {
      std::string names;
      for (size_t i = 0; i < msg->fields.size(); ++i) {
        names += msg->fields[i].name;
        if (i + 1 < msg->fields.size()) {
          names += ",";
        }
      }
      RCLCPP_WARN(
        this->get_logger(),
        "Input cloud missing required fields [x,y,z,intensity,ring,timestamp], got=[%s]",
        names.c_str());
      return;
    }

    std::vector<float> x_vec;
    std::vector<float> y_vec;
    std::vector<float> z_vec;
    std::vector<float> intensity_vec;
    std::vector<uint16_t> ring_vec;
    std::vector<float> ts_vec;
    const size_t total = static_cast<size_t>(msg->width) * static_cast<size_t>(msg->height);
    x_vec.reserve(total);
    y_vec.reserve(total);
    z_vec.reserve(total);
    intensity_vec.reserve(total);
    ring_vec.reserve(total);
    ts_vec.reserve(total);

    const auto * f_x = find_field(*msg, "x");
    const auto * f_y = find_field(*msg, "y");
    const auto * f_z = find_field(*msg, "z");
    const auto * f_intensity = find_field(*msg, "intensity");
    const auto * f_ring = find_field(*msg, "ring");
    const auto * f_timestamp = find_field(*msg, "timestamp");
    if (!f_x || !f_y || !f_z || !f_intensity || !f_ring || !f_timestamp) {
      return;
    }

    float ts_min = std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < total; ++i) {
      const uint8_t * p = msg->data.data() + i * msg->point_step;
      const float x = static_cast<float>(read_as_double(p + f_x->offset, f_x->datatype));
      const float y = static_cast<float>(read_as_double(p + f_y->offset, f_y->datatype));
      const float z = static_cast<float>(read_as_double(p + f_z->offset, f_z->datatype));
      const float intensity =
        static_cast<float>(read_as_double(p + f_intensity->offset, f_intensity->datatype));
      const float t =
        static_cast<float>(read_as_double(p + f_timestamp->offset, f_timestamp->datatype));
      const double ring_d = read_as_double(p + f_ring->offset, f_ring->datatype);
      // Match Python read_points(..., skip_nans=True): drop any point with non-finite xyz/intensity/time.
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) || !std::isfinite(intensity) ||
          !std::isfinite(t)) {
        continue;
      }
      ts_min = std::min(ts_min, t);
      x_vec.push_back(x);
      y_vec.push_back(y);
      z_vec.push_back(z);
      intensity_vec.push_back(intensity);
      ring_vec.push_back(static_cast<uint16_t>(std::max(0.0, std::min(65535.0, ring_d))));
      ts_vec.push_back(t);
    }

    if (ts_vec.empty() || !std::isfinite(ts_min)) {
      return;
    }

    sensor_msgs::msg::PointCloud2 out_msg;
    out_msg.header = msg->header;
    out_msg.header.frame_id = target_frame_id_;
    out_msg.height = 1;
    out_msg.width = static_cast<uint32_t>(ts_vec.size());
    out_msg.is_bigendian = false;
    out_msg.is_dense = true;
    out_msg.fields.resize(6);
    out_msg.fields[0].name = "x";
    out_msg.fields[0].offset = 0;
    out_msg.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
    out_msg.fields[0].count = 1;
    out_msg.fields[1].name = "y";
    out_msg.fields[1].offset = 4;
    out_msg.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
    out_msg.fields[1].count = 1;
    out_msg.fields[2].name = "z";
    out_msg.fields[2].offset = 8;
    out_msg.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
    out_msg.fields[2].count = 1;
    out_msg.fields[3].name = "intensity";
    out_msg.fields[3].offset = 12;
    out_msg.fields[3].datatype = sensor_msgs::msg::PointField::FLOAT32;
    out_msg.fields[3].count = 1;
    out_msg.fields[4].name = "ring";
    out_msg.fields[4].offset = 16;
    out_msg.fields[4].datatype = sensor_msgs::msg::PointField::UINT16;
    out_msg.fields[4].count = 1;
    out_msg.fields[5].name = "time";
    out_msg.fields[5].offset = 20;
    out_msg.fields[5].datatype = sensor_msgs::msg::PointField::FLOAT32;
    out_msg.fields[5].count = 1;
    out_msg.point_step = 24;
    out_msg.row_step = out_msg.point_step * out_msg.width;
    out_msg.data.resize(out_msg.row_step);

    for (size_t i = 0; i < ts_vec.size(); ++i) {
      uint8_t * p = out_msg.data.data() + i * out_msg.point_step;
      std::memcpy(p + 0, &x_vec[i], sizeof(float));
      std::memcpy(p + 4, &y_vec[i], sizeof(float));
      std::memcpy(p + 8, &z_vec[i], sizeof(float));
      std::memcpy(p + 12, &intensity_vec[i], sizeof(float));
      std::memcpy(p + 16, &ring_vec[i], sizeof(uint16_t));
      // velodyne_handler: curvature = time_field / 1000 (ms) -> time_field in microseconds.
      const float t = (ts_vec[i] - ts_min) * 1e6f;
      std::memcpy(p + 20, &t, sizeof(float));
    }

    pub_->publish(out_msg);
  }

  std::string target_frame_id_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr raw_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<RslidarToFastlioCloud>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
