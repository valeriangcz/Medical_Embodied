/**
 * L515 RealSense Camera Node
 *
 * Thin ROS 2 wrapper around librealsense2 C++ API that publishes exactly
 * the same topic set as realsense2_camera so the monitor package needs
 * zero modifications when switching from D455 to L515.
 *
 * Published topics (default names, matching realsense2_camera convention):
 *   /camera/camera/color/image_raw              sensor_msgs/Image (bgr8)
 *   /camera/camera/aligned_depth_to_color/image_raw  sensor_msgs/Image (mono16, mm)
 *   /camera/camera/color/camera_info            sensor_msgs/CameraInfo
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>

#include <thread>
#include <atomic>
#include <memory>
#include <string>
#include <chrono>

class L515CameraNode : public rclcpp::Node
{
public:
  L515CameraNode()
  : Node("l515_camera_node")
  {
    declare_params();
    get_params();
    create_publishers();
    start_pipeline();
    start_camera_thread();

    RCLCPP_INFO(get_logger(),
      "L515 camera node ready — publishing on %s, %s, %s",
      color_topic_.c_str(), depth_topic_.c_str(), cinfo_topic_.c_str());
  }

  ~L515CameraNode() override
  {
    stop_requested_ = true;
    if (pipeline_started_) {
      try { pipe_.stop(); } catch (...) {}
    }
    if (camera_thread_.joinable()) {
      camera_thread_.join();
    }
  }

private:
  // ── parameters ────────────────────────────────────────────────────────
  void declare_params()
  {
    declare_parameter("serial_no", "");
    declare_parameter("color_width", 640);
    declare_parameter("color_height", 480);
    declare_parameter("color_fps", 30);
    declare_parameter("depth_width", 640);
    declare_parameter("depth_height", 480);
    declare_parameter("depth_fps", 30);
    declare_parameter("camera_name", "camera");
    declare_parameter("camera_namespace", "camera");
    declare_parameter("l515_preset", "short_range");
    declare_parameter("publish_depth", true);
    declare_parameter("initial_reset", true);
  }

  void get_params()
  {
    serial_no_        = get_parameter("serial_no").as_string();
    color_width_       = get_parameter("color_width").as_int();
    color_height_      = get_parameter("color_height").as_int();
    color_fps_         = get_parameter("color_fps").as_int();
    depth_width_       = get_parameter("depth_width").as_int();
    depth_height_      = get_parameter("depth_height").as_int();
    depth_fps_         = get_parameter("depth_fps").as_int();
    camera_name_       = get_parameter("camera_name").as_string();
    camera_namespace_  = get_parameter("camera_namespace").as_string();
    l515_preset_       = get_parameter("l515_preset").as_string();
    publish_depth_     = get_parameter("publish_depth").as_bool();
    initial_reset_     = get_parameter("initial_reset").as_bool();

    // Build topic strings to match realsense2_camera convention
    const std::string prefix = "/" + camera_namespace_ + "/" + camera_name_;
    color_topic_  = prefix + "/color/image_raw";
    depth_topic_  = prefix + "/aligned_depth_to_color/image_raw";
    cinfo_topic_  = prefix + "/color/camera_info";
    color_frame_id_ = camera_namespace_ + "_" + camera_name_ + "_color_optical_frame";
    depth_frame_id_ = camera_namespace_ + "_" + camera_name_ + "_depth_optical_frame";
  }

  // ── publishers ────────────────────────────────────────────────────────
  void create_publishers()
  {
    auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
    color_pub_  = create_publisher<sensor_msgs::msg::Image>(color_topic_, qos);
    cinfo_pub_  = create_publisher<sensor_msgs::msg::CameraInfo>(cinfo_topic_, qos);
    if (publish_depth_) {
      depth_pub_ = create_publisher<sensor_msgs::msg::Image>(depth_topic_, qos);
    }
  }

  // ── pipeline setup ────────────────────────────────────────────────────
  void start_pipeline()
  {
    if (initial_reset_) {
      perform_initial_reset();
    }

    if (!serial_no_.empty()) {
      cfg_.enable_device(serial_no_);
    }

    cfg_.enable_stream(RS2_STREAM_COLOR,
      color_width_, color_height_, RS2_FORMAT_RGB8, color_fps_);
    cfg_.enable_stream(RS2_STREAM_DEPTH,
      depth_width_, depth_height_, RS2_FORMAT_Z16, depth_fps_);

    profile_ = pipe_.start(cfg_);
    pipeline_started_ = true;

    apply_l515_preset();

    align_ = std::make_unique<rs2::align>(RS2_STREAM_COLOR);
    publish_camera_info();
  }

  // L515 固件有时会卡死在错误状态（set_xu 报 Device or resource busy、
  // 深度全部失真），启动前做一次硬件复位可恢复。与 realsense2_camera 的
  // initial_reset 参数行为一致；复位后等待设备重新枚举。
  void perform_initial_reset()
  {
    rs2::context ctx;
    if (ctx.query_devices().size() == 0) {
      RCLCPP_WARN(get_logger(), "No device found, skip initial reset");
      return;
    }
    rs2::device dev = ctx.query_devices()[0];
    if (!serial_no_.empty() &&
        dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) != serial_no_) {
      return;  // 指定了序列号但不匹配，不重置
    }

    RCLCPP_INFO(get_logger(), "Performing hardware reset (L515 firmware recovery)...");
    try {
      dev.hardware_reset();
    } catch (const rs2::error & e) {
      RCLCPP_WARN(get_logger(), "hardware_reset failed: %s", e.what());
      return;
    }

    // 注意：复位后不能在本函数内创建新 context 轮询设备——旧 context /
    // 设备句柄还存活，新 context 的 USB 枚举会被旧句柄阻塞（实测死锁）。
    // 固定等待重枚举（通常 3-6 秒），随后 pipe_.start() 用全新 context
    // 自然重新枚举设备。
    std::this_thread::sleep_for(std::chrono::seconds(6));
    RCLCPP_INFO(get_logger(), "Hardware reset done, starting pipeline...");
  }

  void apply_l515_preset()
  {
    // Map preset name to RS2 enum value
    rs2_l500_visual_preset preset = RS2_L500_VISUAL_PRESET_SHORT_RANGE;
    if (l515_preset_ == "no_ambient")       preset = RS2_L500_VISUAL_PRESET_NO_AMBIENT;
    else if (l515_preset_ == "low_ambient")  preset = RS2_L500_VISUAL_PRESET_LOW_AMBIENT;
    else if (l515_preset_ == "max_range")    preset = RS2_L500_VISUAL_PRESET_MAX_RANGE;
    else if (l515_preset_ == "default")      preset = RS2_L500_VISUAL_PRESET_DEFAULT;
    else if (l515_preset_ == "automatic")    preset = RS2_L500_VISUAL_PRESET_AUTOMATIC;

    // librealsense 2.50: iterate sensors to find the depth sensor
    rs2::device dev = profile_.get_device();
    for (auto & s : dev.query_sensors()) {
      if (s.is<rs2::depth_sensor>()) {
        auto depth = s.as<rs2::depth_sensor>();
        if (depth.supports(RS2_OPTION_VISUAL_PRESET)) {
          // 流启动后立即设置 XU 控件偶发 Device or resource busy，
          // 重试多次（间隔递增）直至成功
          for (int attempt = 1; attempt <= 5; ++attempt) {
            try {
              depth.set_option(RS2_OPTION_VISUAL_PRESET, static_cast<float>(preset));
              RCLCPP_INFO(get_logger(), "L515 visual preset set to '%s'",
                          l515_preset_.c_str());
              return;
            } catch (const rs2::error & e) {
              RCLCPP_WARN(get_logger(),
                "Set L515 preset attempt %d/5 failed: %s",
                attempt, e.what());
              std::this_thread::sleep_for(std::chrono::milliseconds(1000 * attempt));
            }
          }
        }
        return;
      }
    }
  }

  void publish_camera_info()
  {
    auto color_stream = profile_.get_stream(RS2_STREAM_COLOR)
                          .as<rs2::video_stream_profile>();
    auto in = color_stream.get_intrinsics();

    auto msg = sensor_msgs::msg::CameraInfo();
    msg.header.stamp    = now();
    msg.header.frame_id = color_frame_id_;
    msg.height = in.height;
    msg.width  = in.width;
    msg.distortion_model = "plumb_bob";

    // K matrix (3x3)
    msg.k[0] = in.fx;  msg.k[1] = 0.0;    msg.k[2] = in.ppx;
    msg.k[3] = 0.0;    msg.k[4] = in.fy;   msg.k[5] = in.ppy;
    msg.k[6] = 0.0;    msg.k[7] = 0.0;     msg.k[8] = 1.0;

    // P matrix (3x4) — same as K with last column = 0 (rectified image)
    msg.p[0] = msg.k[0]; msg.p[1] = msg.k[1]; msg.p[2] = msg.k[2];  msg.p[3] = 0.0;
    msg.p[4] = msg.k[3]; msg.p[5] = msg.k[4]; msg.p[6] = msg.k[5];  msg.p[7] = 0.0;
    msg.p[8] = msg.k[6]; msg.p[9] = msg.k[7]; msg.p[10] = msg.k[8]; msg.p[11] = 0.0;

    // R matrix (3x3 identity)
    msg.r[0] = 1.0; msg.r[1] = 0.0; msg.r[2] = 0.0;
    msg.r[3] = 0.0; msg.r[4] = 1.0; msg.r[5] = 0.0;
    msg.r[6] = 0.0; msg.r[7] = 0.0; msg.r[8] = 1.0;

    // D coefficients — all zeros (L515 images are rectified)
    msg.d.assign(5, 0.0);

    camera_info_msg_ = std::make_shared<sensor_msgs::msg::CameraInfo>(msg);
    cinfo_pub_->publish(*camera_info_msg_);
  }

  // ── camera thread ─────────────────────────────────────────────────────
  void start_camera_thread()
  {
    camera_thread_ = std::thread(&L515CameraNode::camera_loop, this);
  }

  void camera_loop()
  {
    while (rclcpp::ok() && !stop_requested_) {
      process_one_frame();
    }
  }

  void process_one_frame()
  {
    rs2::frameset frames;
    try {
      frames = pipe_.wait_for_frames(500);  // ms timeout
    } catch (...) {
      return;  // timeout or pipeline stopped
    }

    auto aligned = align_->process(frames);

    auto color_frame = aligned.get_color_frame();
    auto depth_frame = aligned.get_depth_frame();

    if (!color_frame) return;

    auto stamp = now();
    publish_color(color_frame, stamp);
    if (depth_pub_ && depth_frame) {
      publish_depth(depth_frame, stamp);
    }
    publish_cinfo(stamp);
  }

  void publish_color(const rs2::video_frame & f, rclcpp::Time stamp)
  {
    const int w = f.get_width();
    const int h = f.get_height();

    // RGB8 (librealsense) → BGR8 (OpenCV / monitor convention)
    cv::Mat rgb(h, w, CV_8UC3, const_cast<void *>(f.get_data()));
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);

    auto msg = sensor_msgs::msg::Image();
    msg.header.stamp    = stamp;
    msg.header.frame_id = color_frame_id_;
    msg.height   = h;
    msg.width    = w;
    msg.encoding = "bgr8";
    msg.is_bigendian = false;
    msg.step     = static_cast<uint32_t>(bgr.step[0]);
    msg.data.assign(bgr.datastart, bgr.dataend);

    color_pub_->publish(msg);
  }

  void publish_depth(const rs2::depth_frame & f, rclcpp::Time stamp)
  {
    auto msg = sensor_msgs::msg::Image();
    msg.header.stamp    = stamp;
    msg.header.frame_id = depth_frame_id_;
    msg.height   = f.get_height();
    msg.width    = f.get_width();
    msg.encoding = "mono16";          // 16-bit unsigned, mm
    msg.is_bigendian = false;
    msg.step     = static_cast<uint32_t>(f.get_width() * 2);

    // 传感器原始 Z16 单位不一定是 1mm（L515 为 0.25mm）。
    // 按帧的实际 units 缩放到 mm 再发布，保证下游按 D455 约定（mono16=mm）处理正确。
    const float scale_to_mm = static_cast<float>(f.get_units()) * 1000.0f;
    const int w = f.get_width(), h = f.get_height();
    std::vector<uint16_t> scaled(static_cast<size_t>(w) * h);
    const auto * src = reinterpret_cast<const uint16_t *>(f.get_data());
    for (size_t i = 0; i < scaled.size(); ++i) {
      scaled[i] = static_cast<uint16_t>(src[i] * scale_to_mm + 0.5f);
    }
    const auto * out = reinterpret_cast<const uint8_t *>(scaled.data());
    msg.data.assign(out, out + scaled.size() * sizeof(uint16_t));

    depth_pub_->publish(msg);
  }

  void publish_cinfo(rclcpp::Time stamp)
  {
    if (!camera_info_msg_) return;
    camera_info_msg_->header.stamp = stamp;
    cinfo_pub_->publish(*camera_info_msg_);
  }

  // ── parameter storage ─────────────────────────────────────────────────
  std::string serial_no_;
  int color_width_, color_height_, color_fps_;
  int depth_width_, depth_height_, depth_fps_;
  std::string camera_name_, camera_namespace_, l515_preset_;
  bool publish_depth_;
  bool initial_reset_{true};

  std::string color_topic_, depth_topic_, cinfo_topic_;
  std::string color_frame_id_, depth_frame_id_;

  // ── ROS publishers ────────────────────────────────────────────────────
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr color_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr cinfo_pub_;
  sensor_msgs::msg::CameraInfo::SharedPtr camera_info_msg_;

  // ── librealsense objects ──────────────────────────────────────────────
  rs2::pipeline       pipe_;
  rs2::config         cfg_;
  rs2::pipeline_profile profile_;
  std::unique_ptr<rs2::align> align_;

  // ── threading ─────────────────────────────────────────────────────────
  std::thread        camera_thread_;
  std::atomic<bool>  stop_requested_{false};
  bool               pipeline_started_{false};
};

// ── entry point ──────────────────────────────────────────────────────────
int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<L515CameraNode>();
    rclcpp::spin(node);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(rclcpp::get_logger("l515_camera"), "Fatal: %s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
