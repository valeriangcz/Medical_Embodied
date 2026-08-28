#include "behaviortree_cpp/bt_factory.h"
#include "behaviortree_cpp/loggers/groot2_publisher.h"
// #include "behaviortree_cpp/loggers/bt_cout_logger.h"
// #include "behaviortree_cpp/loggers/bt_file_logger_v2.h"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <std_msgs/msg/bool.hpp>

#include "interfaces/action/call_nurse.hpp"
#include "interfaces/action/llm_interaction.hpp"
#include "interfaces/action/navigate.hpp"
#include "interfaces/msg/action_status.hpp"
#include "interfaces/msg/battery.hpp"
#include "interfaces/msg/fault.hpp"
#include "interfaces/srv/detect_anomaly.hpp"
#include "interfaces/srv/face_identify.hpp"
#include "interfaces/srv/set_config.hpp"
#include "mode_define.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include <filesystem>

using namespace BT;
using namespace std::chrono_literals;

namespace {

int navTypeFromString(const std::string& s)
{
    if (s == "goal")
    {
        return NAVIGATION::GOAL;
    }
    if (s == "stop")
    {
        return NAVIGATION::STOP;
    }
    if (s == "dock")
    {
        return NAVIGATION::DOCK;
    }
    return NAVIGATION::STOP;
}

int detectModeFromString(const std::string& s)
{
    if (s == "area")
    {
        return DETECT::AREA;
    }
    if (s == "bed")
    {
        return DETECT::BED;
    }
    return DETECT::AREA;
}

int interactionModeFromString(const std::string& s)
{
    if (s == "alert")
    {
        return INTERACTION::ALERT;
    }
    if (s == "passive")
    {
        return INTERACTION::PASSIVE;
    }
    if (s == "interrupt")
    {
        return INTERACTION::INTERUPT;
    }
    return INTERACTION::PASSIVE;
}

struct NodePort //节点使用的端口的名称，不同节点可能有重复使用，比如mode
{
    static constexpr const char* patrol_triggered = "patrol_triggered";
    static constexpr const char* fault_type = "fault_type";
    static constexpr const char* fault_severity = "fault_severity";
    static constexpr const char* battery_soc = "battery_soc";
    static constexpr const char* battery_charging = "battery_charging";
    static constexpr const char* battery_voltage = "battery_voltage";
    static constexpr const char* call_signal = "call_signal";
    static constexpr const char* bed_id = "bed_id";
    static constexpr const char* bed_queue = "bed_queue";
    static constexpr const char* patrol_point = "patrol_point";
    static constexpr const char* mode = "mode";
    static constexpr const char* context = "context";
    static constexpr const char* person_id = "person_id";
    static constexpr const char* face_confidence = "face_confidence";
    static constexpr const char* summary = "summary";
    static constexpr const char* call_nurse = "call_nurse";
};


struct RosContext
{
    rclcpp::Node::SharedPtr node;
    rclcpp::Subscription<interfaces::msg::Battery>::SharedPtr battery_sub;
    rclcpp::Subscription<interfaces::msg::Fault>::SharedPtr fault_sub;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr call_signal_sub;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr patrol_trigger_sub;

    rclcpp::Client<interfaces::srv::DetectAnomaly>::SharedPtr anomaly_client;
    rclcpp::Client<interfaces::srv::FaceIdentify>::SharedPtr face_identify_client;
    rclcpp::Client<interfaces::srv::SetConfig>::SharedPtr loadconfig_client;

    rclcpp_action::Client<interfaces::action::Navigate>::SharedPtr navigate_client;
    rclcpp_action::Client<interfaces::action::LLMInteraction>::SharedPtr llm_client;
    rclcpp_action::Client<interfaces::action::CallNurse>::SharedPtr call_nurse_client;

    double battery_low_threshold = 20.0;
    double battery_full_threshold = 80.0;
};

struct PatrolContext
{
    std::string route_id = "route_a";
    int cycles_total = 1;
    int cycles_remaining = 1;
    int point_index = 0;
    // 巡诊点列表(waypoint index): 不再硬编码, 由 waypoints.yaml 中的 patrol_* 条目解析填充
    std::vector<int> points;
    bool complete = false;
};

// 查找导航 waypoints.yaml 路径:
//   1. ROS 参数 waypoints_config_path 显式指定
//   2. ament index 中的 xjrobot_bridge 包 share/config/waypoints.yaml
//   3. 源码目录回退: workspace/src/nav/xjrobot_bridge/config/waypoints.yaml
std::string find_waypoints_config_path(const rclcpp::Node::SharedPtr& node)
{
    // 1) 参数显式指定
    std::string param_path;
    try {
        param_path = node->declare_parameter<std::string>("waypoints_config_path", "");
    } catch (const std::exception&) {
        param_path = "";
    }
    if (!param_path.empty() && std::ifstream(param_path).good()) {
        return param_path;
    }
    // 2) ament share 目录
    try {
        const std::string share =
            ament_index_cpp::get_package_share_directory("xjrobot_bridge");
        const std::string p = share + "/config/waypoints.yaml";
        if (std::ifstream(p).good()) {
            return p;
        }
    } catch (const std::exception&) {}
    // 3) 源码目录回退: 从可执行文件路径向上定位 workspace 根
    try {
        namespace fs = std::filesystem;
        fs::path exe = fs::read_symlink("/proc/self/exe");
        // .../install/behavior_tree/lib/ros_bt_runner -> 向上4级 = workspace 根
        fs::path root = exe.parent_path().parent_path().parent_path().parent_path();
        fs::path p = root / "src" / "nav" / "xjrobot_bridge" / "config" / "waypoints.yaml";
        if (fs::exists(p)) {
            return p.string();
        }
    } catch (const std::exception&) {}
    return "";
}

// 解析 waypoints.yaml, 提取所有 patrol_* 条目的 waypoint index, 升序排列
std::vector<int> parse_patrol_points_from_waypoints(const std::string& path)
{
    std::vector<int> points;
    if (path.empty()) {
        return points;
    }
    try {
        YAML::Node root = YAML::LoadFile(path);
        YAML::Node params = root["xjrobot_bridge_node"]["ros__parameters"];
        if (!params || !params.IsMap()) {
            // 兼容平铺结构
            params = root;
        }
        YAML::Node waypoints = params["waypoints"];
        if (waypoints && waypoints.IsMap()) {
            for (const auto& it : waypoints) {
                const std::string key = it.first.as<std::string>();
                if (key.rfind("patrol_", 0) == 0) {
                    const int index = it.second["index"].as<int>(-1);
                    if (index >= 0) {
                        points.push_back(index);
                    }
                }
            }
        }
        std::sort(points.begin(), points.end());
    } catch (const std::exception& e) {
        std::cerr << "[ros_bt_runner] 解析 waypoints.yaml 失败: " << e.what() << "\n";
    }
    return points;
}

template <typename T>
void setRootValue(const Blackboard::Ptr& bb, const char* key, const T& value)
{
    if (bb)
    {
        bb->set(key, value);
    }
}

void initializeBlackboardDefaults(const Blackboard::Ptr& bb)
{
    //黑板变量可能与端口的命名不同，端口名 -> 黑板key -> 黑板value
    //端口名 -> 黑板key/value 在行为树中定义
    //这里定义了黑板key -> 黑板 value, 不过在次项目中，为了方便，很多端口与黑板key相同，因此可以这样定义
    bb->set(NodePort::bed_id, -1);
    bb->set(NodePort::bed_queue, std::vector<int>{});
    bb->set(NodePort::call_nurse, false);
    bb->set(NodePort::call_signal, false);
    bb->set(NodePort::person_id, -1);
    bb->set(NodePort::face_confidence, 0.0f);
    bb->set(NodePort::summary, std::string(""));
    bb->set(NodePort::context, std::string(""));
    bb->set(NodePort::patrol_point, -1);
    bb->set(NodePort::patrol_triggered, false);
    bb->set(NodePort::battery_soc, 100.0f);
    bb->set(NodePort::battery_charging, false);
    bb->set(NodePort::battery_voltage, 0.0f);
    bb->set(NodePort::fault_type, std::string(""));
    bb->set(NodePort::fault_severity, 0);
    bb->set("interaction_mode",std::string("none"));
}

class IdleWait : public StatefulActionNode
{
public:
    IdleWait(const std::string& name, const NodeConfig& config) : StatefulActionNode(name, config) {}

    static PortsList providedPorts() { return {}; }
    NodeStatus onStart() override { 
        return NodeStatus::RUNNING; }
    NodeStatus onRunning() override {
        return NodeStatus::RUNNING; }
    void onHalted() override {}
};

class NavgateTo : public StatefulActionNode
{
public:
    NavgateTo(
        const std::string& name,
        const NodeConfig& config,
        const rclcpp_action::Client<interfaces::action::Navigate>::SharedPtr& navigate_client)
        : StatefulActionNode(name, config), navigate_client_(navigate_client) {}

    static PortsList providedPorts()
    {
        return {InputPort<int>("target"), InputPort<std::string>("nav_type")};
    }

    NodeStatus onStart() override
    {
        const auto target = getInput<int>("target").value_or(-1);
        const auto nav_type_str = getInput<std::string>("nav_type").value_or("stop");
        const int nav_type = navTypeFromString(nav_type_str);
        std::cout << "NavgateTo start target_index=" << target << " nav_type=" << nav_type_str << "\n";

        // STOP 前清理 client 上可能残留的 goal（例如 patrol 被 halt 后未 cancel 的目标）。
        if (nav_type == NAVIGATION::STOP) {
            navigate_client_->async_cancel_all_goals();
        }

        if (!navigate_client_->wait_for_action_server(100ms))
        {
            if(nav_type_str=="stop"){
                std::cout << "Navgate server not found, nav type is "<<nav_type_str<<"\n";
                return NodeStatus::SUCCESS;
            }
            std::cout << "[ERR ] NavgateTo no server\n";
            return NodeStatus::FAILURE;
        }
        std::cout << " NavgateTo server Find\n";
        

        interfaces::action::Navigate::Goal goal;
        goal.target_index = target;
        goal.nav_type = nav_type;

        send_future_ = navigate_client_->async_send_goal(goal);
        phase_ = Phase::WAIT_GOAL;
        goal_handle_.reset();
        return NodeStatus::RUNNING;
    }

    NodeStatus onRunning() override
    {
        if (phase_ == Phase::WAIT_GOAL)
        {
            if (send_future_.wait_for(0ms) != std::future_status::ready)
            {
                // std::cout << " NavgateTo waiting goal\n";
                return NodeStatus::RUNNING;
            }
            goal_handle_ = send_future_.get();
            if (!goal_handle_)
            {
                // std::cout << " NavgateTo get empty goal handle\n";
                phase_ = Phase::IDLE;
                return NodeStatus::FAILURE;
            }
            result_future_ = navigate_client_->async_get_result(goal_handle_);
            phase_ = Phase::WAIT_RESULT;
            return NodeStatus::RUNNING;
        }

        if (phase_ == Phase::WAIT_RESULT)
        {
            // std::cout << " NavgateTo get goal handle waiting res\n";
            if (result_future_.wait_for(0ms) != std::future_status::ready)
            {
                return NodeStatus::RUNNING;
            }

            auto wrapped = result_future_.get();
            phase_ = Phase::IDLE;
            goal_handle_.reset();
            const uint8_t status_code = wrapped.result ?
                wrapped.result->status.status : interfaces::msg::ActionStatus::ABORTED;
            std::cout << "NavgateTo complete result_code=" << static_cast<int>(wrapped.code)
                      << " status=" << static_cast<int>(status_code) << "\n";
            if (wrapped.code == rclcpp_action::ResultCode::SUCCEEDED) {
                return (status_code == interfaces::msg::ActionStatus::OK)
                           ? NodeStatus::SUCCESS
                           : NodeStatus::FAILURE;
            }
            // 被 STOP / 新目标抢占时桥接层返回 PREEMPTED，行为树应继续而非重试。
            if (wrapped.code == rclcpp_action::ResultCode::CANCELED &&
                status_code == interfaces::msg::ActionStatus::PREEMPTED)
            {
                return NodeStatus::SUCCESS;
            }
            return NodeStatus::FAILURE;
        }

        return NodeStatus::FAILURE;
    }

    void onHalted() override
    {
        if (goal_handle_) {
            navigate_client_->async_cancel_goal(goal_handle_);
        } else if (phase_ != Phase::IDLE) {
            navigate_client_->async_cancel_all_goals();
        }
        phase_ = Phase::IDLE;
        goal_handle_.reset();
    }

private:
    using GoalHandle = rclcpp_action::ClientGoalHandle<interfaces::action::Navigate>;
    enum class Phase { IDLE, WAIT_GOAL, WAIT_RESULT };

    Phase phase_ = Phase::IDLE;
    rclcpp_action::Client<interfaces::action::Navigate>::SharedPtr navigate_client_;
    GoalHandle::SharedPtr goal_handle_;
    std::shared_future<GoalHandle::SharedPtr> send_future_;
    std::shared_future<GoalHandle::WrappedResult> result_future_;
};

class LLMInteraction : public StatefulActionNode
{
public:
    LLMInteraction(
        const std::string& name,
        const NodeConfig& config,
        const rclcpp_action::Client<interfaces::action::LLMInteraction>::SharedPtr& llm_client)
        : StatefulActionNode(name, config), llm_client_(llm_client) {}

    static PortsList providedPorts()
    {
        return {InputPort<std::string>(NodePort::mode),
                InputPort<int>(NodePort::person_id),
                InputPort<std::string>(NodePort::context),
                OutputPort<bool>(NodePort::call_nurse),
                OutputPort<std::string>(NodePort::summary)};
    }

    NodeStatus onStart() override
    {
        std::cout<<"LLMInteraction start\n";
        const auto mode_str = getInput<std::string>(NodePort::mode).value_or("passive");
        int mode = interactionModeFromString(mode_str);
        const int person_id = getInput<int>(NodePort::person_id).value_or(-1);
        const auto context = getInput<std::string>(NodePort::context).value_or("none");
        std::cout<<"LLMInteraction mode="<<mode_str<<" person_id="<<person_id<<" context="<<context<<"\n";
        if (!llm_client_->wait_for_action_server(100ms))
        {
            std::cout<<"[ERR ] LLMInteraction no server\n";
            return NodeStatus::FAILURE;
        }

        interfaces::action::LLMInteraction::Goal goal;
        goal.mode = mode;
        goal.person_id = person_id;
        goal.context = context;

        send_future_ = llm_client_->async_send_goal(goal);
        phase_ = Phase::WAIT_GOAL;
        goal_handle_.reset();
        return NodeStatus::RUNNING;
    }

    NodeStatus onRunning() override
    {
        if (phase_ == Phase::WAIT_GOAL)
        {
            if (send_future_.wait_for(0ms) != std::future_status::ready)
            {
                return NodeStatus::RUNNING;
            }
            goal_handle_ = send_future_.get();
            if (!goal_handle_)
            {
                phase_ = Phase::IDLE;
                return NodeStatus::FAILURE;
            }
            result_future_ = llm_client_->async_get_result(goal_handle_);
            phase_ = Phase::WAIT_RESULT;
            return NodeStatus::RUNNING;
        }

        if (phase_ == Phase::WAIT_RESULT)
        {
            if (result_future_.wait_for(0ms) != std::future_status::ready)
            {
                return NodeStatus::RUNNING;
            }

            auto wrapped = result_future_.get();
            phase_ = Phase::IDLE;
            goal_handle_.reset();
            if (wrapped.code != rclcpp_action::ResultCode::SUCCEEDED)
            {
                return NodeStatus::FAILURE;
            }

            setOutput(NodePort::call_nurse, wrapped.result->need_call_nurse);
            setOutput(NodePort::summary, wrapped.result->summary);
            std::cout<<"LLMInteraction geting  Result"<<int(wrapped.result->status.status)<<"\n";

            return (wrapped.result->status.status == interfaces::msg::ActionStatus::OK)
                       ? NodeStatus::SUCCESS
                       : NodeStatus::FAILURE;
        }

        return NodeStatus::FAILURE;
    }

    void onHalted() override
    {
        phase_ = Phase::IDLE;
        goal_handle_.reset();
    }

private:
    using GoalHandle = rclcpp_action::ClientGoalHandle<interfaces::action::LLMInteraction>;
    enum class Phase { IDLE, WAIT_GOAL, WAIT_RESULT };

    Phase phase_ = Phase::IDLE;
    rclcpp_action::Client<interfaces::action::LLMInteraction>::SharedPtr llm_client_;
    GoalHandle::SharedPtr goal_handle_;
    std::shared_future<GoalHandle::SharedPtr> send_future_;
    std::shared_future<GoalHandle::WrappedResult> result_future_;
};

class CallDutyNurse : public StatefulActionNode
{
public:
    CallDutyNurse(
        const std::string& name,
        const NodeConfig& config,
        const rclcpp_action::Client<interfaces::action::CallNurse>::SharedPtr& call_nurse_client)
        : StatefulActionNode(name, config), call_nurse_client_(call_nurse_client) {}

    static PortsList providedPorts()
    {
        return {InputPort<int>(NodePort::bed_id),
                InputPort<std::string>(NodePort::summary),
                OutputPort<bool>(NodePort::call_nurse)};
    }

    NodeStatus onStart() override
    {
        std::cout<<"CallDutyNurse start\n";
        const int bed_id = getInput<int>(NodePort::bed_id).value_or(-1);
        const auto summary = getInput<std::string>(NodePort::summary).value_or("unknown");
        if (bed_id >= 0)
        {
            if (pending_bed_ids_.empty() || pending_bed_ids_.back() != bed_id || pending_summaries_.back() != summary)
            {
                pending_bed_ids_.push_back(bed_id);
                pending_summaries_.push_back(summary);
            }
        }

        if (pending_bed_ids_.empty())
        {
            setOutput(NodePort::call_nurse, false);
            return NodeStatus::SUCCESS;
        }

        if (!call_nurse_client_->wait_for_action_server(100ms))
        {
            setOutput(NodePort::call_nurse, true);
            return NodeStatus::FAILURE;
        }

        interfaces::action::CallNurse::Goal goal;
        goal.bed_ids = pending_bed_ids_;
        goal.summarys = pending_summaries_;

        send_future_ = call_nurse_client_->async_send_goal(goal);
        phase_ = Phase::WAIT_GOAL;
        goal_handle_.reset();
        return NodeStatus::RUNNING;
    }

    NodeStatus onRunning() override
    {
        if (phase_ == Phase::WAIT_GOAL)
        {
            if (send_future_.wait_for(0ms) != std::future_status::ready)
            {
                return NodeStatus::RUNNING;
            }
            goal_handle_ = send_future_.get();
            if (!goal_handle_)
            {
                phase_ = Phase::IDLE;
                setOutput(NodePort::call_nurse, true);
                return NodeStatus::FAILURE;
            }
            result_future_ = call_nurse_client_->async_get_result(goal_handle_);
            phase_ = Phase::WAIT_RESULT;
            return NodeStatus::RUNNING;
        }

        if (phase_ == Phase::WAIT_RESULT)
        {
            if (result_future_.wait_for(0ms) != std::future_status::ready)
            {
                return NodeStatus::RUNNING;
            }

            auto wrapped = result_future_.get();
            phase_ = Phase::IDLE;
            goal_handle_.reset();
            const bool ok = wrapped.code == rclcpp_action::ResultCode::SUCCEEDED &&
                            wrapped.result->status.status == interfaces::msg::ActionStatus::OK;
            if (ok)
            {
                pending_bed_ids_.clear();
                pending_summaries_.clear();
            }
            setOutput(NodePort::call_nurse, !ok);
            std::cout<<"Callduty nurse geting  Result"<<int(wrapped.result->status.status)<<"\n";
            return ok ? NodeStatus::SUCCESS : NodeStatus::FAILURE;
        }

        return NodeStatus::FAILURE;
    }

    void onHalted() override
    {
        phase_ = Phase::IDLE;
        goal_handle_.reset();
    }

private:
    using GoalHandle = rclcpp_action::ClientGoalHandle<interfaces::action::CallNurse>;
    enum class Phase { IDLE, WAIT_GOAL, WAIT_RESULT };

    Phase phase_ = Phase::IDLE;
    rclcpp_action::Client<interfaces::action::CallNurse>::SharedPtr call_nurse_client_;
    GoalHandle::SharedPtr goal_handle_;
    
    std::shared_future<GoalHandle::SharedPtr> send_future_;
    std::shared_future<GoalHandle::WrappedResult> result_future_;
    std::vector<int> pending_bed_ids_;
    std::vector<std::string> pending_summaries_;
};

class SelectNextBed : public SyncActionNode
{
public:
    SelectNextBed(const std::string& name, const NodeConfig& config) : SyncActionNode(name, config) {}

    static PortsList providedPorts()
    {
        return {BidirectionalPort<std::vector<int>>(NodePort::bed_queue), OutputPort<int>(NodePort::bed_id)};
    }

    NodeStatus tick() override
    {
        auto queue = getInput<std::vector<int>>(NodePort::bed_queue).value_or(std::vector<int>{});
        if (!queue.empty())
        {
            const int bed_id = queue.front();
            queue.erase(queue.begin());
            setOutput(NodePort::bed_queue, queue);
            setOutput(NodePort::bed_id, bed_id);
            std::cout<<">>>>>>>>>>>>>>>Select next bed"<<"\n";
            std::cout<<"bed queue size=: "<<queue.size()<<"\n";
            std::cout<<"bed queue:";
            for(auto i:queue){
                std::cout<<i<<" ";
            }
            std::cout<<"\n";
            std::cout<<"selected bed: "<<bed_id<<"\n";

            return NodeStatus::SUCCESS;
        }
        setOutput(NodePort::bed_queue, queue);
        setOutput(NodePort::bed_id, -1);
        return NodeStatus::FAILURE;
    }
};

class FaceIdentify : public SyncActionNode
{
public:
    FaceIdentify(
        const std::string& name,
        const NodeConfig& config,
        const rclcpp::Node::SharedPtr& node,
        const rclcpp::Client<interfaces::srv::FaceIdentify>::SharedPtr& face_identify_client)
        : SyncActionNode(name, config), node_(node), face_identify_client_(face_identify_client) {}

    static PortsList providedPorts()
    {
        return {OutputPort<int>(NodePort::person_id), OutputPort<float>(NodePort::face_confidence)};
    }

    NodeStatus tick() override
    {
        auto set_unknown = [this]() {
            setOutput(NodePort::person_id, -1);
            setOutput(NodePort::face_confidence, 0.0f);
            return NodeStatus::SUCCESS;
        };

        auto req = std::make_shared<interfaces::srv::FaceIdentify::Request>();
        auto fut = face_identify_client_->async_send_request(req);
        
        if (rclcpp::spin_until_future_complete(node_, fut, 3s) != rclcpp::FutureReturnCode::SUCCESS)
        {
            return set_unknown();
        }

        const auto res = fut.get();
        setOutput(NodePort::person_id, static_cast<int>(res->person_id));
        setOutput(NodePort::face_confidence, static_cast<float>(res->confidence));
        return NodeStatus::SUCCESS;
    }

private:
    rclcpp::Node::SharedPtr node_;
    rclcpp::Client<interfaces::srv::FaceIdentify>::SharedPtr face_identify_client_;
};

class NextPatrolPoint : public SyncActionNode
{
public:
    NextPatrolPoint(
        const std::string& name,
        const NodeConfig& config,
        const std::shared_ptr<PatrolContext>& patrol_ctx)
        : SyncActionNode(name, config), patrol_ctx_(patrol_ctx) {}

    static PortsList providedPorts() { return {OutputPort<int>(NodePort::patrol_point)}; }

    NodeStatus tick() override
    {
        if (patrol_ctx_->cycles_remaining <= 0)
        {
            return NodeStatus::FAILURE;
        }
        if (patrol_ctx_->points.empty())
        {
            patrol_ctx_->complete = true;
            patrol_ctx_->cycles_remaining = 0;
            return NodeStatus::FAILURE;
        }

        int index = patrol_ctx_->point_index;
        if (index < 0 || index >= static_cast<int>(patrol_ctx_->points.size()))
        {
            index = 0;
        }

        const int point = patrol_ctx_->points[index];
        index += 1;
        if (index >= static_cast<int>(patrol_ctx_->points.size()))
        {
            index = 0;
            patrol_ctx_->cycles_remaining -= 1;
            if (patrol_ctx_->cycles_remaining <= 0)
            {
                patrol_ctx_->complete = true;
            }
        }
        patrol_ctx_->point_index = index;
        setOutput(NodePort::patrol_point, point);
        std::cout<<">>>>>>>>>>>>>>>Select next patrol point ="<<point<<"\n";
        return NodeStatus::SUCCESS;
    }

private:
    std::shared_ptr<PatrolContext> patrol_ctx_;
};

class Detect_BedProcess : public SyncActionNode
{
public:
    Detect_BedProcess(
        const std::string& name,
        const NodeConfig& config,
        const rclcpp::Node::SharedPtr& node,
        const rclcpp::Client<interfaces::srv::DetectAnomaly>::SharedPtr& anomaly_client)
        : SyncActionNode(name, config), node_(node), anomaly_client_(anomaly_client) {}

    static PortsList providedPorts()
    {
        return {InputPort<std::string>(NodePort::mode),
                InputPort<int>("patrol_bed_id"),
                OutputPort<std::vector<int>>(NodePort::bed_queue),
                OutputPort<std::string>("interaction_mode"),
                OutputPort<std::string>(NodePort::context)};
    }

    NodeStatus tick() override
    {
        const auto mode_str = getInput<std::string>(NodePort::mode).value_or("area");
        const int scan_mode = detectModeFromString(mode_str);
        const int patrol_bed_id = getInput<int>("patrol_bed_id").value_or(-1);

        auto req = std::make_shared<interfaces::srv::DetectAnomaly::Request>();
        req->mode = scan_mode;
        req->area_bed_id = patrol_bed_id;

        auto fut = anomaly_client_->async_send_request(req);
        if (rclcpp::spin_until_future_complete(node_, fut, 8s) != rclcpp::FutureReturnCode::SUCCESS)
        {
            return NodeStatus::FAILURE;
        }
        const auto res = fut.get();

        if (scan_mode == DETECT::BED)
        {
            setOutput("interaction_mode", res->is_anomaly ? std::string("alert") : std::string("passive"));
            setOutput(NodePort::context, res->is_anomaly ? res->details : std::string(""));
            return NodeStatus::SUCCESS;
        }
        std::vector<int> bed_queue(res->bed_ids.begin(), res->bed_ids.end());

        std::cout<<"Detect bed process get bed queue:";
        for (auto bed_id :bed_queue)
        {
            std::cout<<bed_id<<" ";
        }
        std::cout<<"\n";
        std::cout<<"bedqueue size: "<<bed_queue.size()<<"\n";

        setOutput(NodePort::bed_queue, bed_queue);
        return NodeStatus::SUCCESS;
    }

private:
    rclcpp::Node::SharedPtr node_;
    rclcpp::Client<interfaces::srv::DetectAnomaly>::SharedPtr anomaly_client_;
};

class WaitChargeUntil : public StatefulActionNode
{
public:
    WaitChargeUntil(
        const std::string& name, 
        const NodeConfig& config,
        const float& battery_full_threshold) : 
        StatefulActionNode(name, config),target_(battery_full_threshold) {}

    static PortsList providedPorts() { return {InputPort<float>(NodePort::battery_soc)}; }

    NodeStatus onStart() override
    {
        current_ = getInput<float>(NodePort::battery_soc).value_or(0.0);
        return NodeStatus::RUNNING;
    }

    NodeStatus onRunning() override
    {
        if (current_ >= target_)
        {
            return NodeStatus::SUCCESS;
        }
        return NodeStatus::RUNNING;
    }

    void onHalted() override {}

private:
    float current_;
    float target_ ;
};

}  // namespace

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    auto ros_node = std::make_shared<rclcpp::Node>("ros_bt_runner");
    auto root_bb = Blackboard::create();
    auto ros_ctx = std::make_shared<RosContext>();
    auto patrol_ctx = std::make_shared<PatrolContext>();
    auto config_loaded = std::make_shared<bool>(false);
    ros_ctx->node = ros_node;

    initializeBlackboardDefaults(root_bb);

    ros_node->declare_parameter("battery_low_threshold", 20.0);
    ros_node->declare_parameter("battery_full_threshold",80.0);
    ros_node->declare_parameter("config_id", std::string("default"));
    ros_node->declare_parameter("max_ticks", 10000000);

    ros_ctx->battery_low_threshold = ros_node->get_parameter("battery_low_threshold").as_double();
    ros_ctx->battery_full_threshold = ros_node->get_parameter("battery_full_threshold").as_double();
    const auto config_id = ros_node->get_parameter("config_id").as_string();

    // 巡诊点列表从 waypoints.yaml 解析 (patrol_* 条目 -> waypoint index), 不再硬编码
    const std::string waypoints_path = find_waypoints_config_path(ros_node);
    patrol_ctx->points = parse_patrol_points_from_waypoints(waypoints_path);
    std::cout << "[ros_bt_runner] waypoints config: "
              << (waypoints_path.empty() ? "(not found, patrol points empty)" : waypoints_path)
              << "\n";
    std::cout << "[ros_bt_runner] patrol points:";
    for (const int p : patrol_ctx->points) {
        std::cout << " " << p;
    }
    std::cout << "\n";

    ros_ctx->battery_sub = ros_node->create_subscription<interfaces::msg::Battery>(
        "/battery", 10, [root_bb](const interfaces::msg::Battery::SharedPtr msg) {
            setRootValue(root_bb, NodePort::battery_soc, msg->soc);
            setRootValue(root_bb, NodePort::battery_charging, msg->charging);
            setRootValue(root_bb, NodePort::battery_voltage, msg->voltage);
        });
    ros_ctx->fault_sub = ros_node->create_subscription<interfaces::msg::Fault>(
        "/fault", 10, [root_bb](const interfaces::msg::Fault::SharedPtr msg) {
            const bool active = msg->severity > 0;
            setRootValue(root_bb, NodePort::fault_type, msg->fault_type);
            setRootValue(root_bb, NodePort::fault_severity, msg->severity);
        });
    ros_ctx->call_signal_sub = ros_node->create_subscription<std_msgs::msg::Bool>(
        "/call_signal", 10, [root_bb](const std_msgs::msg::Bool::SharedPtr msg) {
            setRootValue(root_bb, NodePort::call_signal, msg->data);
        });
    ros_ctx->patrol_trigger_sub = ros_node->create_subscription<std_msgs::msg::Bool>(
        "/patrol_triggered", 10, [root_bb](const std_msgs::msg::Bool::SharedPtr msg) {
            setRootValue(root_bb, NodePort::patrol_triggered, msg->data);
        });

    ros_ctx->anomaly_client = ros_node->create_client<interfaces::srv::DetectAnomaly>("/detect_anomaly");
    ros_ctx->face_identify_client = ros_node->create_client<interfaces::srv::FaceIdentify>("/face_identify");
    ros_ctx->loadconfig_client = ros_node->create_client<interfaces::srv::SetConfig>("/loadconfig/set_config");

    ros_ctx->navigate_client = rclcpp_action::create_client<interfaces::action::Navigate>(ros_node, "navigate");
    ros_ctx->llm_client = rclcpp_action::create_client<interfaces::action::LLMInteraction>(ros_node, "llm_interaction");
    ros_ctx->call_nurse_client = rclcpp_action::create_client<interfaces::action::CallNurse>(ros_node, "call_nurse");

    BehaviorTreeFactory factory;

    factory.registerBuilder<IdleWait>("IdleWait", [](const std::string& name, const NodeConfig& config) {
        return std::make_unique<IdleWait>(name, config);
    });
    factory.registerBuilder<NavgateTo>(
        "NavgateTo", [navigate_client = ros_ctx->navigate_client](const std::string& name, const NodeConfig& config) {
            return std::make_unique<NavgateTo>(name, config, navigate_client);
        });
    factory.registerBuilder<LLMInteraction>(
        "LLMInteraction", [llm_client = ros_ctx->llm_client](const std::string& name, const NodeConfig& config) {
            return std::make_unique<LLMInteraction>(name, config, llm_client);
        });
    factory.registerBuilder<CallDutyNurse>(
        "CallDutyNurse", [call_nurse_client = ros_ctx->call_nurse_client](const std::string& name, const NodeConfig& config) {
            return std::make_unique<CallDutyNurse>(name, config, call_nurse_client);
        });
    factory.registerBuilder<SelectNextBed>("SelectNextBed", [](const std::string& name, const NodeConfig& config) {
        return std::make_unique<SelectNextBed>(name, config);
    });
    factory.registerBuilder<FaceIdentify>(
        "FaceIdentify",
        [node = ros_ctx->node, face_identify_client = ros_ctx->face_identify_client](const std::string& name, const NodeConfig& config) {
            return std::make_unique<FaceIdentify>(name, config, node, face_identify_client);
        });
    factory.registerBuilder<NextPatrolPoint>(
        "NextPatrolPoint", [patrol_ctx](const std::string& name, const NodeConfig& config) {
            return std::make_unique<NextPatrolPoint>(name, config, patrol_ctx);
        });
    factory.registerBuilder<Detect_BedProcess>(
        "Detect_BedProcess",
        [node = ros_ctx->node, anomaly_client = ros_ctx->anomaly_client](const std::string& name, const NodeConfig& config) {
            return std::make_unique<Detect_BedProcess>(name, config, node, anomaly_client);
        });
    factory.registerBuilder<WaitChargeUntil>("WaitChargeUntil", [threshold=ros_ctx->battery_full_threshold](const std::string& name, const NodeConfig& config) {
        return std::make_unique<WaitChargeUntil>(name, config,threshold);
    });

    factory.registerSimpleAction(
        "LoadConfig",
        [loadconfig_client = ros_ctx->loadconfig_client, 
            ros_node_ = ros_ctx->node,
            patrol_ctx, 
            config_loaded, 
            config_id,
            waypoints_path](TreeNode& node) {
            if (loadconfig_client->wait_for_service(2s)){
                auto req = std::make_shared<interfaces::srv::SetConfig::Request>();
                req->config_id = config_id;
                auto fut = loadconfig_client->async_send_request(req);
                if(rclcpp::spin_until_future_complete(ros_node_,fut,2s) == rclcpp::FutureReturnCode::SUCCESS && fut.get()->ok){
                    // 巡诊点列表从 waypoints.yaml 的 patrol_* 条目解析, 不硬编码
                    patrol_ctx->points = parse_patrol_points_from_waypoints(waypoints_path);
                    patrol_ctx->route_id = "route_a";
                    patrol_ctx->cycles_total = 1;
                    patrol_ctx->cycles_remaining = 1;
                    patrol_ctx->point_index = 0;
                    patrol_ctx->complete = false;
                    std::cout << "[LoadConfig] patrol points from waypoints:";
                    for (const int p : patrol_ctx->points) {
                        std::cout << " " << p;
                    }
                    std::cout << "\n";
                    return NodeStatus::SUCCESS;
                }
            }
            std::cout<<"wait loadconfig srv failure\n";
            return NodeStatus::FAILURE;

        });

    factory.registerSimpleAction("SaveContext", [](TreeNode&) { return NodeStatus::SUCCESS; });
    factory.registerSimpleAction("RestoreContext", [](TreeNode&) { return NodeStatus::SUCCESS; });
    factory.registerSimpleAction("AlertChargeFault", [](TreeNode&) { return NodeStatus::SUCCESS; });

    factory.registerSimpleAction(
        "ClearInteractionMode", [](TreeNode& node) {
            node.setOutput(NodePort::mode, std::string("none"));
            return NodeStatus::SUCCESS;
        },
        {BidirectionalPort<std::string>(NodePort::mode)});
    factory.registerSimpleAction(
        "ClearPatrolTrigger", [patrol_ctx](TreeNode& node) {
            node.setOutput(NodePort::patrol_triggered, false);
            patrol_ctx->complete = false;
            patrol_ctx->cycles_remaining = patrol_ctx->cycles_total;
            patrol_ctx->point_index = 0;
            return NodeStatus::SUCCESS;
        },
        {OutputPort<bool>(NodePort::patrol_triggered)});

    factory.registerSimpleCondition("IsBedProcessComplete",[](TreeNode& node){
        const auto bed_queue = node.getInput<std::vector<int>>(NodePort::bed_queue).value_or(std::vector<int>{});
       std::cout<<"Isbedprocess complete get bed queue \n";
        for (auto bed_id : bed_queue)
        {
            std::cout<<"bed_id: "<<bed_id<<"\n";
        }
        std::cout<<"bedqueue size: "<<bed_queue.size()<<"\n";
        return bed_queue.empty()? NodeStatus::SUCCESS:NodeStatus::FAILURE;
    },{InputPort<std::vector<int>>(NodePort::bed_queue)});
    
    factory.registerSimpleCondition("IsPatrolComplete", [patrol_ctx](TreeNode&) {
        if (patrol_ctx->cycles_remaining <= 0) {
            return NodeStatus::SUCCESS;
        }
        // cycles_remaining 仍大于 0，说明巡检尚未完成，应继续走 PatrolRun 分支。
        std::cout << "Patrol have not completed, remaining cycles "
                  << patrol_ctx->cycles_remaining << "\n";
        return NodeStatus::FAILURE;
    });
    factory.registerSimpleCondition(
        "IsPatrolTriggered", [](TreeNode& node) {
            const bool triggered = node.getInput<bool>(NodePort::patrol_triggered).value_or(false);
            return triggered ? NodeStatus::SUCCESS : NodeStatus::FAILURE;
        }, {InputPort<bool>(NodePort::patrol_triggered)});

    factory.registerSimpleCondition(
        "IsCallSignal", [](TreeNode& node) {
            const bool active = node.getInput<bool>(NodePort::call_signal).value_or(false);
            if(active){
                std::cout<<"call signal active\n";
                node.setOutput(NodePort::call_signal,false);
                return NodeStatus::SUCCESS;
            }
            return NodeStatus::FAILURE;
        }, {BidirectionalPort<bool>(NodePort::call_signal)});
    factory.registerSimpleCondition(
        "IsBatteryLow", [threshold = ros_ctx->battery_low_threshold](TreeNode& node) {
            const float battery_soc = node.getInput<float>(NodePort::battery_soc).value_or(100.0f);
            return battery_soc <= threshold ? NodeStatus::SUCCESS : NodeStatus::FAILURE;
        }, {InputPort<float>(NodePort::battery_soc)});
    
    factory.registerSimpleCondition(
        "IsFault",[](TreeNode& node){
            const int severity = node.getInput<int>(NodePort::fault_severity).value_or(1);
            if (severity>0){
                const std::string type = node.getInput<std::string>(NodePort::fault_type).value_or("None");
                //TODO remind the fault type
                std::cout<<"find fault type:"<<type<<" severity:"<<severity<<"\n";
                return NodeStatus::SUCCESS;
            }
            return NodeStatus::FAILURE;
        },{InputPort<std::string>(NodePort::fault_type),InputPort<int>(NodePort::fault_severity)});

    factory.registerSimpleCondition(
        "IsInteractionMode", [](TreeNode& node) {
            const auto mode = node.getInput<std::string>(NodePort::mode).value_or("interrput");
            return (mode == "passive" || mode == "alert") ? NodeStatus::SUCCESS : NodeStatus::FAILURE;
        }, {InputPort<std::string>(NodePort::mode)});


    try
    {
        const auto xml_path = ament_index_cpp::get_package_share_directory("behavior_tree") + "/BH_xml/medical.xml";
        auto tree = factory.createTreeFromFile(xml_path, root_bb);
        //创建 logger ，进行打印
        // StdCoutLogger logger(tree);
        // FileLogger2 logger2(tree,"tree.btlog");
        Groot2Publisher bt_publisher(tree,1667);

        const int max_ticks = ros_node->get_parameter("max_ticks").as_int();
        const float kTickHz = 10;
        rclcpp::Rate rate(kTickHz);

        for (int tick = 0; tick < max_ticks && rclcpp::ok(); ++tick)
        {
            rclcpp::spin_some(ros_node);
            try
            {
                tree.tickOnce();
                // std::cout << "tic tree " << tick << "\n";
            }
            catch (const std::exception& e)
            {
                std::cerr << "[ERROR] tick=" << tick << " tree.tickOnce exception: " << e.what() << "\n";
                throw;
            }
            rate.sleep();
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "[ERROR] " << e.what() << "\n";
        rclcpp::shutdown();
        return 1;
    }

    root_bb.reset();
    ros_node.reset();
    ros_ctx.reset();
    patrol_ctx.reset();
    rclcpp::shutdown();
    return 0;
}
