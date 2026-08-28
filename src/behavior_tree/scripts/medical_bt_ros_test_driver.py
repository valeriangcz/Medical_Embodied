#!/usr/bin/env python3
"""
医疗行为树 ROS 测试驱动节点。

用于在无真实硬件/后端时，模拟传感器输入、导航、LLM 交互等接口，
便于行为树逻辑联调与回归测试。

各模拟模块可通过 ROS 参数独立开关，关闭某项模拟后该接口由真实节点提供。
"""
import threading
import time
from typing import Optional

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.node import Node
from std_msgs.msg import Bool

from interfaces.action import CallNurse, LLMInteraction, Navigate
from interfaces.msg import ActionStatus, Battery, Fault
from interfaces.srv import DetectAnomaly, FaceIdentify, SetConfig

# LLM 交互模式常量
INTERACTION_ALERT = 0      # 主动告警
INTERACTION_PASSIVE = 1    # 被动响应
INTERACTION_INTERRUPT = 2  # 打断/恢复

# 导航类型常量
NAV_GOAL = 0   # 前往目标点
NAV_STOP = 1   # 停止
NAV_DOCK = 2   # 回充/对接

# 异常检测模式常量
DETECT_AREA = 0  # 区域扫描
DETECT_BED = 1   # 单床位检测


class Metrics:
    """统计各模拟接口被调用的次数，便于测试结束后核对行为树是否触发预期动作。"""

    def __init__(self) -> None:
        self.llm_call_response = 0
        self.llm_abnormal = 0
        self.llm_passive = 0
        self.nav_stop = 0
        self.nav_dock = 0
        self.nav_patrol = 0
        self.call_nurse = 0
        self.face_identify_call_signal = 0
        self.face_identify_patrol = 0


class MedicalBtRosTestDriver(Node):
    """行为树测试驱动：提供模拟服务/动作，并按 tick 周期注入测试输入。"""

    def __init__(self) -> None:
        super().__init__('medical_bt_ros_test_driver')
        self.metrics = Metrics()
        self.section_marks = {}
        self.lock = threading.Lock()
        self.patrol_triggered = False
        self.call_signal_face_pending = False
        self._last_call_signal_published = False

        # ---------- 各模拟模块独立开关（默认 True，关闭后由真实节点接管） ----------
        self.declare_parameter('sim_navigate', True)        # 模拟导航 Action（含进度反馈）
        self.declare_parameter('sim_llm', False)             # 模拟 LLM 交互 Action
        self.declare_parameter('sim_call_nurse', False)      # 模拟呼叫护士 Action
        self.declare_parameter('sim_detect_anomaly', True)  # 模拟异常检测服务
        self.declare_parameter('sim_face_identify', True)   # 模拟人脸识别服务
        self.declare_parameter('sim_set_config', True)      # 模拟配置加载服务
        self.declare_parameter('sim_battery', True)         # 模拟 /battery 话题发布
        self.declare_parameter('sim_fault', False)           # 模拟 /fault 话题发布
        self.declare_parameter('sim_call_signal', False)     # 模拟 /call_signal 话题发布
        self.declare_parameter('sim_patrol_trigger', False)  # 模拟 /patrol_triggered 话题发布

        # ---------- tick 循环参数 ----------
        self.declare_parameter('start_delay', 2.0)   # 启动后等待秒数，再开始 tick 循环
        self.declare_parameter('ticks', 20000000)    # 模拟 tick 总次数
        self.declare_parameter('tick_hz', 10)        # 每个 tick 的间隔频率（Hz）

        # ---------- 巡逻相关参数（可由 SetConfig 服务动态覆盖） ----------
        self.declare_parameter('patrol_route_id', 'route_a')
        self.declare_parameter('patrol_cycles', 2)
        self.declare_parameter('patrol_points', ['p0', 'p1'])

        # ---------- 人脸识别模拟反馈（巡诊 / call_signal 分场景） ----------
        self.declare_parameter('patrol_face_person_id', 1)       # 巡诊/床位流程默认识别 person_id
        self.declare_parameter('patrol_face_confidence', 0.9)     # 巡诊/床位流程默认识别置信度
        self.declare_parameter('call_signal_face_person_id', 2)    # call_signal 唤醒后识别 person_id
        self.declare_parameter('call_signal_face_confidence', 0.95)  # call_signal 唤醒后识别置信度

        self.sim_navigate = self._param_bool('sim_navigate')
        self.sim_llm = self._param_bool('sim_llm')
        self.sim_call_nurse = self._param_bool('sim_call_nurse')
        self.sim_detect_anomaly = self._param_bool('sim_detect_anomaly')
        self.sim_face_identify = self._param_bool('sim_face_identify')
        self.sim_set_config = self._param_bool('sim_set_config')
        self.sim_battery = self._param_bool('sim_battery')
        self.sim_fault = self._param_bool('sim_fault')
        self.sim_call_signal = self._param_bool('sim_call_signal')
        self.sim_patrol_trigger = self._param_bool('sim_patrol_trigger')

        self._setup_mock_interfaces()
        self._log_sim_status()

    def _param_bool(self, name: str) -> bool:
        """读取布尔型 ROS 参数。"""
        return bool(self.get_parameter(name).value)

    def _setup_mock_interfaces(self) -> None:
        """按各模块开关，有条件地注册模拟服务、Action 与发布器。"""
        if self.sim_detect_anomaly:
            self.create_service(DetectAnomaly, '/detect_anomaly', self.handle_detect_anomaly)

        if self.sim_face_identify:
            self.create_service(FaceIdentify, '/face_identify', self.handle_face_identify)
            # 监听 call_signal，下一次 FaceIdentify 请求返回呼叫场景专用反馈
            self.create_subscription(Bool, '/call_signal', self._on_call_signal_received, 10)

        if self.sim_set_config:
            self.create_service(SetConfig, '/loadconfig/set_config', self.handle_set_config)

        if self.sim_battery:
            self.battery_pub = self.create_publisher(Battery, '/battery', 10)

        if self.sim_fault:
            self.fault_pub = self.create_publisher(Fault, '/fault', 10)

        if self.sim_call_signal:
            self.call_signal_pub = self.create_publisher(Bool, '/call_signal', 10)

        if self.sim_patrol_trigger:
            self.patrol_trigger_pub = self.create_publisher(Bool, '/patrol_triggered', 1)
            self.publish_patrol_triggered(False)

        if self.sim_navigate:
            self.nav_server = ActionServer(
                self, Navigate, 'navigate',
                execute_callback=self.handle_navigate,
            )

        if self.sim_llm:
            self.llm_server = ActionServer(
                self, LLMInteraction, 'llm_interaction',
                execute_callback=self.handle_llm,
                goal_callback=lambda _req: GoalResponse.ACCEPT,
                cancel_callback=lambda _gh: CancelResponse.ACCEPT,
            )

        if self.sim_call_nurse:
            self.call_nurse_server = ActionServer(
                self, CallNurse, 'call_nurse',
                execute_callback=self.handle_call_nurse,
                goal_callback=lambda _req: GoalResponse.ACCEPT,
                cancel_callback=lambda _gh: CancelResponse.ACCEPT,
            )

    def _log_sim_status(self) -> None:
        """启动时打印各模拟模块的开启/关闭状态。"""
        sim_flags = {
            'sim_navigate': self.sim_navigate,
            'sim_llm': self.sim_llm,
            'sim_call_nurse': self.sim_call_nurse,
            'sim_detect_anomaly': self.sim_detect_anomaly,
            'sim_face_identify': self.sim_face_identify,
            'sim_set_config': self.sim_set_config,
            'sim_battery': self.sim_battery,
            'sim_fault': self.sim_fault,
            'sim_call_signal': self.sim_call_signal,
            'sim_patrol_trigger': self.sim_patrol_trigger,
        }
        enabled = [name for name, on in sim_flags.items() if on]
        disabled = [name for name, on in sim_flags.items() if not on]
        self.get_logger().info(f'模拟已开启: {", ".join(enabled) or "无"}')
        if disabled:
            self.get_logger().info(f'模拟已关闭（由真实节点接管）: {", ".join(disabled)}')
        if self._needs_tick_loop():
            self.get_logger().info('tick 循环将启动（周期性话题模拟已开启）')
        else:
            self.get_logger().info(
                'tick 循环未启动：sim_battery / sim_fault / sim_call_signal / '
                'sim_patrol_trigger 均为 false；Action 与服务模拟仍可正常响应'
            )

    def _needs_tick_loop(self) -> bool:
        """任一周期性输入模拟开启时，需要运行 tick 循环。"""
        return any([
            self.sim_battery,
            self.sim_fault,
            self.sim_call_signal,
            self.sim_patrol_trigger,
        ])

    def _on_call_signal_received(self, msg: Bool) -> None:
        """收到 call_signal=true 时，标记下一次人脸识别走呼叫场景反馈。"""
        if msg.data:
            self._mark_call_signal_face_pending()

    def _mark_call_signal_face_pending(self, tick: Optional[int] = None) -> None:
        """标记：下一次 /face_identify 请求按 call_signal 场景返回模拟结果。"""
        with self.lock:
            self.call_signal_face_pending = True
        if tick is not None:
            self.get_logger().info(
                f'[SIM ] tick={tick} call_signal=true，等待 FaceIdentify 返回呼叫场景反馈'
            )
        else:
            self.get_logger().info('[SIM ] call_signal=true，等待 FaceIdentify 返回呼叫场景反馈')

    def _fill_face_identify_response(
        self,
        response: FaceIdentify.Response,
        person_id: int,
        confidence: float,
        message: str,
    ) -> FaceIdentify.Response:
        response.success = True
        response.person_id = int(person_id)
        response.confidence = float(confidence)
        response.message = message
        return response

    def publish_patrol_triggered(self, value: bool, tick: Optional[int] = None) -> None:
        """发布巡逻触发信号；值未变化时不重复发布。"""
        if not self.sim_patrol_trigger:
            return
        if value == self.patrol_triggered:
            return
        self.patrol_triggered = value
        self.patrol_trigger_pub.publish(Bool(data=value))
        if tick is not None:
            self.get_logger().info(f'[SIM ] tick={tick} patrol_triggered={str(value).lower()}')

    def handle_detect_anomaly(
        self,
        request: DetectAnomaly.Request,
        response: DetectAnomaly.Response,
    ) -> DetectAnomaly.Response:
        """模拟异常检测：区域模式返回固定床位列表，单床模式按 bed_id 奇偶判定是否异常。"""
        if request.mode == DETECT_AREA:
            response.is_anomaly = False
            response.details = 'scan'
            response.bed_ids = [0, 2, 4]
            response.urgencies = [1, 2, 3]
            return response
        is_anomaly = (request.area_bed_id % 2 == 0)
        response.is_anomaly = is_anomaly
        response.details = 'anomaly' if is_anomaly else 'normal'
        response.bed_ids = []
        response.urgencies = []
        return response

    def handle_face_identify(
        self,
        _request: FaceIdentify.Request,
        response: FaceIdentify.Response,
    ) -> FaceIdentify.Response:
        """模拟人脸识别：call_signal 后返回呼叫场景反馈，否则返回巡诊/床位场景反馈。"""
        use_call_signal_scene = False
        with self.lock:
            if self.call_signal_face_pending:
                use_call_signal_scene = True
                self.call_signal_face_pending = False

        if use_call_signal_scene:
            person_id = int(self.get_parameter('call_signal_face_person_id').value)
            confidence = float(self.get_parameter('call_signal_face_confidence').value)
            message = 'call_signal caller identified'
            self.metrics.face_identify_call_signal += 1
            self.get_logger().info(
                f'[SIM ] FaceIdentify(call_signal) person_id={person_id} '
                f'confidence={confidence:.2f}'
            )
            return self._fill_face_identify_response(
                response, person_id, confidence, message
            )

        person_id = int(self.get_parameter('patrol_face_person_id').value)
        confidence = float(self.get_parameter('patrol_face_confidence').value)
        message = 'patrol bed visitor identified'
        self.metrics.face_identify_patrol += 1
        self.get_logger().info(
            f'[SIM ] FaceIdentify(patrol) person_id={person_id} confidence={confidence:.2f}'
        )
        return self._fill_face_identify_response(response, person_id, confidence, message)

    def handle_set_config(
        self,
        request: SetConfig.Request,
        response: SetConfig.Response,
    ) -> SetConfig.Response:
        """模拟配置加载：根据 config_id 更新巡逻路线参数。"""
        config_id = (request.config_id or 'default').strip()
        if config_id == 'default':
            route_id, cycles, points = 'route_a', 2, ['p0', 'p1']
        else:
            route_id, cycles, points = 'route_a', 1, ['p0']
        self.set_parameters([
            rclpy.parameter.Parameter('patrol_route_id', value=route_id),
            rclpy.parameter.Parameter('patrol_cycles', value=cycles),
            rclpy.parameter.Parameter('patrol_points', value=points),
        ])
        response.ok = True
        response.message = 'ok'
        return response

    def handle_navigate(self, goal_handle: ServerGoalHandle) -> Navigate.Result:
        """模拟导航 Action：按类型计数，发送进度反馈后返回成功。"""
        goal = goal_handle.request
        with self.lock:
            if goal.nav_type == NAV_STOP:
                self.metrics.nav_stop += 1
            if goal.nav_type == NAV_DOCK:
                self.metrics.nav_dock += 1
            if goal.nav_type == NAV_GOAL:
                self.metrics.nav_patrol += 1
        time.sleep(0.05)
        feedback = Navigate.Feedback()
        for i in range(20):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result = Navigate.Result()
                result.message = 'Goal canceled'
                print("goal canceled")
                return result
            feedback.progress = (i + 1) / 5.0
            time.sleep(0.5)
            print("sending navigate feed back i=", i)
            goal_handle.publish_feedback(feedback)
        result = Navigate.Result()
        result.status.status = ActionStatus.OK
        result.message = 'ok'
        goal_handle.succeed()
        return result

    def handle_llm(self, goal_handle: ServerGoalHandle) -> LLMInteraction.Result:
        """模拟 LLM 交互 Action：按 mode 计数，告警模式会标记 need_call_nurse。"""
        goal = goal_handle.request
        need_call_nurse = (goal.mode == INTERACTION_ALERT)
        with self.lock:
            if goal.mode == INTERACTION_INTERRUPT:
                self.metrics.llm_call_response += 1
            elif goal.mode == INTERACTION_ALERT:
                self.metrics.llm_abnormal += 1
            elif goal.mode == INTERACTION_PASSIVE:
                self.metrics.llm_passive += 1
        time.sleep(0.05)
        feedback = LLMInteraction.Feedback()
        for i in range(10):
            feedback.partial = str(i / 9.0)
            time.sleep(0.1)
            print("sending llm feed back i=", i)
            goal_handle.publish_feedback(feedback)

        result = LLMInteraction.Result()
        result.status.status = ActionStatus.OK
        result.summary = 'ok'
        result.need_call_nurse = need_call_nurse
        goal_handle.succeed()
        return result

    def handle_call_nurse(self, goal_handle: ServerGoalHandle) -> CallNurse.Result:
        """模拟呼叫护士 Action：发送进度反馈后返回成功。"""
        with self.lock:
            self.metrics.call_nurse += 1
        time.sleep(0.02)
        feedback = CallNurse.Feedback()
        for i in range(10):
            feedback.progress = str(i / 9.0)
            time.sleep(0.1)
            print("sending callnurse feed back i=", i)
            goal_handle.publish_feedback(feedback)
        result = CallNurse.Result()
        result.status.status = ActionStatus.OK
        result.message = 'ok'
        goal_handle.succeed()
        return result

    def publish_inputs(self, tick: int) -> None:
        """按 tick 序号注入电池、故障、呼叫信号等模拟输入。"""
        call_signal = 5 <= tick <= 8 or 95 <= tick <= 110
        battery_soc = 10.0 if 68 <= tick <= 74 else 50.0
        fault_type = ''
        fault_severity = 0
        if 15 <= tick <= 18:
            fault_type, fault_severity = 'localization', 1
        elif 25 <= tick <= 28:
            fault_type, fault_severity = 'navigation', 1
        elif 35 <= tick <= 38:
            fault_type, fault_severity = 'self', 1

        battery = Battery()
        battery.soc = float(battery_soc)
        battery.charging = False
        battery.voltage = 24.0
        if self.sim_battery:
            self.battery_pub.publish(battery)

        fault = Fault()
        fault.fault_type = fault_type
        fault.severity = fault_severity
        fault.details = ''
        if self.sim_fault:
            self.fault_pub.publish(fault)

        if self.sim_call_signal:
            if call_signal and not self._last_call_signal_published:
                self._mark_call_signal_face_pending(tick=tick)
            self.call_signal_pub.publish(Bool(data=call_signal))
            self._last_call_signal_published = call_signal

    def run(self) -> None:
        """主 tick 循环：任一周期性话题模拟开启时运行。"""
        if not self._needs_tick_loop():
            return

        start_delay = float(self.get_parameter('start_delay').value)
        total_ticks = int(self.get_parameter('ticks').value)
        tick_hz = float(self.get_parameter('tick_hz').value)
        period = 1.0 / max(1.0, tick_hz)

        self.get_logger().info(
            f'medical_bt_ros_test_driver tick 循环启动，{start_delay:.1f}s 后开始'
        )
        time.sleep(start_delay)

        for tick in range(total_ticks):
            self.publish_inputs(tick)
            if tick == 2:
                self.publish_patrol_triggered(True, tick=tick)
                print("Publish patrol triggered")
            time.sleep(period)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MedicalBtRosTestDriver()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        node.run()
    finally:
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()