#!/usr/bin/env python3

import os
import signal
import subprocess
import time
from threading import Event
from typing import Tuple

import rclpy
from ament_index_python.packages import get_package_share_directory
from interfaces.action import Docking
from interfaces.srv import ChargeUntil, Dock as DockSrv
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

_DOCKING_STATE_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
)


class ChargeServices(Node):
    TERMINAL_STATES = {"DOCKING_COMPLETED", "DOCKING_FAILED", "ABORTED"}
    ACTIVE_STATES = {
        "SEARCHING_TAG",
        "LATERAL_ALIGNING",
        "PID_APPROACHING",
        "ANGULAR_ALIGNING",
        "RETRYING",
    }

    def __init__(self):
        super().__init__("charge_services")
        self.state_timeout_sec = float(self.declare_parameter("state_timeout_sec", 5.0).value)
        self.enable_apriltag_on_demand = bool(
            self.declare_parameter("enable_apriltag_on_demand", True).value
        )
        self.apriltag_launch_package = str(
            self.declare_parameter("apriltag_launch_package", "charge").value
        )
        self.apriltag_launch_file = str(
            self.declare_parameter("apriltag_launch_file", "tag_realsense_node.launch.py").value
        )
        self.apriltag_camera_name = str(self.declare_parameter("apriltag_camera_name", "/camera").value)
        self.apriltag_image_topic = str(self.declare_parameter("apriltag_image_topic", "image_raw").value)
        default_apriltag_params = os.path.join(
            get_package_share_directory("charge"), "cfg", "tags_36h11_node.yaml"
        )
        self.apriltag_params_file = str(
            self.declare_parameter("apriltag_params_file", default_apriltag_params).value
        )
        self.apriltag_startup_delay_sec = float(
            self.declare_parameter("apriltag_startup_delay_sec", 1.0).value
        )
        self.apriltag_stop_timeout_sec = float(
            self.declare_parameter("apriltag_stop_timeout_sec", 3.0).value
        )
        self.apriltag_start_retry_count = int(
            self.declare_parameter("apriltag_start_retry_count", 2).value
        )
        self.apriltag_start_retry_delay_sec = float(
            self.declare_parameter("apriltag_start_retry_delay_sec", 0.8).value
        )
        self.docking_session_cooldown_sec = float(
            self.declare_parameter("docking_session_cooldown_sec", 5.0).value
        )
        self.docking_completion_timeout_sec = float(
            self.declare_parameter("docking_completion_timeout_sec", 120.0).value
        )
        self.dock_action_name = str(self.declare_parameter("dock_action_name", "dock").value)

        self.last_docking_state = "UNKNOWN"
        self.state_event = Event()
        self.apriltag_process = None
        self.docking_session_active = False
        self.session_seen_active_state = False
        self.last_docking_end_monotonic = None
        self._active_dock_action_goal = None

        self.control_pub = self.create_publisher(String, "/dock/control_cmd", 10)
        self.create_subscription(String, "/dock/control_cmd", self._control_cmd_callback, 10)
        self.create_subscription(
            String, "/docking/state", self._state_callback, _DOCKING_STATE_QOS
        )
        # handle_dock blocks in _run_docking_start_and_wait (cooldown + apriltag
        # startup + state wait). It must NOT share the default callback group with
        # the /docking/state subscription, otherwise _state_callback cannot run
        # while the service handler is blocked and the state wait always times out.
        self._dock_service_cb_group = MutuallyExclusiveCallbackGroup()
        self.create_service(
            DockSrv, "dock", self.handle_dock, callback_group=self._dock_service_cb_group
        )
        self.create_service(ChargeUntil, "charge_until", self.handle_charge)
        self._dock_action_server = ActionServer(
            self,
            Docking,
            self.dock_action_name,
            execute_callback=self._execute_dock_action,
            goal_callback=self._dock_action_goal_callback,
            cancel_callback=self._dock_action_cancel_callback,
            callback_group=ReentrantCallbackGroup(),
        )
        self.get_logger().info(
            "charge_services started (dock service + dock action '%s' with feedback)"
            % self.dock_action_name
        )

    @staticmethod
    def _state_to_progress(state: str, session_seen_active: bool) -> float:
        if state == "DOCKING_COMPLETED":
            return 1.0
        if state == "IDLE" and session_seen_active:
            return 1.0
        if state == "SEARCHING_TAG":
            return 0.1
        if state == "LATERAL_ALIGNING":
            return 0.3
        if state == "PID_APPROACHING":
            return 0.6
        if state == "ANGULAR_ALIGNING":
            return 0.85
        if state == "RETRYING":
            return 0.6
        return 0.0

    def _publish_dock_action_feedback(self) -> None:
        goal_handle = self._active_dock_action_goal
        if goal_handle is None:
            return
        feedback = Docking.Feedback()
        feedback.docking_state = self.last_docking_state
        feedback.progress = self._state_to_progress(
            self.last_docking_state, self.session_seen_active_state
        )
        goal_handle.publish_feedback(feedback)

    def _state_callback(self, msg: String) -> None:
        self.last_docking_state = msg.data.strip()
        self.state_event.set()
        if self.docking_session_active and self.last_docking_state in self.ACTIVE_STATES:
            self.session_seen_active_state = True
        self._publish_dock_action_feedback()
        if (
            self.docking_session_active
            and self.session_seen_active_state
            and self.last_docking_state in self.TERMINAL_STATES
        ):
            self.get_logger().info(
                "Docking session finished with state=%s" % self.last_docking_state
            )
            self._end_docking_session(stop_controller=False)

    def _publish_control(self, command: str) -> None:
        msg = String()
        msg.data = command
        self.control_pub.publish(msg)

    def _wait_for_state(self, accepted_states: set[str], timeout_sec: float) -> bool:
        self.state_event.clear()
        deadline = time.monotonic() + max(timeout_sec, 0.0)
        while rclpy.ok():
            if self.last_docking_state in accepted_states:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)
        return False

    def _wait_for_docking_completion(self, timeout_sec: float) -> bool:
        deadline = time.monotonic() + max(timeout_sec, 0.0)
        while rclpy.ok():
            state = self.last_docking_state
            if state == "DOCKING_COMPLETED":
                return True
            if state in {"DOCKING_FAILED", "ABORTED"}:
                return False
            if state == "IDLE" and self.session_seen_active_state:
                return True
            self._publish_dock_action_feedback()
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)
        return False

    def _prepare_new_docking_session(self) -> None:
        self.last_docking_state = "UNKNOWN"
        self.state_event.clear()
        self.docking_session_active = True
        self.session_seen_active_state = False

    def _record_docking_cooldown(self) -> None:
        self.last_docking_end_monotonic = time.monotonic()

    def _wait_for_docking_cooldown(self) -> None:
        if self.last_docking_end_monotonic is None:
            return
        remaining = self.docking_session_cooldown_sec - (
            time.monotonic() - self.last_docking_end_monotonic
        )
        if remaining <= 0.0:
            return
        self.get_logger().info(
            "Docking cooldown: waiting %.1fs before next start" % remaining
        )
        time.sleep(remaining)

    def _end_docking_session(self, stop_controller: bool = True) -> None:
        self.docking_session_active = False
        if stop_controller:
            self._publish_control("stop")
        if self.enable_apriltag_on_demand:
            self._stop_apriltag_detector()
        self._record_docking_cooldown()

    def _control_cmd_callback(self, msg: String) -> None:
        command = msg.data.strip().lower()
        if not self.enable_apriltag_on_demand:
            return
        if command == "start":
            if not self.docking_session_active:
                self._prepare_new_docking_session()
            if not self._start_apriltag_detector():
                self.docking_session_active = False
                self.get_logger().error("Received /dock/control_cmd start but apriltag failed to start")
        elif command == "stop":
            if self.docking_session_active:
                self._end_docking_session(stop_controller=False)

    def _build_apriltag_launch_cmd(self) -> list[str]:
        return [
            "ros2",
            "launch",
            self.apriltag_launch_package,
            self.apriltag_launch_file,
            f"camera_name:={self.apriltag_camera_name}",
            f"image_topic:={self.apriltag_image_topic}",
            f"apriltag_params_file:={self.apriltag_params_file}",
        ]

    def _wait_for_apriltag_startup_result(self) -> bool:
        if self.apriltag_startup_delay_sec > 0.0:
            time.sleep(self.apriltag_startup_delay_sec)

        if self.apriltag_process is None:
            return False
        if self.apriltag_process.poll() is not None:
            self.get_logger().error(
                "apriltag detector exited unexpectedly after startup, return_code=%s"
                % self.apriltag_process.poll()
            )
            self.apriltag_process = None
            return False
        return True

    def _retry_start_apriltag(self) -> bool:
        max_attempts = max(self.apriltag_start_retry_count + 1, 1)
        launch_cmd = self._build_apriltag_launch_cmd()
        for attempt in range(1, max_attempts + 1):
            self.get_logger().info(
                "Starting apriltag detector (attempt %d/%d): %s"
                % (attempt, max_attempts, " ".join(launch_cmd))
            )
            try:
                self.apriltag_process = subprocess.Popen(launch_cmd, start_new_session=True)
            except Exception as exc:
                self.get_logger().error(f"Failed to start apriltag detector: {exc}")
                self.apriltag_process = None
            else:
                if self._wait_for_apriltag_startup_result():
                    return True

            if attempt < max_attempts:
                time.sleep(max(self.apriltag_start_retry_delay_sec, 0.0))
        return False

    def _start_apriltag_detector(self) -> bool:
        if self.apriltag_process is not None and self.apriltag_process.poll() is None:
            return True
        return self._retry_start_apriltag()

    def _stop_apriltag_detector(self) -> None:
        if self.apriltag_process is None:
            return
        if self.apriltag_process.poll() is not None:
            self.apriltag_process = None
            return

        self.get_logger().info("Stopping apriltag detector")
        try:
            os.killpg(self.apriltag_process.pid, signal.SIGINT)
            self.apriltag_process.wait(timeout=self.apriltag_stop_timeout_sec)
        except subprocess.TimeoutExpired:
            self.get_logger().warn("apriltag detector did not stop on SIGINT, sending SIGTERM")
            try:
                os.killpg(self.apriltag_process.pid, signal.SIGTERM)
                self.apriltag_process.wait(timeout=1.0)
            except Exception as exc:
                self.get_logger().error(f"Failed to force stop apriltag detector: {exc}")
        except Exception as exc:
            self.get_logger().error(f"Failed to stop apriltag detector: {exc}")
        finally:
            self.apriltag_process = None

    def _run_docking_start_and_wait(self) -> Tuple[bool, str]:
        self._wait_for_docking_cooldown()
        self._prepare_new_docking_session()
        if self.enable_apriltag_on_demand and not self._start_apriltag_detector():
            self.docking_session_active = False
            self._record_docking_cooldown()
            return False, "apriltag detector failed to start"

        self._publish_control("start")
        self._publish_dock_action_feedback()
        accepted = self._wait_for_state(
            {
                "SEARCHING_TAG",
                "LATERAL_ALIGNING",
                "PID_APPROACHING",
                "ANGULAR_ALIGNING",
                "RETRYING",
            },
            self.state_timeout_sec,
        )
        if not accepted:
            self.get_logger().error("Dock start rejected or timed out waiting for state transition")
            self._end_docking_session(stop_controller=True)
            return False, "dock start rejected or timed out"

        self.get_logger().info(
            "Docking active, waiting up to %.1fs for completion"
            % self.docking_completion_timeout_sec
        )
        completed = self._wait_for_docking_completion(self.docking_completion_timeout_sec)
        if completed:
            self.get_logger().info("Docking completed successfully")
            return True, "docking completed"
        self.get_logger().error(
            "Docking did not complete within %.1fs, last_state=%s"
            % (self.docking_completion_timeout_sec, self.last_docking_state)
        )
        if self.docking_session_active:
            self._end_docking_session(stop_controller=True)
        return False, "docking failed, last_state=%s" % self.last_docking_state

    def _dock_action_goal_callback(self, goal_request: Docking.Goal) -> GoalResponse:
        return GoalResponse.ACCEPT

    def _dock_action_cancel_callback(self, goal_handle) -> CancelResponse:
        self._end_docking_session(stop_controller=True)
        return CancelResponse.ACCEPT

    def _execute_dock_action(self, goal_handle) -> Docking.Result:
        result = Docking.Result()
        if not bool(goal_handle.request.start):
            self._end_docking_session(stop_controller=True)
            result.ok = True
            result.message = "docking stopped"
            goal_handle.succeed()
            return result

        self._active_dock_action_goal = goal_handle
        try:
            ok, message = self._run_docking_start_and_wait()
            result.ok = ok
            result.message = message
            if ok:
                goal_handle.succeed()
            else:
                goal_handle.abort()
        finally:
            self._active_dock_action_goal = None
        return result

    def handle_dock(self, request, response):
        start = bool(request.start)
        self.get_logger().info(f"dock service requested start={str(start).lower()}")

        if start:
            ok, _message = self._run_docking_start_and_wait()
            response.ok = ok
        else:
            self._end_docking_session(stop_controller=True)
            response.ok = True
        return response

    def handle_charge(self, request, response):
        self.get_logger().info(f"charge_until requested soc_target={request.soc_target:.1f}")
        response.ok = True
        return response

    def destroy_node(self):
        if self.enable_apriltag_on_demand:
            self._stop_apriltag_detector()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ChargeServices()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
