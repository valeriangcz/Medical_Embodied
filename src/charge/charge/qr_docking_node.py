#!/usr/bin/env python3

import math
import time

import rclpy
from geometry_msgs.msg import Twist, Vector3Stamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener


class QrDockingNode(Node):
    IDLE = "IDLE"
    SEARCHING = "SEARCHING_TAG"
    LATERAL_ALIGNING = "LATERAL_ALIGNING"
    PID_APPROACHING = "PID_APPROACHING"
    ANGULAR_ALIGNING = "ANGULAR_ALIGNING"
    RETRYING = "RETRYING"
    COMPLETED = "DOCKING_COMPLETED"
    FAILED = "DOCKING_FAILED"
    ABORTED = "ABORTED"

    def __init__(self) -> None:
        super().__init__("qr_docking_node")

        self.is_running = False
        self.current_state = self.IDLE
        self.is_docking = False
        self.tag_visible = False
        self.last_tag_time = self.get_clock().now()
        self.last_valid_error_time = self.get_clock().now()
        self.start_time = None
        self.completed_stable_count = 0
        # Contact-settle detection: aligned + commanded to reverse onto the contacts
        # but physically not moving (spring/contact reaction exceeds the slow approach
        # force) means the dock is physically engaged.
        self.contact_stall_counter = 0
        self.contact_prev_range = None
        self.retry_count = 0
        self.is_retreating = False
        self.retreat_start_time = None
        self.current_distance_error = float("nan")
        self.current_lateral_error = float("nan")
        self.current_angular_error = float("nan")
        self.last_docking_end_monotonic = None
        # Current docking stage (LATERAL_ALIGNING / PID_APPROACHING / ANGULAR_ALIGNING).
        # Initialized to None so the first valid tag frame picks the stage from the
        # measured distance to the tag.
        self.phase = None

        self.target_distance = self._declare_float("target_distance", 0.315)
        # Calibrated dock pose of the tag in base_link: the y offset and yaw the tag
        # has when the robot is physically docked correctly. 0.0 assumes the tag sits
        # exactly on the charging-contact centerline facing it squarely; measure with
        # `ros2 run tf2_ros tf2_echo base_link dock_frame` at a good manual dock and
        # fill the values in to remove systematic tag/camera mounting offsets.
        self.target_lateral = self._declare_float("target_lateral", 0.0)
        self.target_yaw = self._declare_float("target_yaw", 0.0)
        self.horizontal_tolerance = self._declare_float("horizontal_tolerance", 0.03)
        self.angular_tolerance = self._declare_float("angular_tolerance", 0.05)
        self.linear_max_speed = self._declare_float("linear_max_speed", 0.1)
        self.angular_max_speed = self._declare_float("angular_max_speed", 0.3)
        self.search_angular_speed = self._declare_float("search_angular_speed", 0.25)
        # Proportional gain for re-centering the tag when it drifts toward / past the
        # camera FOV edge, and for the rotate-to-reacquire search below.
        self.search_gain = self._declare_float("search_gain", 1.5)
        # Blind search (tag never seen): flip rotation direction every N seconds.
        self.search_flip_period_sec = self._declare_float("search_flip_period_sec", 8.0)
        # Tag bearing guard (rad, from the rear camera optical axis): beyond this the
        # steering prioritizes pulling the tag back toward the image center so the
        # aligning arcs cannot sweep it out of view. >= pi disables the guard.
        self.fov_guard_rad = self._declare_float("fov_guard_rad", 0.6)
        # Carrot point on the dock line for far-field steering (stages 1/2). The rear
        # camera can only see where it is going, and squaring up with the dock line
        # from a side start points the camera away from the tag, so the robot instead
        # reverses onto the line by aiming its tail at a carrot that slides down the
        # line (from carrot_max_offset ahead of the tag to the dock pose) as the
        # range closes. Inside angular_align_distance the cross-track cascade and its
        # adaptive reverse speed take over for the final trim.
        self.carrot_max_offset = self._declare_float("carrot_max_offset", 1.2)
        self.carrot_margin = self._declare_float("carrot_margin", 0.25)
        # Rotate to aim before driving when the carrot is this far off the rear axis.
        self.aim_drive_threshold = self._declare_float("aim_drive_threshold", 0.35)
        self.last_tag_bearing_err = None
        self.search_start_time = None
        self.search_dir = 1.0
        # -------- Three-stage docking geometry --------
        # Stage 1 (LATERAL_ALIGNING): complete the lateral (y) adjustment at this
        #   distance in front of the tag, then hold there until y is within tolerance.
        self.lateral_align_distance = self._declare_float("lateral_align_distance", 1.0)
        # Stage 2 -> Stage 3 boundary: the PID approach stops at this distance in front
        # of the tag and the heading (yaw) alignment takes over.
        self.angular_align_distance = self._declare_float("angular_align_distance", 0.5)
        # Linear speed caps per stage (m/s).
        self.lateral_align_linear_max_speed = self._declare_float(
            "lateral_align_linear_max_speed", 0.06
        )
        self.angular_align_linear_max_speed = self._declare_float(
            "angular_align_linear_max_speed", 0.03
        )
        self.max_docking_duration_sec = self._declare_float("max_docking_duration_sec", 120.0)
        self.tag_timeout_sec = self._declare_float("tag_timeout_sec", 1.0)
        self.max_tag_lost_sec = self._declare_float("max_tag_lost_sec", 10.0)
        self.tag_startup_grace_sec = self._declare_float("tag_startup_grace_sec", 5.0)
        self.max_retry_count = self._declare_int("max_retry_count", 5)
        self.done_stable_cycles = self._declare_int("done_stable_cycles", 8)
        # Contact settle: in the terminal regime, when aligned (cross-track + yaw
        # within tolerance), inside contact_settle_window short of the dock pose,
        # and commanding a meaningful reverse speed yet the measured range does not
        # change by more than contact_motion_eps for contact_stall_cycles control
        # periods, the contacts are physically engaged — accept the dock instead of
        # deadlocking waiting for centimetres the mechanics cannot travel.
        self.contact_stall_cycles = self._declare_int("contact_stall_cycles", 10)
        self.contact_settle_window = self._declare_float("contact_settle_window", 0.05)
        self.contact_motion_eps = self._declare_float("contact_motion_eps", 0.004)
        self.contact_push_speed = self._declare_float("contact_push_speed", 0.03)
        # Refinement: on first physical contact, accept only if the seat is TIGHT
        # (cross-track / yaw inside these bounds). If the robot has clearly reached
        # the contacts but the pose is still loose, back out, square the heading, and
        # re-approach (one retry per attempt) instead of accepting a sloppy seat.
        # After max_retry_count refinement attempts the loose contact_aligned bounds
        # are used as a fallback so an engaged dock is never hard-failed.
        self.contact_refine_cross_track = self._declare_float(
            "contact_refine_cross_track", 0.03
        )
        self.contact_refine_yaw = self._declare_float("contact_refine_yaw", 0.08)
        # Refine back-out: a contact-seat refinement only needs to release the
        # contacts a few centimetres to regain steering authority, not the full
        # overshoot-retry retreat. Use this short/slow retreat for it.
        self.refine_retreat_duration = Duration(
            seconds=self._declare_float("refine_retreat_duration_sec", 2.5)
        )
        self.refine_retreat_speed = self._declare_float("refine_retreat_speed", 0.08)
        self.refine_retreat_target = self._declare_float("refine_retreat_target_distance", 0.6)
        self.is_refine_retreat = False
        self.stop_publish_count = self._declare_int("stop_publish_count", 5)
        self.tag_timeout = Duration(seconds=self.tag_timeout_sec)
        self.max_tag_lost = Duration(seconds=self.max_tag_lost_sec)
        self.tag_startup_grace = Duration(seconds=self.tag_startup_grace_sec)
        self.max_docking_duration = Duration(seconds=self.max_docking_duration_sec)

        self.linear_kp = self._declare_float("linear_kp", 0.5)
        self.linear_ki = self._declare_float("linear_ki", 0.0)
        self.linear_kd = self._declare_float("linear_kd", 0.1)
        # Heading-loop gain (w = k_psi * (psi - psi_ref), psi_ref from cross-track).
        self.angular_kp = self._declare_float("angular_kp", 1.0)

        self.linear_error_sum_max = self._declare_float("linear_error_sum_max", 1.0)
        self.linear_error_sum = 0.0
        self.last_linear_error = 0.0
        # Low-pass filter on the measured yaw error to reject apriltag orientation jitter.
        self.angular_error_alpha = self._declare_float("angular_error_alpha", 0.4)
        self.angular_error_filtered = 0.0
        # Cross-track -> heading-reference gain (psi_ref = -k * cross_track). Larger
        # values bleed the dock-line lateral offset faster while reversing.
        self.cross_track_k = self._declare_float("cross_track_k", 1.5)
        # Cap on the temporary heading offset the cross-track loop may command.
        self.heading_offset_max = self._declare_float("heading_offset_max", 0.5)

        self.retry_threshold = self._declare_float("retry_threshold", 0.5)
        self.retreat_duration = Duration(seconds=self._declare_float("retreat_duration_sec", 3.0))
        self.retreat_speed = self._declare_float("retreat_speed", 0.1)
        self.retry_stall_linear_speed_threshold = self._declare_float(
            "retry_stall_linear_speed_threshold", 0.01
        )
        self.retry_stall_cycles = self._declare_int("retry_stall_cycles", 8)
        self.final_approach_distance_window = self._declare_float("final_approach_distance_window", 0.2)
        self.final_approach_linear_max_speed = self._declare_float("final_approach_linear_max_speed", 0.05)
        self.stall_retry_counter = 0

        self.base_frame = self._declare_str("base_frame", "base_link")
        self.tag_frame = self._declare_str("tag_frame", "dock_frame")
        self.control_period = self._declare_float("control_period", 0.1)
        self.error_topic = self._declare_str("error_topic", "/docking/current_error")
        self.error_publish_hz = self._declare_float("error_publish_hz", 20.0)
        self.docking_restart_cooldown_sec = self._declare_float("docking_restart_cooldown_sec", 5.0)
        self.state_publish_hz = self._declare_float("state_publish_hz", 2.0)
        self.terminal_state_hold_sec = self._declare_float("terminal_state_hold_sec", 3.0)
        self._terminal_state_until_monotonic = None

        state_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.cmd_vel_pub = self.create_publisher(Twist, "cmd_vel_safe", 10)
        self.state_pub = self.create_publisher(String, "/docking/state", state_qos)
        self.error_pub = self.create_publisher(Vector3Stamped, self.error_topic, 10)
        self.create_subscription(String, "/dock/control_cmd", self.control_callback, 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_timer(self.control_period, self.control_loop)
        self.create_timer(1.0 / max(self.error_publish_hz, 1.0), self.publish_current_error_timer)
        self.create_timer(1.0 / max(self.state_publish_hz, 0.5), self.publish_state_timer)

        self.get_logger().info(
            f"qr_docking_node started, tracking tf {self.base_frame} -> {self.tag_frame}, "
            f"error_topic={self.error_topic}, error_publish_hz={self.error_publish_hz}"
        )

    def _declare_float(self, name: str, default: float) -> float:
        self.declare_parameter(name, default)
        return float(self.get_parameter(name).value)

    def _declare_str(self, name: str, default: str) -> str:
        self.declare_parameter(name, default)
        return str(self.get_parameter(name).value)

    def _declare_int(self, name: str, default: int) -> int:
        self.declare_parameter(name, default)
        return int(self.get_parameter(name).value)

    def _wait_for_restart_cooldown(self) -> None:
        if self.last_docking_end_monotonic is None:
            return
        remaining = self.docking_restart_cooldown_sec - (
            time.monotonic() - self.last_docking_end_monotonic
        )
        if remaining <= 0.0:
            return
        self.get_logger().info(
            "Docking restart cooldown: waiting %.1fs before start" % remaining
        )
        time.sleep(remaining)

    def control_callback(self, msg: String) -> None:
        command = msg.data.strip().lower()
        if command == "start":
            self._wait_for_restart_cooldown()
            if self.is_running:
                self.get_logger().warn("Docking is running, restarting docking flow")
            self.start_docking()
            self.get_logger().info("Docking started")
        elif command == "stop":
            self.stop_docking(self.ABORTED)
            self.get_logger().info("Docking stopped")
        self.publish_state()

    def _is_terminal_state(self, state: str) -> bool:
        return state in {self.COMPLETED, self.FAILED, self.ABORTED}

    def _mark_terminal_state(self, state: str) -> None:
        if self._is_terminal_state(state):
            self._terminal_state_until_monotonic = (
                time.monotonic() + max(self.terminal_state_hold_sec, 0.0)
            )

    def publish_state(self) -> None:
        state_msg = String()
        state_msg.data = self.current_state
        self.state_pub.publish(state_msg)

    def publish_state_timer(self) -> None:
        now = time.monotonic()
        if (
            self._terminal_state_until_monotonic is not None
            and now >= self._terminal_state_until_monotonic
            and self._is_terminal_state(self.current_state)
            and not self.is_docking
        ):
            self._terminal_state_until_monotonic = None
            self.current_state = self.IDLE
        self.publish_state()

    def publish_current_error(
        self, distance_error: float, lateral_error: float, angular_error: float
    ) -> None:
        self.current_distance_error = float(distance_error)
        self.current_lateral_error = float(lateral_error)
        self.current_angular_error = float(angular_error)

    def publish_current_error_timer(self) -> None:
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        # x=dx, y=dy, z=dyaw
        msg.vector.x = self.current_distance_error
        msg.vector.y = self.current_lateral_error
        msg.vector.z = self.current_angular_error
        self.error_pub.publish(msg)

    def get_tag_transform(self):
        try:
            transform = self.tf_buffer.lookup_transform(self.base_frame, self.tag_frame, Time())
            now = self.get_clock().now()
            transform_time = Time.from_msg(transform.header.stamp)
            # Prevent reusing stale TF from previous docking rounds.
            if transform_time.nanoseconds > 0 and (now - transform_time) > self.tag_timeout:
                self.tag_visible = False
                self.get_logger().debug(
                    "Ignoring stale tag transform, age=%.3fs"
                    % ((now - transform_time).nanoseconds / 1e9)
                )
                return None

            self.tag_visible = True
            self.last_tag_time = transform_time if transform_time.nanoseconds > 0 else now
            return transform
        except TransformException as exc:
            if (self.get_clock().now() - self.last_tag_time) > self.tag_timeout:
                self.tag_visible = False
            self.get_logger().debug(f"Cannot get tag transform: {exc}")
            return None

    def start_docking(self) -> None:
        self._terminal_state_until_monotonic = None
        self.is_running = True
        self.is_docking = True
        self.is_retreating = False
        self.is_refine_retreat = False
        self.phase = None
        self.retry_count = 0
        self.completed_stable_count = 0
        self.stall_retry_counter = 0
        self.contact_stall_counter = 0
        self.contact_prev_range = None
        now = self.get_clock().now()
        self.start_time = now
        self.last_valid_error_time = now
        self.current_state = self.SEARCHING
        self.angular_error_filtered = 0.0
        self.reset_pid_errors()
        self.publish_state()

    def reset_pid_errors(self) -> None:
        self.linear_error_sum = 0.0
        self.last_linear_error = 0.0

    def stop_docking(self, state: str = IDLE) -> None:
        self.is_running = False
        self.is_docking = False
        self.is_retreating = False
        self.current_state = state
        self._mark_terminal_state(state)
        self.stall_retry_counter = 0
        self.last_docking_end_monotonic = time.monotonic()
        self.publish_current_error(float("nan"), float("nan"), float("nan"))
        self.send_stop_command()
        self.publish_state()

    def fail_docking(self, reason: str) -> None:
        self.get_logger().error(f"Docking failed: {reason}")
        self.stop_docking(self.FAILED)

    def start_retreating(self, refine: bool = False) -> None:
        self.is_retreating = True
        self.is_refine_retreat = refine
        self.retreat_start_time = self.get_clock().now()
        self.retry_count += 1
        self.linear_error_sum = 0.0
        self.stall_retry_counter = 0
        self.contact_stall_counter = 0
        self.contact_prev_range = None
        self.current_state = self.RETRYING
        self.get_logger().warn(f"Start retry maneuver ({self.retry_count}/{self.max_retry_count})")

    def publish_zero_velocity(self) -> None:
        self.cmd_vel_pub.publish(Twist())

    def send_stop_command(self) -> None:
        cmd = Twist()
        for _ in range(max(self.stop_publish_count, 1)):
            self.cmd_vel_pub.publish(cmd)

    def send_search_command(self) -> None:
        cmd = Twist()
        cmd.angular.z = -self.search_angular_speed
        self.cmd_vel_pub.publish(cmd)

    @staticmethod
    def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def _bearing_err_from(self, raw_x: float, raw_y: float) -> float:
        """Signed tag bearing error from the rear camera optical axis (base -x).

        0 means the tag sits dead-center behind the robot; +/-pi means dead ahead.
        Computed from the RAW tag translation (no calibration offset): it describes
        where the tag is in the image, not where it should be.
        """
        return self._wrap_angle(math.pi - math.atan2(raw_y, raw_x))

    def _aim_control(self, current_distance: float, lateral_error: float,
                     angular_error_filtered: float):
        """Far-field steering: aim the tail at a carrot sliding down the dock line.

        Returns ``(angular_velocity, carrot_bearing)``. The carrot sits on the dock
        line ``carrot_offset`` ahead of the tag, where ``carrot_offset =
        clamp(range - carrot_margin, |target_distance|, carrot_max_offset)``: far
        away it leads the robot onto the line from any start (side approaches
        included), and near the pile it converges to the dock pose. The heading
        servo drives the carrot bearing to zero (tail pointed at the carrot) and
        reversing then tracks the line down to the dock. The caller switches to the
        cross-track cascade once range_to_tag reaches angular_align_distance.
        """
        carrot_offset = min(
            self.carrot_max_offset,
            max(abs(self.target_distance), abs(current_distance) - self.carrot_margin),
        )
        carrot_x = current_distance + carrot_offset * math.cos(angular_error_filtered)
        carrot_y = lateral_error + carrot_offset * math.sin(angular_error_filtered)
        # Bearing of the carrot from the rear axis (CCW-positive); driving it to zero
        # points the tail at the carrot, and theta decays under w = k * theta.
        theta = self._wrap_angle(math.atan2(carrot_y, carrot_x) + math.pi)
        w = max(-self.angular_max_speed, min(self.angular_max_speed, self.angular_kp * theta))
        return w, theta

    def _fov_guard(self, angular_velocity: float, bearing_err: float) -> float:
        """Override steering when the tag drifts toward the camera FOV edge.

        Losing the tag mid-turn freezes the docking (nothing re-acquires it), so once
        the tag bearing leaves the guard cone the heading law is replaced by a
        re-centering command: turning continues only in the direction that pulls the
        tag back toward the optical axis.
        """
        if self.fov_guard_rad >= math.pi or abs(bearing_err) <= self.fov_guard_rad:
            return angular_velocity
        w = -self.search_gain * bearing_err
        return max(-self.angular_max_speed, min(self.angular_max_speed, w))

    @staticmethod
    def _target_sign(target_distance: float, current_distance: float) -> float:
        if abs(target_distance) > 1e-6:
            return math.copysign(1.0, target_distance)
        if abs(current_distance) > 1e-6:
            return math.copysign(1.0, current_distance)
        return 1.0

    def log_status_line(
        self,
        current_distance: float,
        cross_track: float,
        angular_error: float,
        distance_error: float,
    ) -> None:
        self.get_logger().info(
            "state=%s retry=%d/%d current(x=%.3f,cross=%.3f,yaw=%.3f) "
            "target(x=%.3f,y=%.3f,yaw=%.3f) error(dx=%.3f,dcross=%.3f,dyaw=%.3f)"
            % (
                self.current_state,
                self.retry_count,
                self.max_retry_count,
                current_distance,
                cross_track,
                angular_error,
                self.target_distance,
                self.target_lateral,
                self.target_yaw,
                distance_error,
                cross_track,
                angular_error,
            )
        )

    def _initial_phase(self, range_to_tag: float, lateral_aligned: bool) -> str:
        """Pick the starting stage from the current distance to the tag.

        Lateral alignment is always the first priority whenever it is still off, so a
        robot that starts already close to the tag does not skip the lateral fix.
        """
        if not lateral_aligned:
            return self.LATERAL_ALIGNING
        if range_to_tag > self.lateral_align_distance:
            return self.LATERAL_ALIGNING
        if range_to_tag > self.angular_align_distance:
            return self.PID_APPROACHING
        return self.ANGULAR_ALIGNING

    def _linear_pid(self, linear_error: float) -> float:
        """PID on the distance error -> linear velocity."""
        self.linear_error_sum += linear_error
        self.linear_error_sum = max(
            -self.linear_error_sum_max, min(self.linear_error_sum_max, self.linear_error_sum)
        )
        linear_p = self.linear_kp * linear_error
        linear_i = self.linear_ki * self.linear_error_sum
        linear_d = self.linear_kd * (linear_error - self.last_linear_error)
        linear_velocity = linear_p + linear_i + linear_d
        # Forward adjustment uses proportional-only control: avoid I/D overshoot when
        # the robot has to nudge forward after slightly overshooting the target.
        if linear_velocity > 0.0:
            linear_velocity = linear_p
        self.last_linear_error = linear_error
        return linear_velocity

    def _reset_linear_pid(self) -> None:
        self.linear_error_sum = 0.0
        self.last_linear_error = 0.0

    def _cross_track_error(
        self, current_distance: float, lateral_error: float, angular_error: float
    ) -> float:
        """Lateral offset of the robot from the dock line, in the dock frame.

        For a tag seen at (x, y) with yaw error psi in base_link, the robot sits
        ``x*sin(psi) - y*cos(psi)`` off the dock's approach line. Unlike the
        base-frame y this stays meaningful while the robot is yawed, and under
        reverse motion it changes at ``dy = -u * psi``: only a heading offset held
        while moving bleeds it away, which is what the heading controller uses.
        """
        return current_distance * math.sin(angular_error) - lateral_error * math.cos(
            angular_error
        )

    def _heading_control(self, cross_track: float, angular_error: float) -> float:
        """Cascade dock-line controller: servo the heading to a cross-track reference.

        Parallel lateral+yaw feedback has a structural zero eigenvalue in reverse
        docking: the pose settles onto a residual line (y ~ k*psi) instead of the
        origin. Instead the heading tracks ``psi_ref = -k_cross * cross_track``:
        while reversing, that temporary heading offset bleeds the cross-track away;
        as the cross-track closes, psi_ref -> 0 and the heading squares up. All
        three errors converge to zero together.
        """
        psi_ref = -self.cross_track_k * cross_track
        psi_ref = max(-self.heading_offset_max, min(self.heading_offset_max, psi_ref))
        angular_velocity = self.angular_kp * (angular_error - psi_ref)
        return max(-self.angular_max_speed, min(self.angular_max_speed, angular_velocity))

    def _control_lateral_align(
        self,
        current_distance: float,
        range_to_tag: float,
        target_sign: float,
        cross_track: float,
        angular_error_filtered: float,
    ):
        """Stage 1: back up toward the 1 m standoff while closing the cross-track."""
        if range_to_tag > self.lateral_align_distance:
            # Still far: drive backward toward the 1 m standoff.
            linear_target = target_sign * self.lateral_align_distance
            linear_velocity = self._linear_pid(current_distance - linear_target)
            linear_velocity = max(
                -self.lateral_align_linear_max_speed,
                min(self.lateral_align_linear_max_speed, linear_velocity),
            )
            # Near the standoff the PID output can drop below the chassis deadband and
            # leave the robot parked just short of lateral_align_distance, so it can
            # never cross into the next stage. Enforce a minimum approach speed.
            min_approach = self.lateral_align_linear_max_speed * 0.5
            if abs(linear_velocity) < min_approach:
                linear_velocity = target_sign * min_approach
        else:
            # At/past the standoff but cross-track is still off: keep creeping backward
            # so the heading reference keeps bleeding it (cross-track only decays while
            # the robot moves, dy = -u * psi).
            linear_velocity = target_sign * self.lateral_align_linear_max_speed * 0.5

        angular_velocity = self._heading_control(cross_track, angular_error_filtered)
        return linear_velocity, angular_velocity

    def _control_pid_approach(
        self,
        distance_error: float,
        distance_to_target: float,
        cross_track: float,
        angular_error_filtered: float,
    ):
        """Stage 2: PID on distance; heading reference keeps the dock line tracked."""
        linear_velocity = self._linear_pid(distance_error)
        linear_velocity = max(-self.linear_max_speed, min(self.linear_max_speed, linear_velocity))
        if distance_to_target <= self.final_approach_distance_window:
            linear_velocity = max(
                -self.final_approach_linear_max_speed,
                min(self.final_approach_linear_max_speed, linear_velocity),
            )

        angular_velocity = self._heading_control(cross_track, angular_error_filtered)
        return linear_velocity, angular_velocity

    def _control_angular_align(
        self,
        distance_error: float,
        target_sign: float,
        cross_track: float,
        angular_error_filtered: float,
    ):
        """Stage 3: final cross-track + heading trim near the charging position."""
        linear_velocity = self._linear_pid(distance_error)
        linear_velocity = max(
            -self.angular_align_linear_max_speed,
            min(self.angular_align_linear_max_speed, linear_velocity),
        )
        # Cross-track only decays while moving (dy = -u * psi), so keep a minimum
        # reverse speed while it is open — scaled to the cross-track AND to the
        # remaining distance (reversing harder than half the shortfall per second
        # would overshoot the dock pose, and an overshoot pose is exactly where the
        # tag leaves the guard cone and the controller deadlocks). Past the target
        # the PID alone commands motion; the retry branch handles the recovery.
        shortfall = target_sign * distance_error  # > 0 while short of the dock pose
        # Chassis breakaway: the last centimetres command <4 wheel RPM, which static
        # friction eats, leaving the robot parked just short of the contacts. Push a
        # minimum approach speed while still meaningfully short; it releases inside
        # the last ~8 mm so the PID settles without meaningful overshoot.
        if shortfall > 0.008:
            v_approach = min(
                max(abs(linear_velocity), 0.05), self.final_approach_linear_max_speed
            )
            linear_velocity = target_sign * v_approach
        elif abs(cross_track) > 0.5 * self.horizontal_tolerance and shortfall > 0.0:
            v_rev = min(
                0.25 * abs(cross_track),
                0.5 * shortfall,
                self.final_approach_linear_max_speed,
            )
            if v_rev >= 0.02 and abs(linear_velocity) < v_rev:
                linear_velocity = target_sign * v_rev

        angular_velocity = self._heading_control(cross_track, angular_error_filtered)
        return linear_velocity, angular_velocity

    def control_loop(self) -> None:
        if not self.is_docking:
            return
        self.publish_state()

        now = self.get_clock().now()
        if self.start_time is not None and (now - self.start_time) > self.max_docking_duration:
            self.fail_docking("overall docking timeout")
            return

        transform = self.get_tag_transform()
        if transform is None or not self.tag_visible:
            self.current_state = self.SEARCHING
            self.publish_current_error(float("nan"), float("nan"), float("nan"))
            self.log_status_line(float("nan"), float("nan"), float("nan"), float("nan"))
            # Allow apriltag detector startup latency at docking begin.
            if self.start_time is not None and (now - self.start_time) <= self.tag_startup_grace:
                self.publish_zero_velocity()
                return
            if (now - self.last_valid_error_time) > self.max_tag_lost:
                self.fail_docking("tag lost for too long")
                return
            # Rotate to re-acquire the tag instead of freezing: standing still can
            # never bring a side-mounted tag back into the rear camera's FOV. With a
            # last-seen bearing, rotate proportionally toward centering it at the
            # rear; without one, sweep at a fixed speed and flip direction
            # periodically so the search covers a full circle.
            if self.search_start_time is None:
                self.search_start_time = now
            elapsed_s = (now - self.search_start_time).nanoseconds / 1e9
            bearing_err = self.last_tag_bearing_err
            if bearing_err is not None and abs(bearing_err) > 0.15:
                w = -self.search_gain * bearing_err
                w = max(-self.search_angular_speed, min(self.search_angular_speed, w))
            else:
                periods = int(elapsed_s / max(self.search_flip_period_sec, 0.1))
                direction = self.search_dir if periods % 2 == 0 else -self.search_dir
                w = direction * self.search_angular_speed
            cmd = Twist()
            cmd.angular.z = w
            self.cmd_vel_pub.publish(cmd)
            return

        # Plan in base_link frame:
        #   forward/backward -> x axis
        #   lateral          -> y axis
        current_distance = transform.transform.translation.x
        lateral_error = transform.transform.translation.y - self.target_lateral
        # Tag bearing in the raw (uncalibrated) camera geometry, for the FOV guard
        # and for the reacquire rotation above.
        self.last_tag_bearing_err = self._bearing_err_from(
            transform.transform.translation.x, transform.transform.translation.y
        )
        self.search_start_time = None

        q = transform.transform.rotation
        angular_error = self._wrap_angle(
            self._yaw_from_quaternion(q.x, q.y, q.z, q.w) - self.target_yaw
        )

        # Low-pass filter the measured yaw error to reject apriltag orientation jitter
        # before feeding it to the controller (raw value is kept for logging / done test).
        self.angular_error_filtered = (
            self.angular_error_alpha * angular_error
            + (1.0 - self.angular_error_alpha) * self.angular_error_filtered
        )
        self.last_valid_error_time = now
        distance_error = current_distance - self.target_distance
        # Cross-track offset from the dock line (dock frame), computed with the
        # filtered yaw so the heading controller does not chase tag pose jitter.
        cross_track = self._cross_track_error(
            current_distance, lateral_error, self.angular_error_filtered
        )
        self.publish_current_error(distance_error, cross_track, angular_error)

        target_sign = self._target_sign(self.target_distance, current_distance)
        range_to_tag = abs(current_distance)
        distance_to_target = target_sign * distance_error
        lateral_aligned = abs(cross_track) <= self.horizontal_tolerance

        if self.is_retreating:
            self.log_status_line(current_distance, cross_track, angular_error, distance_error)
            if self.is_refine_retreat:
                # Refine retreat: back out to lateral_align_distance so the robot
                # has full steering authority to re-run the full 3-stage approach.
                # Much more effective than a short 2.5 cm wiggle — after reaching
                # the staging distance, reset phase and start over cleanly.
                cmd = Twist()
                cmd.linear.x = -target_sign * self.refine_retreat_speed
                cmd.angular.z = self._fov_guard(
                    max(
                        -self.angular_max_speed,
                        min(self.angular_max_speed, self.angular_kp * self.angular_error_filtered),
                    ),
                    self.last_tag_bearing_err,
                )
                self.cmd_vel_pub.publish(cmd)
                # Stop when we've backed out past the refine target distance.
                if range_to_tag >= self.refine_retreat_target:
                    self.is_retreating = False
                    self.is_refine_retreat = False
                    self.phase = None
                    self.reset_pid_errors()
                    self.get_logger().info(
                        f"Refine back-out reached staging ({range_to_tag:.2f} m); "
                        f"restarting full approach"
                    )
                return

            # Generic overshoot retry retreat — keep the original time-based logic.
            if (self.get_clock().now() - self.retreat_start_time) > self.retreat_duration:
                self.is_retreating = False
                self.is_refine_retreat = False
                self.phase = None
                self.reset_pid_errors()
                self.get_logger().info("Forward retry finished, resume docking")
                return

            cmd = Twist()
            cmd.linear.x = -target_sign * self.retreat_speed
            cmd.angular.z = self._fov_guard(
                max(
                    -self.angular_max_speed,
                    min(self.angular_max_speed, self.angular_kp * self.angular_error_filtered),
                ),
                self.last_tag_bearing_err,
            )
            self.cmd_vel_pub.publish(cmd)
            return

        # -------- Stage selection (monotonic: lateral -> approach -> angular) --------
        if self.phase is None:
            self.phase = self._initial_phase(range_to_tag, lateral_aligned)

        if self.phase == self.LATERAL_ALIGNING:
            # Leave the lateral stage only once the robot is at the standoff AND the
            # lateral offset is already inside tolerance.
            if range_to_tag <= self.lateral_align_distance and lateral_aligned:
                self.phase = self.PID_APPROACHING
                self.reset_pid_errors()
        elif self.phase == self.PID_APPROACHING:
            if range_to_tag <= self.angular_align_distance:
                self.phase = self.ANGULAR_ALIGNING
                self.reset_pid_errors()

        # -------- Retry when past the dock pose with the cross-track still open --------
        # Short of the target the cascade/aim laws have room to converge; at or past
        # it there is none (the overshoot pose throws the tag out of the guard cone),
        # so back off forward and re-approach on the line.
        if (
            range_to_tag < self.retry_threshold
            and abs(cross_track) > self.horizontal_tolerance
            and target_sign * distance_error < -0.03
            and not self.is_retreating
        ):
            if self.retry_count >= self.max_retry_count:
                self.fail_docking("retry count exceeded")
                return
            self.start_retreating()
            self.log_status_line(current_distance, cross_track, angular_error, distance_error)
            return

        # -------- Per-stage control --------
        if range_to_tag <= self.angular_align_distance:
            # Terminal regime: cross-track cascade trims the final pose near the dock
            # (heading reference + adaptive reverse speed; see _control_angular_align).
            self.current_state = self.ANGULAR_ALIGNING
            self.phase = self.ANGULAR_ALIGNING
            linear_velocity, angular_velocity = self._control_angular_align(
                distance_error, target_sign, cross_track, self.angular_error_filtered,
            )
        else:
            # Far field: aim the tail at the dock-line carrot (handles side/off-axis
            # starts; squaring up with the line here would sweep the tag out of view).
            w_aim, theta_aim = self._aim_control(
                current_distance, lateral_error, self.angular_error_filtered
            )
            if self.phase == self.LATERAL_ALIGNING:
                self.current_state = self.LATERAL_ALIGNING
                linear_velocity, _ = self._control_lateral_align(
                    current_distance, range_to_tag, target_sign,
                    cross_track, self.angular_error_filtered,
                )
            else:
                self.current_state = self.PID_APPROACHING
                linear_velocity, _ = self._control_pid_approach(
                    distance_error, distance_to_target, cross_track, self.angular_error_filtered,
                )
            # Aim first, drive later: rotate with the tail pointed at the carrot.
            if abs(theta_aim) > self.aim_drive_threshold:
                linear_velocity = 0.0
            angular_velocity = w_aim

        # Never steer the tag out of the camera's FOV while aligning.
        angular_velocity = self._fov_guard(angular_velocity, self.last_tag_bearing_err)

        # If cross-track stays large while linear speed stalls near zero,
        # proactively trigger another retry maneuver instead of idling in place.
        # Only when already close to the tag: holding at the 1 m standoff is an
        # intentional part of stage 1 and must not trigger a retreat.
        stalled_with_lateral_error = (
            range_to_tag < self.retry_threshold
            and abs(cross_track) > self.horizontal_tolerance
            and abs(linear_velocity) < self.retry_stall_linear_speed_threshold
            and not self.is_retreating
        )
        if stalled_with_lateral_error:
            self.stall_retry_counter += 1
        else:
            self.stall_retry_counter = 0
        if self.stall_retry_counter >= max(self.retry_stall_cycles, 1):
            if self.retry_count >= self.max_retry_count:
                self.fail_docking("retry count exceeded (stall with lateral error)")
                return
            self.start_retreating()
            self.log_status_line(current_distance, cross_track, angular_error, distance_error)
            return

        # -------- Contact settle-out --------
        # The terminal controller commands a slow reverse speed to seat onto the
        # charging contacts. Spring/contact reaction can exceed that force: the
        # robot is physically touching yet cannot travel the last centimetres to
        # target_distance, so |distance_error| never enters the done window and the
        # dock deadlocks until the 120 s timeout. When aligned, within the contact
        # window, commanding a real push, and the measured range stops moving for
        # several cycles, the contacts are mechanically engaged — accept the dock.
        # Physical stall detection: decoupled from pose alignment. From a side
        # approach the first contact often has cross_track > horizontal_tolerance,
        # so gating the stall counter on contact_aligned would leave the robot
        # pushing against the dock forever with no exit path. Detect raw physical
        # engagement first (in window + pushing + range not changing for N cycles),
        # then decide what to do based on pose quality.
        contact_push = abs(linear_velocity) >= self.contact_push_speed
        contact_window = 0.0 <= distance_to_target <= self.contact_settle_window
        # Pose quality measured at stall time — loose vs tight bounds below.
        contact_aligned = (
            abs(cross_track) < self.horizontal_tolerance
            and abs(angular_error) < self.angular_tolerance
        )
        if contact_window and contact_push and not self.is_retreating:
            if (
                self.contact_prev_range is not None
                and abs(range_to_tag - self.contact_prev_range) < self.contact_motion_eps
            ):
                self.contact_stall_counter += 1
            else:
                self.contact_stall_counter = 0
        else:
            self.contact_stall_counter = 0
        self.contact_prev_range = range_to_tag
        contact_stalled = self.contact_stall_counter >= max(self.contact_stall_cycles, 1)

        # On physical contact, accept only a TIGHT seat. A loose first contact
        # (contacts engaged but cross-track / yaw still outside the refine bounds)
        # triggers a back-out + heading-squaring + re-approach instead of finishing
        # on a sloppy pose. After max_retry_count refinement attempts the loose
        # contact bounds are the fallback — an engaged dock is never hard-failed.
        contact_tight = (
            abs(cross_track) < self.contact_refine_cross_track
            and abs(angular_error) < self.contact_refine_yaw
        )
        refine_retreat = False
        if contact_stalled:
            if contact_tight:
                done_contact = True
            elif self.retry_count < 1 and not self.is_retreating:
                # Only one full reset — backing out to staging and re-running the
                # complete 3-stage approach is expensive, one clean retry is enough.
                refine_retreat = True
                done_contact = False
            else:
                done_contact = True
        else:
            done_contact = False

        if refine_retreat:
            self.completed_stable_count = 0
            self.get_logger().warn(
                f"Contact seat loose (cross={cross_track * 100:.1f}cm "
                f"yaw={math.degrees(angular_error):.1f}deg); back out & re-approach "
                f"to refine seat ({self.retry_count + 1}/{self.max_retry_count})"
            )
            self.start_retreating(refine=True)
            self.log_status_line(current_distance, cross_track, angular_error, distance_error)
            return

        done1 = (
            abs(distance_error) < 0.01
            and abs(cross_track) < self.horizontal_tolerance
            and abs(angular_error) < self.angular_tolerance
        )
        done2 = (
            abs(distance_error) < 0.02
            and abs(cross_track) < self.horizontal_tolerance
            and abs(angular_error) < self.angular_tolerance
            and abs(linear_velocity) < 0.01
        )
        if done1 or done2 or done_contact:
            self.completed_stable_count += 1
            if self.completed_stable_count >= self.done_stable_cycles:
                self.current_state = self.COMPLETED
                self.publish_state()
                self.stop_docking(self.COMPLETED)
                if done_contact:
                    self.get_logger().info(
                        "Docking completed: contacts engaged (physical settle)"
                    )
                else:
                    self.get_logger().info("Docking completed with stable convergence")
            else:
                if done_contact:
                    # Keep the seating pressure while confirming the physical settle:
                    # publishing zero here would drop contact_push and re-arm the
                    # stall counter, oscillating before the stable count completes.
                    cmd = Twist()
                    cmd.linear.x = linear_velocity
                    cmd.angular.z = angular_velocity
                    self.cmd_vel_pub.publish(cmd)
                else:
                    self.publish_zero_velocity()
            return
        self.completed_stable_count = 0

        cmd = Twist()
        # Keep linear command sign consistent with distance_error:
        # target behind (negative) -> usually command negative (reverse),
        # only when too close (error > 0) command becomes positive (forward).
        cmd.linear.x = linear_velocity
        cmd.angular.z = angular_velocity
        self.cmd_vel_pub.publish(cmd)
        self.log_status_line(current_distance, cross_track, angular_error, distance_error)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = QrDockingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
