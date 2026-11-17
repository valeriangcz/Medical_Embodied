#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import subprocess
import os
import time

# 把 ros2 生成的接口文件地址加上
import os, sys
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), 
        "../../../install/llm_node_comm/lib/python3.12/site-packages"
    )
)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


# 一个人机对话 session 的结束
from llm_node_comm.msg import (
    DialogSessionFinished,
)


# 提供两个 stop 方法，一个是 非阻塞的topic，一个是 阻塞的service

# 提供一个 service，用来查询当前的状态，返回是否需要呼叫护士


class StopDialogNode(Node):
    def __init__(self):
        # 初始化 Node
        super().__init__("dialog_manager")

        # # 创建阻塞服务，使用 SetBool
        # self.service = rospy.Service("stop_dialog_service", SetBool, self.handle_service)
        # self.get_logger().info("Service 'stop_dialog_service' is ready.")

        # # 发布消息,开始进行语音识别
        # self.start_asr_pub = self.create_publisher(Bool, "start_asr", 10)

        # 人机对话的时候进行打断
        self.stop_hci_dialog_sub = self.create_subscription(
            Bool, "stop_hci_dialog", self.stop_hci_dialog_callback, 10
        )

        # 发送预制问题的时候进行打断
        self.stop_ui_question_sub = self.create_subscription(
            Bool, "stop_ui_question", self.stop_ui_question_callback, 10
        )

        # 订阅 这个人机对话结束的 topic
        self.dialog_session_finished_sub = self.create_subscription(
            DialogSessionFinished,
            "dialog_session_finished", 
            self.dialog_session_finished_callback,
            10,
        )
        # self.dialog_session_finished:bool = False
        # dialog session finished 的返回值
        self.call_nurse = False
        self.call_nurse_reason = "init call nurse reason"

        # 提供一个人机对话的service服务,会一直阻塞到人机对话结束
        # self.hci_service = rospy.Service(
        #     "hci_dialog_service", 
        #     hci_dialog,
        #     self.hci_dialog_callback,
        # )

        print("dialog manager init done.")

    def stop_hci_dialog_callback(self, msg):
        del msg
        # 2026-11：默认不再"重拉全栈"——新架构下节点无需重启，盲重拉在
        # voice_terminal 模式会起出第二套栈（表现为每个回答播两遍）。
        # 需要旧行为时设 VOICE_ALLOW_RELAUNCH=1。
        if os.environ.get("VOICE_ALLOW_RELAUNCH", "0") != "1":
            print("stop_hci_dialog: 已忽略重拉（VOICE_ALLOW_RELAUNCH=0）")
            return
        # 当前 Python 文件所在目录
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # 拼接成绝对路径
        asr_script = os.path.abspath(
            os.path.join(base_dir, "../../../launch/launch_depends_asr_question.sh")
        )
        ui_script  = os.path.abspath(
            os.path.join(base_dir, "../../../launch/launch_depends_ui_question.sh")
        )

        # 调用 bash
        subprocess.run(["bash", asr_script])
        subprocess.run(["bash", ui_script])
        return

    def stop_ui_question_callback(self, msg):
        del msg
        if os.environ.get("VOICE_ALLOW_RELAUNCH", "0") != "1":
            print("stop_ui_question: 已忽略重拉（VOICE_ALLOW_RELAUNCH=0）")
            return
        base_dir = os.path.dirname(os.path.abspath(__file__))
        ui_script  = os.path.abspath(
            os.path.join(base_dir, "../../../launch/launch_depends_ui_question.sh")
        )

        subprocess.run(["bash", ui_script])
        return

    # -----------------------------------------------------------------------
    # 这个是整个 session 结束，目前有两种情况，一个是 timeout, 一个是 call nurse
    # -----------------------------------------------------------------------
    def dialog_session_finished_callback(self, msg):
        # 标志着 dialog session 已经结束了
        self.dialog_session_finished = True
        # 是否需要呼叫护士
        self.call_nurse = msg.call_nurse
        # 是否需要呼叫护士的原因
        self.call_nurse_reason = msg.reason


    # -----------------------------------------------------------------------
    # 这里完成人机交互对话的主流程，是对话服务的回调函数
    # 这个服务是 行为树 进行调用的
    # -----------------------------------------------------------------------
    # def hci_dialog_callback(self, req):
    #     persion_id = req.person_id
    #     context = req.context

    #     # 首先先把 asr 打开,进行语音识别
    #     msg = Bool()
    #     msg.data = True
    #     self.start_asr_pub.publish(msg)

    #     # 然后一直等待 dialog_session_finished msg 的到来
    #     self.dialog_session_finished = False
    #     while not self.dialog_session_finished:
    #         time.sleep(0.5)

    #     # 返回结果给 client
    #     return hci_dialogResponse(
    #         summary=self.call_nurse_reason,
    #         need_call_nurse=self.call_nurse,
    #     )

    def asr_init_done_callback(self, msg: Bool):
        del msg
        self.get_logger().info("Received ASR init done signal.")
        self.dialog_ready = True

    # def handle_service(self, req):
    #     rospy.loginfo("Service called: waiting for initialization to complete...")
        
    #     self.dialog_ready = False

    #     # 重启整个 dialog
    #     # self.restart_llm_node()

    #     # 阻塞直到 初始化完成
    #     while self.dialog_ready:
    #         time.sleep(0.1)

    #     rospy.loginfo("Initialization completed. Executing restart sequence...")

    #     # 返回 bool 类型的结果
    #     return SetBoolResponse(success=True, message="LLM node restarted after init.")





    # def stop_callback(self, msg: Bool):
    #     rospy.loginfo("Received stop command on 'huzhou_llm_stop'. Restarting LLM node...")

    #     self.dialog_ready = False

    #     # 直接重启，不用考虑阻塞的事情
    #     self.restart_llm_node()

    # def restart_llm_node(self):
    #     # 结束 tmux session
    #     try:
    #         subprocess.run(["tmux", "kill-session", "-t", "llm_node"], check=True)
    #         rospy.loginfo("Killed tmux session 'llm_node'.")
    #     except subprocess.CalledProcessError:
    #         rospy.logwarn("tmux session 'llm_node' does not exist or could not be killed.")

    #     # Sleep 0.5s
    #     time.sleep(0.5)

    #     # 运行 launch_llm_node.sh
    #     current_dir = os.path.dirname(os.path.realpath(__file__))
    #     launch_script = os.path.join(current_dir, "../launch/launch_llm_node.sh")
    #     if os.path.exists(launch_script):
    #         subprocess.Popen(["bash", launch_script])
    #         rospy.loginfo(f"Relaunched LLM node using {launch_script}")
    #     else:
    #         rospy.logerr(f"Launch script not found: {launch_script}")


if __name__ == "__main__":
    rclpy.init()
    node = StopDialogNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
