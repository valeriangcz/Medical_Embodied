
import time
import rclpy
import subprocess
from pathlib import Path

from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool
from rclpy.callback_groups import ReentrantCallbackGroup

"""
监听 session 对话结束的消息：

ros2 topic echo /dialog_session_finished

"""

import sys, os
# 把 ros2 生成的 srv 文件的地址加上
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), 
        "../../../install/llm_node_comm/lib/python3.12/site-packages"
    )
)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# 把 ros2 生成的 srv 文件的地址加上
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), 
        "../../../install/interfaces/lib/python3.12/site-packages"
    )
)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from interfaces.action import LLMInteraction
from interfaces.msg import ActionStatus
from llm_node_comm.msg import DialogSessionFinished
from llm_node_comm.srv import TtsOneshot

# 定义告警模式常量（与 action goal.mode 的值保持一致）。
INTERACTION_ALERT = 0

INTERACTION_PASSIVE = 1

INTERACTION_INTERRUPT = 2


AUDIO_ASSETS_DIR = (
    Path(__file__).resolve().parents[2] / "llm_node_comm" / "audio_assets"
)


def play_mp3_non_blocking(filename: str) -> None:
    subprocess.Popen(
        [
            "ffplay",
            "-nodisp",
            "-autoexit",
            "-loglevel",
            "quiet",
            str(AUDIO_ASSETS_DIR / filename),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def play_mp3_blocking(filename: str) -> None:
    subprocess.run(
        [
            "ffplay",
            "-nodisp",
            "-autoexit",
            "-loglevel",
            "quiet",
            str(AUDIO_ASSETS_DIR / filename),
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


class LLMMockServer(Node):

    def __init__(self):

        super().__init__('llm_mock_server')

        self.actionCallbackGroup = ReentrantCallbackGroup()
        self.dialogSessionFinishedCallbackGroup = ReentrantCallbackGroup()
        self.ttsClientCallbackGroup = ReentrantCallbackGroup()

        # 开始进行 asr 
        self.start_asr_pub = self.create_publisher(
            Bool, 'start_asr', 10
        )
        # 接收 对话结束的标志
        self.dialog_session_finished_sub = self.create_subscription(
            DialogSessionFinished, 
            'dialog_session_finished', 
            self.dialog_session_finished_callback, 
            10,
            callback_group=self.dialogSessionFinishedCallbackGroup,
        )
        # 发送一次性消息，进行一次性的语音合成
        self.tts_client = self.create_client(
            TtsOneshot, 
            'tts_one_shot',
            callback_group=self.ttsClientCallbackGroup,
        )

        # 初始化“对话是否结束”标志位，默认 True 表示当前无进行中的会话。
        self.dialog_session_finished = True
        # 是否要呼叫护士
        self.call_nurse = False
        # 为什么呼叫护士
        self.call_nurse_reason = ''

        # action
        self._server = ActionServer(
            self, 
            LLMInteraction, 
            'llm_interaction', 
            execute_callback=self.execute_callback, 
            goal_callback=self.goal_callback, 
            cancel_callback=self.cancel_callback,
            callback_group=self.actionCallbackGroup,
        )

        self.get_logger().info('llm_mock_server started')


    def goal_callback(self, _goal_request):
        # 返回 ACCEPT，表示接受客户端发来的 action 目标。
        return GoalResponse.ACCEPT

    # 处理 cancel 请求：当前策略为全部允许取消。
    def cancel_callback(self, _goal_handle):
        # 返回 ACCEPT，表示同意客户端取消当前目标。
        return CancelResponse.ACCEPT

    # 发送一个对话消息结束标志的 callback
    # 把发送过来的消息（是否要呼叫护士， 为什么要呼叫护士）全部接收到
    def dialog_session_finished_callback(self, msg):

        print("received dialog session finished signal")

        # 收到结束消息后将结束标志置为 True，驱动等待循环退出。
        self.dialog_session_finished = True
        # 记录是否需要呼叫护士。
        self.call_nurse = msg.call_nurse
        # 记录呼叫护士原因文本。
        self.call_nurse_reason = msg.reason

    def execute_callback(self, goal_handle):

        # 当前处在什么模式下
        goal = goal_handle.request

        print(f"received action request, mode:{goal}")

        # 最终返回的结果
        result = LLMInteraction.Result()

        # 处理过程中的结果
        feedback = LLMInteraction.Feedback()

        # 坠床风险提醒 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if goal.mode == INTERACTION_ALERT:

            req = TtsOneshot.Request()
            req.tts_text = '小心哦，您离床边有点近了，请慢慢移动一下。'
            req.block = True

            # 调用服务，进行语音合成
            self.tts_client.call(req)

            result.status.status = ActionStatus.OK
            result.summary = 'alert patient, no need for calling nurse.'
            result.need_call_nurse = False
            goal_handle.succeed()
            return result
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        # --------------------------------------------------
        # 说几句客套话
        # req = TtsOneshot.Request()
        # req.tts_text = '您好，我是小医，您需要什么帮助吗？'
        # req.block = True
        #
        # self.tts_client.call(req)

        # 这个如果不 block 的话，语音识别会识别到机器人说的话
        play_mp3_blocking("xiaoyi_greeting.mp3")
        # --------------------------------------------------

        # 非告警模式先发送一次“收到上下文”的反馈信息。
        feedback.partial = f'Received context for person {goal.person_id}'
        # 发布初始反馈给 action 客户端。
        goal_handle.publish_feedback(feedback)

        # 开始新一轮对话等待前，将会话结束标志置为 False。
        self.dialog_session_finished = False
        
        # 启动语音识别, 语音对话
        msg = Bool()
        msg.data = True
        self.start_asr_pub.publish(msg)

        # 在未收到 dialog_session_finished 前持续轮询等待。
        print("running... ...")
        while not self.dialog_session_finished:

            # 若客户端请求取消，则立即返回 PREEMPTED。
            # if goal_handle.is_cancel_requested:
            #     result.status.status = ActionStatus.PREEMPTED
            #     result.summary = 'llm action was canceled'
            #     result.need_call_nurse = False
            #     goal_handle.canceled()
            #     return result

            # 跟 clinet 说自己正在忙
            feedback.partial = 'running'
            goal_handle.publish_feedback(feedback)

            # 异步等待 0.2 秒，避免忙等并与参考节奏保持一致。
            time.sleep(0.2)

        print("action exited.")

        # end of 'while' 退出循环说明对话已经结束了


        result.status.status = ActionStatus.OK
        # 呼叫护士的原因
        result.summary = self.call_nurse_reason
        # 是否要呼叫护士
        result.need_call_nurse = self.call_nurse

        if not self.call_nurse:
            # --------------------------------------------------
            # 说几句客套话
            # req = TtsOneshot.Request()
            # req.tts_text = "没有其他事情的话，小医先走了，有问题记得叫小医。"
            # req.block = True
            #
            # self.tts_client.call(req)
            
            time.sleep(2.5)
            
            play_mp3_blocking("xiaoyi_goodbye.mp3")
            # --------------------------------------------------


        goal_handle.succeed()

        print(
            f"action finished. call nurse:'{self.call_nurse}', "
            f"reason:'{self.call_nurse_reason}'"
        )

        return result


def main(args=None):
    rclpy.init(args=args)
    node = LLMMockServer()

    executor = MultiThreadedExecutor(num_threads=5)
    executor.add_node(node)
    
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
