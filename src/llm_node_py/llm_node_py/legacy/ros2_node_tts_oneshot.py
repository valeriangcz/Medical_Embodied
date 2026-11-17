#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
============================================================
节点名称：tts_one_shot_service_node

节点功能说明：
    本节点提供一次性阻塞式 TTS 服务。
    每次调用服务都会创建独立的播放实例，互不影响。
    服务调用阻塞，直到语音播放完成才返回。
============================================================
"""

"""
对服务进行测试：
ros2 service call /tts_one_shot llm_node_comm/srv/TtsOneshot "{tts_text: '你好，我在呢。', block: true}"
"""


# import 生成的 service 文件
import sys
import os

# 把 ros2 生成的 srv 文件地址加上
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../../install/llm_node_comm/lib/python3.12/site-packages"
    )
)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# 当前目录
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), 
        "./"
    )
)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


from llm_node_comm.srv import TtsOneshot  # type: ignore

import time
import base64
import threading
import pyaudio
import rclpy
from rclpy.node import Node
import dashscope
from dashscope.audio.qwen_tts_realtime import *
from get_dashscope_key import get_dashscope_key

# =========================
# 全局变量
# =========================
qwen_tts_realtime: QwenTtsRealtime = None  # type: ignore


def init_dashscope_api_key():
    dashscope.api_key = get_dashscope_key()


class OneShotTtsCallback(QwenTtsRealtimeCallback):
    """每次服务调用都会创建独立实例，播放完成触发事件"""

    def __init__(self, node: Node):
        self.node = node
        self.complete_event = threading.Event()

        self.connected = False

        # 初始化 PyAudio 播放器
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,
            output=True,
            frames_per_buffer=1024,
        )

    def on_open(self) -> None:
        print('connection opened')
        self.connected = True

    def on_close(self, close_status_code, close_msg) -> None:
        print("websocket 连接关闭。")
        self.connected = False
        # self.stream.stop_stream()
        # self.stream.close()
        # self.p.terminate()
        # self.node.get_logger().info(
        #     f'WebSocket 连接关闭，释放音频设备: code={close_status_code} msg={close_msg}'
        # )

    # 死等，直到 self.connected 为 True
    def wait_for_connected(self):
        while not self.connected:
            time.sleep(0.1)

    def on_event(self, response: str) -> None:  # type: ignore
        try:
            type_ = response['type']  # type: ignore

            if type_ == 'response.audio.delta':
                pcm = base64.b64decode(response['delta'])  # type: ignore
                self.stream.write(pcm)

            elif type_ == 'session.finished':
                self.node.get_logger().info('一次性 TTS 播放完成')
                self.complete_event.set()

            elif type_ == 'error':
                self.node.get_logger().error(
                    "TTS 错误: %s, %s"
                    % (
                        response['error']['code'],  # type: ignore
                        response['error']['message'],  # type: ignore
                    )
                )

        except Exception as e:
            self.node.get_logger().error(f'[OneShotTTS Error] {e}')

    def wait_until_done(self):
        """阻塞等待语音播放完成"""
        self.complete_event.wait()


class OneShotTtsServiceNode(Node):
    """
    ROS 服务节点，每次调用服务都是一次性、阻塞式播放
    """

    def __init__(self):
        super().__init__('ros_node_tts_one_shot')
        init_dashscope_api_key()

        self.get_logger().info("初始化一次性 TTS 服务节点...")

        self.callback = None
        self.qwen_tts_realtime = None

        # # 创建独立 callback
        # self.callback = OneShotTtsCallback(self)

        # # 初始化全局 TTS 实例，不绑定 callback
        # self.qwen_tts_realtime = QwenTtsRealtime(
        #     model='qwen3-tts-instruct-flash-realtime',
        #     callback=self.callback,
        #     url='wss://dashscope.aliyuncs.com/api-ws/v1/realtime'
        # )
        # self.qwen_tts_realtime.connect()
        # self.get_logger().info("TTS WebSocket 已连接")

        # self.qwen_tts_realtime.update_session(
        #     voice='Cherry',

        #     # 语速调节
        #     speech_rate=0.8,

        #     response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
        #     optimize_instructions=True,
        #     mode='commit'
        # )

        # 注册服务
        self.service = self.create_service(
            TtsOneshot,
            'tts_one_shot',
            self.handle_request
        )

        self.get_logger().info("一次性 TTS 服务节点启动完成，等待调用...")

    def handle_request(self, req: TtsOneshot.Request, res: TtsOneshot.Response):
        text_to_speak = req.tts_text
        block = req.block
        self.get_logger().info(f"收到一次性 TTS 请求: {text_to_speak}")

        self.callback = OneShotTtsCallback(self)

        self.qwen_tts_realtime = QwenTtsRealtime(
            model='qwen3-tts-instruct-flash-realtime',
            callback=self.callback,
            url='wss://dashscope.aliyuncs.com/api-ws/v1/realtime'
        )
        self.qwen_tts_realtime.connect()
        self.get_logger().info("TTS WebSocket 已连接")

        self.qwen_tts_realtime.update_session(
            voice='Cherry',

            # 语速调节
            speech_rate=0.8,

            response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
            optimize_instructions=True,
            mode='commit'
        )

        self.callback.wait_for_connected()


        # 追加文本并提交
        self.qwen_tts_realtime.append_text(text_to_speak)
        self.qwen_tts_realtime.commit()
        self.qwen_tts_realtime.finish()

        if block:
            time.sleep(0.1)
            # 阻塞等待播放完成
            self.callback.wait_until_done()
        print("一次性 TTS 播放完成，服务返回")

        # del self.callback, self.qwen_tts_realtime

        res.result = True
        return res

    def shutdown(self):
        self.get_logger().info("关闭一次性 TTS 服务节点")
        self.qwen_tts_realtime.finish()


def main(args=None):
    rclpy.init(args=args)
    node = OneShotTtsServiceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
