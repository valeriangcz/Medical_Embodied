#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import base64
import threading
import time
import pyaudio

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import dashscope
from dashscope.audio.qwen_tts_realtime import *

# 当前目录
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), 
        "./"
    )
)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

"""
ros2 topic pub -1 /tts_realtime_data std_msgs/msg/String "{data: '[START]'}"

ros2 topic pub -1 /tts_realtime_data std_msgs/msg/String "{data: '你好，我是小医，请问有什么可以帮助你的？'}"

ros2 topic pub -1 /tts_realtime_data std_msgs/msg/String "{data: '[DONE]'}"

"""

from get_dashscope_key import get_dashscope_key

# =========================
# 全局变量（保持你的写法）
# =========================

qwen_tts_realtime: QwenTtsRealtime = None  # type: ignore

DO_VIDEO_TEST = False

tts_status = 'idle'


def init_dashscope_api_key():
    """
        Set your DashScope API-key. More information:
        https://github.com/aliyun/alibabacloud-bailian-speech-demo/blob/master/PREREQUISITES.md
    """

    # 新加坡和北京地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
    # if 'DASHSCOPE_API_KEY' in os.environ:
    #     # load API-key from environment variable DASHSCOPE_API_KEY
    #     dashscope.api_key = os.environ['DASHSCOPE_API_KEY']
    # else:
    #     dashscope.api_key = 'sk-xxxx'  # set API-key manually

    dashscope.api_key = get_dashscope_key()


class MyCallback(QwenTtsRealtimeCallback):
    def __init__(self, node: Node):
        self.node = node
        self.complete_event = threading.Event()

        # 初始化 PyAudio
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,
            output=True,
            frames_per_buffer=8000,
        )

        # 添加 ROS 发布器
        self.pub_finished = self.node.create_publisher(
            String, '/tts_session_finished', 10
        )


    # 当和服务端建立连接完成后，该方法立刻被回调
    def on_open(self) -> None:
        self.node.get_logger().info('connection opened, init player')

    # 当服务已经关闭连接后进行回调
    def on_close(self, close_status_code, close_msg) -> None:
        # close_status_code：关闭WebSocket的状态码。
        # close_msg：关闭WebSocket的关闭信息。

        # 关闭扬声器设备
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        self.node.get_logger().info('connection closed, audio device released')

        self.node.get_logger().info(
            f'connection closed with code: {close_status_code}, msg: {close_msg}, destroy player'
        )

    def on_event(self, response: str) -> None:  # type: ignore
        try:
            global qwen_tts_realtime, tts_status
            type = response['type']  # type: ignore

            if 'session.created' == type:
                self.node.get_logger().info(
                    f"start session: {response['session']['id']}"  # type: ignore
                )

            elif 'response.audio.delta' == type:
                recv_audio_b64 = response['delta']  # type: ignore

                # 把生成的音频播放到扬声器设备
                pcm = base64.b64decode(recv_audio_b64)
                self.stream.write(pcm)

            elif 'response.done' == type:
                # rospy.loginfo(
                #     'response %s dpub_finishedone',
                #     qwen_tts_realtime.get_last_response_id()
                # )
                pass

            elif 'session.finished' == type:
                # 这个 session finished 好像就是当语音全部播放完成之后
                self.node.get_logger().info('session finished')
                self.complete_event.set()

                # 改变当前状态
                tts_status = 'idle'

                # 发布 ROS topic 告诉其他节点当前的session结束了
                # 其实也就是语音已经播放完成了
                time.sleep(0.3)
                msg = String()
                msg.data = "finished"
                self.pub_finished.publish(msg)
                self.node.get_logger().info('published /tts_session_finished')

            elif 'error' == type:
                self.node.get_logger().error(
                    f"error: {response['error']['code']}, {response['error']['message']}"  # type: ignore
                )
            
            # else:
            #     print(f"unknow status: {type}")

        except Exception as e:
            self.node.get_logger().error(f'[Error] {e}')
            return

    def wait_for_finished(self):
        return self.complete_event.wait(timeout=10.0)

    def reset_finished(self):
        self.complete_event.clear()


class QwenRealtimeTtsRosNode(Node):
    def __init__(self):
        super().__init__('qwen_realtime_tts_node')

        init_dashscope_api_key()

        self.get_logger().info('Initializing ...')

        self.callback = MyCallback(self)

        # 初始化 Qwen 实时 TTS
        self.qwen_tts_realtime = QwenTtsRealtime(
            # model='qwen3-tts-instruct-flash-realtime',  # 使用的是哪个模型
            model='qwen3-tts-instruct-flash-realtime',

            callback=self.callback,                     # callback function
            url='wss://dashscope.aliyuncs.com/api-ws/v1/realtime'  # 北京的 api 地址
        )

        self.connect_to_server()

        # 订阅要合成为语音的文本
        self.sub = self.create_subscription(
            String,
            '/tts_realtime_data',
            self.on_text,
            200,
        )

        self.get_logger().info('Qwen realtime TTS ROS node ready')

    def connect_to_server(self):
        print("connect to tts realtime server.")

        self.qwen_tts_realtime.connect()  # 手动建立与服务器的连接

        self.qwen_tts_realtime.update_session(
            voice='Cherry',  # 语音合成所使用的音色
            response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
 
            # 语速调节
            speech_rate=0.8,

            # instructions 相关
            # instructions='热心的女护士，带有关心关切的语气。',
            # 当设置为 True 时，系统将对 instructions 的内容进行语义增强与重写，生成更适合语音合成的内部指令
            optimize_instructions=True,

            # mode='server_commit',  # server commit 模式还是 commit 模式
            mode='commit',
        )

    def on_text(self, msg: String):
        global tts_status

        text = msg.data.strip()
        if not text:
            return
        
        if '[START]' in text:
            if tts_status == "idle":
                self.connect_to_server()
                
            # 是一次对话的开始
            self.get_logger().info("receive [START] flag")
            # 改变状态
            tts_status = 'running'

        elif '[DONE]' in text:
            # 是这次对话的结束

            # 清除对话结束的标志
            self.callback.reset_finished()

            # 发送一个信号，告诉服务器文本输入已经结束了
            self.qwen_tts_realtime.finish()

            if not self.callback.wait_for_finished():
                self.get_logger().warn('wait session finished timeout after [DONE]')

            self.get_logger().info("receive [DONE] flag")
            
        else:


            self.get_logger().info(f'send text: {text}')

            time.sleep(0.1) # 不要发送的太快

            # 将文本片段追加到云端输入文本缓冲区。缓冲区是你可以写入并稍后提交的临时存储。
            # 如果是"server_commit"模式下，服务器决定何时提交并合成文本缓冲区中的文本。
            self.qwen_tts_realtime.append_text("" + text + "。")

            # 在 'commit' 下要手动触发
            self.qwen_tts_realtime.commit()


    def shutdown(self):
        self.get_logger().info('shutdown qwen realtime tts node')
        self.qwen_tts_realtime.finish()


def main(args=None):
    rclpy.init(args=args)

    node = None
    try:
        node = QwenRealtimeTtsRosNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
