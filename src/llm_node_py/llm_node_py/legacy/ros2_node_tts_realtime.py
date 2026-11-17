#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# der 包
# https://www.volcengine.com/docs/6561/1329505?lang=zh

# ali:
# https://help.aliyun.com/zh/model-studio/qwen-tts-realtime?spm=a2c4g.11186623.help-menu-2400256.d_0_4_2_1.297ccb649HUjhr&scm=20140722.H_2938790._.OR_help-T_cn~zh-V_1#f9ec7be148l9g

# https://github.com/aliyun/alibabacloud-bailian-speech-demo/tree/master/samples/conversation/omni


import os, sys
import base64
import threading
import time
import pyaudio
import queue

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from collections import deque

import dashscope
from dashscope.audio.qwen_tts_realtime import *

ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), 
        "./"
    )
)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from get_dashscope_key import get_dashscope_key


# =========================
# 全局变量
# =========================

tts_status = "idle"
pub_finished = None


done_signal_received = False

audio_empty_flag = True


def set_done_signal_received():
    global done_signal_received
    done_signal_received = True

def pub_finished_signal():

    global done_signal_received

    # done_signal_received = False
    if not done_signal_received:
        return
    
    # 发送一个信号，说明这个问题的回答已经结束了
    global pub_finished
    if pub_finished is not None:
        msg = String()
        msg.data = "finished"
        pub_finished.publish(msg)
        print('published /tts_session_finished')
    else:
        print("pub_finished is None")
    
    done_signal_received = False



def init_dashscope_api_key():
    dashscope.api_key = get_dashscope_key()



"""

# 发布问题：
ros2 topic pub --once /question_asr std_msgs/msg/String "{data: '天气今天如何？'}"

python test_ros2_tts.py


# 监听大模型的输出
ros2 topic echo /ali_llm_output
"""


class MyCallback(QwenTtsRealtimeCallback):
    def __init__(self):
        # # 这句话是否已经合成完成
        # self.tts_synthesis_complete_event = False

        # # 合成出来的这句话是否已经播放完成
        # self.play_audio_done_event = False

        self.connected = False

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,
            output=True,
            frames_per_buffer=2048,
        )

        # 音频队列
        self.audio_queue = queue.Queue()

        self.complete_event = threading.Event()

        # 启动音频播放线程
        self.audio_thread = threading.Thread(target=self.audio_worker)
        self.audio_thread.start()

    def reset_event(self):
        self.complete_event = threading.Event()

    def audio_worker(self):

        global audio_empty_flag

        while True:
            try:
                pcm_data = self.audio_queue.get(timeout=0.1)
                if pcm_data is None:
                    break

                # 这个是阻塞的
                audio_empty_flag = True

                self.stream.write(pcm_data)
                

            except queue.Empty:

                audio_empty_flag = False

                # 如果合成完毕，并且 queue 中也空了，说明这句话已经合成并且播放完成了
                # if self.tts_synthesis_complete_event:
                #     self.play_audio_done_event = True
                #     self.tts_synthesis_complete_event = False

                pub_finished_signal()

                continue   # 超时后循环继续，检查 stop_event

    def on_open(self) -> None:
        print('connection opened')
        self.connected = True

    # 在 session finished 之后，这个 on close 似乎会有延迟
    def on_close(self, close_status_code, close_msg) -> None:
        print(f'connection closed with code: {close_status_code}, msg: {close_msg}')
        self.connected = False

        # # 理论上 on close 运行的时候，里面不可能出现 tts_synthesis_complete_event=False
        # if not self.tts_synthesis_complete_event:
        #     # 出现了说明 server 那边卡死了
        #     self.tts_synthesis_complete_event = True

    # 死等，直到 self.connected 为 True
    def wait_for_connected(self):
        while not self.connected:
            time.sleep(0.1)

    def on_event(self, response: str) -> None:
        try:
            global tts_status
            type = response['type']

            if type == 'session.created':
                print(f"start session: {response['session']['id']}")

            elif type == 'response.audio.delta':
                pcm = base64.b64decode(response['delta'])

                # 生产者：将数据放入队列，不阻塞
                self.audio_queue.put(pcm)

                # self.stream.write(pcm)

            elif type == 'session.finished':
                self.complete_event.set()
                print('session finished')

                # self.tts_synthesis_complete_event = True

            elif 'response.done' == type:
                self.complete_event.set()


            elif type == 'error':
                print(f"error: {response['error']}")

        except Exception as e:
            print(f'[Error] {e}')

    def close_audio(self):
        try:
            # 发送终止信号
            self.audio_queue.put(None)
            # 等待线程结束
            if self.audio_thread.is_alive():
                self.audio_thread.join(timeout=2.0)

            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
            print('audio device released')
        except Exception as e:
            print(f'close audio failed: {e}')

    def wait_for_sentence_synthesis(self, timeout=-1):
        # wait_time = 0
        # while not self.tts_synthesis_complete_event:
        #     time.sleep(0.1)
        #     wait_time += 0.1
        #     if timeout > 0 and wait_time > timeout:
        #         return False
            
        return True

    # 等待这个句子播放完成
    def wait_for_finished(self):
        while not self.play_audio_done_event:
            time.sleep(0.1)

    def wait_for_response_done(self, timeout = 2.0):
        # 如果超时而事件没有被 set，它返回 False
        return self.complete_event.wait(timeout=timeout)

# =========================
# ROS Node
# =========================

class QwenRealtimeTtsRosNode(Node):
    def __init__(self):
        super().__init__('qwen_realtime_tts_node')

        init_dashscope_api_key()

        self.get_logger().info('Initializing ...')

        # 发布 tts session finished 的信号
        global pub_finished
        pub_finished = self.create_publisher(
            String, '/tts_session_finished', 10
        )

        # 定时器，每隔 0.3 秒发布信息
        self.timer = self.create_timer(0.3, self.process_queue)

        self.qwen_tts_realtime = None
        self.callback = None
        # self.text_queue = queue.Queue()
        self.text_queue = deque()

        self.connect_to_server()

        self.sub = self.create_subscription(
            String,
            '/tts_realtime_data',
            self.on_text,
            200,
        )

        self.get_logger().info('Qwen realtime TTS ROS node ready')


    # =========================
    # 队列处理
    # =========================
    def process_queue(self):
        global audio_empty_flag

        if self.qwen_tts_realtime and len(self.text_queue) > 0:
            # 从 queue 中取出对应的文本
            try:
                # text = self.text_queue.get_nowait()
                text = self.text_queue.popleft()
            except queue.Empty:
                return

            self.get_logger().info(f'send text: {text}')

            if text == '[DONE]':

                set_done_signal_received()


                return

            # 向 server 追加并提交要进行 tts 的文本
            try:
                print("check connected.")
                # 检查是否 connected:
                if not self.callback.connected:
                    print("reconnect. ")

                    self.callback.close_audio()

                    self.connect_to_server()
                    
                    # # 重新 connect 
                    # self.callback = MyCallback()

                    # self.qwen_tts_realtime = QwenTtsRealtime(
                    #     # model='qwen3-tts-instruct-flash-realtime',
                    #     model='qwen3-tts-flash-realtime',
                    #     callback=self.callback,
                    #     url='wss://dashscope.aliyuncs.com/api-ws/v1/realtime'
                    # )

                    # self.qwen_tts_realtime.connect()

                    # self.qwen_tts_realtime.update_session(
                    #     voice='Cherry',
                    #     response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
                    #     speech_rate=0.8,
                    #     optimize_instructions=True,
                    #     mode='commit',
                    # )

                #     print("wait for qwen server connection ...")
                #     self.callback.wait_for_connected()

                # # 两个全部 clear 掉
                # self.callback.play_audio_done_event = False
                # self.callback.tts_synthesis_complete_event = False

                print("append text + commit.")
                self.qwen_tts_realtime.append_text(text)
                self.qwen_tts_realtime.commit()
                time.sleep(0.1)


                ret = self.callback.wait_for_response_done()
                self.callback.reset_event()

                if not ret:
                    # 超时
                    print("timeout, reconnecting ... ... ")

                    print("wait for audio empyty for reconnect.")

                    # while not audio_empty_flag:
                    #     time.sleep(0.1)

                    self.connect_to_server()
                    self.callback.wait_for_connected()

                    self.text_queue.appendleft(text)





                
                # print("set finish.")
                # self.qwen_tts_realtime.finish()

                # print("wait for tts.")
                # ret = self.callback.wait_for_sentence_synthesis(timeout=2)
                # if ret:
                #     print("wait for finish.")
                #     self.callback.wait_for_finished()
                # else:
                #     print("wait for tts synthesis timeout!")

            except Exception as e:
                print(f'error -> commit({text}) failed: {e}')


    # =========================
    # 连接
    # =========================
    def connect_to_server(self):
        print("connect to server")

        # 每次新建 callback
        self.callback = MyCallback()

        self.qwen_tts_realtime = QwenTtsRealtime(
            # model='qwen3-tts-instruct-flash-realtime',
            model='qwen3-tts-flash-realtime',
            callback=self.callback,
            url='wss://dashscope.aliyuncs.com/api-ws/v1/realtime'
        )

        self.qwen_tts_realtime.connect()

        self.qwen_tts_realtime.update_session(
            voice='Cherry',
            response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
            speech_rate=0.8,
            optimize_instructions=True,
            mode='commit',
        )

    # =========================
    # 接收 ROS 消息
    # =========================
    def on_text(self, msg: String):
        global tts_status

        # 发送过来的文本
        text = msg.data.strip()
        if not text:
            print("ros topic received text msg is None.")
            return

        # START
        if '[START]' in text:
            # # 等待上次的 session close
            # print("wait for last session closing to start a new client.")
            # if self.callback:
            #     self.callback.wait_for_close()

            # if tts_status == "idle":
            #     self.connect_to_server()
            # else:
            #     print(f"tts status error: {tts_status} != 'idle'")

            # tts_status = 'running'
            return

        # # DONE
        # if '[DONE]' in text:
        #     self.text_queue.put('[DONE]')
        #     return

        # 文本入队
        # self.text_queue.put(text)
        self.text_queue.append(text)
        print(f'enqueue: {text}')


    # =========================
    # 关闭
    # =========================
    def shutdown(self):
        return


# =========================
# main
# =========================

def main(args=None):
    rclpy.init(args=args)

    node = QwenRealtimeTtsRosNode()

    try:
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.callback.close_audio()
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()