#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
最小本地 TTS ROS2 服务节点。

服务名称和接口格式对齐 ros2_node_tts_oneshot.py：

# 一次性 tts 服务的调用
ros2 service call /tts_one_shot llm_node_comm/srv/TtsOneshot "{tts_text: '你好，我在呢。', block: true}"

ros2 topic echo /tts_session_finished

ros2 topic pub --once /tts_realtime_data std_msgs/msg/String "{data: '你好，我在呢。'}"
ros2 topic pub --once /tts_realtime_data std_msgs/msg/String "{data: '[DONE]'}"


"""

import os
import sys
import threading
import time
import traceback
from collections import deque

import numpy as np
import pyaudio
import rclpy
import torch
from qwen_tts import Qwen3TTSModel
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

# 快速解码路径：绕过 HF generate 的双层 Python 循环
# （talker 主循环 + 每帧嵌套的 subtalker generate），
# greedy 模式下与官方路径逐 token 一致（见 tts_fast_generate.py 与 bench_tts_fast.py）
from tts_fast_generate import fast_generate_custom_voice


ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../../install/llm_node_comm/lib/python3.12/site-packages",
    )
)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


from llm_node_comm.srv import TtsOneshot


MODEL_PATH = (
    "/home/medical/Medical_Embodied/src/llm_node_py/"
    "Qwen3-TTS-12Hz-0.6B-CustomVoice"
)

# 采样参数取官方 generation_config.json 默认值（top_k=50/top_p=1.0/温度0.9）。
# 快速解码路径下采样开销已不是瓶颈，无需再收紧参数牺牲音色多样性。
TTS_GEN_KWARGS = dict(
    do_sample=True,
    top_k=50,
    top_p=1.0,
    temperature=0.9,
    repetition_penalty=1.05,
)


class LocalTtsServiceNode(Node):
    def __init__(self):
        super().__init__("ros_node_tts_local")

        self.model_lock = threading.Lock()
        self.queueLock = threading.Lock()
        self.playerLock = threading.Lock()
        self.ttsRealtimeQueue = deque()
        self.ttsOneShotServiceCallbackGroup = MutuallyExclusiveCallbackGroup()
        self.ttsRealtimeDataTopicCallbackGroup = MutuallyExclusiveCallbackGroup()
        self.ttsRealtimeQueueCallbackGroup = MutuallyExclusiveCallbackGroup()
        self.model = Qwen3TTSModel.from_pretrained(
            MODEL_PATH,
            device_map="cuda:0",
            dtype=torch.bfloat16,

            # 使用 flash attention2 加速
            attn_implementation="flash_attention_2",
        )

        # 复用 PyAudio 实例，避免每次播放都重新初始化底层音频子系统
        # （每次 PyAudio() + terminate() 都有几十毫秒固定开销）
        self._pyaudio = pyaudio.PyAudio()

        self.service = self.create_service(
            TtsOneshot,
            "tts_one_shot",
            self.handleTtsOneShotService,
            callback_group=self.ttsOneShotServiceCallbackGroup,
        )
        self.ttsSessionFinishedPublisher = self.create_publisher(
            String,
            "/tts_session_finished",
            10,
        )
        self.ttsRealtimeDataSubscriber = self.create_subscription(
            String,
            "/tts_realtime_data",
            self.handleTtsRealtimeDataTopic,
            200,
            callback_group=self.ttsRealtimeDataTopicCallbackGroup,
        )
        self.ttsRealtimeTimer = self.create_timer(
            0.1,
            self.handleTtsRealtimeQueue,
            callback_group=self.ttsRealtimeQueueCallbackGroup,
        )

        self.get_logger().info("本地 TTS 服务节点启动完成，开始预热模型...")
        # 启动后立即做一次短文本推理：让 CUDA kernel JIT 编译、
        # cuBLAS 算法选择、flash attention 配置一次性完成，
        # 避免首个真实请求承担全部冷启动开销。
        self._warmup_model()

    def _warmup_model(self) -> None:
        """对模型做一次短文本推理，触发 CUDA 内核编译与缓存填充。"""
        try:
            t0 = time.perf_counter()
            with self.model_lock:
                _ = self._synth_once("你好。")
            dt = time.perf_counter() - t0
            self.get_logger().info(
                f"模型预热完成，耗时 {dt:.2f}s，后续请求将显著更快"
            )
        except Exception as exc:
            self.get_logger().warn(f"模型预热失败（不影响服务可用性）: {exc}")

    @torch.inference_mode()
    def _synth_once(self, text: str):
        """单次合成：返回 (audio_np, sample_rate)。

        优先走快速解码路径（绕过 HF generate 双层循环，首音延迟显著降低），
        失败时回退官方 generate_custom_voice 保证可用性。
        """
        try:
            wavs, sample_rate = fast_generate_custom_voice(
                self.model,
                text,
                language="Chinese",
                speaker="Serena",
                **TTS_GEN_KWARGS,
            )
        except Exception:
            self.get_logger().error(
                f"快速 TTS 路径失败，回退官方路径:\n{traceback.format_exc()}"
            )
            wavs, sample_rate = self.model.generate_custom_voice(
                text=text,
                language="Chinese",
                speaker="Serena",
                **TTS_GEN_KWARGS,
            )
        return wavs[0], sample_rate

    def runTtsOneShot(self, text: str):
        with self.model_lock:
            return self._synth_once(text)

    def playTtsAudio(self, audio: np.ndarray, sample_rate: int) -> None:
        with self.playerLock:
            stream = self._pyaudio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=sample_rate,
                output=True,
            )

            try:
                stream.write(audio.astype(np.float32).tobytes())
            finally:
                stream.stop_stream()
                stream.close()

    def _synthAndPlay(self, text: str) -> None:
        """整段合成并播放。

        TTS 是 autoregressive 模型，整段合成比切碎合成更高效
        （一次编码文本、KV cache 连续复用），所以这里不按句分段。
        """
        t0 = time.perf_counter()
        audio, sample_rate = self.runTtsOneShot(text)
        t1 = time.perf_counter()
        self.playTtsAudio(audio, sample_rate)
        t2 = time.perf_counter()
        self.get_logger().info(
            f"合成耗时 {t1 - t0:.2f}s，播放耗时 {t2 - t1:.2f}s"
        )

    def publishTtsSessionFinished(self) -> None:
        msg = String()
        msg.data = "finished"
        self.ttsSessionFinishedPublisher.publish(msg)

    # 一次性的 tts 调用，有 阻塞和非阻塞的 区别
    def handleTtsOneShotService(
        self,
        req: TtsOneshot.Request,
        res: TtsOneshot.Response,
    ):
        self.get_logger().info(f"收到本地 TTS 请求: {req.tts_text}")

        try:
            if req.block:
                self._synthAndPlay(req.tts_text)
            else:
                threading.Thread(
                    target=self._synthAndPlay,
                    args=(req.tts_text,),
                    daemon=True,
                ).start()

            res.result = True
        except Exception as exc:
            self.get_logger().error(
                f"本地 TTS 执行失败: {exc}\n{traceback.format_exc()}"
            )
            res.result = False

        return res

    def handleTtsRealtimeDataTopic(self, msg: String) -> None:
        text = msg.data.strip()
        if not text:
            self.get_logger().info("收到空的 /tts_realtime_data 文本，忽略")
            return

        if text == "[START]":
            return

        if text == "[DONE]":
            with self.queueLock:
                self.ttsRealtimeQueue.append(("[DONE]", None, None))
            return

        try:
            audio, sample_rate = self.runTtsOneShot(text)
        except Exception as exc:
            self.get_logger().error(f"/tts_realtime_data 本地 TTS 合成失败: {exc}")
            return

        with self.queueLock:
            self.ttsRealtimeQueue.append((text, audio, sample_rate))

    def handleTtsRealtimeQueue(self) -> None:
        with self.queueLock:
            if not self.ttsRealtimeQueue:
                return
            text, audio, sample_rate = self.ttsRealtimeQueue.popleft()

        print(f"正在播放：{text}")

        # 遇到 "[DONE]" 的时候说明，之前所有的语句都全部合成完了
        if text == "[DONE]":
            self.publishTtsSessionFinished()
            return

        try:
            self.playTtsAudio(audio, sample_rate)
        except Exception as exc:
            self.get_logger().error(f"/tts_realtime_data 本地 TTS 播放失败: {exc}")


def main(args=None):
    rclpy.init(args=args)
    node = LocalTtsServiceNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        try:
            node._pyaudio.terminate()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == "__main__":
    main()
