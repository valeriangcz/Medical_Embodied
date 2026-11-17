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
import json
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

# 文本规范化：LLM 偶发公式符号（=、+、%、3-5、mmol/L 等）→ TTS 可靠朗读的口语
from tts_text_norm import normalize_for_tts


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

# 固定首句（填充词）：收到 [START]（LLM 即将开始生成）时立即播预合成缓存，
# 让患者在 LLM 首包+首单元合成期间先听到回应。置空禁用。
TTS_FILLER_TEXT = os.environ.get("TTS_FILLER_TEXT", "好的，")


class LocalTtsServiceNode(Node):
    def __init__(self):
        super().__init__("ros_node_tts_local")

        self.model_lock = threading.Lock()
        self.queueLock = threading.Lock()
        self.playerLock = threading.Lock()
        # 播报打断：/tts_cancel 到达时立即停止当前播放并清空待播队列
        self.cancelEvent = threading.Event()
        self.ttsRealtimeQueue = deque()
        self.ttsOneShotServiceCallbackGroup = MutuallyExclusiveCallbackGroup()
        self.ttsRealtimeDataTopicCallbackGroup = MutuallyExclusiveCallbackGroup()
        self.ttsRealtimeQueueCallbackGroup = MutuallyExclusiveCallbackGroup()
        try:
            self.model = Qwen3TTSModel.from_pretrained(
                MODEL_PATH,
                device_map="cuda:0",
                dtype=torch.bfloat16,

                # 使用 flash attention2 加速
                attn_implementation="flash_attention_2",
            )
        except ImportError:
            # 未安装 flash-attn 的机器回退 sdpa 注意力（部署机上无需编译 CUDA 扩展）
            self.model = Qwen3TTSModel.from_pretrained(
                MODEL_PATH,
                device_map="cuda:0",
                dtype=torch.bfloat16,
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
        self.ttsCancelSubscriber = self.create_subscription(
            String,
            "/tts_cancel",
            self.handleTtsCancel,
            10,
            callback_group=self.ttsRealtimeDataTopicCallbackGroup,
        )
        self.ttsRealtimeTimer = self.create_timer(
            0.1,
            self.handleTtsRealtimeQueue,
            callback_group=self.ttsRealtimeQueueCallbackGroup,
        )

        # ===== 链路耗时埋点（LATENCY_PROBE=1 时生效，默认不创建话题）=====
        self._latency_probe = os.environ.get("LATENCY_PROBE", "0") == "1"
        self._latency_pub = None
        if self._latency_probe:
            self._latency_pub = self.create_publisher(String, "/latency_event", 200)
            self.get_logger().info(
                "latency probe enabled, publishing /latency_event (tts_local)"
            )

        self.get_logger().info("本地 TTS 服务节点启动完成，开始预热模型...")
        # 启动后立即做一次短文本推理：让 CUDA kernel JIT 编译、
        # cuBLAS 算法选择、flash attention 配置一次性完成，
        # 避免首个真实请求承担全部冷启动开销。
        self._warmup_model()

        # 预合成填充词缓存（收到 [START] 时 0 延迟入播队列）
        self._filler_audio = None
        self._filler_sr = None
        if TTS_FILLER_TEXT:
            try:
                t0 = time.perf_counter()
                with self.model_lock:
                    self._filler_audio, self._filler_sr = self._synth_once(TTS_FILLER_TEXT)
                self.get_logger().info(
                    f"填充词缓存完成: {TTS_FILLER_TEXT!r} "
                    f"({len(self._filler_audio) / self._filler_sr:.1f}s, "
                    f"{time.perf_counter() - t0:.1f}s)")
            except Exception as exc:
                self.get_logger().warn(f"填充词预合成失败（禁用填充词）: {exc}")
                self._filler_audio = None

    def _lat(self, stage: str, **detail) -> None:
        """把本节点内部时间戳发布到 /latency_event（仅在 LATENCY_PROBE=1 时）。"""
        if self._latency_pub is None:
            return
        try:
            payload = {
                "node": "tts_local",
                "stage": stage,
                "t_ns": time.time_ns(),
                "detail": detail,
            }
            msg = String()
            msg.data = json.dumps(payload, ensure_ascii=False)
            self._latency_pub.publish(msg)
        except Exception:
            pass

    def _warmup_model(self) -> None:
        """对模型做一次短文本推理，触发 CUDA 内核编译与缓存填充。"""
        try:
            t0 = time.perf_counter()
            with self.model_lock:
                _ = self._synth_once("你好。")
            dt = time.perf_counter() - t0
            self._lat("model_warmup", dur_ms=dt * 1000.0, ok=True)
            self.get_logger().info(
                f"模型预热完成，耗时 {dt:.2f}s，后续请求将显著更快"
            )
        except Exception as exc:
            self._lat("model_warmup", ok=False)
            self.get_logger().warn(f"模型预热失败（不影响服务可用性）: {exc}")

    @torch.inference_mode()
    def _synth_once(self, text: str):
        """单次合成：返回 (audio_np, sample_rate)。

        优先走快速解码路径（绕过 HF generate 双层循环，首音延迟显著降低），
        失败时回退官方 generate_custom_voice 保证可用性。
        合成前做文本规范化（公式符号 → 口语，见 tts_text_norm.py）。
        """
        norm = normalize_for_tts(text)
        if norm != text:
            self.get_logger().info(f"文本规范化: {text!r} -> {norm!r}")
        try:
            wavs, sample_rate = fast_generate_custom_voice(
                self.model,
                norm,
                language="Chinese",
                speaker="Serena",
                **TTS_GEN_KWARGS,
            )
        except Exception:
            self.get_logger().error(
                f"快速 TTS 路径失败，回退官方路径:\n{traceback.format_exc()}"
            )
            wavs, sample_rate = self.model.generate_custom_voice(
                text=norm,
                language="Chinese",
                speaker="Serena",
                **TTS_GEN_KWARGS,
            )
        return wavs[0], sample_rate

    def runTtsOneShot(self, text: str):
        with self.model_lock:
            return self._synth_once(text)

    def playTtsAudio(self, audio: np.ndarray, sample_rate: int) -> bool:
        """播放音频；返回 False 表示被 /tts_cancel 中途打断。"""
        with self.playerLock:
            if self.cancelEvent.is_set():
                return False
            stream = self._pyaudio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=sample_rate,
                output=True,
            )
            interrupted = False
            try:
                data = audio.astype(np.float32)
                # 按 100ms 分块写入，便于及时响应打断
                block = max(int(sample_rate * 0.1), 1)
                for start in range(0, len(data), block):
                    if self.cancelEvent.is_set():
                        interrupted = True
                        break
                    stream.write(data[start:start + block].tobytes())
            finally:
                stream.stop_stream()
                stream.close()
            return not interrupted

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
            # 新一轮对话：复位打断状态
            if self.cancelEvent.is_set():
                self.cancelEvent.clear()
                self.get_logger().info("收到 [START]，复位 /tts_cancel 状态")
            self._lat("rx_control", control="[START]")
            # 固定首句：LLM 生成期间先给患者一个即时回应
            if self._filler_audio is not None:
                with self.queueLock:
                    self.ttsRealtimeQueue.append(
                        ("[填充词]", self._filler_audio, self._filler_sr))
                self._lat("filler_enqueued", dur_s=round(
                    len(self._filler_audio) / self._filler_sr, 2))
            return

        if text == "[DONE]":
            self._lat("rx_control", control="[DONE]")
            with self.queueLock:
                self.ttsRealtimeQueue.append(("[DONE]", None, None))
            return

        self._lat("rx", text_len=len(text), head=text[:24])
        try:
            synth_start_ns = time.time_ns()
            self._lat("synth_start", text_len=len(text))
            audio, sample_rate = self.runTtsOneShot(text)
            synth_end_ns = time.time_ns()
            dur_ms = (synth_end_ns - synth_start_ns) / 1e6
            audio_sec = (
                float(len(audio)) / float(sample_rate) if sample_rate else 0.0
            )
            self._lat(
                "synth_end",
                text_len=len(text),
                dur_ms=dur_ms,
                audio_sec=audio_sec,
                rtf=(dur_ms / 1000.0 / audio_sec if audio_sec > 0 else 0.0),
            )
        except Exception as exc:
            self._lat("synth_error", text_len=len(text))
            self.get_logger().error(f"/tts_realtime_data 本地 TTS 合成失败: {exc}")
            return

        with self.queueLock:
            self.ttsRealtimeQueue.append((text, audio, sample_rate))

    def handleTtsCancel(self, msg: String) -> None:
        """立即打断播报：清空队列并让正在播放的音块尽快停止。"""
        reason = msg.data.strip() or "cancel"
        self.cancelEvent.set()
        with self.queueLock:
            dropped = len(self.ttsRealtimeQueue)
            self.ttsRealtimeQueue.clear()
        self._lat("cancel", reason=reason, dropped=dropped)
        self.get_logger().info(
            f"收到 /tts_cancel({reason})，已清空 {dropped} 条待播文本并打断当前播放"
        )

    def handleTtsRealtimeQueue(self) -> None:
        if self.cancelEvent.is_set():
            return
        with self.queueLock:
            if not self.ttsRealtimeQueue:
                return
            text, audio, sample_rate = self.ttsRealtimeQueue.popleft()

        print(f"正在播放：{text}")

        # 遇到 "[DONE]" 的时候说明，之前所有的语句都全部合成完了
        if text == "[DONE]":
            self._lat("session_finished")
            self.publishTtsSessionFinished()
            return

        play_start_ns = time.time_ns()
        self._lat("play_start", text_len=len(text), head=text[:24])
        interrupted = False
        try:
            interrupted = not self.playTtsAudio(audio, sample_rate)
        except Exception as exc:
            self.get_logger().error(f"/tts_realtime_data 本地 TTS 播放失败: {exc}")
        finally:
            self._lat(
                "play_end",
                text_len=len(text),
                interrupted=interrupted,
                dur_ms=(time.time_ns() - play_start_ns) / 1e6,
            )


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
