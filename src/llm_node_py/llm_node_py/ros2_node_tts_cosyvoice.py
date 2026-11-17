#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""CosyVoice2 本地 TTS ROS2 节点 —— 接口与 ros2_node_tts_local.py 完全一致。

替换链路中的 TTS 引擎：Qwen3-TTS-0.6B → CosyVoice2-0.5B（流式架构、中英混、零样本音色）。
对外契约不变，上游（stream_text 断句器 / dialog_manager / nurse_alert）无需任何改动：

  # 一次性 tts 服务的调用
  ros2 service call /tts_one_shot llm_node_comm/srv/TtsOneshot "{tts_text: '你好，我在呢。', block: true}"

  ros2 topic pub --once /tts_realtime_data std_msgs/msg/String "{data: '你好，我在呢。'}"
  ros2 topic pub --once /tts_realtime_data std_msgs/msg/String "{data: '[DONE]'}"
  ros2 topic echo /tts_session_finished

环境变量：
  COSYVOICE_RUNTIME   运行时根目录（含 CosyVoice 仓库 / CosyVoice2-0.5B 模型 / deps）
                      默认 /home/artrc/cosyvoice_runtime
  TTS_PROMPT_WAV      音色提示音频（16s 内，克隆音色的"声纹"）
                      默认 cosyvoice_assets/prompt_serena.wav（与现役 Serena 音色同源）
  TTS_PROMPT_TEXT     提示音频对应的文字（必须与音频逐字一致）
  TTS_SPK_ID          注册后的音色 id，默认 xiaoyi
  TTS_SPEED           语速 0.6~1.4，默认 1.0
  TTS_FILLER_TEXT     固定首句/填充词（收到 [START] 立即播，默认"好的，"；置空禁用）
  LATENCY_PROBE       1 时发布 /latency_event 埋点

文本预处理：合成前调用 tts_text_norm.normalize_for_tts，把 LLM 偶发的公式符号
（=、+、%、3-5、mmol/L 等）改写成 TTS 可靠朗读的口语。

运行时依赖首次搭建（本机已完成，重启不丢）：
  COSYVOICE_RUNTIME/CosyVoice          https://github.com/FunAudioLLM/CosyVoice (含 Matcha 子模块)
  COSYVOICE_RUNTIME/CosyVoice2-0.5B    ModelScope iic/CosyVoice2-0.5B
  COSYVOICE_RUNTIME/deps               推理依赖（transformers 4.51.3 等，PYTHONPATH 注入，不污染 conda 环境）
                                       deps/whisper 为本地桩（CosyVoice 只用 log_mel_spectrogram）
"""

import ctypes
import json
import os
import re
import sys
import threading
import time
import traceback
import queue
from collections import deque

# 必须在 import torch 之前设置：显存分配器分段扩展，
# 实测配合权重 fp16 把进程占用从 2946MiB 降到 1600MiB
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pyaudio
import rclpy
import torch
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

# ---------- CosyVoice 运行时引导（必须在 import cosyvoice 之前） ----------
COSY_RUNTIME = os.environ.get("COSYVOICE_RUNTIME", "/home/artrc/cosyvoice_runtime")
COSY_REPO = os.path.join(COSY_RUNTIME, "CosyVoice")
COSY_DEPS = os.path.join(COSY_RUNTIME, "deps")
COSY_MODEL_DIR = os.environ.get(
    "COSYVOICE_MODEL_DIR", os.path.join(COSY_RUNTIME, "CosyVoice2-0.5B"))
_ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "cosyvoice_assets")
TTS_PROMPT_WAV = os.environ.get(
    "TTS_PROMPT_WAV", os.path.join(_ASSET_DIR, "prompt_serena.wav"))
TTS_PROMPT_TEXT = os.environ.get(
    "TTS_PROMPT_TEXT", "您好，我是血透巡诊机器人小医，今天由我为您做透析前的健康指导。")
TTS_SPK_ID = os.environ.get("TTS_SPK_ID", "xiaoyi")
TTS_SPEED = float(os.environ.get("TTS_SPEED", "1.0"))
# TensorRT 加速 flow 解码（2026-11-10 起默认开启，用户确认允许）：
# 引擎已构建并缓存于模型目录（flow.decoder.estimator.fp16.mygpu.plan，170MB，
# 换 GPU 需删除重建，构建约 78s）。
# 实测（RTX 3080）：RTF 0.437→0.385（约 12% 提速），显存 1600→2036MiB。
# 需要省显存时设 TTS_LOAD_TRT=0 关闭。
TTS_LOAD_TRT = os.environ.get("TTS_LOAD_TRT", "1") == "1"
# 播放队列深度上限（工业背压）：满时等待，超时丢最老保最新
PLAYBACK_QMAX = int(os.environ.get("VOICE_TTS_QMAX", "15"))


class _StreamSplitter:
    """断句器（TTS 进程内）：切分送 CosyVoice2 的合成单元。

    粒度实测（2026-11-09，campplus 声纹一致性量化）：
      逐标点子句切（旧行为）：2~6 字碎片独立生成，各段音色/韵律各自漂移
      （短文本声纹相似度低至 0.73），总 RTF 3.65，拼接后听感"很乱"。
      句级切分（默认）：12~30 字单元，总 RTF 1.78，单元内韵律连贯。
    TTS_SPLIT_COMMAS=1 可回退旧逐子句行为。
    """

    PATTERN = re.compile(r"[^，。；：、！？]*[，。；：、！？]?")
    SENTENCE_MARKS = "。；！？"

    def __init__(self):
        self.buffer = ""
        self.first_flushed = False
        # 逗号是否切断（默认否：并入整句，避免碎片化合成）
        self.split_commas = os.environ.get("TTS_SPLIT_COMMAS", "0") == "1"

    def reset(self):
        self.buffer = ""
        self.first_flushed = False

    def feed(self, text):
        """按序切分：以逗号结尾的碎片必须并入紧随的完整句一起发出。
        旧实现把碎片留在 buffer 而先发后面的完整句，导致句子顺序颠倒
        （实测："第二句是解释说明，继续补充第二句。" → 先播后半句）。"""
        self.buffer += text
        units, pending = [], ""
        for part in self.PATTERN.findall(self.buffer):
            part = part.strip()
            if not part:
                continue
            if part[-1] in self.SENTENCE_MARKS:
                self.first_flushed = True
                units.append(pending + part)      # 前缀碎片 + 本句，顺序保护
                pending = ""
            elif self.split_commas:
                self.first_flushed = True
                units.append(pending + part)
                pending = ""
            else:
                pending += part                   # 逗号/顿号等并入下一句
        self.buffer = pending
        return units

    def flush(self):
        if self.buffer.strip():
            out, self.buffer = self.buffer.strip(), ""
            self.first_flushed = True
            return [out]
        return []
# 固定首句（填充词）：收到 [START]（LLM 即将开始生成）时立即播报，
# 让患者在 LLM 首包+首单元合成期间先听到回应。置空禁用。
TTS_FILLER_TEXT = os.environ.get("TTS_FILLER_TEXT", "好的，")

for _p in (COSY_DEPS, COSY_REPO, os.path.join(COSY_REPO, "third_party", "Matcha-TTS")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# torch>=2.6 默认 weights_only=True，CosyVoice 的 ckpt 加载需要放开
_orig_torch_load = torch.load


def _torch_load(f, *a, **k):
    k.setdefault("weights_only", False)
    return _orig_torch_load(f, *a, **k)


torch.load = _torch_load

import cosyvoice.utils.file_utils as _cosy_fu  # noqa: E402
import soundfile as _sf  # noqa: E402


def _load_wav(wav, target_sr, min_sr=16000):
    """等效替换 cosyvoice 的 load_wav（torchaudio 2.11 的 load 需要 torchcodec）。"""
    import torchaudio.functional as _taf
    if isinstance(wav, (str, os.PathLike)):
        data, sr = _sf.read(wav, dtype="float32", always_2d=True)
        speech = torch.from_numpy(data[:, 0])[None, :]
    else:  # 张量输入：CosyVoice 内部约定为 16k 提示音
        speech = wav if wav.dim() == 2 else wav[None]
        sr = 16000
    if speech.shape[0] > 1:
        speech = speech.mean(dim=0, keepdim=True)
    return _taf.resample(speech, sr, target_sr) if sr != target_sr else speech


_cosy_fu.load_wav = _load_wav

from cosyvoice.cli.cosyvoice import CosyVoice2  # noqa: E402

# ---------- 文本规范化（数学符号 → 口语） ----------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tts_text_norm import normalize_for_tts  # noqa: E402

ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../../install/llm_node_comm/lib/python3.12/site-packages",
    )
)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from llm_node_comm.srv import TtsOneshot  # noqa: E402


class CosyVoiceTtsServiceNode(Node):
    def __init__(self):
        super().__init__("ros_node_tts_cosyvoice")

        self.model_lock = threading.Lock()
        self.queueLock = threading.Lock()
        self.playerLock = threading.Lock()
        self.cancelEvent = threading.Event()
        self.ttsRealtimeQueue = deque()
        self.ttsOneShotServiceCallbackGroup = MutuallyExclusiveCallbackGroup()
        self.ttsRealtimeDataTopicCallbackGroup = MutuallyExclusiveCallbackGroup()
        self.ttsRealtimeQueueCallbackGroup = MutuallyExclusiveCallbackGroup()

        self.get_logger().info(f"加载 CosyVoice2: {COSY_MODEL_DIR} (TRT={'on' if TTS_LOAD_TRT else 'off'})")
        t0 = time.perf_counter()
        self.model = CosyVoice2(COSY_MODEL_DIR, load_jit=False, load_trt=TTS_LOAD_TRT,
                                load_vllm=False, fp16=True)
        self.get_logger().info(
            f"CosyVoice2 加载完成 {time.perf_counter() - t0:.1f}s, sr={self.model.sample_rate}")

        # 注册克隆音色（一次性；后续每次合成跳过提示音特征提取）
        if not os.path.exists(TTS_PROMPT_WAV):
            raise FileNotFoundError(
                f"音色提示音频不存在: {TTS_PROMPT_WAV}（设置 TTS_PROMPT_WAV 指向 16k/24k wav）")
        prompt16k = _load_wav(TTS_PROMPT_WAV, 16000)
        mi = self.model.frontend.frontend_zero_shot(
            TTS_PROMPT_TEXT, TTS_PROMPT_TEXT, prompt16k, self.model.sample_rate, "")
        self.model.frontend.spk2info[TTS_SPK_ID] = mi
        self._prompt16k = prompt16k
        self.get_logger().info(
            f"音色注册完成: {TTS_SPK_ID} <- {os.path.basename(TTS_PROMPT_WAV)}")

        # 权重真 fp16：fp16=True 只是前向 autocast，权重仍按 fp32 驻留显存。
        # 实测（RTX 3080）：2946MiB → 1600MiB（llm+flow half），RTF 0.60→0.44，
        # ASR 回环转写与 fp32 完全一致。hift(vocoder) 保持 fp32（conv1d 不兼容）。
        # TRT 启用时 flow 解码已由 fp16 TRT 引擎接管，flow 编码器默认保持 fp32
        #（避免与 TRT 引擎的 dtype 摩擦），可用 TTS_FLOW_FP16=1 强制。
        if os.environ.get("TTS_FP16_WEIGHTS", "1") == "1":
            try:
                inner = self.model.model
                inner.llm.half()
                flow_half = os.environ.get("TTS_FLOW_FP16", "0" if TTS_LOAD_TRT else "1") == "1"
                if flow_half:
                    inner.flow.half()
                torch.cuda.empty_cache()
                self.get_logger().info(
                    f"权重已转 fp16（llm{' +flow' if flow_half else ''}），显存约省 1.3GB")
            except Exception as exc:
                self.get_logger().warn(f"fp16 权重转换失败（保持 fp32 权重）: {exc}")

        self._pyaudio = pyaudio.PyAudio()

        self.service = self.create_service(
            TtsOneshot, "tts_one_shot", self.handleTtsOneShotService,
            callback_group=self.ttsOneShotServiceCallbackGroup)
        self.ttsSessionFinishedPublisher = self.create_publisher(
            String, "/tts_session_finished", 10)
        # 断句器内嵌（原 stream_text 节点职责）：直接消费 /ali_llm_output 原始流，
        # 按标点子句切分后送合成，/tts_realtime_data 中间话题消失
        self._splitter = _StreamSplitter()
        self.ttsRealtimeDataSubscriber = self.create_subscription(
            String, "/ali_llm_output", self.handleAliLlmOutput, 200,
            callback_group=self.ttsRealtimeDataTopicCallbackGroup)
        self.ttsCancelSubscriber = self.create_subscription(
            String, "/tts_cancel", self.handleTtsCancel, 10,
            callback_group=self.ttsRealtimeDataTopicCallbackGroup)
        self.ttsRealtimeTimer = self.create_timer(
            0.1, self.handleTtsRealtimeQueue,
            callback_group=self.ttsRealtimeQueueCallbackGroup)

        self._latency_probe = os.environ.get("LATENCY_PROBE", "0") == "1"
        self._latency_pub = None
        if self._latency_probe:
            self._latency_pub = self.create_publisher(String, "/latency_event", 200)
            self.get_logger().info(
                "latency probe enabled, publishing /latency_event (tts_cosyvoice)")

        self.get_logger().info("CosyVoice2 TTS 服务节点启动完成，开始预热模型...")
        self._warmup_model()

        # C1 内存卸载：音色已注册（缓存 spk），speech_tokenizer/campplus 仅用于
        # 注册/克隆音色——卸载可省 RAM（克隆新音色需重启节点）。
        if os.environ.get("VOICE_UNLOAD_TOKENIZER", "1") == "1":
            try:
                fe = self.model.frontend
                fe.speech_tokenizer_session = None
                fe.campplus_session = None
                import gc as _gc
                _gc.collect()
                self.get_logger().info("tokenizer/campplus 会话已卸载（省 RAM）")
            except Exception as exc:
                self.get_logger().warn(f"tokenizer 卸载失败（保留）: {exc}")

        # 预合成填充词缓存（0 延迟可播）。首轮 TTS 单元用流式合成时，
        # 该状态位为 True，播完即复位。
        self._awaiting_first_unit = False
        self._playback_qmax = PLAYBACK_QMAX
        self._session_gen = 0   # 会话代数：换轮后丢弃旧轮仍在合成中的单元
        self._filler_audio = None
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

        # 加载期内存回收：fp32→fp16 转换后的 CPU 侧临时张量、tokenizer 会话等
        # 已不再需要，但 glibc/PyTorch 分配器不主动归还 OS，RSS 会常驻在峰值
        #（实测 ~5.1GB）。这里 gc + malloc_trim 把可回收部分还给系统（给导航省 RAM）。
        def _trim_ram():
            import gc
            gc.collect()
            freed = 0
            try:
                libc = ctypes.CDLL("libc.so.6")
                freed = libc.malloc_trim(0)
            except Exception:
                pass
            try:
                rss = int(open("/proc/self/statm").read().split()[1]) * 4 // 1024
            except Exception:
                rss = -1
            self.get_logger().info(f"加载期内存回收完成(malloc_trim={freed})，RSS≈{rss}MB")

        threading.Thread(target=_trim_ram, daemon=True).start()

    # 合成工作线程：单元队列 → CosyVoice2 合成 → 播放队列（与播放重叠）
        self._synthQ = queue.Queue()
        self._synth_worker = threading.Thread(target=self._synth_loop, daemon=True)
        self._synth_worker.start()

    # ---------- 延迟埋点（与 tts_local 同款） ----------
    def _lat(self, stage: str, **detail) -> None:
        if self._latency_pub is None:
            return
        try:
            payload = {"node": "tts_cosyvoice", "stage": stage,
                       "t_ns": time.time_ns(), "detail": detail}
            msg = String()
            msg.data = json.dumps(payload, ensure_ascii=False)
            self._latency_pub.publish(msg)
        except Exception:
            pass

    def _warmup_model(self) -> None:
        try:
            t0 = time.perf_counter()
            with self.model_lock:
                _ = self._synth_once("你好。")
            dt = time.perf_counter() - t0
            self._lat("model_warmup", dur_ms=dt * 1000.0, ok=True)
            self.get_logger().info(f"模型预热完成，耗时 {dt:.2f}s")
        except Exception as exc:
            self._lat("model_warmup", ok=False)
            self.get_logger().warn(f"模型预热失败（不影响服务可用性）: {exc}")

    @torch.inference_mode()
    def _synth_once(self, text: str, stream: bool = False, on_chunk=None):
        """单次合成：返回 (audio_np_float32, sample_rate)。

        stream=False：整单元合成后一次性返回（默认，与 tts_local 语义一致）。
        stream=True：流式合成，每得到一个音频块即调用 on_chunk(audio, sr)
        （用于首轮 TTS 单元边合成边播，首音提前约 0.3~0.9s），
        同时仍返回拼接后的完整音频供队列使用。
        """
        norm = normalize_for_tts(text)
        if norm != text:
            self.get_logger().info(f"文本规范化: {text!r} -> {norm!r}")
        pieces = []
        for chunk in self.model.inference_zero_shot(
                norm, TTS_PROMPT_TEXT, self._prompt16k,
                zero_shot_spk_id=TTS_SPK_ID, stream=stream, speed=TTS_SPEED):
            piece = chunk["tts_speech"][0].float().cpu().numpy()
            pieces.append(piece)
            if on_chunk is not None:
                on_chunk(piece.astype(np.float32), self.model.sample_rate)
        if not pieces:
            return np.zeros(0, dtype=np.float32), self.model.sample_rate
        return np.concatenate(pieces).astype(np.float32), self.model.sample_rate

    def runTtsOneShot(self, text: str):
        with self.model_lock:
            return self._synth_once(text)

    def runTtsOneShotStreaming(self, text: str, on_chunk):
        """流式合成整单元：每个音频块通过 on_chunk(audio, sr) 即时回调。"""
        with self.model_lock:
            return self._synth_once(text, stream=True, on_chunk=on_chunk)

    def playTtsAudio(self, audio: np.ndarray, sample_rate: int) -> bool:
        """播放音频；返回 False 表示被 /tts_cancel 中途打断。"""
        with self.playerLock:
            if self.cancelEvent.is_set():
                return False
            stream = self._pyaudio.open(
                format=pyaudio.paFloat32, channels=1, rate=sample_rate, output=True)
            interrupted = False
            try:
                data = audio.astype(np.float32)
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
        t0 = time.perf_counter()
        audio, sample_rate = self.runTtsOneShot(text)
        t1 = time.perf_counter()
        self.playTtsAudio(audio, sample_rate)
        t2 = time.perf_counter()
        self.get_logger().info(f"合成耗时 {t1 - t0:.2f}s，播放耗时 {t2 - t1:.2f}s")

    def publishTtsSessionFinished(self) -> None:
        msg = String()
        msg.data = "finished"
        self.ttsSessionFinishedPublisher.publish(msg)

    def handleTtsOneShotService(self, req: TtsOneshot.Request, res: TtsOneshot.Response):
        self.get_logger().info(f"收到 CosyVoice2 TTS 请求: {req.tts_text}")
        try:
            if self.cancelEvent.is_set():
                # 显式 oneshot（如呼叫护士播报）要覆盖上一条 /tts_cancel：
                # 等主播放线程退出后再清标志，否则 playTtsAudio 直接返回、播报静音。
                time.sleep(0.12)
                self.cancelEvent.clear()
            if req.block:
                self._synthAndPlay(req.tts_text)
            else:
                threading.Thread(target=self._synthAndPlay, args=(req.tts_text,),
                                 daemon=True).start()
            res.result = True
        except Exception as exc:
            self.get_logger().error(
                f"CosyVoice2 TTS 执行失败: {exc}\n{traceback.format_exc()}")
            res.result = False
        return res

    def handleAliLlmOutput(self, msg: String) -> None:
        """/ali_llm_output 原始流：[START]/[DONE] 控制 + 断句器切分出 TTS 单元。"""
        text = msg.data.strip()
        if not text:
            return

        if text == "[START]":
            # 新一轮对话新高一：打断上一轮残留播放并清空未播队列，
            # 否则上一轮说"结束"时没播完的单元会留到下次唤醒抢播（实测 2026-11-09）
            self.cancelEvent.set()
            time.sleep(0.05)                      # 让播放线程尽快停在当前音频块
            with self.queueLock:
                dropped = len(self.ttsRealtimeQueue)
                self.ttsRealtimeQueue.clear()
            self.cancelEvent.clear()
            # 会话代数+1：丢弃上一轮滞留在合成队列/合成中的单元（否则它们会在
            # 下一轮唤醒后补播——实测"没回答，重新唤醒后先播上一题"的根因）
            self._session_gen += 1
            drained = 0
            while True:
                try:
                    item = self._synthQ.get_nowait()
                    self._synthQ.task_done()
                    if item[0] == "unit":
                        drained += 1
                except queue.Empty:
                    break
            self.get_logger().info(
                f"收到 [START]，清掉上轮残留 {dropped} 条(播放)/{drained} 条(待合成)，"
                f"会话代数->{self._session_gen}")
            self._splitter.reset()
            # 本轮第一个 TTS 单元走流式合成（首块即入播）
            self._awaiting_first_unit = True
            self._lat("rx_control", control="[START]")
            # 固定首句：LLM 生成期间先给患者一个即时回应，
            # 把"说完→听到回应"的体感从 4~5s 压到 ~2s
            if self._filler_audio is not None:
                with self.queueLock:
                    self.ttsRealtimeQueue.append(
                        ("[填充词]", self._filler_audio, self._filler_sr))
                self._lat("filler_enqueued", dur_s=round(
                    len(self._filler_audio) / self._filler_sr, 2))
            return

        if text == "[DONE]":
            self._lat("rx_control", control="[DONE]")
            # 冲刷断句器残留，再入队 [DONE] 结束标记
            for unit in self._splitter.flush():
                self._handle_unit(unit)
            self._synthQ.put(("done", None, None, self._session_gen))
            return

        # LLM 增量文本 → 断句器 → TTS 单元
        for unit in self._splitter.feed(text):
            self._handle_unit(unit)

    def _enqueue_playback(self, entry) -> None:
        """入播放队列，带深度护栏（工业背压）：
        队列满时等待最多 10s，仍满则丢最老一条保最新（防无限积压）。"""
        deadline = time.time() + 10.0
        while True:
            with self.queueLock:
                if len(self.ttsRealtimeQueue) < self._playback_qmax:
                    self.ttsRealtimeQueue.append(entry)
                    return
                overflow = True
            if time.time() > deadline:
                with self.queueLock:
                    if self.ttsRealtimeQueue:
                        dropped = self.ttsRealtimeQueue.popleft()
                        self.get_logger().warn(
                            f"播放队列满(>{self._playback_qmax})，丢弃最老: "
                            f"{dropped[0][:16]}")
                    self.ttsRealtimeQueue.append(entry)
                return
            time.sleep(0.05)

    def _synth_loop(self) -> None:
        """合成线程：unit → 合成 → 播放队列；done → 转发 [DONE] 结束标记。"""
        while True:
            item = self._synthQ.get()
            try:
                kind = item[0]
                item_gen = item[3] if len(item) > 3 else self._session_gen
                if kind == "done":
                    if item_gen == self._session_gen:
                        with self.queueLock:
                            self.ttsRealtimeQueue.append(("[DONE]", None, None))
                    continue
                text = item[1]
                # 排空合并：已排队的后续单元并入本次合成（单次生成音频连续，
                # 消除"短首句播完、次句还在合成"的句间空隙；上限 120 字）
                done_flag = False
                merged = 0
                while len(text) < 120:
                    try:
                        nxt = self._synthQ.get_nowait()
                    except queue.Empty:
                        break
                    self._synthQ.task_done()
                    if nxt[0] == "done":
                        if nxt[3] == self._session_gen:
                            done_flag = True
                        break
                    if nxt[3] != item_gen:
                        continue          # 跨轮次滞留单元丢弃
                    text += nxt[1]
                    merged += 1
                if merged:
                    self.get_logger().info(f"合并 {merged} 个后续单元一起合成")

                def _enqueue_chunk(audio, sample_rate, _text=text):
                    with self.queueLock:
                        self.ttsRealtimeQueue.append(
                            (f"{_text[:12]}…", audio, sample_rate))

                # stream_first 已在 _handle_unit 入队时捕获（避免与回调线程竞态）
                stream_first = item[2]
                try:
                    synth_start_ns = time.time_ns()
                    self._lat("synth_start", text_len=len(text), stream_first=stream_first)
                    def _chunk_gate(a, s, _g=item_gen, _t=text):
                        if _g == self._session_gen:
                            self._enqueue_playback((f"{_t[:12]}…", a, s))

                    if stream_first:
                        audio, sample_rate = self.runTtsOneShotStreaming(
                            text, _chunk_gate)
                    else:
                        audio, sample_rate = self.runTtsOneShot(text)
                    synth_end_ns = time.time_ns()
                    dur_ms = (synth_end_ns - synth_start_ns) / 1e6
                    audio_sec = float(len(audio)) / float(sample_rate) if sample_rate else 0.0
                    self._lat("synth_end", text_len=len(text), dur_ms=dur_ms,
                              audio_sec=audio_sec,
                              rtf=(dur_ms / 1000.0 / audio_sec if audio_sec > 0 else 0.0))
                except Exception as exc:
                    self._lat("synth_error", text_len=len(text))
                    self.get_logger().error(f"文本单元 CosyVoice2 合成失败: {exc}")
                    continue

                if item_gen != self._session_gen:
                    self.get_logger().info(
                        f"丢弃过期合成结果(代数{item_gen}->{self._session_gen}): {text[:12]}")
                    continue
                if not stream_first:
                    self._enqueue_playback((text, audio, sample_rate))
                    if os.environ.get("TTS_DEBUG_DUP") == "1":
                        self.get_logger().info(f"[D2]合成完成入播:{text[:12]}")
                if done_flag:
                    with self.queueLock:
                        self.ttsRealtimeQueue.append(("[DONE]", None, None))
            finally:
                self._synthQ.task_done()

    def _handle_unit(self, text: str) -> None:
        """单元入合成队列并立即返回；合成在独立线程，与播放重叠，
        消除句末标点处的"合成等待"间隙。"""
        self._lat("rx", text_len=len(text), head=text[:24])
        stream_first = self._awaiting_first_unit
        self._awaiting_first_unit = False
        self._synthQ.put(("unit", text, stream_first, self._session_gen))
        if os.environ.get("TTS_DEBUG_DUP") == "1":
            self.get_logger().info(f"[D1]入合成队列:{text[:12]}")


    def handleTtsCancel(self, msg: String) -> None:
        reason = msg.data.strip() or "cancel"
        self.cancelEvent.set()
        with self.queueLock:
            dropped = len(self.ttsRealtimeQueue)
            self.ttsRealtimeQueue.clear()
        # 同时丢弃已在合成队列/合成中的旧单元：否则 cancel 后它们仍可能在
        # 下一轮/oneshot 时被补播（结束对话/呼叫护士路径）。
        self._session_gen += 1
        drained = 0
        while True:
            try:
                item = self._synthQ.get_nowait()
                self._synthQ.task_done()
                if item[0] == "unit":
                    drained += 1
            except queue.Empty:
                break
        self._splitter.reset()
        self._lat("cancel", reason=reason, dropped=dropped, drained=drained,
                  gen=self._session_gen)
        self.get_logger().info(
            f"收到 /tts_cancel({reason})，已清空 {dropped} 条待播/"
            f"{drained} 条待合成并打断当前播放，会话代数->{self._session_gen}")

    def handleTtsRealtimeQueue(self) -> None:
        if self.cancelEvent.is_set():
            return
        with self.queueLock:
            if not self.ttsRealtimeQueue:
                return
            text, audio, sample_rate = self.ttsRealtimeQueue.popleft()

        # 带 epoch：可精确统计播放次数（原来 print 无时间戳，被仪表盘逐帧重绘无法去重）
        self.get_logger().info(f"正在播放：{text}")
        if os.environ.get("TTS_DEBUG_DUP") == "1":
            with self.queueLock:
                _ql = len(self.ttsRealtimeQueue)
                _nxt = (self.ttsRealtimeQueue[0][0][:12]
                        if self.ttsRealtimeQueue else "-")
            self.get_logger().info(f"[D3]出队播放:{text[:12]} 剩余{_ql} 下一个={_nxt}")

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
            self.get_logger().error(f"/tts_realtime_data CosyVoice2 播放失败: {exc}")
        finally:
            self._lat("play_end", text_len=len(text), interrupted=interrupted,
                      dur_ms=(time.time_ns() - play_start_ns) / 1e6)


def main(args=None):
    rclpy.init(args=args)
    node = CosyVoiceTtsServiceNode()
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
