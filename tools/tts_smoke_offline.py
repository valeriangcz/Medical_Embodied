#!/usr/bin/env python3
"""CosyVoice2 离线合成冒烟验证：加载 TRT 缓存引擎 → 注册 Serena 音色 → 合成一句到 /tmp/tts_smoke.wav。
用法: PYTHONNOUSERSITE=1 <conda_env_python> tools/tts_smoke_offline.py
"""
import os
import sys
import time

RUNTIME = os.environ.get("COSYVOICE_RUNTIME", "/home/medical/cosyvoice_runtime")
MODEL_DIR = os.path.join(RUNTIME, "CosyVoice2-0.5B")
ASSET = "/home/medical/Medical_Embodied/src/llm_node_py/cosyvoice_assets/prompt_serena.wav"
OUT = "/tmp/tts_smoke.wav"
PROMPT_TEXT = "您好，我是血透巡诊机器人小医，今天由我为您做透析前的健康指导。"

for _p in (os.path.join(RUNTIME, "deps"),
           os.path.join(RUNTIME, "CosyVoice"),
           os.path.join(RUNTIME, "CosyVoice", "third_party", "Matcha-TTS")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

_orig_torch_load = torch.load


def _torch_load(f, *a, **k):
    k.setdefault("weights_only", False)
    return _orig_torch_load(f, *a, **k)


torch.load = _torch_load

import cosyvoice.utils.file_utils as _cosy_fu  # noqa: E402


def _load_wav(wav, target_sr, min_sr=16000):
    import torchaudio.functional as _taf
    if isinstance(wav, (str, os.PathLike)):
        data, sr = sf.read(wav, dtype="float32", always_2d=True)
        speech = torch.from_numpy(data[:, 0])[None, :]
    else:
        speech = wav if wav.dim() == 2 else wav[None]
        sr = 16000
    if speech.shape[0] > 1:
        speech = speech.mean(dim=0, keepdim=True)
    return _taf.resample(speech, sr, target_sr) if sr != target_sr else speech


_cosy_fu.load_wav = _load_wav

from cosyvoice.cli.cosyvoice import CosyVoice2  # noqa: E402

print(f"[LOAD] GPU={torch.cuda.get_device_name(0)} (TRT 缓存引擎复用)")
t0 = time.perf_counter()
model = CosyVoice2(MODEL_DIR, load_jit=False, load_trt=True,
                   load_vllm=False, fp16=True)
print(f"[LOAD] 完成 {time.perf_counter() - t0:.1f}s, sr={model.sample_rate}")

prompt16k = _load_wav(ASSET, 16000)
mi = model.frontend.frontend_zero_shot(PROMPT_TEXT, PROMPT_TEXT,
                                       prompt16k, model.sample_rate, "")
model.frontend.spk2info["xiaoyi"] = mi
print("[VOICE] 音色 xiaoyi 注册完成")

TEXT = "你好，我在呢。透析间期请注意控制饮水量。"
t1 = time.perf_counter()
chunks = []
for item in model.inference_zero_shot(TEXT, PROMPT_TEXT, prompt16k,
                                      stream=True, speed=1.0):
    speech = item["tts_speech"] if isinstance(item, dict) else item
    chunks.append(speech.squeeze().cpu())
audio = torch.cat(chunks)
dt = time.perf_counter() - t1
dur = audio.numel() / model.sample_rate
sf.write(OUT, audio.numpy(), model.sample_rate)
print(f"[OK] 合成 {dur:.2f}s 音频耗时 {dt:.2f}s (RTF≈{dt / dur:.2f}) -> {OUT}")
assert dur > 0.5, "音频过短，疑似合成异常"
