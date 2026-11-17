#!/usr/bin/env python3
"""离线重建 CosyVoice2 TRT 引擎（flow.decoder.estimator.fp16.mygpu.plan）。

用法（conda env qwen_tts_online）：
  PYTHONNOUSERSITE=1 python tools/build_trt_engine_offline.py

与 ros2_node_tts_cosyvoice.py 的运行时引导保持一致：
sys.path 注入 deps/CosyVoice/Matcha-TTS，torch.load 放开 weights_only，
load_wav 换 soundfile 实现（此处构建引擎用不到，仅为启动路径等价）。
构建耗时约 1~3 分钟（RTX 4060），产物缓存于模型目录，节点启动直接复用。
"""
import os
import sys
import time

RUNTIME = os.environ.get("COSYVOICE_RUNTIME", "/home/medical/cosyvoice_runtime")
MODEL_DIR = os.path.join(RUNTIME, "CosyVoice2-0.5B")
PLAN = os.path.join(MODEL_DIR, "flow.decoder.estimator.fp16.mygpu.plan")

for _p in (os.path.join(RUNTIME, "deps"),
           os.path.join(RUNTIME, "CosyVoice"),
           os.path.join(RUNTIME, "CosyVoice", "third_party", "Matcha-TTS")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if os.path.exists(PLAN):
    print(f"[SKIP] 引擎已存在: {PLAN}")
    sys.exit(0)

import torch  # noqa: E402

_orig_torch_load = torch.load


def _torch_load(f, *a, **k):
    k.setdefault("weights_only", False)
    return _orig_torch_load(f, *a, **k)


torch.load = _torch_load

import cosyvoice.utils.file_utils as _cosy_fu  # noqa: E402
import soundfile as _sf  # noqa: E402


def _load_wav(wav, target_sr, min_sr=16000):
    import torchaudio.functional as _taf
    if isinstance(wav, (str, os.PathLike)):
        data, sr = _sf.read(wav, dtype="float32", always_2d=True)
        speech = torch.from_numpy(data[:, 0])[None, :]
    else:
        speech = wav if wav.dim() == 2 else wav[None]
        sr = 16000
    if speech.shape[0] > 1:
        speech = speech.mean(dim=0, keepdim=True)
    return _taf.resample(speech, sr, target_sr) if sr != target_sr else speech


_cosy_fu.load_wav = _load_wav

from cosyvoice.cli.cosyvoice import CosyVoice2  # noqa: E402

print(f"[BUILD] GPU={torch.cuda.get_device_name(0)} 模型={MODEL_DIR}")
t0 = time.perf_counter()
model = CosyVoice2(MODEL_DIR, load_jit=False, load_trt=True,
                   load_vllm=False, fp16=True)
dt = time.perf_counter() - t0
assert os.path.exists(PLAN), "构建结束后 plan 文件不存在"
size_mb = os.path.getsize(PLAN) / 1e6
print(f"[OK] TRT 引擎构建完成 {dt:.1f}s, plan={size_mb:.0f}MB, sr={model.sample_rate}")
