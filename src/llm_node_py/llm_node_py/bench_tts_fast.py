#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tts_fast_generate 正确性 + 性能基准（v2）。

v1 结论修正：
  - greedy 必须同时关掉 subtalker 采样（subtalker_dosample=False）才是确定性参考；
    v1 只固定了 talker，subtalker 仍在随机采样，比对结果无效。
  - 性能对比必须交替 A/B（笔记本 4060 有热降频，先后跑会不公平）。

流程：
  1. 预热（官方路径）
  2. 双 greedy 确定性比对：官方 codes_a vs patch后官方 codes_c vs 完整快速 codes_b
  3. 逐阶段剖析：talker.forward / subtalker / talker采样 各占多少
  4. 交替 A/B x2：官方 vs 快速，同条件对比 RTF
  5. codec 解码单独计时 + 音频健全性
"""

import os
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwen_tts import Qwen3TTSModel
import tts_fast_generate as fast

MODEL_PATH = "/home/medical/Medical_Embodied/src/llm_node_py/Qwen3-TTS-12Hz-0.6B-CustomVoice"
TEXT = "你好，我在呢,今天天气很不错，适合出去玩。"
SAMPLING = dict(do_sample=True, top_k=10, top_p=0.9, temperature=0.9, repetition_penalty=1.05)

PROF = dict(talker_fwd=0.0, sub=0.0, frames=0)


def audio_stats(wavs, sr):
    a = np.asarray(wavs[0])
    return dict(
        duration_s=round(len(a) / sr, 2),
        rms=round(float(np.sqrt(np.mean(a**2))), 4),
        finite=bool(np.isfinite(a).all()),
        peak=round(float(np.max(np.abs(a))), 3),
    )


def codes_from_stock(wrapper, input_ids, *, do_sample, subtalker_dosample):
    with torch.inference_mode():
        gen_kwargs = wrapper._merge_generate_kwargs(
            do_sample=do_sample,
            top_k=None,
            top_p=None,
            temperature=None,
            repetition_penalty=1.05,
            subtalker_dosample=subtalker_dosample,
        )
        codes_list, _ = wrapper.model.generate(
            input_ids=[input_ids],
            instruct_ids=[None],
            languages=["Chinese"],
            speakers=["Serena"],
            non_streaming_mode=True,
            **gen_kwargs,
        )
    return codes_list[0]


def codes_from_fast(wrapper, input_ids, *, do_sample, subtalker_do_sample):
    with torch.inference_mode():
        return fast.fast_generate_codes(
            wrapper,
            input_ids,
            "Chinese",
            "Serena",
            do_sample=do_sample,
            top_k=10,
            top_p=0.9,
            temperature=0.9,
            repetition_penalty=1.05,
            subtalker_do_sample=subtalker_do_sample,
            subtalker_top_k=50,
            subtalker_top_p=1.0,
            subtalker_temperature=0.9,
            max_new_tokens=8192,
        )


def diff_report(name_a, a, name_b, b):
    if a.shape == b.shape and torch.equal(a, b):
        print(f"  {name_a} == {name_b}: True")
        return True
    print(f"  {name_a} == {name_b}: False  形状 {tuple(a.shape)} vs {tuple(b.shape)}")
    n = min(a.shape[0], b.shape[0])
    frame_diff = (a[:n] != b[:n]).any(dim=-1)
    idx = torch.nonzero(frame_diff)
    print(f"  差异帧数: {idx.numel()}/{n}, 首个差异帧: {idx[0].item() if idx.numel() else '无'}")
    if idx.numel():
        i0 = idx[0].item()
        cb = (a[i0] != b[i0]).nonzero().flatten().tolist()
        print(f"  帧 {i0} 差异 codebook 位: {cb}")
        print(f"    a: {a[i0].tolist()}")
        print(f"    b: {b[i0].tolist()}")
    return False


def run_stock_sampled(wrapper, text):
    t0 = time.perf_counter()
    wavs, sr = wrapper.generate_custom_voice(
        text=text, language="Chinese", speaker="Serena", **SAMPLING
    )
    dt = time.perf_counter() - t0
    return wavs, sr, dt


def run_fast_sampled(wrapper, text):
    t0 = time.perf_counter()
    with torch.inference_mode():
        wavs, sr = fast.fast_generate_custom_voice(
            wrapper, text, language="Chinese", speaker="Serena", **SAMPLING
        )
    dt = time.perf_counter() - t0
    return wavs, sr, dt


def main():
    print("=== 加载模型 ===", flush=True)
    t0 = time.perf_counter()
    wrapper = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    print(f"加载耗时 {time.perf_counter() - t0:.2f}s\n", flush=True)

    print("=== 1. 官方路径预热 ===", flush=True)
    t0 = time.perf_counter()
    wrapper.generate_custom_voice(
        text="你好。", language="Chinese", speaker="Serena", **SAMPLING
    )
    print(f"预热耗时 {time.perf_counter() - t0:.2f}s\n", flush=True)

    input_ids = wrapper._tokenize_texts([wrapper._build_assistant_text(TEXT)])[0]
    print(f"文本 token 数: {input_ids.shape[1]}", flush=True)

    print("\n=== 2. 双 greedy 确定性比对（talker 与 subtalker 都关采样） ===", flush=True)
    t0 = time.perf_counter()
    codes_a = codes_from_stock(
        wrapper, input_ids, do_sample=False, subtalker_dosample=False
    )
    print(f"官方双 greedy: {time.perf_counter() - t0:.2f}s, 形状 {tuple(codes_a.shape)}", flush=True)

    patched_fn = fast.patch_subtalker(wrapper.model)
    t0 = time.perf_counter()
    codes_c = codes_from_stock(
        wrapper, input_ids, do_sample=False, subtalker_dosample=False
    )
    print(f"patch后官方 talker 循环双 greedy: {time.perf_counter() - t0:.2f}s", flush=True)
    ok_c = diff_report("codes_a", codes_a, "codes_c", codes_c)

    t0 = time.perf_counter()
    codes_b = codes_from_fast(
        wrapper, input_ids, do_sample=False, subtalker_do_sample=False
    )
    print(f"完整快速路径双 greedy: {time.perf_counter() - t0:.2f}s", flush=True)
    ok_b = diff_report("codes_a", codes_a, "codes_b", codes_b)
    if not (ok_c and ok_b):
        print("!! 等价性验证未通过，后续性能数据仅供参考", flush=True)

    print("\n=== 3. 快速路径（CUDA graph 引擎）计时 ===", flush=True)
    t0 = time.perf_counter()
    wavs, sr, _ = run_fast_sampled(wrapper, TEXT)
    dt_total = time.perf_counter() - t0
    n_frames = int(round(len(wavs[0]) / sr * 12))
    print(
        f"总合成 {dt_total:.2f}s, 音频 {len(wavs[0]) / sr:.2f}s "
        f"(约 {n_frames} 帧, {dt_total / max(n_frames, 1) * 1000:.1f}ms/帧)"
    )
    print(f"  音频统计: {audio_stats(wavs, sr)}", flush=True)

    print("\n=== 4. 交替 A/B x2（A=未patch官方基线, B=快速路径） ===", flush=True)
    for i in range(2):
        fast.unpatch_subtalker(wrapper.model, None)
        wavs, sr, dt = run_stock_sampled(wrapper, TEXT)
        dur = len(wavs[0]) / sr
        print(f"  [A{i + 1}] 官方  {dt:.2f}s 音频 {dur:.2f}s RTF={dt / dur:.2f}", flush=True)
        fast.patch_subtalker(wrapper.model)
        wavs, sr, dt = run_fast_sampled(wrapper, TEXT)
        dur = len(wavs[0]) / sr
        print(f"  [B{i + 1}] 快速  {dt:.2f}s 音频 {dur:.2f}s RTF={dt / dur:.2f}", flush=True)

    print("\n=== 完成 ===", flush=True)


if __name__ == "__main__":
    main()
