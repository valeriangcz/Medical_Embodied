#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用本地 Qwen3-TTS 模型生成语音，并保存为 mp3 文件。

默认模型名：
Qwen3-TTS-12Hz-0.6B-CustomVoice

说明：
1. 本文件只负责“生成并保存 mp3”，不播放音频。
2. 先将模型输出保存为临时 wav，再调用 ffmpeg 转为 mp3。
3. 若模型路径与默认值不同，可通过传入 model_path 或环境变量
   QWEN_TTS_MODEL_PATH 覆盖。
"""

import os
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
import torch
from qwen_tts import Qwen3TTSModel


DEFAULT_MODEL_PATH = (
    "/home/medical/Medical_Embodied/src/llm_node_py/"
    "Qwen3-TTS-12Hz-0.6B-CustomVoice"
)


def resolve_model_path(cli_model_path: str | None) -> str:
    if cli_model_path:
        return cli_model_path

    env_model_path = os.environ.get("QWEN_TTS_MODEL_PATH")
    if env_model_path:
        return env_model_path

    candidate_paths = [
        DEFAULT_MODEL_PATH,
        str(
            Path(__file__).resolve().parents[1]
            / "Qwen3-TTS-12Hz-0.6B-CustomVoice"
        ),
    ]

    for candidate in candidate_paths:
        if os.path.isdir(candidate):
            return candidate

    return DEFAULT_MODEL_PATH


def save_float_audio_to_wav(audio: np.ndarray, sample_rate: int, wav_path: str) -> None:
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16)

    with wave.open(wav_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())


def convert_wav_to_mp3(wav_path: str, mp3_path: str) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            wav_path,
            "-codec:a",
            "libmp3lame",
            "-qscale:a",
            "2",
            mp3_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def save_mp3(
    text: str,
    output_path: str,
    model_path: str | None = None,
    speaker: str = "Serena",
    language: str = "Chinese",
    instruct: str | None = None,
    device: str = "cuda:0",
) -> str:
    model_path = resolve_model_path(model_path)
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    wavs, sample_rate = model.generate_custom_voice(
        text=text,
        language=language,
        speaker=speaker,
        instruct=instruct,
    )

    output_file = Path(output_path).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        temp_wav_path = temp_wav.name

    try:
        save_float_audio_to_wav(wavs[0], sample_rate, temp_wav_path)
        convert_wav_to_mp3(temp_wav_path, str(output_file))
    finally:
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

    return str(output_file)


if __name__ == "__main__":
    audio_dir = (
        Path(__file__).resolve().parents[2]
        / "llm_node_comm"
        / "audio_assets"
    )

    phrases = [
        # ("你好！我在呢！", "hello_i_am_here.mp3"),
        # ("小医听到啦", "xiaoyi_heard_you.mp3"),
        # ("您好，我是小医，您需要什么帮助吗？", "xiaoyi_greeting.mp3"),
        # ("没有其他事情的话，小医先走啦，有问题记得叫小医哦。", "xiaoyi_goodbye.mp3"),
    ]

    for text, filename in phrases:
        saved_path = save_mp3(
            text=text,
            output_path=str(audio_dir / filename),
            speaker="Serena",
            language="Chinese",
            instruct="用温柔、自然的语气说话",
        )
        print(f"已生成: {saved_path}")
