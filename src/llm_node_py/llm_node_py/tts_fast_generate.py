#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen3-TTS 0.6B (12Hz CustomVoice) 快速解码路径。

背景：官方推理链路里，每个音频帧（12Hz）除了 1 次 talker 单步解码外，
还会在 talker.forward 的 decode 分支里嵌套调用一次完整的 HuggingFace
`code_predictor.generate()`（2 token prefill + num_code_groups-1=15 步解码）。
每帧 17 次模型 forward 全部经过 HF generate 的 Python 机器（采样器构建、
mask 创建、每步 GPU->CPU 同步），小模型上 CPU 开销远超 GPU 计算，
实测 GPU 利用率只有 ~33%，25 字合成 ~5s（RTF≈1.0）。

本模块绕过两层 HF generate 循环，手写精简解码循环：
  1. talker 主循环：DynamicCache + 每帧 1 token forward + 手工采样
     （重复惩罚 / suppress_tokens / min_new_tokens / temperature / top_k / top_p，
     语义与 HF LogitsProcessor 一一对应）
  2. subtalker：把嵌套 generate 换成 1 次 prefill + 14 步解码的裸循环
     （通过实例级 monkey-patch code_predictor.generate，talker.forward
     的 decode 分支无需改动，rope/mask/尾文本逻辑全部保持官方实现）

正确性验证：greedy（do_sample=False）是确定性的，本路径与官方路径
输出的 codec 序列应当逐 token 完全一致（见 bench_tts_fast.py）。
"""

import types
from types import SimpleNamespace

import torch
from transformers.cache_utils import DynamicCache

import tts_cuda_graphs as _graph

_EMPTY_LONG = None


def _empty_long(device):
    global _EMPTY_LONG
    if _EMPTY_LONG is None or _EMPTY_LONG.device != device:
        _EMPTY_LONG = torch.empty(0, dtype=torch.long, device=device)
    return _EMPTY_LONG


def _sample_token(
    logits: torch.Tensor,
    *,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    prev_tokens: torch.Tensor,
    suppress_idx: torch.Tensor = None,
    eos_token_id: int = None,
    min_new_tokens: int = 0,
) -> torch.Tensor:
    """对单个位置的 logits 采样，返回 shape (1,) 的 token id。

    logits: 1D (V,)，调用方必须传入新算出的 logits（本函数原地修改）。
    prev_tokens: 已生成的 token（用于重复惩罚）。
    处理顺序与 HF _get_logits_processor 一致：
    suppress -> repetition_penalty -> min_new_tokens -> temperature -> top_k -> top_p。
    """
    if suppress_idx is not None:
        logits.index_fill_(0, suppress_idx, float("-inf"))
    if repetition_penalty and repetition_penalty != 1.0 and prev_tokens.numel() > 0:
        score = logits[prev_tokens]
        logits[prev_tokens] = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
    if (
        min_new_tokens
        and eos_token_id is not None
        and prev_tokens.numel() < min_new_tokens
    ):
        logits[eos_token_id] = float("-inf")
    if not do_sample:
        return logits.argmax().view(1)
    if temperature and temperature != 1.0:
        logits = logits / temperature
    if top_k:
        k = min(max(int(top_k), 1), logits.shape[-1] - 1)
        kth = torch.topk(logits, k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
        sorted_indices_to_remove[-1] = False
        indices_to_remove = sorted_indices_to_remove.scatter(
            0, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).view(1)


def _fast_subtalker_generate(
    code_predictor,
    inputs_embeds: torch.Tensor,
    *,
    max_new_tokens: int,
    do_sample: bool,
    top_p: float,
    top_k: int,
    temperature: float,
) -> torch.Tensor:
    """subtalker（code_predictor）的裸解码循环，替代嵌套的 HF generate。

    inputs_embeds: (1, 2, H) = [talker 上一帧末隐状态, talker 刚采样 token 的 embedding]。
    返回 (1, max_new_tokens) 的 codebook token 序列。
    采样语义与 HF 对该次 generate 应用的处理器一致（temperature/top_k/top_p，
    无重复惩罚/抑制/最短长度），逐帧缓存新建，位置编号与官方对齐。
    """
    pmodel = code_predictor.model
    proj = code_predictor.small_to_mtp_projection
    is_identity_proj = isinstance(proj, torch.nn.Identity)
    heads = code_predictor.lm_head
    codec_embedding = pmodel.codec_embedding
    device = inputs_embeds.device

    cache = DynamicCache()
    x = proj(inputs_embeds) if not is_identity_proj else inputs_embeds
    out = pmodel(
        inputs_embeds=x,
        past_key_values=cache,
        use_cache=True,
        cache_position=torch.arange(x.shape[1], device=device),
    )
    logits = heads[0](out.last_hidden_state[:, -1])[0]

    sparams = dict(
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=1.0,
        prev_tokens=_empty_long(device),
    )
    # 位置编号缓冲：fill_ 纯 GPU 操作，避免每步 torch.tensor(..., device=cuda) 的 H2D 拷贝
    pos_buf = torch.empty(1, dtype=torch.long, device=device)
    tokens = []
    seq_len = x.shape[1]
    for step in range(max_new_tokens):
        tok = _sample_token(logits, **sparams)
        tokens.append(tok)
        if step == max_new_tokens - 1:
            break
        emb = codec_embedding[step](tok.view(1, 1))
        if not is_identity_proj:
            emb = proj(emb)
        pos_buf.fill_(seq_len)
        out = pmodel(
            inputs_embeds=emb,
            past_key_values=cache,
            use_cache=True,
            cache_position=pos_buf,
        )
        seq_len += 1
        logits = heads[step + 1](out.last_hidden_state[:, -1])[0]
    return torch.cat(tokens).view(1, -1)


def patch_subtalker(core) -> None:
    """把 core.talker.code_predictor.generate 替换为快速版本（实例级，幂等）。"""
    cp = core.talker.code_predictor
    if getattr(cp, "_fast_generate_patched", False):
        return

    def fast_generate(
        self,
        inputs_embeds=None,
        max_new_tokens=0,
        do_sample=True,
        top_p=1.0,
        top_k=50,
        temperature=0.9,
        **kwargs,
    ):
        # 优先走 CUDA graph 引擎（随机采样、参数匹配时）；
        # greedy / 参数不匹配 / 引擎构建失败时走手写裸循环。
        engine = _graph.get_subtalker_engine(self, temperature=temperature, top_k=top_k)
        if (
            engine is not None
            and do_sample
            and max_new_tokens == engine.max_steps
            and (top_p is None or top_p >= 1.0)
            and engine.temperature == temperature
            and engine.top_k == top_k
            and inputs_embeds.shape[0] == 1
        ):
            sequences = engine.run(
                inputs_embeds[:, 0:1], inputs_embeds[:, 1:2]
            )
            return SimpleNamespace(sequences=sequences)

        sequences = _fast_subtalker_generate(
            self,
            inputs_embeds,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
        )
        # talker.forward 只使用 .sequences
        return SimpleNamespace(sequences=sequences)

    cp.generate = types.MethodType(fast_generate, cp)
    cp._fast_generate_patched = True
    return fast_generate


def unpatch_subtalker(core, patched_fn) -> None:
    """恢复官方 code_predictor.generate（bench 基线用；节点运行不需要）。"""
    cp = core.talker.code_predictor
    if getattr(cp, "_fast_generate_patched", False):
        cp.__dict__.pop("generate", None)
        cp._fast_generate_patched = False
    if patched_fn is not None:
        # 重新挂载必须绑定 self（patched_fn 是未绑定的局部函数）
        cp.generate = types.MethodType(patched_fn, cp)
        cp._fast_generate_patched = True


def _build_custom_voice_prefill(core, input_ids, language: str, speaker: str):
    """复刻 Qwen3TTSForConditionalGeneration.generate 中 batch=1、
    custom_voice、non_streaming_mode=True 的 talker 前置 embedding 构建。"""
    talker = core.talker
    cfg = core.config
    tcfg = cfg.talker_config
    device = input_ids.device

    if speaker:
        spk_id = tcfg.spk_id[speaker.lower()]
        speaker_embed = talker.get_input_embeddings()(
            torch.tensor(spk_id, device=device, dtype=input_ids.dtype)
        )
    else:
        speaker_embed = None

    if language.lower() == "auto":
        language_id = None
    else:
        language_id = tcfg.codec_language_id[language.lower()]
    if (
        language.lower() in ["chinese", "auto"]
        and speaker
        and tcfg.spk_is_dialect[speaker.lower()] != False
    ):
        language_id = tcfg.codec_language_id[tcfg.spk_is_dialect[speaker.lower()]]

    tts_bos_embed, tts_eos_embed, tts_pad_embed = talker.text_projection(
        talker.get_text_embeddings()(
            torch.tensor(
                [[cfg.tts_bos_token_id, cfg.tts_eos_token_id, cfg.tts_pad_token_id]],
                device=device,
                dtype=input_ids.dtype,
            )
        )
    ).chunk(3, dim=1)

    codec_prefill_list = [[
        tcfg.codec_think_id,
        tcfg.codec_think_bos_id,
        language_id,
        tcfg.codec_think_eos_id,
    ]]
    codec_embed0 = talker.get_input_embeddings()(
        torch.tensor(codec_prefill_list, device=device, dtype=input_ids.dtype)
    )
    codec_embed1 = talker.get_input_embeddings()(
        torch.tensor(
            [[tcfg.codec_pad_id, tcfg.codec_bos_id]],
            device=device,
            dtype=input_ids.dtype,
        )
    )
    if speaker_embed is None:
        codec_embed = torch.cat([codec_embed0, codec_embed1], dim=1)
    else:
        codec_embed = torch.cat(
            [codec_embed0, speaker_embed.view(1, 1, -1), codec_embed1], dim=1
        )

    role_embed = talker.text_projection(talker.get_text_embeddings()(input_ids[:, :3]))
    _talker_embed = torch.cat(
        (
            tts_pad_embed.expand(-1, codec_embed.shape[1] - 2, -1),
            tts_bos_embed,
        ),
        dim=1,
    ) + codec_embed[:, :-1]
    talker_input = torch.cat((role_embed, _talker_embed), dim=1)

    # 官方 non_streaming 分支先 append "text 首token + codec 末列" 再整体
    # [:, :-1] 移除——净效果就是保持 cat(role, _talker) 不变（长度 = role+codec）。
    # 这里等价实现，切勿再切片，否则会丢掉 codec 流的起始锚点列，
    # 导致 prefill 比 official 短一列（23 vs 24），subtalker 输入随之偏移。
    text_len = input_ids[:, 3:-5].shape[1]
    talker_input = torch.cat(
        [
            talker_input,
            torch.cat(
                (
                    talker.text_projection(
                        talker.get_text_embeddings()(input_ids[:, 3:-5])
                    ),
                    tts_eos_embed,
                ),
                dim=1,
            )
            + talker.get_input_embeddings()(
                torch.tensor(
                    [[tcfg.codec_pad_id] * (text_len + 1)],
                    device=device,
                    dtype=input_ids.dtype,
                )
            ),
            tts_pad_embed
            + talker.get_input_embeddings()(
                torch.tensor(
                    [[tcfg.codec_bos_id]], device=device, dtype=input_ids.dtype
                )
            ),
        ],
        dim=1,
    )
    trailing_text_hidden = tts_pad_embed
    return talker_input, trailing_text_hidden, tts_pad_embed


def fast_generate_codes(
    wrapper,
    input_ids: torch.Tensor,
    language: str,
    speaker: str,
    *,
    do_sample: bool,
    top_k: int,
    top_p: float,
    temperature: float,
    repetition_penalty: float,
    subtalker_do_sample: bool,
    subtalker_top_k: int,
    subtalker_top_p: float,
    subtalker_temperature: float,
    max_new_tokens: int,
    min_new_tokens: int = 2,
) -> torch.Tensor:
    """快速生成 talker codec 序列，返回 (T, num_code_groups)。

    与官方 Qwen3TTSForConditionalGeneration.generate 在 batch=1、
    non_streaming_mode=True、无 instruct、无 voice clone 时输出同构
    （不含 eos 的 codec 帧序列）。
    """
    core = wrapper.model
    talker = core.talker
    tcfg = core.config.talker_config
    device = input_ids.device

    patch_subtalker(core)

    prefill, trailing_text_hidden, tts_pad_embed = _build_custom_voice_prefill(
        core, input_ids, language, speaker
    )

    suppress_idx = torch.tensor(
        [
            i
            for i in range(tcfg.vocab_size - 1024, tcfg.vocab_size)
            if i != tcfg.codec_eos_token_id
        ],
        device=device,
        dtype=torch.long,
    )
    eos_token_id = tcfg.codec_eos_token_id

    tparams = dict(
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        suppress_idx=suppress_idx,
        eos_token_id=eos_token_id,
        min_new_tokens=min_new_tokens,
    )

    cache = DynamicCache()
    prefill_len = prefill.shape[1]
    # attention_mask=None：batch=1 无 padding，position 由 cache_position 决定，
    # 与官方全 1 mask + rope_deltas=0 的路径数值一致，省去每帧 ones 分配
    out = talker.forward(
        inputs_embeds=prefill,
        attention_mask=None,
        past_key_values=cache,
        use_cache=True,
        cache_position=torch.arange(prefill_len, device=device),
        trailing_text_hidden=trailing_text_hidden,
        tts_pad_embed=tts_pad_embed,
        output_hidden_states=False,
    )

    # talker 主循环：每帧 1 token，decode 分支内部会调用被 patch 的
    # code_predictor.generate（快速版）补齐该帧剩余 15 个 codebook，
    # 该帧完整 codec_ids 从 out.hidden_states[1] 取回。
    # 注意必须显式传 subtalker 采样参数，否则 decode 分支拿到 None
    # 会落入 patched generate 的默认值（do_sample=True）。
    sub_kwargs = dict(
        subtalker_dosample=subtalker_do_sample,
        subtalker_top_k=subtalker_top_k,
        subtalker_top_p=subtalker_top_p,
        subtalker_temperature=subtalker_temperature,
    )
    pos_buf = torch.empty(1, dtype=torch.long, device=device)
    frames = []
    gen_toks = []
    logits = out.logits[0, -1]
    tok = _sample_token(logits, prev_tokens=_empty_long(device), **tparams)
    while tok.item() != eos_token_id:
        gen_toks.append(tok)
        if len(gen_toks) >= max_new_tokens:
            break
        n = len(frames) + 1
        pos_buf.fill_(prefill_len + n - 1)
        out = talker.forward(
            input_ids=tok.view(1, 1),
            attention_mask=None,
            past_key_values=cache,
            use_cache=True,
            cache_position=pos_buf,
            past_hidden=out.past_hidden,
            generation_step=out.generation_step,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
            output_hidden_states=False,
            **sub_kwargs,
        )
        frames.append(out.hidden_states[1])
        logits = out.logits[0, -1]
        tok = _sample_token(
            logits, prev_tokens=torch.cat(gen_toks), **tparams
        )

    if not frames:
        return torch.empty(
            0, tcfg.num_code_groups, dtype=torch.long, device=device
        )
    return torch.cat(frames, dim=0)


def fast_generate_custom_voice(
    wrapper,
    text: str,
    language: str = "Chinese",
    speaker: str = "Serena",
    *,
    do_sample: bool = True,
    top_k: int = 10,
    top_p: float = 0.9,
    temperature: float = 0.9,
    repetition_penalty: float = 1.05,
    subtalker_do_sample: bool = True,
    subtalker_top_k: int = 50,
    subtalker_top_p: float = 1.0,
    subtalker_temperature: float = 0.9,
    max_new_tokens: int = 8192,
):
    """对外入口：与 wrapper.generate_custom_voice(...) 同返回值 (wavs, sr)。"""
    input_ids = wrapper._tokenize_texts([wrapper._build_assistant_text(text)])[0]
    codes = fast_generate_codes(
        wrapper,
        input_ids,
        language,
        speaker,
        do_sample=do_sample,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        subtalker_do_sample=subtalker_do_sample,
        subtalker_top_k=subtalker_top_k,
        subtalker_top_p=subtalker_top_p,
        subtalker_temperature=subtalker_temperature,
        max_new_tokens=max_new_tokens,
    )
    wavs, fs = wrapper.model.speech_tokenizer.decode([{"audio_codes": codes}])
    return wavs, fs
