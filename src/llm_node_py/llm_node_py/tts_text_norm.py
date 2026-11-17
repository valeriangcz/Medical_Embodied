#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TTS 文本规范化：把 LLM 输出里 TTS 读不出/读不对的数学符号与单位转成口语。

背景：qwen-plus 偶尔在口语回答里输出公式（如"每天喝水量=前一天尿量+500毫升"），
Qwen3-TTS / CosyVoice2 对 "="、"+"、"%"、范围号等要么跳过要么读成乱码，
导致播报出错。本模块在 TTS 合成前做一次轻量改写。

用法（在 TTS 节点合成前调用）：
    from tts_text_norm import normalize_for_tts
    text = normalize_for_tts(raw_text)

自测：python3 tts_text_norm.py
"""

import re

# ---- 符号 → 口语词（顺序无关，逐个替换） ----
_SYMBOL_WORDS = [
    ("==", "等于"),
    ("=", "等于"),
    ("+", "加"),
    ("×", "乘以"),
    ("x", "乘以"),   # 仅在数字夹缝中出现时替换（见 _NUM_X_NUM）
    ("÷", "除以"),
    ("≥", "大于等于"),
    ("≤", "小于等于"),
    (">=", "大于等于"),
    ("<=", "小于等于"),
    (">", "大于"),
    ("<", "小于"),
    ("℃", "度"),
    ("°C", "度"),
    ("°c", "度"),
]

# ---- 常见单位（仅紧跟数字时替换，避免误伤普通英文） ----
_UNIT_WORDS = [
    (r"(\d(?:\.\d+)?)\s*μmol/L", r"\1微摩尔每升"),
    (r"(\d(?:\.\d+)?)\s*mmol/L", r"\1毫摩尔每升"),
    (r"(\d(?:\.\d+)?)\s*mmHg", r"\1毫米汞柱"),
    (r"(\d(?:\.\d+)?)\s*kg\b", r"\1公斤"),
    (r"(\d(?:\.\d+)?)\s*ml\b", r"\1毫升"),
    (r"(\d(?:\.\d+)?)\s*ml/min\b", r"\1毫升每分"),
    (r"(\d(?:\.\d+)?)\s*cm\b", r"\1厘米"),
]

# ---- 百分号：数字提到"百分之"前面（"5%" → "百分之5"） ----
# "N%-M%" 范围先整体处理，避免中间的 "-" 在百分号替换后失去数字邻居
_PCT_RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[％%]\s*[~～—–-]\s*(\d+(?:\.\d+)?)\s*[％%]")
_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[％%]")

# ---- 数字范围号（3-5 / 3～5 / 3~5 → 3到5）；只匹配数字夹缝，避免误伤负号 ----
_RANGE_RE = re.compile(r"(\d)\s*[~～—–-]\s*(?=\d)")

# ---- "3x5" 形式的字母 x 当乘号 ----
_NUM_X_NUM_RE = re.compile(r"(?<=\d)\s*[xX]\s*(?=\d)")

# ---- markdown 残留符号（TTS 读不出来或读成杂音） ----
_MD_RE = re.compile(r"[#*`_\\|]+")


def normalize_for_tts(text: str) -> str:
    """把公式符号/单位改写成 TTS 可靠朗读的口语；无命中时原样返回。"""
    if not text:
        return text
    out = text

    # 1) 百分号提前（先于通用符号替换，避免 % 被漏读）
    out = _PCT_RANGE_RE.sub(r"百分之\1到百分之\2", out)
    out = _PCT_RE.sub(r"百分之\1", out)

    # 2) 数字范围（3-5 → 3到5）
    out = _RANGE_RE.sub(r"\1到", out)

    # 3) 单位（紧跟数字的 ml/kg/mmol/L 等）
    for pat, rep in _UNIT_WORDS:
        out = re.sub(pat, rep, out, flags=re.IGNORECASE)

    # 4) 通用符号 → 口语词
    for sym, word in _SYMBOL_WORDS:
        if sym == "x":
            continue  # 字母 x 只按数字夹缝规则处理
        out = out.replace(sym, word)
    out = _NUM_X_NUM_RE.sub("乘以", out)

    # 5) markdown 残留
    out = _MD_RE.sub("", out)

    return out


if __name__ == "__main__":
    CASES = [
        # 来自真实链路（qwen-plus 实际输出过的公式）
        ("每天喝水量=前一天尿量+500毫升", "每天喝水量等于前一天尿量加500毫升"),
        ("体重增长别超干体重的5%", "体重增长别超干体重的百分之5"),
        ("控制在3%-5%之间", "控制在百分之3到百分之5之间"),
        ("透析间隔24小时", "透析间隔24小时"),
        ("干体重60kg，超滤2L/2h", "干体重60公斤，超滤2L/2h"),
        ("血钾3.5-5.5mmol/L", "血钾3.5到5.5毫摩尔每升"),
        ("肌酐450μmol/L", "肌酐450微摩尔每升"),
        ("血压<140/90", "血压小于140/90"),
        ("热敷15-30分钟", "热敷15到30分钟"),
        ("3x5分钟", "3乘以5分钟"),
        (" Brenm*强**调** ", " Brenm强调 "),
    ]
    ok = True
    for raw, want in CASES:
        got = normalize_for_tts(raw)
        mark = "PASS" if got == want else "FAIL"
        if got != want:
            ok = False
        print(f"[{mark}] {raw!r} -> {got!r}"
              + ("" if got == want else f"  (期望 {want!r})"))
    print("\n全部通过" if ok else "\n存在失败用例")
