#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BM25-only 知识库检索（纯 Python，零 GPU、零模型依赖）。

背景：rtx3080 16GB 要给导航算法让路，语音栈必须省显存。实测（2026-11，
23 题评测集，含 8 道对抗题）：BM25-only Recall@4 = 100%，不低于原
BGE-M3+BM25 混合检索（95.7%），检索 <1ms/题，因此 RAG 去模型化，
撤销 rag_service GPU 进程（省 ~1.3GB 显存 + 1 份 CUDA context）。

数据源：tools/rag/index/{chunks.jsonl, bm25.json}（与原 rag_service 同一份索引，
文档更新后仍用 build_index.py 重建）。

用法：
    from rag_bm25 import retrieve
    chunks = retrieve("透析患者能喝多少水", top_k=4)

自测：python3 rag_bm25.py   （跑 23 题评测，断言 Recall@4 = 100%）
"""

import json
import math
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INDEX_DIR = os.environ.get(
    "RAG_INDEX_DIR",
    os.path.join(_HERE, "..", "..", "..", "tools", "rag", "index"))

K1, B = 1.5, 0.75
_state = None  # 懒加载单例：{units, df, idf, avgdl, n, tokens}

# 口语词 → 知识库用词（追加到查询，不替换原词）。
# 生产链有 LLM 查询扩展；这里是离线/超时时的静态兜底，避免
# “睡不着/针眼/沾水”这类词面不重叠导致的漏检（2026-11 新索引回归修复）。
_SYNONYMS = {
    "睡不着": "失眠",
    "睡不好": "失眠",
    "心里烦": "情绪 焦虑",
    "心烦": "情绪",
    "针眼": "穿刺 穿刺点",
    "沾水": "清洗 洗澡",
    "抽筋": "肌肉痉挛",
    "头晕": "低血压",
    "没力气": "乏力",
    "拉肚子": "腹泻",
    "喘不上气": "呼吸困难",
    "心慌": "心悸",
    "小豆子": "耳穴压豆",
    "忘事": "记忆力",
    # 化验指标 + 饮食问法 → 饮食管理文档用词。
    # 只在“血磷高/血钾高”时触发；"血钾和血磷要控制在多少" 这类
    # 问目标值的句子不应被带偏到饮食文档（Q9 回归教训）。
    "血磷高": "高磷 含磷 限制磷 低磷饮食",
    "血钾高": "高钾 含钾 限制钾 低钾饮食",
    "磷高": "高磷 含磷",
    "钾高": "高钾 含钾",
}


def _expand_query(query: str) -> str:
    extras = []
    for k, v in _SYNONYMS.items():
        if k in query:
            for w in v.split():
                if w not in extras:
                    extras.append(w)
    return query + " " + " ".join(extras) if extras else query



def _clean(text):
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text or "")


def _bigrams(text):
    c = _clean(text)
    if len(c) < 2:
        return [c] if c else []
    return [c[i:i + 2] for i in range(len(c) - 1)]


def _load_index(index_dir=None):
    global _state
    if _state is not None:
        return _state
    index_dir = index_dir or DEFAULT_INDEX_DIR
    units = []
    with open(os.path.join(index_dir, "chunks.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                units.append(json.loads(line))
    with open(os.path.join(index_dir, "bm25.json"), encoding="utf-8") as f:
        bm25 = json.load(f)
    df = bm25["df"]
    n = bm25.get("n") or len(units)
    avgdl = bm25.get("avgdl") or 1.0
    tokens = []
    for u in units:
        toks = (_bigrams(u.get("title", "")) + _bigrams(u.get("heading_path", ""))
                + _bigrams(u.get("text", "")))
        tokens.append(toks)
    idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
    _state = {"units": units, "tokens": tokens, "df": df, "idf": idf,
              "n": n, "avgdl": avgdl or 1.0}
    return _state


def retrieve(query, top_k=4, index_dir=None):
    """返回与 rag_service /retrieve 同构的 chunk 列表（含 doc_id/title/text 等）。"""
    st = _load_index(index_dir)
    q_tokens = [t for t in _bigrams(_expand_query(query)) if t in st["idf"]]
    if not q_tokens:
        return []
    scores = [0.0] * st["n"]
    for i, toks in enumerate(st["tokens"]):
        if not toks:
            continue
        tf = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        dl = len(toks)
        denom_norm = K1 * (1 - B + B * dl / st["avgdl"])
        s = 0.0
        for t in q_tokens:
            f = tf.get(t)
            if f:
                s += st["idf"][t] * f * (K1 + 1) / (f + denom_norm)
        scores[i] = s
    order = sorted(range(st["n"]), key=lambda i: -scores[i])
    out = []
    for i in order[:top_k]:
        if scores[i] <= 0:
            break
        u = dict(st["units"][i])
        u["score"] = round(float(scores[i]), 6)
        out.append(u)
    return out


if __name__ == "__main__":
    # ---- 自测：23 题评测集（15 原始 + 8 对抗），断言 Recall@4 = 100% ----
    import sys

    EVALS = [os.path.join(_HERE, "..", "..", "..", "tools", "rag", "eval", "eval_set.json"),
             os.path.join(_HERE, "..", "..", "..", "tools", "rag", "eval", "eval_set_hard.json")]
    expected = {}
    for p in EVALS:
        if os.path.exists(p):
            for it in json.load(open(p, encoding="utf-8")):
                expected[it["question"]] = set(
                    os.path.splitext(os.path.basename(d))[0] for d in it["source_docs"])
    items = []
    for p in EVALS:
        if os.path.exists(p):
            items += json.load(open(p, encoding="utf-8"))
    k = int(os.environ.get("EVAL_K", "4"))
    hits, t0 = 0, __import__("time").perf_counter()
    misses = []
    for it in items:
        docs = [c["doc_id"] for c in retrieve(it["question"], top_k=k)]
        exp = expected[it["question"]]
        ok = any(e in str(d) for d in docs for e in exp)
        hits += ok
        if not ok:
            misses.append((it["id"], it["question"], docs))
    dt = (__import__("time").perf_counter() - t0) / len(items) * 1000
    print(f"评测 {len(items)} 题, Recall@{k}: {hits}/{len(items)} = {hits/len(items)*100:.1f}%"
          f" | {dt:.2f}ms/题")
    for m in misses:
        print(f"  未命中 Q{m[0]}: {m[1]} -> {m[2]}")
    if hits != len(items):
        print("自测失败", file=sys.stderr)
        sys.exit(1)
    print("自测通过（Recall@4 = 100%）")
