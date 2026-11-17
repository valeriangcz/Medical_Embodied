#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""混合检索：BGE-M3 稠密余弦 + 字符 bigram BM25，RRF 融合。"""

import json
import os
import re
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INDEX = os.environ.get("RAG_INDEX_DIR", os.path.join(HERE, "index"))
DEFAULT_MODEL = os.environ.get("RAG_EMBED_MODEL", "/home/artrc/medical_ws/models/bge-m3")

K1 = 1.5
B = 0.75

# 患者口语 → 知识库标准用语的轻量查询扩展（只做追加，不替换原 query）
# 血管通路类型路由：患者问“针眼/内瘘”时不该被“导管护理”抢答（反过来也一样）。
# 只在 query 明确指向某类通路时，对文档做轻量加权/降权，不改变排序机制。
AVF_DOCS = {
    "03_avf_infection", "04_avf_occlusion", "05_avf_post_dialysis",
    "06_new_avf_puncture", "09_avf_surgery", "10_exercise", "17_avf_home_care",
}
CATH_DOCS = {"16_catheter_care"}
AVG_DOCS = {"18_avg_complications"}
AVF_HINTS = ("内瘘", "瘘", "针眼", "针口", "穿刺点", "穿刺部位", "穿刺口",
             "胳膊", "手臂", "震颤", "杂音", "吻合口", "热敷", "止血")
AVG_HINTS = ("人工血管", "人造血管", "移植血管", "graft", "AVG")
CATH_HINTS = ("导管", "置管", "中心静脉", "长期管", "临时管", "隧道", "卡夫", "cuff")

# 中医适宜技术类文档（耳穴、体部艾灸/贴敷等）
TCM_DOCS = {"19_tcm_ear", "20_tcm_body"}
TCM_HINTS = ("中医", "艾灸", "贴敷", "耳穴", "耳豆", "火罐", "拔罐", "刮痧",
             "熏蒸", "督脉灸", "穴位", "经络", "中药")

COLLOQUIAL_MAP = {
    "针眼": "穿刺点 穿刺口 穿刺部位",
    "针口": "穿刺点 穿刺口",
    "沾水": "接触水 清洗 洗澡 淋浴",
    "洗澡": "淋浴 擦浴 沾水",
    "胳膊": "肢体 手臂",
    "瘘": "内瘘",
    "瘘管": "内瘘",
    "瘘堵了": "内瘘闭塞 血流不畅",
    "不跳了": "震颤消失 杂音消失",
    "透析完": "透析后",
    "喝水": "水分摄入 控水",
    "涨水": "体重增长 控水",
    "痒": "皮肤瘙痒 皮肤痒",
    "心情": "情绪 情绪管理 心情低落",
    "情绪": "情绪管理 心理",
    "贴敷": "穴位贴敷 中药贴敷",
    "艾灸": "艾灸 中医适宜技术",
    "化验": "抽血 检验项目 检查",
    "抽血": "检验项目 抽血检查",
    "抽筋": "肌肉痉挛",
    "头晕": "低血压 头晕",
    "没力气": "贫血 乏力",
    # ASR 同音误识别兜底：患者词在流式 ASR 里常被识别成同音字，
    # 这里把常见错字也映射到正确术语，避免检索跑偏。
    "第一次": "首次",
    "准备": "透析前 准备 注意事项 带好物品",
    "去之前": "透析前 准备",
    "吃东西": "进食 透析中进食",
    "饿了": "进食 透析中进食",
    "睁眼": "针眼 穿刺点",
    "整眼": "针眼 穿刺点",
    "真眼": "针眼 穿刺点",
    "整烟": "针眼 穿刺点",
    "真烟": "针眼 穿刺点",
    "展水": "沾水 接触水",
    "沾睡": "沾水 接触水",
    # 患者的其它口语说法：不点名"导管/内瘘"时也能落到正确文档
    "碰水": "沾水 接触水 清洗",
    "湿水": "沾水 接触水",
    "管子": "导管 置管",
    "那条管": "导管 置管",
    "这条管": "导管 置管",
    "那条线": "导管 置管",
    "这条线": "导管 置管",
}


def expand_query(query: str) -> str:
    """把口语词对应的标准术语追加到 query 末尾，帮助稠密与 BM25 同时命中。"""
    extra = []
    for k, v in COLLOQUIAL_MAP.items():
        if k in query and v not in query:
            extra.append(v)
    if not extra:
        return query
    return query + " " + " ".join(dict.fromkeys(extra))
DENSE_TOP = 30
SPARSE_TOP = 30
# 稠密为主、BM25 为辅：字符 bigram BM25 对「透析/患者」这类全库高频词很敏感，
# 低权重 + 过滤高频 query token 后只用于兜住药名/指标名等精确词。
DENSE_WEIGHT = 1.0
SPARSE_WEIGHT = 0.35
STOP_DF_RATIO = 0.30
# 只有 query 里出现「稀有词」（df 很小，如 内瘘/血钾/促红素/KtV）时才启用 BM25，
# 否则纯语义 query（如「每天能喝多少水」）会被通用 bigram 拉偏。
RARE_DF = 8


def clean_text(text: str) -> str:
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text or "")


def bigrams(text: str):
    c = clean_text(text)
    if len(c) < 2:
        return [c] if c else []
    return [c[i:i + 2] for i in range(len(c) - 1)]


class Retriever:
    def __init__(self, index_dir=DEFAULT_INDEX, model_path=DEFAULT_MODEL, device="cuda:0"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.index_dir = index_dir
        self.embeddings = np.load(os.path.join(index_dir, "embeddings.npy"))
        self.chunks = []
        with open(os.path.join(index_dir, "chunks.jsonl"), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.chunks.append(json.loads(line))
        with open(os.path.join(index_dir, "bm25.json"), encoding="utf-8") as f:
            bm25 = json.load(f)
        self.bm_tokens = bm25["tokens"]
        self.df = bm25["df"]
        self.avgdl = bm25["avgdl"] or 1.0
        self.n = bm25["n"] or len(self.chunks)
        with open(os.path.join(index_dir, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)

        self.device = device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True,
            dtype=torch.float16 if self.device != "cpu" else torch.float32,
        ).to(self.device).eval()

        # 预计算 idf
        import math
        self.idf = {
            t: math.log(1 + (self.n - d + 0.5) / (d + 0.5))
            for t, d in self.df.items()
        }

    # ---------- 稠密 ----------
    def _embed_query(self, query: str) -> np.ndarray:
        encoded = self.tokenizer(
            [query], padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with self.torch.inference_mode():
            out = self.model(**encoded)
            vec = out.last_hidden_state[:, 0, :]
            vec = self.torch.nn.functional.normalize(vec.float(), p=2, dim=1)
        return vec.cpu().numpy().astype(np.float32)[0]

    def _dense(self, qvec: np.ndarray, top_k: int):
        scores = self.embeddings @ qvec  # 均归一化 -> 余弦
        idx = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx]

    # ---------- BM25 ----------
    def _bm25(self, query: str, top_k: int):
        # 过滤掉全库高频 bigram（如「透析」「患者」「每天」），避免通用词主导排序
        stop_df = max(int(STOP_DF_RATIO * self.n), 3)
        q_tokens = [
            t for t in bigrams(query)
            if t in self.idf and self.df.get(t, 0) <= stop_df
        ]
        if not q_tokens:
            return []
        scores = np.zeros(self.n, dtype=np.float32)
        for i, toks in enumerate(self.bm_tokens):
            if not toks:
                continue
            dl = len(toks)
            tf = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            s = 0.0
            for t in q_tokens:
                f = tf.get(t)
                if not f:
                    continue
                denom = f + K1 * (1 - B + B * dl / self.avgdl)
                s += self.idf[t] * f * (K1 + 1) / denom
            scores[i] = s
        idx = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx if scores[i] > 0]

    # ---------- 融合 ----------
    def _has_rare_token(self, query: str) -> bool:
        for t in bigrams(query):
            d = self.df.get(t)
            if d is not None and d <= RARE_DF:
                return True
        return False

    def retrieve(self, query: str, top_k: int = 4, rrf_k: int = 60):
        t0 = time.perf_counter()
        expanded = expand_query(query)
        qvec = self._embed_query(expanded)
        t1 = time.perf_counter()
        dense = self._dense(qvec, DENSE_TOP)
        # 仅当 query 含稀有词时才叠加 BM25（避免通用词污染语义排序）
        if self._has_rare_token(expanded):
            sparse = self._bm25(expanded, SPARSE_TOP)
        else:
            sparse = []
        t2 = time.perf_counter()

        fused = {}
        dense_scores = {}
        for rank, (i, score) in enumerate(dense):
            fused[i] = fused.get(i, 0.0) + DENSE_WEIGHT / (rrf_k + rank + 1)
            dense_scores[i] = score
        sparse_scores = {}
        for rank, (i, score) in enumerate(sparse):
            fused[i] = fused.get(i, 0.0) + SPARSE_WEIGHT / (rrf_k + rank + 1)
            sparse_scores[i] = score

        # 中医适宜技术路由：问到艾灸/贴敷/耳穴时，优先中医类文档
        tcm_hit = any(h in expanded for h in TCM_HINTS)
        if tcm_hit:
            for i in list(fused):
                doc = self.chunks[i].get("doc_id", "")
                if doc in TCM_DOCS:
                    fused[i] *= 1.18

        # 血管通路类型路由：按 query 指向的通路类型对候选做轻量加权
        q_low = expanded
        avf_hit = any(h in q_low for h in AVF_HINTS)
        cath_hit = any(h in q_low for h in CATH_HINTS)
        avg_hit = any(h in q_low for h in AVG_HINTS)
        if avg_hit and not cath_hit:
            for i in list(fused):
                doc = self.chunks[i].get("doc_id", "")
                if doc in AVG_DOCS:
                    fused[i] *= 1.12
        elif avf_hit and not cath_hit:
            for i in list(fused):
                doc = self.chunks[i].get("doc_id", "")
                if doc in AVF_DOCS:
                    fused[i] *= 1.12
                elif doc in CATH_DOCS:
                    fused[i] *= 0.90
        elif cath_hit and not avf_hit:
            for i in list(fused):
                doc = self.chunks[i].get("doc_id", "")
                if doc in CATH_DOCS:
                    fused[i] *= 1.12
                elif doc in AVF_DOCS:
                    fused[i] *= 0.90

        ranked = sorted(fused.items(), key=lambda kv: -kv[1])
        results = []
        for i, fused_score in ranked[:top_k]:
            chunk = dict(self.chunks[i])
            chunk["score"] = round(float(fused_score), 6)
            chunk["dense_score"] = round(float(dense_scores.get(i, 0.0)), 6)
            chunk["bm25_score"] = round(float(sparse_scores.get(i, 0.0)), 6)
            results.append(chunk)
        t3 = time.perf_counter()

        took = {
            "embed_ms": round((t1 - t0) * 1000, 2),
            "search_ms": round((t2 - t1) * 1000, 2),
            "fuse_ms": round((t3 - t2) * 1000, 2),
            "total_ms": round((t3 - t0) * 1000, 2),
        }
        return results, took


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--index-dir", default=DEFAULT_INDEX)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()
    r = Retriever(args.index_dir, args.model)
    chunks, took = r.retrieve(args.query, args.top_k)
    print(f"query: {args.query}\n took: {took}")
    for c in chunks:
        print(f"\n[{c['score']:.4f}] {c['title']} / {c['heading_path']} ({c['id']})\n{c['text'][:200]}")
