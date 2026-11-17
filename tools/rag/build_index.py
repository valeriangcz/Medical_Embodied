#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建本地 RAG 索引：BGE-M3 稠密向量 + 字符 bigram BM25 统计。

用法：
  PY=/home/artrc/miniconda3/envs/qwen_tts_online/bin/python
  $PY tools/rag/build_index.py --docs tools/rag/docs --out tools/rag/index \
      --model /home/artrc/medical_ws/models/bge-m3

产物（--out 目录）：
  embeddings.npy  float32 [N,1024]，已 L2 归一化
  chunks.jsonl    每行一个切块 {id, doc_id, title, source, heading_path, text}
  bm25.json       {tokens:[...], df:{...}, avgdl, n}
  meta.json       构建信息与文档哈希
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time

import numpy as np

CHUNK_MIN = 300
CHUNK_MAX = 600
CHUNK_OVERLAP = 90
MAX_TOKENS = 512


def clean_text(text: str) -> str:
    """BM25 用：去掉标点空白，保留中英文数字。"""
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text or "")


def bigrams(text: str):
    c = clean_text(text)
    if len(c) < 2:
        return [c] if c else []
    return [c[i:i + 2] for i in range(len(c) - 1)]


def split_markdown(path: str):
    """按 markdown 标题分节，返回 [(heading_path, paragraph_text), ...]。"""
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    sections = []
    heading_stack = []
    buf = []

    def flush():
        text = "\n".join(buf).strip()
        if text:
            sections.append(("/".join(heading_stack) if heading_stack else "", text))
        buf.clear()

    for line in lines:
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            # 用 level 维护标题栈（h1 作为文档标题时也记录）
            while heading_stack and len(heading_stack) >= level:
                heading_stack.pop()
            heading_stack.append(title)
            continue
        if line.lstrip().startswith(">"):
            # 来源声明不进入索引正文：旧版并入后会形成孤立短块，
            # BM25 长度归一化会让这些短块（路径含"透析/内瘘"等词）
            # 抢占 Top-4（实测 Recall@4 从 100% 掉到 87%）。
            continue
        buf.append(line)
    flush()
    return sections


def chunk_sections(sections, doc_id, title, source):
    """把分节内容合并/切分成 300-600 字的块，块内保留标题路径。"""
    chunks = []
    for heading_path, text in sections:
        # 按空行拆段，再按长度贪心合并
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        cur = ""
        for p in paras:
            candidate = (cur + "\n" + p).strip() if cur else p
            if len(candidate) <= CHUNK_MAX:
                cur = candidate
                continue
            if cur:
                chunks.append((heading_path, cur))
            # 单段超长：按句子滑窗切
            if len(p) > CHUNK_MAX:
                start = 0
                while start < len(p):
                    piece = p[start:start + CHUNK_MAX]
                    chunks.append((heading_path, piece))
                    if start + CHUNK_MAX >= len(p):
                        break
                    start += CHUNK_MAX - CHUNK_OVERLAP
                cur = ""
            else:
                cur = p
        if cur:
            if len(cur) < CHUNK_MIN and chunks and chunks[-1][0] == heading_path:
                # 过短则并入上一块（仍受上限约束）
                merged = chunks[-1][1] + "\n" + cur
                if len(merged) <= CHUNK_MAX + 120:
                    chunks[-1] = (heading_path, merged)
                    continue
            chunks.append((heading_path, cur))

    out = []
    for idx, (heading_path, text) in enumerate(chunks):
        out.append(
            {
                "id": f"{doc_id}#{idx}",
                "doc_id": os.path.splitext(os.path.basename(source))[0],
                "title": title,
                "source": source,
                "heading_path": heading_path,
                "text": text,
            }
        )
    return out


def encode(model, tokenizer, texts, batch_size, device, torch, max_length=MAX_TOKENS):
    """BGE-M3 稠密向量：CLS pooling + L2 归一化，fp16 推理。"""
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.inference_mode():
            outputs = model(**encoded)
            # 优先取 CLS（BGE 系列）——[batch, hidden]
            vec = outputs.last_hidden_state[:, 0, :]
            vec = torch.nn.functional.normalize(vec.float(), p=2, dim=1)
        all_vecs.append(vec.cpu().numpy().astype(np.float32))
    if not all_vecs:
        return np.zeros((0, 1024), dtype=np.float32)
    return np.vstack(all_vecs)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default=os.path.join(here, "docs"))
    ap.add_argument("--out", default=os.path.join(here, "index"))
    ap.add_argument("--model", default="/home/artrc/medical_ws/models/bge-m3")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-length", type=int, default=MAX_TOKENS)
    args = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    docs = sorted(f for f in os.listdir(args.docs) if f.endswith(".md") and f != "MANIFEST.md")
    if not docs:
        print(f"[ERROR] {args.docs} 下没有 md 文档", file=sys.stderr)
        return 1

    print(f"[build] docs={len(docs)}, model={args.model}")
    device = args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.float16 if device != "cpu" else torch.float32
    )
    model.to(device).eval()
    print(f"[build] model loaded in {time.perf_counter()-t0:.1f}s on {device}")

    chunks = []
    hashes = {}
    for filename in docs:
        path = os.path.join(args.docs, filename)
        with open(path, "rb") as f:
            hashes[filename] = hashlib.sha256(f.read()).hexdigest()[:16]
        with open(path, encoding="utf-8") as f:
            first_heading = ""
            for line in f:
                m = re.match(r"^#\s+(.*)$", line)
                if m:
                    first_heading = m.group(1).strip()
                    break
        sections = split_markdown(path)
        doc_chunks = chunk_sections(sections, filename, first_heading or filename, filename)
        chunks.extend(doc_chunks)
    print(f"[build] {len(chunks)} chunks from {len(docs)} docs")

    t1 = time.perf_counter()
    vectors = encode(
        model, tokenizer, [c["text"] for c in chunks],
        args.batch_size, device, torch, args.max_length,
    )
    embed_s = time.perf_counter() - t1
    print(f"[build] embedded {len(chunks)} chunks in {embed_s:.1f}s "
          f"({embed_s/max(len(chunks),1)*1000:.0f} ms/chunk)")

    # BM25 统计（字符 bigram）
    tokens = [bigrams(c["text"]) for c in chunks]
    df = {}
    for toks in tokens:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    avgdl = sum(len(t) for t in tokens) / max(len(tokens), 1)

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "embeddings.npy"), vectors)
    with open(os.path.join(args.out, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "bm25.json"), "w", encoding="utf-8") as f:
        json.dump({"tokens": tokens, "df": df, "avgdl": avgdl, "n": len(chunks)},
                  f, ensure_ascii=False)
    meta = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        "dim": int(vectors.shape[1]),
        "n_chunks": len(chunks),
        "n_docs": len(docs),
        "device": device,
        "embed_seconds": round(embed_s, 2),
        "docs_sha256": hashes,
        "chunk_min": CHUNK_MIN,
        "chunk_max": CHUNK_MAX,
        "chunk_overlap": CHUNK_OVERLAP,
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[build] done -> {args.out} (n_chunks={len(chunks)}, dim={vectors.shape[1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
