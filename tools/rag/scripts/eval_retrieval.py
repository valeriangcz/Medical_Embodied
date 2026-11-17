#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 0 检索评测：对 eval_set.json 的每题调用本地 RAG 服务，计算 Recall@k。

用法（服务需已启动：tools/rag/rag_service.py）：
  /home/artrc/miniconda3/envs/qwen_tts_online/bin/python \
      tools/rag/scripts/eval_retrieval.py --rag-url http://127.0.0.1:8765 --top-k 5

输出：
  - 控制台表格：题目 / 类型 / 命中 / 命中文档 / 检索耗时
  - tools/rag/scripts/out/retrieval_eval.json
"""

import argparse
import json
import os
import statistics
import urllib.request


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EVAL = os.path.join(HERE, "..", "eval", "eval_set.json")
DEFAULT_OUT = os.path.join(HERE, "out", "retrieval_eval.json")


def load_eval(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def retrieve(rag_url, query, top_k, timeout=5.0):
    url = rag_url.rstrip("/")
    if not url.endswith("/retrieve"):
        url += "/retrieve"
    payload = json.dumps({"query": query, "top_k": top_k}, ensure_ascii=False).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def doc_hit(chunks, expected_docs):
    """hit 判定：返回的 chunk 的 doc_id / source 是否覆盖任一 expected 文档。"""
    expected = set()
    for d in expected_docs:
        expected.add(d)
        expected.add(os.path.basename(d))
        expected.add(os.path.splitext(os.path.basename(d))[0])
    for c in chunks:
        for key in ("doc_id", "source", "id"):
            val = str(c.get(key) or "")
            if val in expected or os.path.basename(val) in expected:
                return True
            for e in expected:
                if e and (e in val):
                    return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rag-url", default="http://127.0.0.1:8765")
    ap.add_argument("--eval", default=DEFAULT_EVAL)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    items = load_eval(args.eval)
    rows = []
    for item in items:
        q = item["question"]
        try:
            data = retrieve(args.rag_url, q, args.top_k)
            chunks = data.get("chunks") or []
            took = (data.get("took_ms") or {}).get("total_ms")
            # 服务端耗时可能缺失，用响应时间兜底
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {q}: {exc}")
            chunks, took = [], None
        hit = doc_hit(chunks, item.get("source_docs") or [])
        top_titles = [c.get("title") or c.get("doc_id") for c in chunks[:3]]
        rows.append(
            {
                "id": item.get("id"),
                "question": q,
                "type": item.get("type"),
                "expected_docs": item.get("source_docs") or [],
                "hit": hit,
                "top_titles": top_titles,
                "took_ms": took,
                "chunks": [
                    {
                        "id": c.get("id"),
                        "doc_id": c.get("doc_id"),
                        "title": c.get("title"),
                        "score": c.get("score"),
                    }
                    for c in chunks
                ],
            }
        )

    n = len(rows)
    hits = sum(1 for r in rows if r["hit"])
    print(f"\n=== 检索评测 Recall@{args.top_k}: {hits}/{n} = {hits / max(n, 1):.1%} ===")
    print(f"{'id':>3} {'type':8s} {'hit':4s} {'took':>8s}  question / top1")
    tooks = []
    for r in rows:
        if r["took_ms"]:
            tooks.append(r["took_ms"])
        t = f"{r['took_ms']:.1f}ms" if r["took_ms"] else "-"
        top1 = (r["top_titles"] or ["-"])[0]
        print(
            f"{str(r['id']):>3} {str(r['type']):8s} {'Y' if r['hit'] else 'N':4s} "
            f"{t:>8s}  {r['question'][:24]:24s} | {top1}"
        )
    if tooks:
        print(
            f"\n检索耗时：mean {statistics.mean(tooks):.1f}ms, "
            f"median {statistics.median(tooks):.1f}ms, max {max(tooks):.1f}ms"
        )
    by_type = {}
    for r in rows:
        by_type.setdefault(r["type"], [0, 0])
        by_type[r["type"]][1] += 1
        if r["hit"]:
            by_type[r["type"]][0] += 1
    print("\n按题型：")
    for k, (h, total) in sorted(by_type.items()):
        print(f"  {k:12s} {h}/{total}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "rag_url": args.rag_url,
                "top_k": args.top_k,
                "recall": hits / max(n, 1),
                "n": n,
                "rows": rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n结果已保存：{args.out}")


if __name__ == "__main__":
    main()
