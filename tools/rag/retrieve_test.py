#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检索自测：命令行查询并打印耗时与 top 片段。"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retriever import Retriever, DEFAULT_INDEX, DEFAULT_MODEL  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", "-q", action="append", required=True, help="可重复多次")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--index-dir", default=DEFAULT_INDEX)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    r = Retriever(args.index_dir, args.model)
    print(f"index: {args.index_dir} n_chunks={r.n} dim={r.embeddings.shape[1]}")
    for q in args.query:
        chunks, took = r.retrieve(q, args.top_k)
        print(f"\n=== {q}")
        print(f"    took: {took}")
        for c in chunks:
            print(f"    - [{c['score']:.4f}] {c['title']} | {c['heading_path']} | {c['id']}")
            print(f"      {c['text'][:120].replace(chr(10), ' ')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
