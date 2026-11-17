#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本地 RAG HTTP 服务（标准库 ThreadingHTTPServer）。

接口契约（ROS 医疗节点按此调用）：
  GET  /health   -> {"ok":true,"n_chunks":N,"model":"...","index":"..."}
  POST /retrieve  请求 {"query":"...","top_k":4}
                  -> {"ok":true,"took_ms":{...},"chunks":[{id,doc_id,title,source,text,score}]}
  GET  /stats    -> 服务统计

环境变量：RAG_PORT(8765) RAG_INDEX_DIR RAG_EMBED_MODEL RAG_DEVICE(cuda:0|cpu)
"""

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from retriever import Retriever, DEFAULT_INDEX, DEFAULT_MODEL  # noqa: E402

PORT = int(os.environ.get("RAG_PORT", "8765"))
STATS = {"requests": 0, "retrieve": 0, "errors": 0, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
RETRIEVER = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send_json(self, code: int, payload) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        STATS["requests"] += 1
        if self.path.startswith("/health"):
            self._send_json(200, {
                "ok": True,
                "n_chunks": RETRIEVER.n,
                "model": RETRIEVER.meta.get("model"),
                "index": RETRIEVER.index_dir,
                "built_at": RETRIEVER.meta.get("built_at"),
                "dim": RETRIEVER.meta.get("dim"),
                "device": RETRIEVER.device,
            })
        elif self.path.startswith("/stats"):
            self._send_json(200, {"ok": True, **STATS})
        else:
            self._send_json(404, {"ok": False, "error": "not found"})

    def do_POST(self):  # noqa: N802
        STATS["requests"] += 1
        if not self.path.startswith("/retrieve"):
            self._send_json(404, {"ok": False, "error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length)
            req = json.loads(raw.decode("utf-8") or "{}")
            query = (req.get("query") or "").strip()
            top_k = int(req.get("top_k") or 4)
            if not query:
                self._send_json(400, {"ok": False, "error": "empty query"})
                return
            top_k = max(1, min(top_k, 20))
            chunks, took = RETRIEVER.retrieve(query, top_k)
            STATS["retrieve"] += 1
            self._send_json(200, {"ok": True, "took_ms": took, "chunks": chunks})
            print(f"[rag] {query[:40]!r} -> {len(chunks)} chunks {took['total_ms']}ms", flush=True)
        except Exception as exc:  # noqa: BLE001
            STATS["errors"] += 1
            print(f"[rag][ERROR] {exc}", flush=True)
            self._send_json(500, {"ok": False, "error": str(exc)})

    def log_message(self, fmt, *args):  # 降低访问日志噪音
        return


def main():
    global RETRIEVER
    index_dir = os.environ.get("RAG_INDEX_DIR", DEFAULT_INDEX)
    model = os.environ.get("RAG_EMBED_MODEL", DEFAULT_MODEL)
    print(f"[rag] loading index from {index_dir}", flush=True)
    t0 = time.perf_counter()
    device = os.environ.get("RAG_DEVICE", "cuda:0")
    RETRIEVER = Retriever(index_dir, model, device=device)
    print(
        f"[rag] ready: n_chunks={RETRIEVER.n}, dim={RETRIEVER.embeddings.shape[1]}, "
        f"device={RETRIEVER.device}, load {time.perf_counter()-t0:.1f}s, port={PORT}",
        flush=True,
    )
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
