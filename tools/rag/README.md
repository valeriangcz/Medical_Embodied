# 本地 RAG 知识库检索服务（Phase 0）

给血透巡诊机器人「小医」的医疗问答提供**本地知识库检索**，替代原先完全依赖云端百炼应用的方案。

```
/llm_to_dashapp ──► ros2_node_llm_medical (RAG)  ──►  rag_service :8765 (BGE-M3 混合检索)
                                 │                        │
                                 ├─ 检索结果注入 prompt ◄─┘
                                 ▼
                       dashscope qwen-plus-latest（OpenAI 兼容，流式）
                                 ▼
                          /ali_llm_output → 断句 → 本地 TTS
```

## 目录

| 路径 | 说明 |
| --- | --- |
| `docs/` | **用户资料库**（由 `source/2.md` 拆分，21 篇 Markdown，v1.0-user-split；需院方审定） |
| `docs_example/` | 早期示例知识库（18 篇，仅供对比演示，**不参与索引**） |
| `source/2.md` | 用户提供的原始资料（16,778 字，未拆分原件） |
| `eval/` | 用户库评测集 15 题（fact 11 / multi 2 / safety 2）；`eval_set_example_docs.json` 为旧示例库评测集 |
| `build_index.py` | 切块 + BGE-M3 向量化 + BM25 统计，产出 `index/` |
| `retriever.py` | 混合检索（稠密余弦 + 字符 bigram BM25，稀有词门控 + 加权 RRF） |
| `rag_service.py` | 标准库 HTTP 服务（`/health` `/retrieve` `/stats`），默认 127.0.0.1:8765 |
| `retrieve_test.py` | 命令行检索自测 |
| `scripts/eval_retrieval.py` | Recall@k 评测（走 HTTP 服务） |
| `scripts/compare_backends.py` | 旧百炼应用 vs 新 RAG+直连 的答案/延迟 A/B |
| `index/` | 索引产物（embeddings.npy / chunks.jsonl / bm25.json / meta.json） |

## 依赖与环境

- Python：`/home/artrc/miniconda3/envs/qwen_tts_online/bin/python`（torch 2.14 + CUDA、transformers 4.57）
- 不需要 faiss / sentence-transformers / jieba：向量检索用 numpy，BM25 用字符 bigram 纯 Python 实现
- 嵌入模型：`/home/artrc/medical_ws/models/bge-m3`（BGE-M3，约 2.2 GB，来自 ModelScope）

## 快速开始

```bash
PY=/home/artrc/miniconda3/envs/qwen_tts_online/bin/python
cd /home/artrc/medical_ws/Medical_Embodied_dev

# 1) 建索引（文档变更后重建）
$PY tools/rag/build_index.py --docs tools/rag/docs --out tools/rag/index \
    --model /home/artrc/medical_ws/models/bge-m3

# 2) 起服务（也可由 tools/latency_probe/start_stack.sh 一并拉起）
$PY tools/rag/rag_service.py                       # 默认 127.0.0.1:8765

# 3) 自测
$PY tools/rag/retrieve_test.py -q "内瘘出血止不住怎么办" -q "血钾高有什么表现" --top-k 3
curl -s http://127.0.0.1:8765/health

# 4) 检索评测 / 答案 A/B
$PY tools/rag/scripts/eval_retrieval.py --rag-url http://127.0.0.1:8765 --top-k 5
$PY tools/rag/scripts/compare_backends.py --limit 15
```

## 接口契约

```http
GET /health
→ {"ok":true,"n_chunks":104,"model":"/home/artrc/medical_ws/models/bge-m3",
   "index":".../tools/rag/index","built_at":"2026-10-28T10:42:08","dim":1024,"device":"cuda:0"}

POST /retrieve
{"query":"透析患者能喝多少水","top_k":4}
→ {"ok":true,
   "took_ms":{"embed_ms":11.2,"search_ms":0.6,"fuse_ms":0.1,"total_ms":12.1},
   "chunks":[{"id":"01_water_control.md#1","doc_id":"01_water_control",
              "title":"饮水与容量控制：两次透析之间怎么喝水",
              "source":"01_water_control.md","heading_path":"...#为什么透析患者要控制饮水",
              "text":"...","score":0.0239,"dense_score":0.7057,"bm25_score":0.0}]}

GET /stats → 服务调用统计
```

## 实测指标（2026-10-28，RTX 3080 Laptop）

| 项 | 数值 |
| --- | --- |
| 索引规模 | 21 篇文档 → 120 个切块（300–600 字/块，重叠 90） |
| 建索引耗时 | 模型加载 1.6s + 向量化 1.2s（10 ms/块） |
| 检索耗时 | 稳态 9–22 ms（首次含模型预热 165–174 ms） |
| Recall@5（用户库） | **15/15 = 100%**（fact 11、multi 2、safety 2 全部在 top-5 命中） |
| Recall@5（旧示例库） | 14/15 = 93.3%（已归档，仅作参考） |

## 设计说明

1. **切块**：按 markdown 标题分节 → 段落贪心合并到 300–600 字；超长段落滑窗切分（重叠 90 字）；块内保留 `heading_path` 便于引用与排错。
2. **稠密检索**：BGE-M3 CLS pooling + L2 归一化，余弦相似度，top-30。
3. **BM25**：字符 bigram（无分词依赖）。中文里「透析/患者」这类词几乎每篇都有，纯 BM25 会把通用词当信号，
   因此做了两道保护：**(a)** 过滤全库 df > 30% 的高频 bigram；**(b)** 只有 query 中出现"稀有词"（df ≤ 8，如 内瘘/血钾/促红素/KtV）时才启用 BM25。
4. **融合**：加权 RRF（稠密权重 1.0、BM25 权重 0.35）——稠密负责语义，BM25 只兜精确词。

## 已知限制

- 知识库为**用户提供的科室资料**（含机构名称、咨询电话、制度班次），未经院方审定，上线前必须核对并替换为最新版本（见 `docs/MANIFEST.md`「待院方确认事项」）。
- 检索评测集仅 15 题，Recall 说明力有限；正式上线前建议扩到 100–300 题。
- 单进程 HTTP 服务，适合单机机器人；多机器人/高并发需要换 Qdrant/Milvus 并做模型副本。
- 服务启动需加载 BGE-M3（约 3–7s），生产上应由 systemd/launch 常驻而不是随对话启动。

## 生产提换建议

1. 用院方最新审定文档替换/更新 `docs/`（保留 MANIFEST 版本登记与去重说明），重建索引。
2. 把 `rag_service.py` 用 systemd 常驻，失败自动重启。
3. 扩评测集到 100+ 题，纳入 CI（每次知识库更新跑 Recall@5 + 答案评审）。
4. 如需更高检索质量，可再加 `bge-reranker-v2-m3` 精排（当前未启用，预计 +20–60ms）。
