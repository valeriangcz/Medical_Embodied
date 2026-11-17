# P1 优化报告：首句 8 字 + 语义端点 + 输入压缩（2026-10-29）

> 目标：在 Phase 0（本地 RAG + qwen-plus 直连）基础上继续压缩「患者说完 → 机器人首音」。
> 上一轮优化后的基线见 `note/latency_optimization_report.md`（6266 ms → 3189 ms）。

## 一、结论先说

| 指标（说完→首音） | P1 前（同 prompt 8 字对照组） | P1 后 | 变化 |
| --- | ---: | ---: | ---: |
| 中位数 | **2985 ms** | **2578 ms** | **−407 ms（−14%）** |
| 均值 | 3578 ms | 2705 ms | −873 ms |
| 非 LLM 段（端点 + TTS 等） | 1738 ms | 1463 ms | −275 ms |
| 会话总时长（中位） | 14836 ms | 12800 ms | −2036 ms |

三句话：

1. **确定性收益来自「语义端点」**：句末带语气词（吗/么/呢…）时，端点静音从 0.93 s 降到 0.56 s，**每次 −374 ms（中位）**；本轮 5 条问句命中 3 条。
2. **大模型首块仍是最大变量**：TTFT 中位只降 87 ms，但均值在 0.5–6 s 间抖动，端到端均值差主要来自它 —— 不能算作 P1 的收益。
3. **TTS 首句合成基本没变**（595 → 590 ms）：模型本来就先给短句，`FIRST_FRAGMENT_MAX_CHARS=8` 兜底全程未触发，说明 prompt 已经足够。

## 二、改了什么

| 改动 | 开关 / 默认 | 说明 |
| --- | --- | --- |
| 语义端点 | `ASR_SEMANTIC_ENDPOINT=1`（默认开）、`ASR_SEMANTIC_RULE=0.55` | 流式识别文本 ≥12 字节、稳定静音 ≥0.55 s、且句末是 `了 吗 呢 吧 啊 呀 嘛 么 的 哦 嗯` 时提前判停；否则回落到 sherpa 库端点（0.8 s 规则，实测静音 0.93–1.03 s） |
| 首句 ≤8 字 | prompt 第 2.1 条 | 「第一句必须极短：不超过 8 个字…首句不要带逗号」 |
| 输入压缩 | `RAG_CHUNK_MAX_CHARS=350` | 每块资料截断到 350 字（实测未带来可测 TTFT 收益，保留为默认） |
| 首段兜底 | `FIRST_FRAGMENT_MAX_CHARS=8` | 断句器首段超 8 字强制切出；本轮 10 轮均未触发 |
| 上下文压缩（实验项） | `RAG_TOP_K=3`、`MED_LLM_HISTORY_PAIRS=1` | 对照组为 4 / 3；收益未证实，**建议默认保持 4 / 3** |

实现位置：

- 语义端点：`src/llm_node_cpp/src/asr_stream.cpp`（`ends_with_sentence_final()`、`ProcessAudio()` 端点判定），头文件 `src/llm_node_cpp/include/llm_node_cpp/asr_stream.h` 新增 `last_partial_text_ / last_partial_change_ / semantic_endpoint_enabled_ / semantic_silence_s_`。
- 埋点：端点事件带 `detail.reason = semantic|lib` 与 `detail.silence_ms`；探针用它计算「语音结束」时刻（`tools/latency_probe/latency_probe.py`，指标 `asr_endpoint_silence_used_ms`），不再依赖命令行假设的静音阈值。
- 首句与输入压缩：`src/llm_node_py/llm_node_py/ros2_node_llm_medical.py`。

## 三、测试口径

- 5 条问句（同一天、同一批合成音频）：
  `u01` 我下周第一次透析，去之前要准备什么？ / `u05` 透析完针眼什么时候能沾水？ / `u16` 我身上的导管能不能洗澡？ / `u05b` 针眼沾水要紧吗？ / `u16b` 身上的导管能洗澡吗？
- 两组各跑 2 轮，共 20 轮，全部 `status=ok`：
  - **base 组**：`ASR_SEMANTIC_ENDPOINT=0 RAG_TOP_K=4 MED_LLM_HISTORY_PAIRS=3 RAG_CHUNK_MAX_CHARS=100000 FIRST_FRAGMENT_MAX_CHARS=0`
  - **p1 组**：`ASR_SEMANTIC_ENDPOINT=1 ASR_SEMANTIC_RULE=0.55 RAG_TOP_K=3 MED_LLM_HISTORY_PAIRS=1 RAG_CHUNK_MAX_CHARS=350 FIRST_FRAGMENT_MAX_CHARS=8`
  - 两组都已带「首句 ≤8 字」prompt（因此本表**不**包含 prompt 缩短本身的收益；相对于 10-29 上午的 optC 组，prompt 缩短另有约 −0.4 s）。
- 唤醒→等 8 s→播问句，轮间 22 s（ASR 15 s 回 KWS 定时器），`pw-play --target latency_sink` 走虚拟回环。
- 数据：`tools/latency_probe/records/{gbase,gbaseb,p1,p1b}_*_round00.json`。

## 四、组件级结果（10 轮，单位 ms）

| 环节 | base 中位 | p1 中位 | 差 |
| --- | ---: | ---: | ---: |
| 端点静音（说完→判停） | 940 | 566 | **−374** |
| 判停→问题发布（含 nurse 并行） | ≈10 | ≈10 | 0 |
| 路由 + RAG 检索 | ≈25 | ≈25 | 0 |
| LLM 首块（TTFT） | 1316 | 1229 | −87 |
| 首块→首段文本 | 30 | 45 | +15 |
| TTS 首句合成 | 595 | 590 | −5 |
| 合成→开始播放 | 30 | 25 | −5 |
| **说完→首音（中位）** | **2985** | **2578** | **−407** |

逐问句「说完→首音」（两轮，ms）：

| 问句 | base | p1 | 端点 |
| --- | ---: | ---: | --- |
| u01 第一次透析准备 | 2898 / 5686 | 2497 / 2634 | 语义（句末「么」）555 |
| u05 针眼何时沾水 | 2182 / 2416 | 2837 / 3315 | 库判停 951 |
| u16 导管能否洗澡 | 2306 / 7753 | 2178 / 4304 | 库判停 940 |
| u05b 针眼沾水要紧吗 | 2727（新音频） | 2522（新音频） | 语义 561 |
| u16b 导管能洗澡吗 | 3605 / 3073 | 2161 / 2859 | 语义 561 |

u05 反而变慢是 TTFT 抖动（两组同问句 TTFT 分别 609/1361 ms）；u16b 变快除端点 380 ms 外还叠加了 TTFT 抖动。

## 五、为什么只快了 0.4 s

1. **语义端点只覆盖语气词结尾**。本轮 5 条里 3 条以「吗/么」收尾才吃到 −374 ms；「能沾水？」「行不行？」这类结尾仍要等库判停 0.93 s。要把覆盖率做上去，需要 VAD 尾音 + 语义停顿判断，而不是词表。
2. **库端点自身有 0.13 s 余量**：规则设 0.8 s，实测判停静音 0.93–1.03 s（sherpa 的尾随空白帧计数粒度 + 我们的轮询周期）。先把这个 0.13 s 吃掉是几乎零风险的下一个 100 ms。
3. **TTFT 抖动 ±1 s 以上**，任何单轮对比都会被它淹没。后续要么固定区域/模型做 A/B，要么用 30 轮 P50/P90 做验收。

## 六、顺带修掉的两个质量问题（比省时更重要）

1. **ASR 认错「针眼」导致答非所问**：患者说「针眼沾水要紧吗」，流式 ASR 输出「睁眼/整烟/偷洗完整烟」——检索拿不到「针眼」，把导管护理文档顶到第一，答成「导管不能沾水」。
   - 修复：把 `针眼 / 穿刺点 / 内瘘 / 瘘管 / 沾水` 加进 `~/medical_ws/models/sherpa/sherpa-onnx-streaming-zipformer-bilingual-zh-en/hotwords_cn.txt`，实测识别恢复为「针眼沾水要紧吗？」（2/2）。
   - 兜底：`tools/rag/retriever.py` 的 `COLLOQUIAL_MAP` 增加同音写法（睁眼/整眼/真眼/整烟→针眼，展水/沾睡→沾水）。
2. **检索把通路类型搞混**：问内瘘相关问题会被「血液透析导管的日常护理」抢答（反之亦然）。
   - 修复：`retriever.py` 增加血管通路类型路由——query 命中 AVF 词（内瘘/针眼/穿刺点/震颤…）时 AVF 文档 ×1.12、导管文档 ×0.90；命中 AVG 词（人工血管/移植血管）时 18 号文档 ×1.12；命中导管词（导管/置管/中心静脉…）时反向加权。
   - 回归：15 题评测集 Recall@5 仍是 **15/15=100%**；问「针眼沾水」top1 变为 `05_avf_post_dialysis`（24 h 内不沾水），问「人工血管内瘘」top1 为 `18_avg_complications`。

## 七、复测命令

```bash
cd /home/artrc/medical_ws/Medical_Embodied_dev/tools/latency_probe
# P1 版
LATENCY_PROBE=1 RAG_ENABLE=1 MED_LLM_BACKEND=direct NURSE_PARALLEL=1 \
ASR_ENDPOINT_RULE2=0.8 ASR_SEMANTIC_ENDPOINT=1 ASR_SEMANTIC_RULE=0.55 \
SPLITTER_EXTRA_PUNCT=1 FIRST_FRAGMENT_MAX_CHARS=8 \
RAG_TOP_K=3 MED_LLM_HISTORY_PAIRS=1 RAG_CHUNK_MAX_CHARS=350 bash start_stack.sh

# 批量（5 问句；间隔需 ≥20 s 等 ASR 回 KWS）
ENDPOINT_SILENCE=0.8 bash /tmp/run_p1_batch.sh p1 22
```

## 八、下一步建议（按性价比）

| 优先级 | 动作 | 预期 | 风险 |
| --- | --- | ---: | --- |
| P2-1 | TTS 边合成边播（首句分块推流） | −300~450 ms | 中；需改播放器与节点接口 |
| P2-2 | 唤醒后预热：KWS 命中即启动 ASR 解码上下文、提前建连接 | −100~200 ms | 低 |
| P2-3 | 端点余量归零（库判停 0.8 s 对齐到 0.8 s 实测）+ 尾音 VAD 语义判停 | −100~250 ms | 中；需测误切率 <5% |
| P2-4 | LLM 换 `qwen-flash` 或区域就近 | −400~700 ms | 需先跑 15 题质量回归 |
| P3 | 首句模板缓存（高频固定问题） | −700~800 ms | 只覆盖固定问法 |

验收口径建议：30–100 轮、报 P50/P90、端点误切率 <5%、LLM/检索改动必须跑 `tools/rag/scripts/eval_retrieval.py` 与人工要点核对。
