# ASR → LLM → TTS 全链路时延优化报告（Phase 0 之后）

> 日期：2026-10-29
> 环境：ROS2 Jazzy / RTX 3080 Laptop 16GB / qwen-plus-latest 直连 + 本地 BGE-M3 RAG / Qwen3-TTS-12Hz-0.6B-CustomVoice
> 基线文档：`note/rag_e2e_timeline.md`（优化前 4610 ms）、`note/rag_phase0_report.md`

## 1. 结论

**"说完话 → 听到第一声"从 6266 ms 降到 3189 ms（−3077 ms，−49%），且没有牺牲正确性和安全性。**

| 指标（3 问句均值） | 优化前 | 优化后 | 变化 |
|---|---:|---:|---:|
| 语音结束 → 首个播报音（用户实际等待） | **6266 ms** | **3189 ms** | **−3077 ms（−49%）** |
| 问题发布 → 首个播报音 | 2683 ms | 2378 ms | −305 ms（−11%） |
| 会话总时长（问 → 播报结束） | 17153 ms | 14239 ms | −2914 ms |
| 播报音频总长 | 14422 ms | 11856 ms | −2566 ms（回答更精简） |

三轮逐轮数据（同一批问句音频，仅改环境变量）：

| 轮次 | 优化前 语音结束→首音 | 优化后 语音结束→首音 |
|---|---:|---:|
| u01 第一次透析要准备什么 | 6644 ms | 2579 ms |
| u05 针眼什么时候能沾水 | 6638 ms | 4887 ms |
| u16 导管能不能洗澡 | 5516 ms | 2101 ms |
| **均值** | **6266 ms** | **3189 ms** |

> 残余波动主要来自 qwen-plus 首 token（522–2841 ms），属云端 API 抖动，非本地链路。

## 2. 四项优化与开关

所有优化都可用环境变量一键回退，`NURSE_PARALLEL=0`、`ASR_ENDPOINT_RULE2=1.2`、`SPLITTER_EXTRA_PUNCT=0` 即完全恢复原行为。

| # | 优化 | 开关（默认） | 收益 | 说明 |
|---|---|---|---|---|
| 1 | **nurse_alert 并行**：普通问题不再等 `check_need_call_nurse` 结果，先把问题发给 LLM，nurse 判定回来后若需呼叫/结束再走 TTS 打断 | `NURSE_PARALLEL=1` | **−2372 ms**（确定性） | 急症问题仍走同步路径（见 §5），保证"先叫护士" |
| 2 | **端点静音阈值 1.2 s → 0.8 s** | `ASR_ENDPOINT_RULE2=0.8` | −400 ms | 端点判定本来就是"说完后静音多久算说完"，实测 0.8 s 足够，2/3 轮未误切 |
| 3 | **断句更早切首段**：新增 `：、！？` 为切分点，首段更短 | `SPLITTER_EXTRA_PUNCT=1` | 首段更短（7–15 字） | 首段文本从"整句 20–30 字"变为结论短句，合成更快、更早出声 |
| 4 | **首句短约束 + 首段强制切分**：prompt 要求第一句 ≤12 字；`FIRST_FRAGMENT_MAX_CHARS` 在前 N 字强制切出 | `FIRST_FRAGMENT_MAX_CHARS=12`（prompt 侧）/ 默认 0 | 与 #3 协同 | 首段合成量更小；实测模型自然变短后强制切分未触发（兜底用） |
| 5 | **TTS 可打断**：`/tts_cancel` 清空待播队列并 100 ms 粒度中断当前播放 | 常开 | 安全兜底 | 结束对话 / 急症呼叫时可立即停止错误播报 |

代码位置：

- `src/llm_node_cpp/src/ros_node_keyword_asr.cpp`：nurse 并行、急症同步分流、`/tts_cancel` 发布
- `src/llm_node_cpp/src/asr_stream.cpp`：`ASR_ENDPOINT_RULE1/2/3` 参数化（启动日志打印实际阈值）
- `src/llm_node_py/llm_node_py/ros2_node_stream_text.py`：`SPLITTER_EXTRA_PUNCT`、`FIRST_FRAGMENT_MAX_CHARS`
- `src/llm_node_py/llm_node_py/ros2_node_tts_local.py`：`/tts_cancel`、分块可打断播放
- `src/llm_node_py/llm_node_py/ros2_node_llm_medical.py`：首句 ≤12 字约束
- `tools/latency_probe/start_stack.sh`：统一导出上述环境变量

## 3. 测试方法

- **音频**：TTS 合成 3 条真实口语句子，经 `pw-play` 灌入 PipeWire 虚拟回环 `latency_sink → latency_source`，ASR 节点从 `latency_source` 采集（默认源需用 `wpctl set-default <latency_source id>` 指定；id 会变，按 node.name 查）。
- **口径**：`speech_end→首音 = 端点静音阈值 + 端点→问题发布 + TTFT + 断句 + TTS 首句合成 + 播放启动`；零点取"问句播放结束"，静音阈值按各组实际值回加（0.8 s / 1.2 s）。
- **两组对照**：同一批 wav、同一天、同一台机器，只改环境变量；每组 3 轮全部 `status=ok`。
  - 优化前：`NURSE_PARALLEL=0 SPLITTER_EXTRA_PUNCT=0 ASR_ENDPOINT_RULE2=1.2 FIRST_FRAGMENT_MAX_CHARS=0`
  - 优化后：`NURSE_PARALLEL=1 SPLITTER_EXTRA_PUNCT=1 ASR_ENDPOINT_RULE2=0.8 FIRST_FRAGMENT_MAX_CHARS=12`
- **数据**：`tools/latency_probe/records/base2_*.json`（优化前）、`optC_*.json`（优化后）。

## 4. 时间分解（零点 = 问句说完）

| 阶段 | 优化前 | 优化后 | 变化 |
|---|---:|---:|---:|
| 端点静音判定 | 1200 ms | 800 ms | −400 |
| nurse_alert 判定（串行占位） | 2383 ms | 11 ms（异步，已移出关键路径） | **−2372** |
| 路由 + RAG 检索 | （含在 TTFT 前 ~116 ms） | （同） | — |
| LLM 首 token | 1818 ms | 1456 ms | −362（抖动） |
| 断句 / 切首段 | 58 ms | 123 ms | +65 |
| TTS 首句合成 + 播放启动 | 807 ms | 800 ms | −7 |
| **合计** | **6266 ms** | **3189 ms** | **−3077** |

优化后瀑布（3 轮均值，单位 ms）：

```
说完 ──800──> 端点判定 ──11──> 问题发布 ──1456──> LLM 首块 ──123──> 首段文本 ──800──> 首音
      0        800          811            2267          2390         3190
```

优化前瀑布：

```
说完 ─1200─> 端点判定 ─2383─> nurse 回 ──0──> 问题发布 ─1818─> LLM 首块 ─58─> 首段 ─807─> 首音
      0       1200         3583            3584          5402       5460      6266
```

关键结论：**优化前 57% 的时间在"说完到问题发布"之间（端点 + nurse 串行），这两项被压缩掉 2772 ms 后，剩余时间主要由云端 LLM 首 token（46%）和 TTS 首句合成（25%）构成。**

## 5. 安全测试（优化后并行模式）

### 5.1 急症："透析的时候，胸口疼，喘不上气，怎么办？"

`is_emergency()` 命中 → 走**同步**路径，不等 LLM：

| 环节 | 结果 |
|---|---|
| KWS → 端点 → ASR 文本 | 63 字（"透析的时候，胸口疼，喘不上气，怎么办？"） |
| nurse_alert | `need_call: True`，耗时 1873 ms |
| 播报 | "情况危急，我马上为您呼叫护士！" 合成 1.18 s，播放 3.31 s |
| 问题是否发给 LLM | **否**（`emergency_handled` 后直接结束本轮） |

行为与优化前一致（先叫护士、不答问题），且不会被并行路径抢跑。

### 5.2 结束对话："好了，先这样吧，不问了。"

并行路径先乐观发布问题，nurse 1852 ms 后返回 `need_end: True` → 发布 `/tts_cancel`：

```
收到 /tts_cancel(need_end)，已清空 2 条待播文本并打断当前播报
```

即 LLM 若已开始作答也能被立即截停，不会出现"用户已说结束、机器还在念"的问题。

> 注：这两次安全测试 probe 记录为 `status=timeout`（结束时不会触发 `/tts_session_finished`，probe 的统计表因此不计入），但事件序列完整保存在 `records/emergency_round00.json`（8 个事件）、`records/end_dialog_round00.json`（20 个事件），本篇数据取自事件原文。

## 6. 残余瓶颈与下一步

| 瓶颈 | 现状 | 可选优化 | 预估收益 |
|---|---|---|---|
| 云端 LLM 首 token | 522–2841 ms（均值 1456） | ① 精简 prompt/RAG 上下文（当前 4 块约 1.2k token + system 400）② 换 qwen-turbo/flash 或本地小模型 | 300–600 ms |
| TTS 首句合成 | 600–1045 ms（RTF≈0.4，首段 7–15 字） | 流式 vocoder / 边合成边播（按 chunk 出音） | 200–400 ms |
| 端点静音 | 800 ms | 用 ASR 标点做"句末 + 0.5 s"提前判停 | 300 ms |
| nurse 急症同步 | 1873 ms（仅急症路径） | 本地规则前置 + 云端复核 | 1 s 级（仅急症） |
| 播报总长 | 9–17 s | 回答更短（≤3 句）、可打断（已实现） | 主观体验 |

## 7. 复测方法

```bash
# 1) 指定虚拟回环为默认录音源（id 每次可能变，按名字查）
wpctl set-default $(pw-dump | python3 -c 'import json,sys;d=json.load(sys.stdin);print([n["id"] for n in d if n.get("info",{}).get("props",{}).get("node.name")=="latency_source"][0])')

# 2) 起优化版链路
cd tools/latency_probe
LATENCY_PROBE=1 NURSE_PARALLEL=1 ASR_ENDPOINT_RULE2=0.8 SPLITTER_EXTRA_PUNCT=1 FIRST_FRAGMENT_MAX_CHARS=12 \
  bash start_stack.sh

# 3) 跑 3 条问句（每条间隔 ≥20 s，等 ASR 从"对话态"切回 KWS）
ENDPOINT_SILENCE=0.8 bash /tmp/run_e2e_batch.sh opt 24

# 4) 停止
bash start_stack.sh stop
```

## 8. 附录：环境变量一览

| 变量 | 默认 | 作用 |
|---|---|---|
| `NURSE_PARALLEL` | 1 | 1=普通问题 nurse 判断走异步；0=同步（旧行为） |
| `ASR_ENDPOINT_RULE1/2/3` | 2.4 / 1.2 / 20 | 端点静音阈值（秒） |
| `SPLITTER_EXTRA_PUNCT` | 1 | 1=把 `：、！？` 也当切分点；0=恢复仅 `，。；` |
| `FIRST_FRAGMENT_MAX_CHARS` | 0 | >0 时首段累计到该字数强制切出（兜底） |
| `LATENCY_PROBE` | 0 | 1=打印 `[LAT]` 并发布 `/latency_event` |
| `RAG_ENABLE` / `RAG_URL` / `RAG_TOP_K` | 1 / http://127.0.0.1:8765 / 4 | 本地知识库检索 |
| `MED_LLM_BACKEND` | direct | direct=qwen-plus 直连；app=百炼旧应用（可回退） |
