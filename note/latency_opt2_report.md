# 第二轮时延优化实测报告（闲聊快速通道 + judge 收紧 + 输入瘦身 + LLM 超时兜底）

日期：2026-10-30
基线：`note/latency_50x2_report.md`（同一天 10:15–10:36，kb50 / chat50 各 50 轮）
本轮：11:00–11:36，40 轮（kb20 + chat_fast10 + chat_gen10），另加 2 组隔离实验

## 结论先说

| 优化项 | 性质 | 实测收益 |
| --- | --- | --- |
| 闲聊快速通道（固定句式不调 judge） | 确定性 | 寒暄/笑话类单轮省掉一次 judge：中位 344 ms、均值 404 ms（n=10）；opt2 批次 10 条闲聊里 8 条走了快速通道 |
| judge 超时 2.0 → 1.2 s | 确定性（最坏情况） | 闲聊最坏少等 0.8 s |
| RAG 输入瘦身（top4/350 → top3/300） | 概率性 | 同段交替实测 TTFT 中位 743 → 578 ms（差 ~0.1 s，噪声内）；prompt 约短 25% |
| LLM 超时兜底（10 s、`max_retries=0`） | 健壮性 | 旧逻辑遇到 API 卡顿会整轮卡 60 s；新逻辑 1.25 s 内播兜底话术并结束本段 |
| TTS | 已到边界 | 已是 bf16 + flash-attn + 自定义快速解码；首句合成 ≈ 65 ms/字 + 240 ms，再压只能靠更短首句 |

**总体**：确定性环节合计约省 0.3–0.5 s/轮（闲聊），医学路径基本不变。今天 11:00 之后 qwen-plus 出现多次 15–34 s 的首块卡顿，全链路单轮数字波动很大（同一配置下 E2E 从 1.0 s 到 33 s 都有），所以**读分阶段中位数、不要读单轮均值**。

## 一、改了什么

1. **闲聊快速通道**（`ros2_node_llm_medical.py`）
   - 新增 `_CHITCHAT_RE` / `_looks_like_chitchat()`：问候、再见、道谢、笑话/故事/睡前故事、唱歌、背诗、谜语、绕口令、陪聊、你是谁/你会做什么、夸夸我、无聊等固定句式。
   - 命中即 `chat_mode=True`，**不再启动 judge 线程**；`intent_source=chitchat`。
   - 开关：代码内默认开；不想要可以删正则或把 `_looks_like_chitchat` 直接返回 False。
2. **judge 超时收紧**：`MED_INTENT_TIMEOUT` 默认 2.0 → 1.2 s（join 上限 1.7 s）。
3. **RAG 输入瘦身**：`RAG_TOP_K=3`、`RAG_CHUNK_MAX_CHARS=300`（`start_stack.sh` 默认值）。
4. **LLM 超时兜底**（`_stream_direct`）：`MED_LLM_TIMEOUT` 默认 10 s、OpenAI 客户端 `max_retries=0`；超时/断流时发布 `[START]` + 「抱歉，我这会儿有点忙，请稍后再问一次。」并正常 `[DONE]`，TTS 不会等不到结束信号。
   - 旧行为：异常时发 `[ERROR]`，TTS 忽略 → 等 60 s 超时、整轮卡死。
5. **TTS 未改**：查证 `ros2_node_tts_local.py` 已用 bf16 + `flash_attention_2` + `fast_generate_custom_voice`（绕过 HF generate 双层循环），fp16/compile 无空间。

## 二、分阶段结果

### 2.1 医学路径（kb20，20 轮，11:27–11:33）

| 阶段 | 基线 kb50 (ms) | 优化后 kb20 (ms) |
| --- | --- | --- |
| RAG 检索 | 13 / 11 / 17 | 27 / 11 / 19 |
| LLM 首块 TTFT | 859 / 728 / 1406 | 4788 / 771 / 16591 |
| 首块→首段文本（断句） | 87 / 71 / 117 | 113 / 75 / 220 |
| TTS 首句合成 | 681 / 605 / 814 | 666 / 545 / 860 |
| 合成完成→播放 | 49 / 40 / 83 | 46 / 33 / 73 |
| 问句→首音 | 1685 / 1503 / 2273 | 5637 / 1812 / 17375 |

（mean / median / p90）

医学路径各"确定性"环节（RAG、断句、TTS、播放）都持平或略快；拉高均值的是 4 轮 17–33 s 的 LLM 卡顿（同一时段 chat 组、隔离实验里也出现）。

### 2.2 闲聊路径（chat_fast10 与 chat_gen10，各 10 轮）

| 组 | 走快速通道 | judge 次数 | 问句→首音 (mean/median/p90) |
| --- | --- | --- | --- |
| 基线 chat50（10:26） | 0 | 48 | 2339 / 2202 / 3285 |
| 闲聊快速通道 chat_fast10 | 8/10 | 2 | 2723 / **1634** / 2473 |
| 一般闲聊 chat_gen10（仍需 judge） | 0/10 | 7 | 2564 / **2334** / 3347 |

快速通道组中位比基线少 **568 ms**，与 judge 单次中位 344 ms + 排队/波动量级一致；需要 judge 的一般闲聊仍维持 ~2.3 s。

### 2.3 隔离实验：RAG 输入瘦身对 TTFT 的影响

同一时段交替调用（A=top4/chunk350，B=top3/chunk300，各 20 次，配对 15 组）：

- A：中位 743 ms，B：中位 578 ms；配对中位差 ≈ 0（噪声大）。
- 结论：瘦身主要收益是 prompt 短 ~25%（更省 token、更不易触发长尾），TTFT 改善在 0.1 s 量级、无法与 API 抖动区分。

### 2.4 健壮性验证：LLM 超时兜底

`MED_LLM_TIMEOUT=0.35` 强制触发超时，注入「透析完针眼什么时候能沾水」：

- 事件：`llm_error`（Request timed out）→ `llm_done`(chars=19)；
- TTS 实际播报：「我这会儿有点忙，」「请稍后再问一次。」
- 本段 1.25 s 结束，probe 状态 `ok`（旧逻辑此处会卡 60 s 超时）。

## 三、剩余瓶颈与下一步

1. **云端 LLM 抖动是最大不确定项**：正常 TTFT 0.5–1.5 s，但今天多次出现 15–34 s 卡顿。建议：
   - 首块 2 s 未到时，先播一句预录「我这就帮您查一下，请稍等」，掩盖等待；
   - 或把 `MED_LLM_TIMEOUT` 压到 6–8 s，超时直接兜底重试一次。
2. **闲聊模型对比**：交错测 qwen-plus vs qwen-flash 首块中位 591 vs 763 ms（另一次 954 vs 593）——两个窗口方向相反，说明差异被抖动掩盖；暂不换模型。
3. **TTS 首句**：合成时间 ≈ 65 ms/字 + 240 ms；把首段从 7.7 字压到 5 字理论省 ~0.17 s，但会切在词中间，暂不做；若要可调 `FIRST_FRAGMENT_MAX_CHARS`。
4. **建议的验收口径**：安静窗口、30 轮以上、看 P50/P90；重要改动前后各跑一次同批问句。

## 四、复测命令

```bash
cd /home/artrc/medical_ws/Medical_Embodied_dev
bash tools/latency_probe/start_stack.sh
source /opt/ros/jazzy/setup.bash && source /home/artrc/medical_ws/setup_env.sh
PY=/home/artrc/miniconda3/envs/qwen_tts_online/bin/python

# 医学：20~50 条
$PY tools/latency_probe/latency_probe.py --inject-file tools/latency_probe/questions/kb50.txt \
  --tag kb50 --repeat 0 --interval 2.0 --timeout 45 --warmup 3.0

# 闲聊（快速通道 / 一般闲聊）
$PY tools/latency_probe/latency_probe.py --inject-file /tmp/bench/chat_fast10.txt --tag chatfast --repeat 0 --warmup 3.0

# 超时兜底
MED_LLM_TIMEOUT=0.35 bash tools/latency_probe/start_stack.sh
$PY tools/latency_probe/latency_probe.py --inject-text "透析完针眼什么时候能沾水" --tag timeout_check
```

## 五、环境变量（本轮新增/变更）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `MED_LLM_TIMEOUT` | 10 | 单次 LLM 请求超时（秒），超时播兜底话术并结束本段 |
| `MED_INTENT_TIMEOUT` | 1.2 | judge 超时（秒），超时按闲聊处理 |
| `RAG_TOP_K` | 3 | 进入 prompt 的检索块数 |
| `RAG_CHUNK_MAX_CHARS` | 300 | 每块截断字数 |
| （代码内）`_CHITCHAT_RE` | 开 | 固定闲聊句式跳过 judge；改正则后重启 `llm_medical` 生效 |
