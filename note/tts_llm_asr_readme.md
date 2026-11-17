# TTS-LLM-ASR 语音交互链路说明

> 语音链路总览：麦克风采集 → ASR 识别 → 问题路由 → 大模型生成（流式） → 本地 TTS 合成 → 扬声器播放，全程支持对话打断。
> 交付形态：三个 tmux session（`depends_asr_question` / `depends_ui_question` / `dialog_manager`），由 `launch/launch_llm_nodes.sh` 一键拉起。

## 一、链路总览

```
 麦克风(portaudio)
      │ 16kHz 音频流
      ▼
┌──────────────────────────┐  唤醒词命中(KWS) + 语音端点检测(VAD endpoint)
│  ros_node_keyword_asr    │──► /start_asr, /call_signal, /send_question_to_ui
│  (C++, sherpa-onnx)      │
└──────────┬───────────────┘
           │ 识别文本
           ▼
     /question_asr
           │
           ▼
┌──────────────────────────┐  关键词"天气" ─► /llm_to_openai（联网问答）
│ ros2_node_question_router│  其余文本   ─► /llm_to_dashapp（医疗知识库）
│ (错别字纠正: 首卫生→手卫生)│
└──────┬──────────┬────────┘
       ▼          ▼
┌────────────┐ ┌──────────────┐   历史    ┌──────────────────┐
│llm_network │ │ llm_medical  │◄────────► │dialog_history    │
│OpenAI 兼容 │ │百炼应用(小医) │ /send_to_ │(/get_llm_dialog_ │
│qwen-plus   │ │b7ed8dad5c0c… │ history   │history 服务存取   │
└─────┬──────┘ └──────┬───────┘           └──────────────────┘
      │    流式文本     │
      └───────┬────────┘
              ▼
        /ali_llm_output ──► nurse_alert(check_need_call_nurse 判断是否呼叫护士)
              │
              ▼
      /tts_realtime_data (文本流, [DONE] 结束)
              │
              ▼
┌──────────────────────────┐  合成音频 ─► 扬声器(pyaudio)
│  ros2_node_tts_local     │──► /tts_session_finished
│  本地 Qwen3-TTS 0.6B     │
└──────────────────────────┘

打断控制: ros2_node_dialog_manager 订阅 /stop_hci_dialog、/stop_ui_question
         (Bool)，人机对话/预制问题播报过程中随时可打断。
```

## 二、节点清单

### ASR（语音识别）

| 节点 | 语言 | 说明 |
| --- | --- | --- |
| `ros_node_keyword_asr`（llm_node_cpp） | C++ | 生产节点。sherpa-onnx 流式 ASR + 关键词唤醒（KWS），portaudio 采集，speexdsp 重采样。发布 `/question_asr` |
| `ros2_node_asr.py`（llm_node_py） | Python | Paraformer + Silero-VAD 方案的参考实现（未接入 ROS 话题），生产链路未使用 |

- ASR 模型：`/opt/sherpa-onnx/sherpa-onnx-streaming-zipformer-bilingual-zh-en`（流式中英双语 zipformer，模型地址见 `src/llm_node_cpp/src/asr.md`）
- KWS 模型：`/opt/sherpa-onnx/kws-zipformer-wenetspeech`，唤醒词表 `outputMykeywords.txt`（可自定义）
- 端点检测三规则（详见 `asr.md`）：rule1 静音 2.4s / rule2 静音 1.2s / rule3 语音超长 20s。**注意：sherpa 三规则为 OR，且 rule2 对任意含语音句子成立，实测端点延迟为 1.2s 左右（rule2 先触发），不是 2.4s**（见 `latency_report.md`）

### LLM（大模型）

| 节点 | 调用方式 | 输入话题 | 输出话题 |
| --- | --- | --- | --- |
| `ros2_node_question_router.py` | 纯 ROS 分流 | `/question_asr` | `/llm_to_openai`、`/llm_to_dashapp` |
| `ros2_node_llm_medical.py` | 百炼 `Application.call`，app_id `b7ed8dad5c0c483d8e865a3be0b4e088`（"小医"人设，血透巡诊场景） | `/llm_to_dashapp` | `/ali_llm_output`（流式） |
| `ros2_node_llm_network.py` | OpenAI 兼容接口 `qwen-plus-latest`（base_url: dashscope compatible-mode） | `/llm_to_openai` | `/ali_llm_output`（流式） |
| `ros2_node_dialog_history.py` | 历史存储 | `/send_to_dialog_history` | 服务 `/get_llm_dialog_history` |
| `ros2_node_nurse_alert.py` | 呼叫护士判断 | — | 服务 `check_need_call_nurse` |

- API key 统一从 `src/llm_node_py/key/dashscope.key` 读取（`get_dashscope_key()`），单行文本。
- 行为树侧通过 `LLMInteraction` action（mode: alert/passive/interrupt）经 `dialog` 包的 `bt_hci_interface.py` 进入本链路。

### TTS（语音合成）

| 节点 | 方式 | 状态 |
| --- | --- | --- |
| `ros2_node_tts_local.py` | **本地模型** `Qwen3-TTS-12Hz-0.6B-CustomVoice`（约 2.5GB，位于 `src/llm_node_py/` 下），GPU bf16 推理，音色 `Serena` | **启用**（launch 中唯一未注释的 TTS） |
| `ros2_node_tts_oneshot.py` | 云端 `qwen3-tts-instruct-flash-realtime` WebSocket | 已注释停用 |
| `ros2_node_tts_realtime.py` | 云端实时 TTS | 已注释停用 |

- 输入：话题 `/tts_realtime_data`（String 文本流，发 `[DONE]` 表示结束）；服务 `tts_one_shot`（`llm_node_comm/srv/TtsOneshot`）
- 输出：pyaudio 播放；完成后发布 `/tts_session_finished`
- 快速解码：优先 `tts_fast_generate.py` 的快速路径（绕过 HF generate 双层循环），失败自动回退官方 `generate_custom_voice`
- 注意：模型加载优先 flash-attention2，未安装 flash-attn 的机器自动回退 sdpa（RTX 3080 实测可用，预热后 RTF≈1.0）

## 三、启动与测试

```bash
# 0) 环境（需 conda 环境承载全部 Python 依赖）
conda activate qwen_tts_online

# 1) 一键启动（tmux 三个 session: depends_asr_question / depends_ui_question / dialog_manager）
cd ~/medical_ws/Medical_Embodied_dev/launch && ./launch_llm_nodes.sh
tmux attach -t depends_asr_question   # 观察各节点输出

# 2) 测试 LLM 路由 + 生成（无需麦克风）
ros2 topic pub --once /question_asr std_msgs/msg/String "{data: '天气今天如何？'}"
ros2 topic echo /ali_llm_output        # 观察流式输出

# 3) 测试本地 TTS
ros2 service call /tts_one_shot llm_node_comm/srv/TtsOneshot "{tts_text: '你好，我在呢。', block: true}"
# 或流式接口（对话链路实际走的方式）:
ros2 topic pub --once /tts_realtime_data std_msgs/msg/String "{data: '你好，我在呢。'}"
ros2 topic pub --once /tts_realtime_data std_msgs/msg/String "{data: '[DONE]'}"
ros2 topic echo /tts_session_finished
```

## 四、模型与凭据清单

| 项 | 位置 | 说明 |
| --- | --- | --- |
| Qwen3-TTS 本地模型 | `src/llm_node_py/Qwen3-TTS-12Hz-0.6B-CustomVoice` | ModelScope 下载，gitignore 不入库 |
| ASR/KWS ONNX 模型 | `/opt/sherpa-onnx/…`（代码内路径） | 不在仓库内，需按 `asr.md` 链接自行下载 |
| dashscope API key | `src/llm_node_py/key/dashscope.key` | 单行文本，勿提交/外传 |
| 百炼应用 | app_id `b7ed8dad5c0c483d8e865a3be0b4e088` | 需 key 所属工作空间有该应用权限 |
| C++ sherpa-onnx 库 | `/opt/sherpa-onnx/aarch64-install-static-lib`（llm_node_cpp 编译期硬编码） | 本机已部署 x86_64 动态库 |

## 五、本机部署状态（2026-10-27）

| 环节 | 状态 |
| --- | --- |
| LLM（医疗应用/联网问答/历史/护士告警） | ✅ key 已配置，Application 与 OpenAI 兼容模式均实测 200 |
| TTS（本地 Qwen3-TTS） | ✅ 模型已下载，GPU 合成实测通过（`/tmp/tts_local_test.wav`） |
| ASR（C++ 关键词唤醒 + 流式识别） | ✅ 模型已下载到 `~/medical_ws/models/sherpa/`（`/opt` 无写权限）：ASR/KWS/标点齐全；C++ 节点已支持 `SHERPA_MODEL_ROOT` 等环境变量覆盖（不设时仍用 `/opt/sherpa-onnx` 默认路径），`LATENCY_PROBE=1` 可开链路埋点。离线验证与回环见 `~/medical_ws/models/model_validation.md`、`models/audio_setup.md` |


## 六、全链路时延实测（2026-10-27 补充）

实测脚本、原始数据与结论见：

- 报告：`note/latency_report.md`（链路图、分段时延、瓶颈与优化建议）
- 工具：`tools/latency_probe/`（`latency_probe.py` 探针、`report_latency.py`、`start_stack.sh` 精简启动、`audio/` 测试音频、`records/` 原始数据）
- 模型：`~/medical_ws/models/sherpa/`（ASR/KWS/标点）

关键数字（RTX 3080 Laptop，本机回环注入）：说话结束→首音，联网问答 ≈5.3s、医疗知识库 ≈12.4s；首音后 TTS（RTF≈0.40）不拖尾。最大瓶颈为百炼应用首块 5.6–10.9s 与 `nurse_alert` 串行 1.6s。


## 七、Phase 0：本地 RAG 知识库（2026-10-28）

医疗问答链路已从「云端百炼应用（检索+生成在云端）」改为「**本地 RAG 检索 + qwen-plus 直连**」：

- 报告：`note/rag_phase0_report.md`
- 知识库与检索服务：`tools/rag/`（18 篇示例宣教文档、15 题评测集、BGE-M3 混合检索服务 :8765、`README.md`）
- 节点开关：`MED_LLM_BACKEND=direct|app`、`RAG_ENABLE=1|0`、`RAG_URL`、`RAG_TOP_K`、`MED_LLM_MODEL`；旧百炼路径保留可回退

关键实测（同日对比）：

| 指标 | 旧（百炼应用） | 新（RAG+直连） |
| --- | ---: | ---: |
| 医疗路径 说话结束→首音 | 12.38 s | **5.03 s（−59%）** |
| 医疗路径 问题→首音（注入） | 8.08 s | **2.65 s（−67%）** |
| LLM 首块 | 6.85 s | **1.12 s** |
| 检索耗时 / Recall@5 | — | 14 ms / 93.3% |
| 15 题中"数据库中没有知识" | 10 题 | **0 题** |

启动方式（含 RAG 服务，自动等待 `/health`）：

```bash
LATENCY_PROBE=1 RAG_ENABLE=1 MED_LLM_BACKEND=direct bash tools/latency_probe/start_stack.sh
```

## 八、时延优化（2026-10-29）

在 Phase 0 基础上继续压缩「说完 → 首音」：**6266 ms → 3189 ms（−49%）**，详细对比、时间分解与安全测试见 `note/latency_optimization_report.md`。

四项优化（均可环境变量回退）：

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `NURSE_PARALLEL` | 1 | 普通问题不等 `check_need_call_nurse`，先发 LLM；nurse 判定需呼叫/结束时用 `/tts_cancel` 打断。急症关键词仍走同步路径先叫护士 |
| `ASR_ENDPOINT_RULE2` | 1.2 | 说话后静音多久判定说完（优化版用 0.8 秒） |
| `SPLITTER_EXTRA_PUNCT` | 1 | 断句把 `：、！？` 也当切分点，首段更短、更早出声；`0` 恢复仅 `，。；` |
| `FIRST_FRAGMENT_MAX_CHARS` | 0 | >0 时首段累计到该字数强制切出（配合 prompt 首句 ≤12 字，作兜底） |

启动（优化版）：

```bash
cd tools/latency_probe
LATENCY_PROBE=1 NURSE_PARALLEL=1 ASR_ENDPOINT_RULE2=0.8 SPLITTER_EXTRA_PUNCT=1 bash start_stack.sh
```

回退到优化前行为：`NURSE_PARALLEL=0 ASR_ENDPOINT_RULE2=1.2 SPLITTER_EXTRA_PUNCT=0 FIRST_FRAGMENT_MAX_CHARS=0`。

安全行为（已实测）：

- 急症（胸痛/喘不上气等）→ 同步 nurse 判定（约 1.9 s）→ 直接播「情况危急，我马上为您呼叫护士！」，**不把问题发给 LLM**。
- 用户说「不问了」等结束语 → 并行路径下 nurse 判定 `need_end` 后发 `/tts_cancel`，清空待播文本并中断当前播报。

## 九、P1：语义端点 + 首句 8 字 + 输入压缩（2026-10-29 下午）

详细数据见 `note/latency_p1_report.md`。要点：

- 「说完→首音」中位 **2985 → 2578 ms（−407 ms，−14%）**，确定性收益几乎全部来自语义端点。
- 新增变量：

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `ASR_SEMANTIC_ENDPOINT` | 1 | 语义端点总开关；`0` 只用 sherpa 库端点 |
| `ASR_SEMANTIC_RULE` | 0.55 | 句末是语气词（了吗呢吧啊呀嘛么的哦嗯）时，静音超过该值即判停（秒） |
| `RAG_CHUNK_MAX_CHARS` | 350 | 每块检索资料进入 prompt 前截断的字数（加 `…`） |

- 端点事件现在带 `detail.reason`（`semantic`/`lib`）与 `detail.silence_ms`，探针据此算「语音结束」（指标 `asr_endpoint_silence_used_ms`），不再依赖命令行假设。
- 已知局限：语义端点只覆盖语气词结尾（本轮 5 条命中 3 条）；非语气词结尾仍等库判停（实测 0.93–1.03 s）。
- 质量问题（顺带修复）：
  - 流式 ASR 会把「针眼」识别成「睁眼/整烟」→ 已把 `针眼 穿刺点 内瘘 瘘管 沾水` 加入 `~/medical_ws/models/sherpa/sherpa-onnx-streaming-zipformer-bilingual-zh-en/hotwords_cn.txt`；`tools/rag/retriever.py` 的 `COLLOQUIAL_MAP` 也加了同音兜底。
  - 检索新增「血管通路类型路由」（AVF / AVG / 导管三类文档加权），避免问内瘘被导管护理文档抢答；15 题 Recall@5 仍为 15/15。

## 十、意图分流：非医学问题正常回答（2026-10-30）

目标：医学/护理问题仍然只依据 RAG 资料（资料不足才说"没有"），**其他问题像陪护一样正常回答**，不再对闲聊、家常、心情一律拒答。

实现（`src/llm_node_py/llm_node_py/ros2_node_llm_medical.py`）：

- 新增关键词分类器 `_is_medical(text)`（`_MEDICAL_TERMS` 列表，覆盖透析/通路/化验/用药/症状/饮食/运动/中医/情志等约 100 个词）。
- `handle_query()` 先分流：
  - `intent=medical` → 照旧跑 RAG → `MED_SYSTEM_PROMPT`（只依据资料、拒答规则、首句 ≤8 字）；
  - `intent=chat` → **跳过 RAG 检索**（省 10–20 ms 与无关资料干扰）→ `MED_CHAT_PROMPT`（正常、简短、1–2 句 ≤40 字，不提资料），并把问题原文直接作为 user 消息。
- 埋点新增 `intent`（`detail.kind=medical/chat`、`chat_mode`），日志同时打印 `intent=... chat_mode=...`。
- 拒答话术改为「这个问题我资料里没有。建议问护士或医生。」——不再重复两遍"资料里没有"，TTS 只切 2 段。

### 二次复核：轻量 LLM 意图判断（judge）

关键词判为「闲聊」时再让快模型复核一次，防止「我身上这条线能不能碰水」这类不用医学术语的问法被漏判：

- 判词模型 `qwen-flash`（单次实测 0.23–0.33 s；链路并发下 0.4–1.3 s），提示词「判断以下问题是否与医学、健康、护理、透析、用药有关。只回答一个字：医 或 聊。」，`temperature=0`、`max_tokens=2`。
- 只在「关键词判闲聊」时触发（医学关键词命中则 0 开销）；judge 线程与 `_get_history()` 并行，取完历史后 `join(timeout=MED_INTENT_TIMEOUT+0.5)`。
- judge 判「医」→ 按医学处理（跑 RAG + 医学提示词）；判失败/超时 → 回退关键词结论。
- 埋点 `intent_judge(model, ms, answer, kind)` / `intent_judge_error(...)`；`intent` 事件带 `source`（keyword / llm:qwen-flash / keyword-fallback）、`judge_kind`、`judge_ms`。
- 检索侧配套：`COLLOQUIAL_MAP` 增加「那条线/这条线/管子/碰水」等口语别名，让 judge 判为医学后能命中 `16_catheter_care`。

judge 时延实测（n=6，`intent_judge` 事件 `detail.ms`）：mean 789 ms、median 690 ms、min 414、max 1326 ms。

### 输出清理：禁止表情符号

闲聊模型仍会偶尔输出 "😄"，TTS 只能念出「正在播放：😄」。现在两处兜底：

- `MED_CHAT_PROMPT` 明确禁止表情符号/颜文字；
- `_stream_direct` 对每个流式 delta 应用 `_EMOJI_RE`（emoji、dingbats、箭头、变体选择符、ZWJ）过滤，空 delta 直接丢弃——医学/闲聊两条路径都生效。
- 同时用 `_QUOTE_RE` 删掉中文引号 `“”「」『』`：否则引号会被断句成单独的「“”」片段，TTS 白播一段。

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `MED_CHAT_ENABLE` | 1 | `1`=非医学问题正常闲聊；`0`=恢复到旧行为（所有问题都只依据资料） |
| `MED_INTENT_JUDGE` | 1 | 关键词判闲聊后再让快模型复核意图 |
| `MED_INTENT_MODEL` | qwen-flash | 判词模型（qwen-flash 实测最稳；qwen3.5-flash 要 3–18 s） |
| `MED_INTENT_TIMEOUT` | 2.0 | judge 超时（秒），超时回退关键词结论 |

实测（2026-10-30，`--inject-text` 注入）：

| 问句 | 分流 | 回答 | 端到端 |
| --- | --- | --- | ---: |
| 给我讲个笑话吧 | chat | 「为什么透析机从不生气？因为它心里有'滤'量，啥都过滤掉啦～」 | 2.26 s |
| 你是谁呀 | chat | 「我是小医，血透中心的巡诊机器人，陪您聊聊天、说说话～」 | 1.18 s |
| 我家猫最近不爱吃饭怎么办 | chat | 「猫挑食可能跟心情、环境或身体有关，建议带它看看兽医哦～」 | 2.08 s |
| 陪我聊聊天吧 | chat | 「好呀～今天窗外阳光暖暖的，您想聊点啥？」 | 1.97 s |
| 透析完针眼什么时候能沾水 | medical | 「24小时后能沾水。透析后当天不能洗澡…」 | 1.72 s |
| 我身上的导管能不能洗澡 | medical | 「绝对禁止洗澡。淋浴、盆浴、游泳都不行…」 | 3.05 s |
| 透析患者能喝咖啡吗 | medical | 「资料里没有。建议问护士或医生。」 | 1.19 s |
| 吃降压药能喝酒吗 | medical | 同上（不再用通用医学知识越界作答） | 2.20 s |
| 我身上这条线能不能碰水 | medical（judge 413.6 ms） | 「绝对不能碰水。临时导管禁止淋浴、盆浴或游泳…」 | 1.59 s |
| 我想问问那条管洗澡行不行 | medical（关键词命中「洗澡」） | 「绝对不行。淋浴、盆浴、游泳都会让导管进水，容易感染…」 | 1.74 s |
| 给我讲个笑话吧（复测） | chat（judge 575.8 ms） | 「今天血压稳吗？稳得很——像我刚交的医保，一分没少！」（无 emoji） | 2.63 s |

边界与已知局限：

- 两级分类：关键词先判（医学词命中即医学，0 时延），只对「关键词判闲聊」的问题加 judge 复核；judge 回退时仍按闲聊处理，`MED_CHAT_PROMPT` 第 4 条兜底为「这个我不太确定，建议问护士或医生」，不会给出医学建议。
- judge 有开销：闲聊问题端到端 1.2–3.5 s（judge 0.4–1.3 s + 生成）；医学关键词路径不增加时延。
- 医学相关但资料没有的问题（咖啡、饮酒、怀孕…）仍按原有规则拒答；这类"边缘医学问题"如需放宽，应在 `MED_SYSTEM_PROMPT` 增加"资料没有时可给一般性提醒"的规则，而不是并入闲聊。
- 扩充医学术语：直接改 `_MEDICAL_TERMS` 列表后重启 `llm_medical` 节点即可。
- 路由层：`/llm_to_openai`（天气等网络问题）仍走 `llm_network` 节点，不经过本分流。
- 50 组知识库对话 + 50 组闲聊对话的分阶段实测：见 [latency_50x2_report.md](latency_50x2_report.md)（知识库组问句→首音 1685 ms，闲聊组 2339 ms）。

## 十一、第二轮优化：闲聊快速通道 + LLM 超时兜底（2026-10-30）

按 [latency_50x2_report.md](latency_50x2_report.md) 的建议做了三处改动（第四项 TTS 查证后判定已无可换空间）：

| 优化 | 开关 | 效果 |
| --- | --- | --- |
| 闲聊快速通道：寒暄/笑话/故事/唱歌/陪聊等固定句式直接走闲聊，不再调 judge | `MED_CHITCHAT_FAST=1`（默认） | 10 条寒暄中 8 条省掉 judge（单次中位 344 ms、均值 404 ms）；顺带修正「你会做什么」曾被 judge 误判为医学 |
| judge 超时 2.0 s → 1.2 s | `MED_INTENT_TIMEOUT=1.2` | 闲聊最坏情况少等 0.8 s |
| RAG 输入瘦身：top_k 4→3、每块 350→300 字 | `RAG_TOP_K=3`、`RAG_CHUNK_MAX_CHARS=300` | 同段交替实测：TTFT 中位 743→578 ms（n=15 配对，噪声内，取 0.1 s 量级）；prompt 短约 25% |
| **LLM 超时兜底**：`MED_LLM_TIMEOUT=10`、`max_retries=0`，超时/断流时播「抱歉，我这会儿有点忙，请稍后再问一次。」并 `[DONE]` | `MED_LLM_TIMEOUT` | 用 0.35 s 强制超时验证：本段 1.25 s 正常结束（旧逻辑等 60 s 超时，整轮卡死） |

注意：本机 11:00 之后 qwen-plus 出现多次 15–34 s 的首块卡顿（同一时段、同一条链路，与配置无关），全链路单轮时延因此波动很大；建议以「分阶段中位数」和「配对 A/B」读数据，不要用单轮均值。

### 复测方法

```bash
# 分阶段时延（每段 mean/median/p90）
bash tools/latency_probe/start_stack.sh
source /opt/ros/jazzy/setup.bash && source /home/artrc/medical_ws/setup_env.sh
$PY tools/latency_probe/latency_probe.py --inject-file tools/latency_probe/questions/kb50.txt \
  --tag kb50 --repeat 0 --interval 2.0 --timeout 45 --warmup 3.0

# 验证超时兜底（故意把超时压到 0.35 s）
MED_LLM_TIMEOUT=0.35 bash tools/latency_probe/start_stack.sh
$PY tools/latency_probe/latency_probe.py --inject-text "透析完针眼什么时候能沾水" --tag timeout_check
```

报告：[latency_opt2_report.md](latency_opt2_report.md)

## 十二、LLM 接入方式切换：dashscope SDK + MaaS 专有部署（2026-10-30）

医疗节点的云端生成从「`openai` SDK 直连兼容接口 + qwen-plus」换成「`dashscope` SDK 原生接口 + 专有 MaaS 部署的 `deepseek-v4.1-flash-1017`」。

| 项目 | 值 |
| --- | --- |
| SDK | `dashscope`（`MultiModalConversation.call`，`stream=True, incremental_output=True`） |
| 地址 | `dashscope.base_http_api_url = MED_LLM_BASE_URL`，默认 `https://llm-wd6fnilu0sk3blx6.cn-beijing.maas.aliyuncs.com/api/v1` |
| 模型 | `deepseek-v4.1-flash-1017`（思考模型，**必须 `enable_thinking=False`**，否则首句要等推理结束） |
| 首块 | 中位 681 ms（kb10）；生成段 328 ms，比 qwen-plus 快约一倍 |
| 回退 | `MED_LLM_PROVIDER=openai MED_LLM_MODEL=qwen-plus-latest` 即回原路径 |

要点：

- 地址只认 `.../api/v1` 下的**原生多模态路径**或 `compatible-mode/v1`；`/services/aigc/text-generation/generation` 会 400 `url error`。
- 主机名第七位是数字 `0`（`llm-wd6fnilu0sk3blx6`）；写成字母 `o` 会 403。
- 流式消费放在 worker 线程 + `queue.Queue`，首块最多等 `MED_LLM_TIMEOUT`（10 s）、之后每块 5 s，超时播「抱歉，我这会儿有点忙，请稍后再问一次。」并正常 `[DONE]`。
- judge 用 `MED_INTENT_BASE_URL`（默认公网兼容接口）与主模型地址解耦，`qwen-flash` 不受影响。
- 新模型更严格，首轮把「资料没逐字回答」当成没资料（拒答 4/10）；放宽 `MED_SYSTEM_PROMPT` 规则 1（有相关内容就用自己的话讲）+ 检索补「准备/首次/吃东西」口语别名后拒答 0/10。「干体重」资料里没有定义，仍会拒答（知识库缺口）。

新增环境变量：`MED_LLM_PROVIDER`（默认 `dashscope`）、`MED_LLM_ENABLE_THINKING`（默认 `0`）、`MED_INTENT_BASE_URL`；`MED_LLM_BASE_URL` / `MED_LLM_MODEL` 默认值随 provider 变化。

报告（地址实测、思考开关、分阶段数据、拒答修复、回退命令）：[llm_provider_switch_report.md](llm_provider_switch_report.md)

## 十三、接入本机 ollama 模型（2026-10-30）

医疗节点支持第三路接入：`MED_LLM_PROVIDER=ollama`，走本机 ollama 的 OpenAI 兼容接口（`http://127.0.0.1:11434/v1`，模型 `qwen3:8b`）。

```bash
# 用本机模型启动整条链路
USE_OLLAMA=1 bash tools/latency_probe/start_stack.sh
# 回云端（默认）
bash tools/latency_probe/start_stack.sh
```

要点：

- qwen3 默认**先思考**，且 OpenAI 兼容接口不认 `think:false`。节点在 `MED_LLM_NO_THINK=1`（默认）时给系统提示追加 `/no_think`，首块从"无正文"恢复到 ~107 ms。
- `MED_LLM_MAX_TOKENS` 对本地模型影响更大：本地生成约 1 s/80 字，答越长越吃亏。
- judge 仍走云端 `qwen-flash`（`MED_INTENT_BASE_URL`），与主模型解耦；本地部署时可考虑也换成本地小请求。
- TTFT 埋点已改到发请求之前（ollama 会先备好首 token 再回响应头，否则 TTFT 会算成 0 ms）。

实测（各 10 轮，中位）：

| 场景 | 云端 | 本机 ollama |
| --- | --- | --- |
| 医学问题 问句→首音 | 1746 ms | 1990 ms（首块更快，但生成慢、首句更长） |
| 闲聊 问句→首音 | 1648 ms | **964 ms** |

注意：qwen3:8b 对"只依据资料"的约束比云端模型松（会用自身知识补答案），也不严格遵守"首句≤8 字"，医学链路的质量需要额外把控。详细数据与建议见 [ollama_local_model_report.md](ollama_local_model_report.md)。

## 十四、本地 LLM 优化（2026-10-30 下午）

针对 ollama qwen3:8b 的四个改动（默认生效，无需额外开关）：

| 项 | 做法 | 效果（医学 kb10，中位） |
| --- | --- | --- |
| 首句短化 | 系统提示加 3 组 few-shot 示例（首句都是 ≤8 字短句） | 首段 14 字 → **8 字**；TTS 首句合成 1342 → **1013 ms** |
| 答案瘦身 | `MED_LLM_MAX_TOKENS` 本地默认 220（云端仍 400） | 回答 46 → 36 字；音频 −480 ms，会话 −1.2 s |
| 预热 | `start_stack.sh` 在 ollama 模式下先发一个 `keep_alive=30m` 的预热请求 | 消除首轮 3~5 s 冷启动 |
| 本地 judge | `MED_INTENT_PROVIDER=ollama`：judge 走 `/api/chat`（`think=false` + `/no_think`） | judge 695 → **77 ms**；闲聊复核路径端到端 2041 → **1300 ms** |

实测（各 10 轮）：

| 指标（中位） | 本地未优化 | 本地优化后 | 云端 deepseek MaaS |
| --- | --- | --- | --- |
| 医学 问句→首音 | 1990 ms | **1666 ms** | 1746 ms |
| 医学 LLM 生成 | 1045 ms | 928 ms | 328 ms |
| 闲聊（快速通道） | 964 ms | 1053 ms | 1648 ms |
| 闲聊（走 judge） | 2041 ms | **1300 ms** | ~2100 ms |

### 第二轮：服务端与模型组合（2026-10-30 傍晚）

在代码优化之后又动了 ollama 服务端与模型组合：

| 项 | 做法 | 效果 |
| --- | --- | --- |
| flash attention + KV 量化 | 重启 `ollama serve` 带 `OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0` | 8b 生成 28 → **55 tok/s**；KV 576 → **153 MiB**/模型 |
| 上下文 4096 → 2048 | `OLLAMA_CONTEXT_LENGTH=2048` | 8b 显存 5.02 → **4.60 GiB**；医学链路中位基本不变 |
| 4b 闲聊模型 | `MED_CHAT_MODEL=qwen3:4b`（4b 从 hf-mirror 的 GGUF + `ollama create` 得到） | 闲聊快速通道 964 → **944 ms**（生成 618 → 326）；带 judge 1300 → **1131 ms** |
| BGE-M3 挪 CPU | `RAG_DEVICE=cpu` | 每问检索 11 → **298 ms**，已回退到 GPU |
| `PYTHONNOUSERSITE=1` | start_stack 顶部导出 | 修 `~/.local` numpy 1.26.4 遮蔽 env numpy 导致的 transformers 导入失败 |

医学链路端到端（中位）：第一轮 1666 ms；开服务端 env + 双模型 1961 ms；开服务端 env + 只留 8b 1768 ms。即服务端 env 本身近乎中性，
**双模型常驻会额外拖慢医学约 190 ms**（显存/带宽竞争），按业务配比选择：

```bash
# 医学为主：不加载 4b
MED_CHAT_MODEL= USE_OLLAMA=1 bash tools/latency_probe/start_stack.sh
# 闲聊为主（默认）
USE_OLLAMA=1 bash tools/latency_probe/start_stack.sh
```

细节与逐项归因见 [local_llm_optimization_report.md](local_llm_optimization_report.md)（第九~十四节）。
