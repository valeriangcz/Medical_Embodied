# Phase 0 报告：本地 RAG 知识库 + 医疗问答直连改造

- 日期：2026-10-28
- 目标：把医疗问答从「云端百炼应用（检索+生成都在云端）」改为「**本地 RAG 检索 + 云端 qwen-plus 直连**」，在不降低质量的前提下压低首音延迟、让知识库可审计可替换。
- 结论：**达到目标**。医疗路径首音延迟降 59%–67%，且答复质量显著优于现状——现状百炼应用在 15 题中 **10 题回答"数据库中还没有该项知识"**。

---

## 一、改动架构

```
/llm_to_dashapp
      │
      ▼
ros2_node_llm_medical（本次改造）
      ├─ 1. 取对话历史（/get_llm_dialog_history，最多 3 对）
      ├─ 2. 本地 RAG 检索：POST http://127.0.0.1:8765/retrieve  (top_k=4)
      │       └─ rag_service: BGE-M3 稠密 + 字符 bigram BM25（稀有词门控）+ 加权 RRF
      ├─ 3. 组装 system prompt（人设 + 只依据资料/资料不足拒答/短句口语化/急症立即叫护士）
      │       + 检索片段（带标题来源）
      ├─ 4. dashscope OpenAI 兼容接口 qwen-plus-latest 流式生成
      └─ 5. 逐块发 /ali_llm_output，末尾 [DONE]；问答写入 dialog_history
              └─ 原样保留旧路径：MED_LLM_BACKEND=app 可切回百炼应用做 A/B
```

**向后兼容**：话题与服务接口（`/llm_to_dashapp`、`/ali_llm_output`、dialog_history）不变，下游断句器、TTS、行为树无感知。
**开关**：`MED_LLM_BACKEND=direct|app`、`RAG_ENABLE=1|0`、`RAG_URL`、`RAG_TOP_K`、`MED_LLM_MODEL`、`LATENCY_PROBE=1`。

## 二、知识库与检索

| 项 | 数值 |
| --- | --- |
| 文档 | 18 篇中文血透患者教育短文（示例资料，**非诊疗依据**，正式部署需替换为院方审定文档） |
| 索引 | 104 个切块，300–600 字/块、重叠 90 字，保留标题路径 |
| 嵌入模型 | BGE-M3（/home/artrc/medical_ws/models/bge-m3，2.2 GB，GPU fp16） |
| 建索引 | 模型加载 2.1 s + 向量化 0.6 s（6 ms/块） |
| **检索耗时** | **稳态 9–22 ms**（首次含预热 165–174 ms） |
| **Recall@5** | **93.3%（14/15）**：fact 8/9、multi 3/3、safety 2/2、out_of_scope 1/1 |

唯一未命中：「透析前需要做哪些准备？」top1 误召回到「出行和旅行」——语义相近，属可接受误差（top-4 内含正确文档则大模型仍可答对）。

**检索设计要点**：字符 bigram BM25 在中文里会被「透析/患者」这类全库高频词带偏（初版把饮水问题排到了血压文档），因此做了两道保护——过滤 df>30% 的高频 bigram，并且**仅当 query 含稀有词（df≤8，如内瘘/血钾/促红素/KtV）时才启用 BM25**；融合用加权 RRF（稠密 1.0 / BM25 0.35）。修正后 Recall@5 从不可用提升到 93.3%。

## 三、延迟对比（同日实测）

### 3.1 文本注入（隔离 ASR/nurse，只比 LLM+TTS）

| 指标 | 旧：百炼应用 | 新：RAG+qwen-plus 直连 | 变化 |
| --- | ---: | ---: | ---: |
| 检索 | —（在云端应用内） | 14 ms | — |
| **LLM 首块 TTFT** | **6852 ms**（6045–8171） | **1117 ms**（483–1648） | **−83.7%** |
| LLM 首块→结束 | 约 1.2 s | 0.97 s | — |
| 断句首句 | 0.17 s | 0.17 s | — |
| TTS 首句合成 | 1.20 s | 1.25 s | — |
| **端到端 question→首音** | **8083 ms** | **2652 ms** | **−67.2%** |
| 会话总时长 | 10787 ms | 17150 ms | 见说明① |

> ① 会话总时长变长是因为新方案的答案更完整（旧应用大量回答"数据库中没有知识"，答案短）。首音延迟才是交互体验指标。

### 3.2 声学全链路（唤醒词→ASR→nurse→LLM→TTS）

| 指标 | 旧（10-27，百炼应用） | 新（10-28，RAG+直连） | 变化 |
| --- | ---: | ---: | ---: |
| ASR 唤醒→start_asr | 2.6 ms | 5.5 ms | — |
| **ASR 说话结束→端点** | ~1.3 s（rule2） | ~1.3 s | 不变 |
| **nurse_alert 判断** | 1668 ms | 1296 ms | 云波动 |
| **LLM TTFT** | 1103 ms（天气）/ 8258 ms（医疗） | **1165 ms** | 医疗路径 −86% |
| 断句首句 | 133 ms | 134 ms | — |
| **TTS 首音** | 1122 ms | 1123 ms | 不变 |
| **说话结束→首音** | **12380 ms** | **5032 ms** | **−59.4%** |
| **问题发布→首音** | 9447 ms | 2423 ms | **−74.3%** |
| 会话总时长 | 35694 ms | 9188 ms | — |

### 3.3 新方案首音预算（声学，mean）

| 阶段 | 耗时 | 占比 |
| --- | ---: | ---: |
| ① ASR 端点静音（rule2=1.2 s + 分块/lookahead） | ≈1.3 s | 26% |
| ② ASR→nurse 请求 | 13 ms | — |
| ③ nurse_alert 云端判断（串行阻塞） | 1296 ms | 26% |
| ④ 路由 + RAG 检索注入 | 15 ms | — |
| ⑤ LLM 首块（qwen-plus） | 1165 ms | 23% |
| ⑥ 断句首个短句 | 134 ms | 3% |
| ⑦ **TTS 首句合成** | 1123 ms | 22% |
| ⑧ 队列→开始播放 | 43 ms | 1% |
| **合计** | **≈5.0 s** | 100% |

## 四、答复质量 A/B（15 题，同题同参数）

| 指标 | 旧：百炼应用 | 新：RAG+直连 |
| --- | ---: | ---: |
| **答复"数据库中还没有该项知识"** | **10 / 15 题** | **0 / 15** |
| 人工要点覆盖（Lead 按期望要点逐题判定，满分 45） | 13 / 45 = **28.9%** | 43 / 45 = **95.6%** |
| 自动 bigram 覆盖率（弱指标，仅作方向参考） | 0.02 | 0.29 |
| 安全题（内瘘震动消失 / 胸痛气促）合规 | 2/2 呼叫医护 | 2/2 立即呼叫护士 |
| 超出知识库问题（保健品"修复肾脏"） | 回答"没有该项知识" | 明确否定 + 建议问医生 |
| 答案平均长度 | 68 字 | 64 字 |
| 出现违禁词"着急" | 1 题 | 0 题 |
| TTFT（同批问题） | 10350 ms | 857 ms |

**结论**：
1. 现状百炼应用的**知识库并没有覆盖这批血透宣教内容**，10/15 题直接说"没有知识"——这是比速度更严重的问题；
2. 新方案回答全部有检索依据、无编造，安全题按预期立即升级到护士，超纲题能正确拒答；
3. 旧应用在皮肤瘙痒题给出「中药全身熏蒸」这类知识库外建议，属**无依据建议风险**；新方案严格限制在资料内。

> 自动 bigram 覆盖率对同义改写很敏感（例如"百分之三到五" vs "3%–5%"），数值偏低不代表答案错，故以**人工要点判定**为准，并在报告中同时保留自动指标作趋势参考。人工判定由 Lead 逐题完成，属主观评分，正式验收建议由临床人员复核。

### 质量样例

| 问题 | 旧（百炼应用） | 新（RAG+直连） |
| --- | --- | --- |
| 透析了还要不要多吃肉？ | "数据库中没有该项知识…不要自行多吃肉" | "透析后可以吃肉，而且需要吃够优质蛋白…具体吃多少建议听营养师" |
| 血钾高有什么感觉？ | "数据库中还没有该项知识" | "血钾高常常没有明显感觉，可能悄悄影响心跳，严重时心跳骤停…定期抽血查钾" |
| 内瘘摸不到震动了怎么办？ | 立即呼叫医护 + 三条注意事项（较长） | "震颤消失是急症，要立即通知护士…我马上帮您呼叫护士"（适合语音） |
| 透析时胸口疼、喘不上气 | 立即告诉护士 + 连续追问（较长） | "我马上帮您呼叫护士。请别动，先按呼叫器"（最短路升级） |

## 五、结论与 Phase 1 建议

**Phase 0 目标已达成**：
- 医疗路径首音 **12.4 s → 5.0 s（声学全链路）**、**8.1 s → 2.7 s（不含 ASR/nurse）**；
- 知识库从"云端不可控、10/15 题无知识"变为"本地 18 篇可审计、Recall@5 93.3%、答复全部有据"；
- 改造为增量式：接口不变、旧路径保留可回退、`RAG_ENABLE=0` 可一键降级。

**新瓶颈（按收益排序）**：
| 优先级 | 瓶颈 | 代价 | 建议 |
| --- | --- | ---: | --- |
| P0 | `nurse_alert` 串行在问题之前 | 1.3–2.3 s | 与主 LLM 并行发起（need_call 只影响是否呼叫护士）；或对科普类问题规则短路 |
| P1 | ASR 端点静音 | 1.3 s | rule2 1.2→0.8 s，评估截断率 |
| P1 | TTS 首句合成 | 1.1 s | 首句再细分 / 固定开场白预合成 / 装 flash-attn2 / CUDA Graphs |
| P2 | LLM 首块 | 1.1 s | 已直连；可换 qwen-flash 系、或本地 4B/8B（Phase 1 评估） |

**Phase 1 建议**：按上一轮方案装 vLLM + Qwen2.5-7B-AWQ / Qwen3-4B-AWQ，复用本 RAG 与评测集，对比「本地模型 vs qwen-plus 直连」的延迟与质量；本报告的 15 题评测集、`compare_backends.py`、`latency_probe` 可直接复用。

## 六、交付物与复现

**新增/修改文件**

| 路径 | 说明 |
| --- | --- |
| `src/llm_node_py/llm_node_py/ros2_node_llm_medical.py` | 改造：RAG 注入 + 直连 qwen-plus + 引用/拒答约束 + `/latency_event` 埋点；保留百炼分支 |
| `tools/rag/docs/`（18 篇 + MANIFEST） | 示例血透宣教知识库 |
| `tools/rag/eval/`（15 题 + README） | 检索与答案评测集 |
| `tools/rag/build_index.py` / `retriever.py` / `rag_service.py` / `retrieve_test.py` / `index/` | RAG 引擎与索引 |
| `tools/rag/scripts/eval_retrieval.py` / `compare_backends.py` / `out/` | 评测脚本与 A/B 原始结果 |
| `tools/rag/README.md` | 用法、接口契约、设计说明 |
| `tools/latency_probe/start_stack.sh` | 增加 rag_service 启停与 `/health` 就绪等待 |
| `tools/latency_probe/records/p0_medical*`、`p0_voice_medical*` | 本次延迟原始数据 |

**复现**

```bash
source /opt/ros/jazzy/setup.bash && source ~/medical_ws/setup_env.sh
PY=$HOME/miniconda3/envs/qwen_tts_online/bin/python
cd ~/medical_ws/Medical_Embodied_dev

# 建索引（文档变更后）
$PY tools/rag/build_index.py --docs tools/rag/docs --out tools/rag/index \
    --model $HOME/medical_ws/models/bge-m3

# 起全链路（含 RAG 服务），医疗节点默认 direct + RAG
LATENCY_PROBE=1 RAG_ENABLE=1 MED_LLM_BACKEND=direct \
    bash tools/latency_probe/start_stack.sh

# 延迟：文本注入
$PY tools/latency_probe/latency_probe.py --record-dir tools/latency_probe/records \
    --tag p0_medical --repeat 3 --inject-text "肾病患者运动时有哪些注意事项？" \
    --warmup 10 --interval 3

# 延迟：声学全链路（先把默认麦克风切到 latency_source，见 models/audio_setup.md）
$PY tools/latency_probe/latency_probe.py --record-dir tools/latency_probe/records \
    --tag p0_voice_medical --repeat 2 \
    --wake-wav tools/latency_probe/audio/wake_xiaoyi.wav \
    --question-wav tools/latency_probe/audio/q_medical_exercise.wav \
    --play-cmd "pw-play --target latency_sink {wav}" --pre-delay 0.3 \
    --warmup 10 --interval 25 --endpoint-silence 1.3 --timeout 120

# 质量：检索 Recall@5 / 答案 A/B
$PY tools/rag/scripts/eval_retrieval.py --rag-url http://127.0.0.1:8765 --top-k 5
$PY tools/rag/scripts/compare_backends.py --limit 15
# A/B 切换旧后端做延迟对照：MED_LLM_BACKEND=app RAG_ENABLE=0 bash tools/latency_probe/start_stack.sh
```

## 七、局限与风险

1. **知识库内容为示例**，未经临床审定；`docs/` 必须由院方替换并做版本登记（MANIFEST 已留字段）。
2. 评测集 15 题规模小、人工评分为主，尚不足以支撑上线决策；建议扩到 100–300 题并由临床复核。
3. 现状百炼应用的"知识库缺失"结论基于当前 app_id 配置；若院方另有应用配置，需要以其实际知识库重新对比。
4. `rag_service` 目前为单进程 HTTP 服务，随 `start_stack.sh` 启动；生产应 systemd 常驻并加健康检查/重启。
5. 声学测量使用虚拟声卡回环，不含真实环境噪声；真实麦克风下的 ASR 首字识别可能更差，但不改变本报告的链路耗时结论。
6. rertriever 的 BM25 采用字符 bigram（无分词器），对同义词泛化弱；若召回不足，Phase 1 可加 reranker 或改用 jieba/词级 BM25。

---

## 附：知识库切换到用户资料（2026-10-29）

用户提供了真实科室资料 `tools/rag/source/2.md`（16,778 字，连续排版、无 Markdown 标题、含重复段落），已完成拆分与结构化，并替换为正式索引源：

| 变更 | 说明 |
| --- | --- |
| 新增 `tools/rag/source/2.md` | 用户原始资料原件（未拆分，保留可追溯） |
| 新增 `tools/rag/docs/`（21 篇 + MANIFEST） | 由 2.md 拆分：首次透析、控水、内瘘感染/闭塞/透析后护理/新瘘穿刺/手术宣教/居家护理、制度、并发症、运动、检验、药物、血压、情志、八段锦、导管护理、AVG 并发症、中医适宜技术（耳部/体部）、四季养生 |
| `tools/rag/docs_example/` | 此前的 18 篇示例文档归档，**不参与索引** |
| `tools/rag/eval/eval_set.json` | 重写为用户库 15 题（fact 11 / multi 2 / safety 2）；旧示例库评测集改名 `eval_set_example_docs.json` |
| 索引 | 重建：21 篇 → **120 个切块**；Recall@5 **15/15 = 100%**；检索 10–22 ms（首次含预热 170 ms） |
| 去重处理 | 热敷要点、传染病监测、血清肿、瘙痒预防等原文重复段落各保留一处；导管护理/内瘘手术宣教做了同主题合并（详见 `docs/MANIFEST.md`） |

**注意**：用户资料含具体机构名称、咨询电话、班次制度与中医适宜技术项目，已在 `docs/MANIFEST.md`「待院方确认事项」中列出需核对项；上线前须由院方审定。


---

## 附二：端到端语音问答实测（2026-10-29，用户资料库）

详见 `note/rag_e2e_voice_test.md`。3 轮声学注入实测结果：

| 指标 | 数值 |
| --- | --- |
| 答复正确率 | 3/3（均有检索依据、无编造） |
| 问题→首音 | 1.99 s（1.61–2.35 s） |
| **说话结束→首音** | **4.87 s**（4.57–5.27 s） |
| nurse_alert | 1.57 s（1.11–2.33 s，最大瓶颈） |
| LLM 首块 | 1.01 s（0.63–1.45 s） |
| TTS 首句合成 | 0.90 s（0.58–1.18 s） |

本轮修复：口语提问（"针眼""沾水""胳膊"）召回错误 → 文档补别名 + 检索器加口语查询扩展，复测答复正确。
