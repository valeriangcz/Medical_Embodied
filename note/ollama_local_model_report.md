# 接入本机 ollama 模型实测（2026-10-30）

## 一、结论

| 项目 | 云端（ds2_kb10） | 本机 ollama（ol_kb10） |
| --- | --- | --- |
| 模型 | deepseek-v4.1-flash-1017（MaaS） | qwen3:8b Q4_K_M（ollama 0.40.13） |
| LLM 首块 TTFT 中位 | 681 ms | **552 ms** |
| 首块→[DONE] 中位 | 328 ms | 1045 ms（本地 8B 生成慢 3 倍） |
| TTS 首句合成 中位 | 758 ms | 1342 ms（首句更长，见下） |
| 问句→首音 中位 | 1746 ms | 1990 ms |
| 闲聊快路径 问句→首音 中位 | 1648 ms | **964 ms**（快 684 ms） |
| 拒答（kb10） | 0/10 | 0/10 |

一句话：**接得上、能用；闲聊链路明显更快（本地首块 172 ms），医学链路反而略慢——因为 qwen3:8b 不守"首句≤8 字"，TTS 首句合成被拉长；另外它的资料约束比云端模型松，会用自己的知识补答案。**

## 二、环境与模型

- `/home/artrc/.ollama/models/` 下只有一个模型：`qwen3:8b`（qwen3-8b-Q4_K_M.gguf，8.2B，上下文 40960，能力 `tools/thinking/completion`）。
- ollama 服务已在跑：`127.0.0.1:11434`，版本 0.40.13（CLI 不在 PATH 里，但 API 可用）。
- GPU 占用：ollama 5.3 GB + TTS 2.6 GB + BGE-M3 1.3 GB = 10.4/16 GB；TTS 的 RTF 从 0.32 升到 0.367（约 +15%，显存竞争）。

## 三、接入方式与关键坑

1. **走 OpenAI 兼容接口**：`http://127.0.0.1:11434/v1`，`model=qwen3:8b`，`api_key` 随便填。
2. **qwen3 默认会思考**：OpenAI 兼容接口不认 `think:false`，会先吐 `reasoning` 字段（实测 184 字推理、正文一个字都没出来就撞 `max_tokens`）。
   - 解法：系统提示末尾加 `/no_think`（实测首块 107 ms、总 669 ms）；
   - 原生 `/api/chat` 的 `think=false` 也有效，但要自己写 NDJSON 解析，没必要。
3. **TTFT 埋点要放在发请求之前**：ollama 会先把首 token 备好再回响应头，`create()` 返回时已经过去 270~3700 ms；之前把 `llm_request` 记在 `create()` 之后，TTFT 被算成 0.4 ms。已修正。
4. judge 仍走云端 `qwen-flash`（`MED_INTENT_BASE_URL` 与主模型解耦），本地模型只负责生成。

## 四、代码改动

`src/llm_node_py/llm_node_py/ros2_node_llm_medical.py`

- `MED_LLM_PROVIDER` 增加 `ollama`：
  - 默认 `MED_LLM_MODEL=qwen3:8b`、`MED_LLM_BASE_URL=http://127.0.0.1:11434/v1`；
  - `MED_LLM_NO_THINK=1`（默认）时，`_stream_openai` 自动在 system 消息末尾追加 `/no_think`。
- 客户端拆分：`self.client` 走主模型地址（openai/ollama），新增 `self.judge_client` 专供 judge（公网兼容接口），互不影响。
- `llm_request` 埋点前移到发起请求之前，TTFT 口径对本地/云端一致。
- 顺手修了两个提示词问题（对本地模型影响明显）：
  - 规则 2.1 强化："第一句必须是不超过 8 个字的独立短句，并以句号结尾，不要把两个信息合并进第一句"；
  - 闲聊人设新增规则 5："不要主动给健康建议（多喝水/多运动/注意饮食）"——实测「早上好」原本会回"记得多喝水哦"（对血透患者是错误建议），修改后回"今天感觉怎么样？"。

`tools/latency_probe/start_stack.sh`

```bash
# 切到本机 ollama
USE_OLLAMA=1 bash tools/latency_probe/start_stack.sh
```

## 五、实测数据（各 10 轮，全部 ok）

### 医学问题（kb10，单位 ms，mean/median/p90）

| 阶段 | 云端 ds2_kb10 | 本地 ol_kb10 |
| --- | --- | --- |
| RAG 检索 | 13/12/16 | 42/14/47 |
| LLM 首块 TTFT | 837/681/1305 | 631/552/908 |
| 首块→[DONE] | 378/328/481 | 980/1045/1352 |
| 首块→首段文本 | 78/71/92 | 160/157/214 |
| TTS 首句合成 | 807/758/1202 | 1352/1342/1521 |
| **问句→首音** | **1767/1746/2182** | **2199/1990/2708** |

本地首块更快，但生成慢（1045 ms vs 328 ms），且首句更长导致 TTS 首句合成多花约 580 ms：

| 指标 | 云端 | 本地 |
| --- | --- | --- |
| 首段字数中位 | 6 | **14** |
| MANIFEST 里"首句≤8字"遵守率 | 高 | 低（qwen3:8b 不严格执行） |

### 闲聊快路径（chat10，寒暄/笑话/陪聊）

| 阶段 | 云端 | 本地 |
| --- | --- | --- |
| LLM 首块 TTFT | 855/784/1144 | **181/172/192** |
| TTS 首句合成 | 591/550/723 | 780/726/918 |
| **问句→首音** | 1600/1648/1830 | **1064/964/1206** |

本地闲聊比云端快约 0.68 s，这是 ollama 最划算的用法。

### 闲聊非快路径（chatgen10，走 judge 的常识问题）

judge 711 ms + 本地生成：TTFT 1050/1174/1351、问句→首音 2134/2041/2645。judge 这一步在本地部署里显得很贵，可以考虑把 judge 也换成本地 `/no_think` 小请求（约 100–200 ms）。

## 六、质量问题（比时延更需要注意）

1. **资料约束变松**：云端 deepseek 会严格"资料没有就说没有"，qwen3:8b 会用自己的知识补。
   - 例：「干体重是什么意思？」知识库只有"干体重 60 kg"的举例、没有定义，云端拒答；本机答"干体重是指透析后身体实际的体重，通常比实际体重稍轻一些"——医学上没错，但**不再是"只依据资料"**。
   - 例：早先测「透析患者能喝咖啡吗」本机直接答"可以适量喝咖啡"，云端按规则拒答。
   - 如果要守住"只依据 RAG"的边界，本机模型需要更强的指令（或做检索分数阈值 + 强制兜底），代价是拒答率回升。
2. **首句不够短**：qwen3:8b 无视"≤8 字"，首段中位 14 字，直接拖慢 TTS 首句合成。可以强制 `FIRST_FRAGMENT_MAX_CHARS=8` 机械切分，但会在第 8 个字硬切、听感可能变差——不建议默认开。
3. **闲聊会主动给健康建议**：已用提示词规则 5 挡住（"多喝水"这类对血透患者是反向建议）。

## 七、建议

- **想省时延/省钱**：做"闲聊走本地、医学走云端"的混合路由——聊天 0.96 s（本地）、医学 1.75 s（云端），两段都取最优；改动只是在 `handle_query` 里按 `chat_mode` 选 provider（约 30 行）。
- **想完全离线/隐私**：全量走本地可以接受 2.0 s 的医学首音，但要接受"资料约束变松"，并考虑把 `MED_LLM_MAX_TOKENS` 压到 250 左右（本地生成慢，答越长越吃亏）。
- **judge 也本地化**：用 `/no_think` 的 qwen3 小请求替代 qwen-flash，预计省 300–500 ms（仅闲聊路径受益）。

## 八、复测

```bash
USE_OLLAMA=1 bash tools/latency_probe/start_stack.sh
source /opt/ros/jazzy/setup.bash && source /home/artrc/medical_ws/setup_env.sh
$PY tools/latency_probe/latency_probe.py --inject-file tools/latency_probe/questions/kb50.txt \
  --tag ol_kb50 --repeat 0 --interval 2.0 --timeout 60 --warmup 3.0
```

记录：`tools/latency_probe/records/ol_kb10_*`、`ol_chat10_*`、`ol_chatgen10_*`、`ol_fix5_*`、`ol_smoke_*`。
