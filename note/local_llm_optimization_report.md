# 本地 LLM（ollama qwen3:8b）优化报告

日期：2026-10-30 · 机器：RTX 3080 Laptop 16 GB · 链路：ASR → router/RAG → LLM → TTS

## 一、结论

| 指标（中位） | 本地未优化 | 本地优化后 | 云端 deepseek MaaS |
| --- | --- | --- | --- |
| 医学：LLM 首块 | 552 ms | **507 ms** | 681 ms |
| 医学：LLM 生成 | 1045 ms | 928 ms | **328 ms** |
| 医学：TTS 首句合成 | 1342 ms | **1013 ms** | 758 ms |
| 医学：问句→首音 | 1990 ms | **1666 ms** | 1746 ms |
| 医学：会话总时长 | 12413 ms | **11194 ms** | ~13 s |
| 闲聊（快速通道） | 964 ms | 1053 ms | 1648 ms |
| 闲聊（走 judge） | 2041 ms | **1300 ms** | ~2100 ms |

一句话：**本地链路的首音已经追平云端（1666 vs 1746 ms），闲聊和 judge 路径反超；剩下的差距在生成速度（28 token/s）和"只依据资料"的遵守度。**

## 二、机器与模型事实

- 模型 `qwen3:8b` = qwen3-8b-Q4_K_M.gguf，8.2B，ctx 40960，5.0 GB。
- ollama 0.40.13，`127.0.0.1:11434`，进程 `/home/artrc/.local/ollama/bin/ollama serve`（PID 1080211，手工启动、无 systemd 单元）。
- 实测吞吐（原生 `/api/chat`，118 prompt tokens）：**prefill 105 ms（≈1100 tok/s）**、**生成 28 tok/s**。
- 显存：ollama 5340 MiB + TTS 2592 MiB + BGE-M3 1312 MiB ≈ 10.4 / 16 GB；TTS RTF 0.367（云端 0.32，约 +15%，显存竞争所致）。

## 三、本次做了哪四项优化

### 1. 首句短化：few-shot 示例（最有效）

qwen3:8b 不遵守"首句 ≤8 字"这类抽象规则（首段中位 14 字 → TTS 首句多花约 580 ms）。改成在 `MED_SYSTEM_PROMPT` 里给 3 组带答案的示例后，模型开始模仿：

```
【示例】
问：透析完针眼什么时候能沾水？
答：先别沾水。透析后 24 小时内穿刺点不能清洗，当天也别洗澡。
问：内瘘摸不到震颤了怎么办？
答：先别慌。摸不到震颤要立即就医，可能是内瘘堵塞。
问：透析的时候能吃东西吗？
答：尽量别吃。透析中进食容易引起低血压、抽筋，实在饿了先问护士。
```

结果：首段 14 字 → **8 字**，TTS 首句合成 1342 → **1013 ms**（TTS 合成耗时与首段字数近似线性）。

### 2. 答案瘦身：本地默认 `max_tokens=220`

本地生成 28 tok/s，答得越长越吃亏。`MED_LLM_MAX_TOKENS` 在 provider=ollama 时默认 220（云端仍 400，env 可覆盖）。回答 46 → 36 字，音频 −480 ms、会话 −1.2 s。

### 3. 预热 + keep_alive

`start_stack.sh` 在 `USE_OLLAMA=1` 时先发一个 `keep_alive=30m` 的预热请求（`/api/chat`，`num_predict=1`）。原来首轮冷启动 3.1 s（模型加载进显存），现在无感。

### 4. 本地 judge

闲聊意图复核原来走云端 qwen-flash（实测 414–790 ms）。新增 `MED_INTENT_PROVIDER=ollama`：judge 用本机 `/api/chat`（`think=false` + `/no_think`），**46–113 ms、12 条判定全对**。走 judge 的闲聊路径 2041 → **1300 ms**。

`start_stack.sh` 的 `USE_OLLAMA=1` 分支默认把三个环境变量都切到本地：`MED_LLM_PROVIDER=ollama`、`MED_INTENT_PROVIDER=ollama`、`MED_INTENT_MODEL=qwen3:8b`。

## 四、分组数据（各 10 轮）

医学（kb10，同一批问句）：

| 指标 | 未优化 mean / med | 优化后 mean / med |
| --- | --- | --- |
| LLM 首块 | 631 / 552 | 512 / 507 |
| LLM 生成 | 980 / 1045 | 942 / 928 |
| TTS 首句合成 | 1352 / 1342 | 1113 / 1013 |
| 问句→首音 | 2199 / 1990 | 1783 / 1666 |
| 会话总时长 | 11870 / 12413 | 11173 / 11194 |
| 音频总长 | 9666 / 9886 | 9374 / 9403 |
| 首段字数 | 13.3 / 14 | 9.6 / 8 |

闲聊（chat10，快速通道）：问句→首音 1064 / 964 → 1146 / 1053（略慢，属噪声；首段 4–5 字的短寒暄本身已接近下限）。
闲聊（chatgen10，走 judge）：问句→首音 2134 / 2041 → **1287 / 1300**。

质量抽查（kb10 回答）：0 拒答，答案均有资料依据，例如"带好水杯、纸巾、医疗卡和透析卡。穿舒适衣服，别带贵重物品。"；"尽量别吃。透析中进食容易引起低血压、抽筋，实在饿了先问护士。"

测量口径：`tools/latency_probe/latency_probe.py --inject-file /tmp/bench/kb10.txt --repeat 0 --interval 2.0 --timeout 60 --warmup 6.0`。首轮曾因 DDS 发现慢出现 1 次 timeout（round00 无任何事件），`--warmup 6.0` 重跑后 10/10。

## 五、剩下的瓶颈（量化）

1. **生成速度 28 tok/s**：80 字答案 ≈ 100–120 token ≈ 3.5–4.3 s，是本地医学链路最长的一段。想再快只能换小模型/更好量化/更强显卡。
2. **prefill**：118 token 需 105 ms；实际 prompt ~1400 字（约 900 token），靠 ollama 的前缀缓存才压到 500 ms 级——**保持 system prompt 和检索块顺序稳定**能继续吃这个红利。
3. **TTS 首句合成**：本地首段 8 字 → 1013 ms，仍高于云端 758 ms，因为 GPU 显存与 ollama 竞争（RTF 0.367 vs 0.32）。
4. **质量**：8B 模型对"资料里没有就不说"的遵守弱于云端（干体重、咖啡这类边界问题会补自己的知识）。医学安全场景需要额外约束。

## 六、建议（按代价从低到高）

### A. 立刻可做（无需改服务）

- 医学答案 `max_tokens` 再压到 160：本地答 1–3 句足够，可再省 300–500 ms。
- 首段兜底：`FIRST_FRAGMENT_MAX_CHARS=10`（现在模型自愿 8 字，加个保险；越小首音越快，但太碎会听感差）。

### B. 需要重启 ollama 服务（服务是手工启动的，建议你确认时机）

```bash
kill 1080211
OLLAMA_FLASH_ATTENTION=1 \
OLLAMA_KV_CACHE_TYPE=q8_0 \
OLLAMA_KEEP_ALIVE=30m \
OLLAMA_NUM_PARALLEL=1 \
OLLAMA_MAX_LOADED_MODELS=1 \
nohup /home/artrc/.local/ollama/bin/ollama serve > /tmp/ollama.log 2>&1 &
```

- `OLLAMA_FLASH_ATTENTION=1`：长上下文 prefill 更快（本机未验证，建议重启后复测 kb10）。
- `OLLAMA_KV_CACHE_TYPE=q8_0`：KV cache 减半，给 TTS 腾 1–2 GB，有望把 TTS 首句拉回云端水平。
- `OLLAMA_NUM_PARALLEL=1` / `OLLAMA_MAX_LOADED_MODELS=1`：避免显存被多请求切分。

### C. 换模型 / 分场景

- 拉 `qwen3:4b`（约 2.5 GB）专供闲聊：生成约快 2×，闲聊首音可到 0.9 s 以内；医学仍用 8b 或云端。
- 混合路由（约 30 行）：闲聊→本地（1.05 s、零成本），医学→云端（1.75 s、资料约束可靠）。目前本地医学首音 1.67 s 已略快于云端 1.75 s，但质量风险要自己权衡。

### D. 架构层（收益大、改动大）

- 首句与正文分两次生成：第一次只出 ≤8 字短句（本地约 150 ms），播放期间生成正文，首音可压到 1.2–1.3 s。
- 高频固定问题预写话术、跳过 LLM（首音可到 1 s 内）。

## 七、复测命令

```bash
# 本地优化配置
(setsid env USE_OLLAMA=1 bash tools/latency_probe/start_stack.sh &)
sleep 45
source /opt/ros/jazzy/setup.bash && source /home/artrc/medical_ws/setup_env.sh
/home/artrc/miniconda3/envs/qwen_tts_online/bin/python tools/latency_probe/latency_probe.py \
  --inject-file /tmp/bench/kb10.txt --tag ol3_kb10 --repeat 0 --interval 2.0 --timeout 60 --warmup 6.0
# 回云端
(setsid bash tools/latency_probe/start_stack.sh &)
```

## 八、涉及文件

- `src/llm_node_py/llm_node_py/ros2_node_llm_medical.py`：few-shot、本地 max_tokens、本地 judge（`_judge_intent_local`）
- `tools/latency_probe/start_stack.sh`：`USE_OLLAMA=1` 分支（provider/judge 切换 + 预热）
- 数据：`tools/latency_probe/records/{ol,ol2,ol2b}_*`

---

# 第二轮：服务端与模型级优化（2026-10-30 18:00–18:40）

第一轮只改了代码（few-shot、max_tokens、预热、本地 judge）。这一轮动的是 **ollama 服务端**和**模型组合**，目标是把「本地模型 28 tok/s 生成太慢」这条硬约束往前推。

## 九、做了什么

| # | 改动 | 怎么生效 | 代价 / 风险 |
|---|---|---|---|
| 1 | **flash attention + KV q8_0** | 重启 `ollama serve` 时带 `OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0` | V-cache 量化需 flash attn，已同时开；精度影响仅 KV |
| 2 | **上下文 4096 → 2048** | `OLLAMA_CONTEXT_LENGTH=2048` | 长资料（top_k=3×300 字 + 3 轮历史）约 900–1300 token，仍在 2048 内 |
| 3 | **qwen3:4b 专供闲聊** | `MED_CHAT_MODEL=qwen3:4b`，仅闲聊分支覆盖模型 | 多占 2.55 GiB 显存；`/no_think` 必须带，否则先吐 reasoning |
| 4 | **BGE-M3 挪 CPU**（已回退） | `RAG_DEVICE=cpu` | 实测每问 +287 ms，显存够就别挪 |
| 5 | `PYTHONNOUSERSITE=1` | start_stack 顶部导出 | 修 17:51 出现的 `~/.local` numpy 1.26.4 遮蔽 env numpy 2.4.2 导致的 transformers 导入失败 |

qwen3:4b 用不了 `ollama pull`（CDN 域名被解析到 198.18.2.26，报 `redirect target not allowed`），改从 hf-mirror 直接下 GGUF（2.33 GiB，12–13 MB/s），`ollama create` 只写 `FROM`，模板从 GGUF 元数据继承。

服务端 KV 实测：`llama_kv_cache: size = 153.00 MiB (2048 cells, 36 layers), K (q8_0): 76.50 MiB, V (q8_0): 76.50 MiB`（原 f16@4096 约 576 MiB/模型）。

## 十、吞吐与显存（warm）

| 指标 | 优化前 | 优化后 |
|---|---|---|
| qwen3:8b 生成 | 28 tok/s | **55 tok/s** |
| qwen3:4b 生成 | — | **84–92 tok/s** |
| 8b prefill | — | 465 token 冷 243 ms（1915 tok/s），命中缓存 18 ms |
| 8b 显存（权重+KV） | 5.02 GiB @ctx4096 f16 | **4.60 GiB @ctx2048 q8_0** |
| 4b 显存 | — | 2.55 GiB @ctx2048 q8_0 |
| GPU 整机占用 | 10.5 GiB（8b+TTS+BGE） | 12.6 GiB（8b+4b+TTS+BGE）/ 10.0 GiB（只 8b） |

## 十一、端到端对比（各 10 轮，中位 ms）

| 批次 | 配置 | 医学首块 | 生成 | TTS 首句 | **问句→首音** | 闲聊(快速) | 闲聊(judge) | 首段字数 |
|---|---|---|---|---|---|---|---|---|
| ol_kb10 | 原始 8b | 552 | 1045 | 1342 | **1990** | 964 | 2041 | 14 |
| ol2b_kb10 | +第一轮代码优化 | 507 | 928 | 1013 | **1666** | 1053 | 1300 | 8 |
| ol3_kb10 | +服务端env +4b +BGE在CPU | 815 | 966 | 1252 | **2286** | 944 | 1131 | 10 |
| ol4_kb10 | +服务端env +4b，BGE回GPU | 547 | 1087 | 1229 | **1961** | — | — | 10 |
| ol5_kb10 | +服务端env，只留 8b | 520 | 881 | 1133 | **1768** | — | — | 8 |

逐项归因（对医学链路，中位）：

- **few-shot 首句短化**：首段 14→8 字，TTS 首句合成 1342→1013（**−329 ms**），E2E 1990→1666（−324）。
- **答案 max_tokens 220**：音频 9886→9403（−483），会话时长约 −1.2 s。
- **本地 judge**：闲聊走 judge 的分支 2041→1300（**−741**），judge 本身 46–113 ms（云端 qwen-flash 是 414–790）。
- **4b 闲聊模型**：闲聊快速通道 964→944（生成 618→326，被 TTS/网络吃掉一部分）；带 judge 的闲聊 1300→1131。
- **服务端 env（flash attn+q8_0+ctx2048）**：医学链路**基本中性**（ol5 vs ol2b：首块 +13、生成 −47、TTS +120、E2E +102，都在噪声内）。
- **BGE 放 CPU**：每次检索 11→298 ms（**+287**），是本轮唯一的确定性劣化，已回退到 GPU。
- **双模型常驻**：ol4（双模型）vs ol5（只 8b）医学 E2E +193 ms，主要是生成和 TTS 被显存/带宽竞争拖慢。

## 十二、结论与建议

1. 服务端 env 免费可开：KV 每模型省 423 MiB、吞吐翻倍，对端到端影响在噪声内。
2. 4b 闲聊模型是「用医学 190 ms 换闲聊 110 ms」，按业务配比选：

```bash
# 医学为主：不加载 4b（闲聊也走 8b）
MED_CHAT_MODEL= USE_OLLAMA=1 bash tools/latency_probe/start_stack.sh
# 闲聊为主：默认配置（8b 医学 + 4b 闲聊）
USE_OLLAMA=1 bash tools/latency_probe/start_stack.sh
```

3. 真正的下一步不是继续压 ollama，而是**分场景路由**：闲聊→本地 4b（944 ms），医学→云端 deepseek（1746 ms）或本地 8b（1768 ms），两者已在同一量级，取决于是否想省钱/怕断网。
4. 剩余瓶颈仍是生成：本地 8b 回答 36 字要 ~0.9 s；云端首块 ~700 ms 但生成 320 ms。想再快只能让回答更短（上限已压到 220 token，实测中位 36 字）。

## 十三、复测命令

```bash
# 服务端（手工启动的服务，改了 env 要重启）
OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_CONTEXT_LENGTH=2048 \
OLLAMA_KEEP_ALIVE=30m OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=2 \
  nohup /home/artrc/.local/ollama/bin/ollama serve >/tmp/ollama_serve.log 2>&1 &

# 链路（注意用 setsid 后台起，且命令行里不要出现 llm_node_py/ros2_node_ 字面量，否则被脚本自己的 pkill 误杀）
(setsid env USE_OLLAMA=1 bash Medical_Embodied_dev/tools/latency_probe/start_stack.sh > /tmp/stack.log 2>&1 &)

# 批次
source /opt/ros/jazzy/setup.bash && source /home/artrc/medical_ws/setup_env.sh
PYTHONNOUSERSITE=1 /home/artrc/miniconda3/envs/qwen_tts_online/bin/python \
  Medical_Embodied_dev/tools/latency_probe/latency_probe.py --inject-file /tmp/bench/kb10.txt \
  --tag ol5_kb10 --repeat 0 --interval 2.0 --timeout 60 --warmup 6.0
```

## 十四、涉及文件

- `src/llm_node_py/llm_node_py/ros2_node_llm_medical.py`（`MED_CHAT_MODEL`、`_request_model`）
- `tools/rag/rag_service.py`（`RAG_DEVICE`）
- `tools/latency_probe/start_stack.sh`（`PYTHONNOUSERSITE`、`RAG_DEVICE`、`MED_CHAT_MODEL`、双模型预热）
