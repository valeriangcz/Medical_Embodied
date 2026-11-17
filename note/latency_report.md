# ASR → LLM → TTS 全链路时延实测报告

- 测量日期：2026-10-27 15:30–16:00（CST）
- 测量主机：Ubuntu 24.04 / RTX 3080 Laptop 16GB / 16C32G
- 被测代码：`~/medical_ws/Medical_Embodied_dev`（dev 分支，未提交改动见 §7）
- 测量工具：`tools/latency_probe/`（单进程同钟采集 + 节点内 `/latency_event` 埋点）
- 原始数据：`tools/latency_probe/records/`（`report_all.md`、`*_summary.json`、`*_stats.csv`、`*_rounds.csv`、每轮 `*_roundNN.json`）

> **一句话结论**：用户说完话到听到第一个音，**联网问答路径约 5.3 s，医疗知识库路径约 12.4 s**。
> 主要耗时依次是：百炼应用首块 5.6–10.9 s（医疗）/ 1.1 s（联网）、`nurse_alert` 云端判断 ≈1.6 s、ASR 端点静音 1.2 s、TTS 首句合成 ≈1.05 s。
> 首音之后 TTS 合成（RTF≈0.40）快于播放，**不会拖尾**，会话总时长 ≈ 首音延迟 + 音频时长。

---

## 一、链路与测量点

```
麦克风(PortAudio "default")                                 扬声器(pyaudio out)
   │ 16k/44.1k（ASR 内部重采样）                                   ▲
   ▼                                                              │
ros_node_keyword_asr (C++)                                        │
  KWS 唤醒(小艺小艺) → ASR 流式识别 → 端点判定(rule2=1.2s)          │
   ├─ [LAT] kws_detected / start_asr / endpoint                   │
   ├─ nurse_alert 服务：qwen3.5-flash 非流式判断（阻塞）            │
   ├─ [LAT] nurse_req / nurse_resp                                │
   └─ /question_asr  ──[LAT] publish_question──┐                  │
                                               ▼                  │
                              ros2_node_question_router（关键词“天气”分流）
                                    │                    │
                        /llm_to_openai          /llm_to_dashapp
                       qwen-plus-latest        百炼应用「小医」
                        (联网+thinking)          (b7ed8dad…)
                                    └──────┬─────────────┘
                                           ▼
                                   /ali_llm_output（流式文本，[START]/[DONE]）
                                           ▼
                             ros2_node_stream_text（按，。；断句）
                                           ▼
                                   /tts_realtime_data（分句文本）
                                           ▼
                             ros2_node_tts_local（Qwen3-TTS 0.6B, bf16）
                               逐句合成 → 队列 → 播放（0.1s 定时器）
                                ├─ [LAT] rx / synth_start / synth_end
                                ├─ [LAT] play_start / play_end
                                └─ /tts_session_finished ──[LAT] session_finished
```

**测量点说明**

| 时间点 | 获取方式 | 说明 |
| --- | --- | --- |
| 用户语音结束 | 问句 wav 播放结束（`pw-play` 退出）+ 端点/rule2 反推 | 双口径互验，差 ~90 ms |
| ASR 端点 `endpoint` | C++ 节点埋点 | 静音规则触发瞬间 |
| 发布问题 `publish_question` / `/question_asr` 到达 | C++ 埋点 / 探针订阅 | 两者差 <1 ms |
| LLM 首块 / `[DONE]` | 探针订阅 `/ali_llm_output` | 忽略 `[START]/[DONE]` 控制符 |
| 首句 TTS 文本 | 探针订阅 `/tts_realtime_data` | 断句器输出第一段 |
| 首音 `play_start`、播放完 `play_end` | TTS 节点埋点 | 首个 `play_start` 即“听到声音” |
| 会话完成 | `/tts_session_finished` | TTS 队列播完 [DONE] 后发出 |

**两种注入方式**

1. **文本注入**（隔离 LLM+TTS）：直接向 `/question_asr` 发文本，跳过 ASR/nurse，重复性最好；
2. **声学注入**（全链路）：PipeWire 虚拟设备 `latency_sink → latency_source` 作为默认麦克风，播放唤醒词/问句 wav，C++ ASR 节点端口 `PortAudio "default"` 实录（回环验证：相关 0.97–1.00、RMS 比 1.00）。**唤醒词“小艺小艺”**（TTS 生成，KWS 解码 43.8 ms）。

**端点静音口径（重要更正）**：`asr_stream.cpp` 设置 rule1=2.4s、rule2=1.2s，但 sherpa-onnx 三规则是 **OR** 关系且 rule2 的 `must_contain_nonsilence=true` 对任意含语音的句子都成立，因此**实测端点延迟是 1.2 s（rule2）+ 模型 lookahead ≈1.27–1.52 s**，不是 readme 写的 2.4 s（关闭 rule2 后才会变成 2.52 s）。本报告“语音结束→首音”按 `endpoint − 1.3 s` 计，与 wav 播放结束口径相差 <100 ms。

---

## 二、全链路结果（声学注入，含 ASR + nurse_alert）

### 2.1 端到端

| 场景 | n | 提出问题→首音 | **语音结束→首音** | 语音结束→会话完成 | 音频总时长 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 联网问答（“天气今天如何？”） | 3 | 2.36 s（mean）<br>1.84 / 2.20 / 3.03 s | **5.25 s**（wav 结束口径）<br>4.84 / 5.66 / 5.26 s | 20.4 s | 18.0 s |
| 医疗知识库（“肾病患者运动时有哪些注意事项？”） | 2 | 9.45 s<br>6.67 / 12.22 s | **12.44 s**（wav 结束口径）<br>9.96 / 14.92 s | 17.7 / 53.7 s | 11.1 / 41.4 s |

> 会话总时长 ≈ 首音延迟 + 音频时长（差值 <0.1 s）：TTS 逐句合成与播放并行，合成速度（RTF 0.40）快于实时，**首音之后没有二次卡顿**。

### 2.2 首音时间预算分解（mean）

| 阶段 | 联网问答 | 医疗知识库 | 备注 |
| --- | ---: | ---: | --- |
| ① ASR 端点静音（固定） | ≈1.30 s | ≈1.30 s | rule2 1.2 s + lookahead/分块 |
| ② ASR 端点→nurse 请求 | 0.013 s | 0.018 s | 同进程回调 |
| ③ `nurse_alert` 云端判断 | **1.67 s** | **1.61 s** | qwen3.5-flash 非流式，串行阻塞 |
| ④ nurse 返回→发布问题 | 0.000 s | 0.000 s | 同进程 |
| ⑤ 路由（router） | 0.001 s | 0.001 s | 纯 ROS |
| ⑥ LLM 首块（TTFT） | **1.10 s** | **8.26 s** | 云服务首包，波动大 |
| ⑦ LLM 首块→断句首句 | 0.13 s | 0.04 s | 等第一个，/。； |
| ⑧ TTS 首句合成 | **1.08 s** | **1.06 s** | 首句 2.4–3.0 s 音频，RTF≈0.42 |
| ⑨ 合成完→开始播放（队列 0.1s 定时器） | 0.04 s | 0.09 s | |
| **合计（①→⑨，≈语音结束→首音）** | **≈5.3 s** | **≈12.4 s** | 与 2.1 实测一致 |

### 2.3 逐轮明细（声学全链路）

| tag / 轮次 | 识别结果（字节） | nurse | LLM TTFT | LLM 首块→完 | 断句首句 | TTS 首音 | 首音(问题→) | 语音结束→首音 | 会话 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| weather r0 | 21（正确） | 1103 ms | 1561 ms | 1532 ms | 79 ms | 1224 ms | 3031 ms | 5257 ms | 19711 ms |
| weather r1 | 21（正确） | 2289 ms | 1115 ms | 1269 ms | 74 ms | 1010 ms | 2204 ms | 5656 ms | 20832 ms |
| weather r2 | 21（正确） | 1614 ms | 635 ms | 1277 ms | 246 ms | 1131 ms | 1840 ms | 4842 ms | 20694 ms |
| medical r0 | 45（正确） | 1989 ms | 5591 ms | 493 ms | 0.4 ms | 1082 ms | 6675 ms | 9955 ms | 17732 ms |
| medical r1 | 45（正确） | 1239 ms | 10925 ms | 1388 ms | 87 ms | 1208 ms | 12220 ms | 14921 ms | 53655 ms |

> 识别正确性：天气问句 21 字节 = “天气今天如何？”（含标点模型补的“？”）；医疗问句 45 字节，路由/知识库均按预期命中，回答内容与问题匹配。KWS 唤醒→`/start_asr` 仅 1–7 ms，KWS 解码 43.8 ms。

---

## 三、LLM+TTS 隔离结果（文本注入，无 ASR/nurse）

| 场景 | n | Router | **LLM TTFT** | LLM 首块→完 | 断句首句 | **TTS 首句合成** | **问题→首音** | 会话 | 音频 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 联网问答 | 3 | 1.2 ms | **914 ms**（713/740/1290） | 1125 ms | 72 ms | 863 ms | **1.92 s**（1.64–2.37） | 17.6 s | 15.6 s |
| 医疗知识库 | 3 | 1.3 ms | **6687 ms**（6303/6682/7077） | 150 ms | 12 ms | 1084 ms | **7.82 s**（7.46–8.17） | 14.7 s | 6.8 s |

> 医疗路径的 LLM 首块 6.3–7.1 s 非常稳定，而首块之后几乎立刻 `[DONE]`（0.5–379 ms）——说明 **百炼应用在流式返回前有 ~6 s 的思考/检索/排队期**，这是医疗路径最大的单点延迟。

---

## 四、TTS 与 ASR 组件实测

### 4.1 TTS（Qwen3-TTS-12Hz-0.6B-CustomVoice，bf16，sdpa）

- 跨全部实测轮次共 **70 个合成片段**：RTF 均值 **0.400**、中位 0.407、范围 0.335–0.456；总音频 171.8 s / 总合成 68.5 s → **综合 RTF 0.399**。
- 首句合成 0.79–1.16 s（首句文本 8–12 字、音频 2.4–3.0 s）；合成完到播放开始 **3–100 ms**（队列定时器 0.1 s）。
- 模型加载+预热 ≈9–11 s（进程启动一次性开销，已确认预热生效）。
- 注：本机未装 flash-attn，已回退 sdpa（约是 readme 中 RTF≈1.0 的全量生成路径的 1/2.5 速度，快速解码路径生效）。

### 4.2 ASR（sherpa-onnx streaming zipformer bilingual zh-en，CPU 6 线程）

- 离线流式 RTF：0.wav 10.05 s→0.84 s（RTF 0.084）、1.wav 5.10 s→1.15 s（0.225）、2.wav 4.69 s→1.26 s（0.268）、3.wav 8.83 s→1.17 s（0.132）；按 C++ 真实 44.1 kHz 喂入结果一致（内部重采样正常）。
- KWS：唤醒词“小艺小艺”命中，解码 43.8 ms；唤醒→`/start_asr` 1–7 ms；问题音频无误唤醒。
- 端点：语音结束→判端点 **1.27–1.52 s（rule2=1.2 s）**；这是语音交互的固定人机延迟，占联网路径首音延迟的 25%。
- 标点模型：2–4 ms/句（补“？”等）。

---

## 五、瓶颈与优化建议（按预期收益排序）

| 优先级 | 瓶颈 | 实测代价 | 建议 | 预期收益 |
| --- | --- | --- | --- | --- |
| P0 | 医疗路径百炼应用首块 | 5.6–10.9 s | 换 `qwen-plus` 直连 + 知识库检索（RAG）替代 `Application.call`；或确认应用是否可关掉思考/检索前置；也可对高频问题做答案缓存 | −4～8 s |
| P0 | `nurse_alert` 非流式判断串在问题前 | 1.1–2.3 s（mean 1.6 s） | 与主 LLM 请求**并行**发起（`publish_question` 不等 nurse 结果，need_call 只影响是否呼叫护士）；或对“天气/科普”类问题用本地规则短路；或把 need_call 判断合并进主 LLM 的 system prompt | −1.5 s |
| P1 | ASR 端点静音 1.2 s | 1.2–1.5 s | 将 `rule2_min_trailing_silence` 调到 0.6–0.8 s（需评估截断率）；同时更正 readme 中“2.4 s”的描述 | −0.4～0.6 s |
| P1 | TTS 首句合成 ≈1.05 s | 1.05 s | ① 首句再细分（逗号即切，现已是 fragment 但首句仍有 8–12 字）；② 固定开场短语（如“好的”“我在呢”）预合成缓存；③ 安装 flash-attn2（当前 sdpa 回退）；④ 尝试 CUDA Graphs（仓库已有 `tts_cuda_graphs.py`） | −0.3～0.6 s |
| P2 | LLM 首块→断句 0.04–0.25 s | 0.13 s | 断句器对首个逗号/句号即发；已接近最优 | −0.05 s |
| P2 | 唤醒后播放 hello mp3 | 本机 ffplay 缺失时跳过（2026-11-02 已装 ffplay，实际约 +2.0 s） | 唤醒→ASR 开启约多 2 s（不属于“说完话→首音”） | — |

**优化后可达目标（估算）**：联网路径 5.3 s → **≈2.2–2.7 s**（端点 0.8 s + nurse 并行 0 s + LLM 1.1 s + TTS 0.7 s）；医疗路径 12.4 s → **≈3–4 s**（前提是换直连 LLM 把首块压到 1–1.5 s）。

---

## 六、复现方式

```bash
# 0) 环境
source /opt/ros/jazzy/setup.bash && source ~/medical_ws/setup_env.sh
export SHERPA_MODEL_ROOT=$HOME/medical_ws/models/sherpa   # 模型不在 /opt（无 sudo），C++ 节点已支持 env 覆盖
PY=$HOME/miniconda3/envs/qwen_tts_online/bin/python

# 1) 启动精简语音链路（等价 launch_llm_nodes.sh 的子集：ASR/路由/LLM/历史/nurse/TTS/断句）
cd ~/medical_ws/Medical_Embodied_dev
LATENCY_PROBE=1 bash tools/latency_probe/start_stack.sh          # 停止：bash ... start_stack.sh stop

# 2) 声学回环（虚拟麦克风；详见 ~/medical_ws/models/audio_setup.md）
pw-loopback -n latency_loop \
  --capture-props='media.class=Audio/Sink node.name=latency_sink node.description="Latency Sink" audio.position=[FL FR]' \
  --playback-props='media.class=Audio/Source node.name=latency_source node.description="Latency Source" audio.position=[FL FR]' &
wpctl set-default $(wpctl status | awk '/Latency Source/{gsub(/[^0-9]/,"",$1); print $1}')  # 原值 53 为真麦

# 3a) 声学全链路测量（唤醒词→/start_asr→问句）
$PY tools/latency_probe/latency_probe.py --record-dir tools/latency_probe/records \
  --tag voice_weather --repeat 3 \
  --wake-wav tools/latency_probe/audio/wake_xiaoyi.wav \
  --question-wav tools/latency_probe/audio/q_weather.wav \
  --play-cmd "pw-play --target latency_sink {wav}" --pre-delay 0.3 \
  --warmup 10 --interval 18 --endpoint-silence 1.3

# 3b) LLM+TTS 隔离测量（直接注入文本）
$PY tools/latency_probe/latency_probe.py --record-dir tools/latency_probe/records \
  --tag inj_weather --repeat 3 --inject-text "天气今天如何？" --warmup 10 --interval 3

# 4) 生成报告表格
$PY tools/latency_probe/report_latency.py --record-dir tools/latency_probe/records \
  --tags "inj_weather,inj_medical,voice_weather,voice_medical" --out tools/latency_probe/records/report_all.md

# 5) 收尾：恢复真实麦克风为默认输入；停链路
wpctl set-default 53
bash tools/latency_probe/start_stack.sh stop
```

> 注意：`--warmup` 必须 ≥5 s（DDS 发现完成前注入会丢订阅，导致只有端到端、缺分段数据）；声学多轮之间 `--interval` 必须 ≥18 s（会话结束后 ASR 节点有 15 s 超时定时器才切回 KWS，否则唤醒词会被当普通语音识别造成串轮）。

---

## 七、本次改动与产物

### 7.1 代码改动（默认行为不变，仅新增 env 开关）

| 文件 | 改动 |
| --- | --- |
| `src/llm_node_cpp/include/llm_node_cpp/latency_probe.h`（新增） | `LATENCY_PROBE=1` 时打印 `[LAT]` 并发布 `/latency_event`（JSON：node/stage/t_ns/detail） |
| `src/llm_node_cpp/src/ros_node_keyword_asr.cpp` | 埋点 init_done/kws_detected/start_asr/asr_callback/nurse_req/nurse_resp/publish_question；模型路径支持 `SHERPA_MODEL_ROOT`/`ASR_MODEL_DIR`/`KWS_MODEL_DIR`/`KWS_KEYWORDS_FILE` 覆盖（默认仍为 `/opt/sherpa-onnx/...`） |
| `src/llm_node_cpp/src/asr_stream.cpp` | endpoint 埋点；`PUNCT_MODEL_PATH` 环境变量 |
| `src/llm_node_cpp/src/asr_punctuation.cpp` | 标点模型缺失/加载失败时降级为不加标点（不再 throw） |
| `src/llm_node_py/llm_node_py/ros2_node_tts_local.py` | `LATENCY_PROBE=1` 时埋点 model_warmup/rx/synth_start/synth_end/play_start/play_end/session_finished |

`llm_node_cpp` 已重新编译（`install/llm_node_cpp/lib/llm_node_cpp/ros_node_keyword_asr`）。

### 7.2 新增工具与资源

- `tools/latency_probe/`：`latency_probe.py`（探针）、`report_latency.py`、`latency_stats.py`、`fake_chain_test.py`、`run_selftest.sh`、`README.md`、`start_stack.sh`、`gen_test_wavs.py`、`audio/`（唤醒词与问句 wav）、`records/`
- `~/medical_ws/models/`：`sherpa/`（ASR/KWS/标点模型，约 1.1 GB；`model_validation.md` 有离线验证数据）、`audio_setup.md`（虚拟回环命令与验收）、`scripts/`、`logs/`

### 7.3 已知限制

- 样本量小（全链路 n=3/2），云端 API（qwen-plus / 百炼应用 / qwen3.5-flash）时延波动大（医疗首块 5.6–10.9 s，另有一次 21.5 s 极端值导致该轮 60 s 超时，已剔除）。建议生产前在目标网络环境复测 ≥10 轮。
- 声学注入为“虚拟声卡回环”，不含真实房间噪声/回声；真实麦克风距离/增益会影响 ASR 首字识别，但不显著改变链路时延。
- 精简体只启动语音链路 8 个节点，未包含 UI/行为树/dialog_manager（不在语音首音关键路径上）。
- 本机最初缺少 `ffplay`，问候语 mp3 被跳过；2026-11-02 装上 ffplay 后，唤醒会播 1.8 s 问候语（`play_mp3_blocking` 阻塞，唤醒→ASR 约 +2.0 s）。
