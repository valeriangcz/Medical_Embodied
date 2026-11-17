# 语音链路（ASR→LLM→TTS）部署说明 —— 精简版 v9（2026-11-09：13→6 进程）

> 本文是当前生产链路的**唯一有效说明**。历史过程见 git log 与各 note 报告；
> 回归工具：`tools/voice_regression/real_pipeline_record_v7.py`。

## 一、当前链路形态

```
问题语音(注入/麦克风) ──► ros_node_keyword_asr (C++, sherpa 流式 zipformer)
                          · 4 线程；语义端点 0.55s / rule2 1.2s
                          ▼
                    /question_asr
                          ▼
              ros2_node_llm_medical（网关，2026-11 并入 4 个节点）
                          · 入口纠错：typo_map v2 + 句首称呼正则
                          · 分流：医学关键词表 + 意图复核（qwen-flash）→ 医疗/闲聊
                          · RAG：进程内 BM25（rag_bm25.py，零 GPU，~1ms）
                          · LLM：云端 qwen-plus-latest（禁思考，流式）；
                            天气闲聊走 dashscope 联网搜索（原 llm_network 能力）
                          · 内嵌 dialog_history（list+服务，口径不变）
                          · 内嵌 /check_end_dialog 服务（原 check_end_session）
                          ▼
                    /ali_llm_output
                          ▼
              ros2_node_tts_cosyvoice（CosyVoice2-0.5B，权重 fp16，占用 1.6GB）
                          · 内嵌断句器（原 stream_text，按标点子句切分）
                          · tts_text_norm + wetext、填充词、首单元流式
                          · tts_text_norm：符号/单位转口语（=、+、%、3-5、mmol/L）
                          · wetext：阿拉伯数字 → 汉语读法（35→三十五）
                          · 填充词"好的，"：[START] 时 0 延迟播预合成缓存
                          · 首单元流式合成（首块即入播）
                          ▼
                    扬声器（/tts_session_finished 结束信号）
```

实测（回归录音 v8）：说完→听到"好的，" **0.66~1.2s**；说完→正式回答 2.4~3.0s；
数字朗读/符号规范化/纠错全链路生效。
**GPU 占用 ~1.6GB（仅 TTS 单进程）；ASR 4 线程；RAG 纯 CPU。
进程数 13 Python → 6（gateway/tts/nurse_alert/dialog_manager/bt_hci + ASR C++），
retired 节点在 src/llm_node_py/llm_node_py/legacy/（含 UI 文字问答通道，UI 移除后停用）。**

## 二、启动方式

### 方式一（推荐）：前台状态终端（工业运行台）

```bash
cd ~/medical_ws/Medical_Embodied_dev
./voice_start.sh
```

- 前台实时仪表盘：7 节点存活/PID/重启次数/CPU/内存 + 流水线状态 + 合并事件流（0.6s 刷新）
- **启动预检**：模型/密钥/索引/音频设备 12 项检查，缺失即拒绝启动并列出清单
- **看门狗自愈**：节点崩溃自动重启（冷却 2/5/10/20/30s，稳定 10 分钟清零；`--no-supervise` 关闭）
- **日志持久化**：`logs/voice_<时间戳>/<节点>.log`（10MB 轮转保留 3 份）
- **单实例锁**：`/tmp/voice_stack.lock`，双栈拒绝启动（双栈=每个回答播两遍）
- 关闭终端 / Ctrl+C = 全链路退出（子进程 PDEATHSIG 双保险）

### 方式二（兼容）：tmux 三会话

```bash
# 0) 关键环境（必须）：PYTHONNOUSERSITE=1
#    原因：2026-10-30 起 ~/.local 里有 numpy 1.26.4 会遮蔽 conda 环境的 2.4.2，
#    导致 transformers 导入即崩。所有启动脚本/终端都要带。
export PYTHONNOUSERSITE=1

# 1) ROS + 工作空间 overlay（rclpy、llm_node_comm 类型支持）
source /opt/ros/jazzy/setup.bash
source ~/medical_ws/Medical_Embodied_dev/install/setup.bash

# 2) 一键拉起（tmux 三会话，脚本内部用 conda env qwen_tts_online 的 python）
cd ~/medical_ws/Medical_Embodied_dev/launch
./launch_llm_nodes.sh        # = depends_asr_question + depends_ui_question + dialog_manager
```

### 启动后验证

```bash
ps aux | grep -cE "ros_node_tts_cosyvoic[e]"   # 必须为 1（双栈=回答播两遍）
```
```bash
# 0) 关键环境（必须）：PYTHONNOUSERSITE=1
#    原因：2026-10-30 起 ~/.local 里有 numpy 1.26.4 会遮蔽 conda 环境的 2.4.2，
#    导致 transformers 导入即崩。所有启动脚本/终端都要带。
export PYTHONNOUSERSITE=1

# 1) ROS + 工作空间 overlay（rclpy、llm_node_comm 类型支持）
source /opt/ros/jazzy/setup.bash
source ~/medical_ws/Medical_Embodied_dev/install/setup.bash

# 2) 一键拉起（tmux 三会话，脚本内部用 conda env qwen_tts_online 的 python）
cd ~/medical_ws/Medical_Embodied_dev/launch
./launch_llm_nodes.sh        # = depends_asr_question + depends_ui_question + dialog_manager
```

- 启动后各节点自动预热（TTS 约 3s、ASR 即时），首个真实请求无冷启动。
- **不再需要**：`tools/rag/rag_service.py`（检索已进程内化）；ollama 服务。
- 回归验证（改完任何环节后必跑）：

```bash
PYTHONNOUSERSITE=1 /home/artrc/miniconda3/envs/qwen_tts_online/bin/python \
  src/llm_node_py/llm_node_py/rag_bm25.py          # Recall@4 必须 100%
PYTHONNOUSERSITE=1 /home/artrc/miniconda3/envs/qwen_tts_online/bin/python \
  src/llm_node_py/llm_node_py/tts_text_norm.py     # 11 用例必须全过
PYTHONNOUSERSITE=1 /home/artrc/miniconda3/envs/qwen_tts_online/bin/python \
  tools/voice_regression/real_pipeline_record_v8.py  # 全链路录音回归（CosyVoice2 版）
```

## 三、配置方式（环境变量）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `RAG_MODE` | `bm25` | `http` 回退旧 127.0.0.1:8765 服务（排障/对比用，需另起 rag_service） |
| `RAG_INDEX_DIR` | `tools/rag/index` | BM25 索引目录（chunks.jsonl + bm25.json） |
| `RAG_TOP_K` / `RAG_CHUNK_MAX_CHARS` | 4 / 350 | 注入 prompt 的资料条数与每块字数 |
| `TTS_FILLER_TEXT` | `好的，` | 固定首句；置空禁用 |
| `TTS_FP16_WEIGHTS` | 1 | CosyVoice2 权重真 fp16（2946→1600MiB，RTF 0.60→0.44；hift 保持 fp32） |
| `PYTORCH_CUDA_ALLOC_CONF` | expandable_segments:True | 节点内自动设置，勿覆盖 |
| `TTS_SPK_ID` / `TTS_PROMPT_WAV` / `TTS_PROMPT_TEXT` / `TTS_SPEED` | xiaoyi / 项目资产 / — / 1.0 | 克隆音色定义（换音色只需换提示音+文本） |
| `MED_LLM_BACKUP_MODEL` | qwen-flash | 备用模型：主模型失败自动切换（`MED_LLM_FAILOVER=0` 关闭） |
| `MED_LLM_READ_TIMEOUT` / `MED_LLM_CONNECT_TIMEOUT` | 4.0 / 3.0 | LLM 分项超时：流式卡顿 4s 快速失败切备用，正常长回答不受影响 |
| `MED_LLM_KEEPALIVE` | 60 | 云端连接保活间隔（秒，0 关闭）——消除建连等待（中位 1.44s→~0） |
| `MED_LLM_PREWARM` | 1 | 启动/唤醒（start_asr）时云端连接预热 |
| `MED_ASR_CORRECT` / `MED_ASR_CORRECT_TIMEOUT` | 1 / 1.0 | ASR 语义纠错层（默认 qwen-flash；`MED_AUX_LOCAL=1` 走本地 llama.cpp，失败回退云端） |
| `MED_RAG_EXPAND` | 1 | RAG 查询扩展：口语改述→检索关键词（默认 qwen-flash；`MED_AUX_LOCAL=1` 走本地） |
| `ASR_SESSION_TIMEOUT` | 15 | 会话静默超时（秒）：长回答场景调大（如 30~60），否则播放中途被判超时 |
| `VOICE_TTS_QMAX` | 15 | TTS 播放队列深度上限（满时背压，超时丢最老保最新） |
| `VOICE_UNLOAD_TOKENIZER` | 1 | 音色注册后卸载 tokenizer 会话（省 RAM；克隆新音色需重启） |
| `TTS_SPLIT_COMMAS` | 0 | 0=句级合成单元（默认，音色稳）；1=逐子句（旧行为，碎片音色漂移） |
| `TTS_LOAD_TRT` | **1** | TRT 加速 flow 解码（引擎已缓存 .plan，默认开启）：RTF 0.44→0.39、显存 1600→2036MiB；省显存时置 0 |
| `TTS_FLOW_FP16` | TRT 关=1 / TRT 开=0 | flow 编码器权重 fp16（与 TRT 的 dtype 兼容默认值已按实测配好） |
| `MED_LLM_PROVIDER` / `MED_LLM_MODEL` | openai / qwen-plus-latest | 可切 dashscope 原生（deepseek-v4.1-flash-1017）、ollama（qwen3:8b）或本地 llama.cpp（见第七节） |
| `MED_LLM_BASE_URL` | dashscope 兼容地址 | 本地 llama.cpp 时设 `http://127.0.0.1:8080/v1` |
| `MED_LLM_CHAT_TEMPLATE_KWARGS` | 空 | 本地 llama.cpp 关思考：`{"enable_thinking": false}`（顶层 enable_thinking 对 llama-server 无效） |
| `MED_AUX_LOCAL` | 0 | 1=ASR纠错+RAG查询扩展优先走本地 llama.cpp（失败回退云端；每题省 ~0.7–1.0s） |
| `MED_ASR_CORRECT_LOCAL` / `MED_RAG_EXPAND_LOCAL` | 跟随 MED_AUX_LOCAL | 单项覆盖开关 |
| `MED_LLM_LOCAL_BASE_URL` / `MED_LLM_LOCAL_MODEL` | http://127.0.0.1:8080/v1 / qwen-local | 本地辅助调用服务地址/模型 |
| `MED_LLM_ENABLE_THINKING` | 0 | 必须保持 0，否则首句晚数秒 |
| `MED_INTENT_JUDGE` / `MED_INTENT_MODEL` | 1 / qwen-flash | 闲聊关键词复核（云端） |
| `ASR_ENDPOINT_RULE1/2/3` | 2.4 / 1.2 / 20 | rule2 **勿低于 1.2**：0.8s 实测会在句中停顿误截断（详见 asr_stream.cpp 注释） |
| `ASR_SEMANTIC_ENDPOINT` / `ASR_SEMANTIC_RULE` | 开 / 0.55 | 语义端点开关与文本稳定阈值 |
| `LATENCY_PROBE` | 0 | 1 时发布 /latency_event 埋点 |

文档更新流程：改 `tools/rag/docs/*.md` → `build_index.py` 重建 `tools/rag/index/` →
跑 `rag_bm25.py` 自测 → 重启 llm_medical（BM25 索引进程内懒加载）。

## 四、部署方式（换机/机器人）

1. **代码**：`git clone https://github.com/valeriangcz/Medical_Embodied.git`
   （或拷贝工作空间），进入 `Medical_Embodied_dev`：

   ```bash
   source /opt/ros/jazzy/setup.bash
   colcon build --packages-select llm_node_cpp llm_node_comm
   ```

2. **模型资产**（不在 git 里，需单独拷贝）：
   - CosyVoice2 运行时 `/home/artrc/cosyvoice_runtime/`（~12GB：模型 4.9GB + deps
     含 tensorrt-cu12 10.13.3.9 与 CUDA12 运行库；节点自举路径，无需 PYTHONPATH）
   - TRT 引擎缓存 `CosyVoice2-0.5B/flow.decoder.estimator.fp16.mygpu.plan`（170MB，
     GPU 相关，换卡后删掉重建，构建仅 78s）与 `src/llm_node_py/cosyvoice_assets/`（提示音）
   - `src/llm_node_py/Qwen3-TTS-12Hz-0.6B-CustomVoice/`（2.4GB，备用引擎）
   - ASR/KWS sherpa 模型目录（`asr_stream.cpp` 的 `asrBasePath` 指向处，
     含 `hotwords_cn.txt`——注：热词在该模型 BPE 配置下实测无效，保留仅作记录）
   - **本地大模型** `models/qwen3.5-2b-glm-deepseek/`（3.3GB：Q4_K_M/Q6_K +
     mmproj）与 **llama.cpp 运行时** `tools/llama.cpp-b12428/`（1.1GB，含
     CUDA 12.8 运行库）。一键包已包含；手工部署时放在 `<根>/models`、`<根>/tools`，
     `voice_env.sh` 会导出 `LOCAL_LLM_DIR / LLAMA_CPP_DIR / LLAMA_CUDA_LIB`
   - `tools/rag/index/`（无则在本机跑 `tools/rag/build_index.py` 生成）
3. **Python 环境**：conda env `qwen_tts_online`（依赖清单
   `llm_python_requirements.txt`，torch 2.14+cu130 / transformers 4.57.3 / numpy==2.4.2）；
   **部署后务必 `pip install --user` 的包清点一遍，严禁用户级 numpy 遮蔽**。
4. **密钥**：`src/llm_node_py/key/dashscope.key`（单行文本）。
5. **音频设备**：播放走 PortAudio 默认输出；离线联调用 PipeWire 回环
   （见 `models/audio_setup.md`）。
6. **验收**：跑第二节的三条自测/回归，并人工试听
   `models/testaudio/qa_fullchain_v7_qwen3tts.wav` 对照效果。

## 五、备用路径（当前链路之外的保留项）

- **Qwen3-TTS 回切**：launch 里把 `ros2_node_tts_cosyvoice.py` 换回
  `ros2_node_tts_local.py`（显存 2.6GB，数字朗读原生正确；无克隆能力）。
- **ollama 本地 LLM 降级**：`MED_LLM_PROVIDER=ollama`（qwen3:8b，断网兜底；
  注意资料约束偏松，只建议闲聊路径）。
- **本地 llama.cpp 模型**：见第七节（qwen3.5-2b 蒸馏模型，速度快但医疗问答质量
  明显低于 qwen-plus，默认不切换）。
- `RAG_MODE=http`：旧 GPU 混合检索（BGE-M3），排障时对比用。


## 六、已知云端特性与近期优化摘要

- **qwen-plus 偶发流式停顿（~3%）**：服务端随机 >4s 无输出（32 次压测实测，`-latest` 别名与固定版无差异）。
  处理链：读超时 4s 快速失败 → 备用模型 qwen-flash 自动切换 → **前缀去重续播**（不重播已播开头）。
  最坏情况从旧配置的 10s 静音+重播整段，压到 ~4s 停顿后无缝续播。
- **qwen3.8-flash 评测结论**（23 题真实链路）：TTFT 555ms（qwen-plus 608ms）但要点覆盖均值 49.7% vs 58.7%，
  1 题边界编造——医疗场景质量优先，默认仍 qwen-plus-latest；切换用 `MED_LLM_MODEL=qwen3.8-flash`（代码已显式关思考）。
- **填充词"好的，"即时响应**：[START] 在问题入口发布，语义纠错/RAG 扩展/云端建连全部与填充词播放并行。
- **合成排空合并**：已排队的后续单元并入同一次合成（≤120 字），消除短首句后的句间空隙。
- **systemd 服务化（无头部署）**：
  ```bash
  sudo cp deploy/voice-chain.service /etc/systemd/system/
  sudo systemctl daemon-reload && sudo systemctl enable --now voice-chain
  ```
  与前台 `./voice_start.sh` 互斥（单实例锁自动仲裁）；开机自启、崩溃 5s 重启、journald + logs/voice_* 双日志。

## 六、一键部署包与脚本（deploy_voice_chain.sh）

仓库根目录的 `deploy_voice_chain.sh` 提供三个子命令：

### 1) 打包（源机）
```bash
cd ~/medical_ws/Medical_Embodied_dev && ./deploy_voice_chain.sh pack
# 产出 ~/medical_ws/deploy_bundle_<日期>.tar.zst（约 20GB）+ .sha256
# 内容：源码树（含 install 预编译产物）+ models/sherpa + Qwen3-TTS 备用引擎
#       + cosyvoice_runtime（含 TRT 引擎与 CUDA 库）+ rag index + dashscope key
#       + 本地模型 qwen3.5-2b-glm-deepseek（3.3GB）
#       + llama.cpp 运行时（含 CUDA12.8 库，1.1GB）
# 排除：build/log/logs/.git、tools/voice_regression/out（测试产物）
# 注：bge-m3 不再打包（BM25 检索不需要；仅 RAG_MODE=http 回退时自备）
```

### 2) 安装（目标机）
```bash
./deploy_voice_chain.sh install deploy_bundle_<日期>.tar.zst [目标根目录] [--build]
# 自动：校验 sha256 → 解压 → 密钥归位 → conda env 检查/创建 → colcon 编译
#       → 生成 voice_env.sh（PYTHONNOUSERSITE + ROS source + TRT 库路径
#         + LOCAL_LLM_DIR / LLAMA_CPP_DIR / LLAMA_CUDA_LIB）
```

### 3) 验证（目标机）
```bash
./deploy_voice_chain.sh verify
# 依次检查：BM25 自测(Recall@4=100%) / 规范化 11 用例 / 关键文件 /
#   本地模型与 llama.cpp 资产 / 2025 资料 32 篇 + 138 chunks /
#   ASR 二进制已更新（无旧提示后缀）/ MED_AUX_LOCAL 支持 / llama.cpp CUDA 设备 /
#   TRT 引擎可加载
```

### 4) 启动（目标机）
```bash
source ~/medical_ws/voice_env.sh
cd ~/medical_ws/Medical_Embodied_dev

# 方式 A：云端答题（默认）
./voice_start.sh --no-bt

# 方式 B：云端答题 + 本地辅助（ASR纠错/RAG扩展走本地，推荐）
./llama_local_start.sh
MED_AUX_LOCAL=1 ./voice_start.sh --no-bt

# 方式 C：全本地模型（断网/降级）
./llama_local_start.sh && ./voice_start_local.sh --no-bt
```

## 七、本地大模型（llama.cpp）接入

用户本地模型 `~/models/qwen3.5-2b-glm-deepseek/`（GGUF, Q4_K_M/Q6_K）由
`llama-server` 提供 OpenAI 兼容接口，网关按普通 openai provider 接入：

```bash
# 0) 安装包环境（导出 LOCAL_LLM_DIR / LLAMA_CPP_DIR / LLAMA_CUDA_LIB）
source ~/medical_ws/voice_env.sh

# 1) 启动推理服务（安装包自带模型与 CUDA 库；已运行则直接跳过）
cd ~/medical_ws/Medical_Embodied_dev
./llama_local_start.sh

# 2) 本地模型启动整条语音链
./voice_start_local.sh              # 等价 voice_start.sh 全部参数
./voice_start_local.sh --no-bt      # 纯语音链
```

手工启动等价命令（源码机路径）：

```bash
llama-server -m ~/models/qwen3.5-2b-glm-deepseek/qwen3.5-2b-glm-deepseek-Q6_K.gguf \
  -ngl 99 -fa on -c 8192 --host 127.0.0.1 --port 8080 -a qwen-local
```

脚本内部导出（也可手工设置）：

```bash
export MED_LLM_PROVIDER=openai
export MED_LLM_BASE_URL=http://127.0.0.1:8080/v1
export MED_LLM_MODEL=qwen-local
export MED_LLM_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'
```

> 必须用 `chat_template_kwargs` 关思考：llama-server 忽略顶层 `enable_thinking`，
> 否则流里只有 `reasoning_content`，TTS 要等整段推理结束才出声。

**实测结论（2026-11-16，23 题同口径）**：本地 2B 首 delta 快约 1.1s、生成快 2.3 倍，
但首回答音频起播与云端持平（TTS 首句合成 ~2s 主导）；要点覆盖率中位 33% vs 云端 67%，
21/23 回答过度触发“资料里没有”，2 题编造。**默认保持 qwen-plus-latest**，
本地模型仅作断网降级/闲聊通道。完整数据见 `note/local_llm_qwen35_2b_eval.md`。

### 本地辅助调用（推荐开启；答题仍走云端）

ASR 语义纠错和 RAG 查询扩展**每题都会调用**，原来各走一次云端（合计 ~0.9–1.3s/题）。
`MED_AUX_LOCAL=1` 时这两项优先走本地 llama.cpp，失败自动回退云端 qwen-flash：

```bash
MED_AUX_LOCAL=1 ./voice_start.sh --no-bt
# 单项覆盖：MED_ASR_CORRECT_LOCAL=1 / MED_RAG_EXPAND_LOCAL=1
```

前置：llama-server 在 `127.0.0.1:8080`（命令见上）。注意必须带 CUDA 运行库，
否则 llama.cpp 会**静默退回 CPU**（辅助调用从 ~130ms 变 500–900ms）：

```bash
CUDA_LIB=/home/artrc/tools/llama.cpp-b12428/cudart-llama-b12428-bin-ubuntu-cuda-12.8-x64
LD_LIBRARY_PATH=$CUDA_LIB llama-server -m ... -ngl 99 -fa on -c 8192 \
  --host 127.0.0.1 --port 8080 -a qwen-local
# 自检：应显示 CUDA0；nvidia-smi 应能看到 llama-server 占 ~2.7GB
```

**实测（2026-11-16，23 题同口径，答题仍为 qwen-plus）**：

| 指标 | 全云端辅助 | MED_AUX_LOCAL=1 |
| --- | ---: | ---: |
| RAG 检索（含扩展） | 558ms | **168ms** |
| LLM 首包（注入→首 delta） | 2689ms | **1202ms** |
| 首回答音频起播 | 3660ms | **2126ms** |
| 端到端 | 18496ms | **15152ms** |
| 辅助调用本地占比 | 0/45 | **45/45（0 回退）** |

质量：覆盖中位 67%、均值 63.1%、编造 0、安全 2/2——与云端基线（67%/64.2%/0/2）持平。
