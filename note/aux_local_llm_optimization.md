# ASR 纠错 + RAG 查询扩展本地化（2026-11-16）

## 背景

ASR 语义纠错与 RAG 查询扩展**每题都调用**，原先各走一次云端 qwen-flash：
- RAG 查询扩展：中位 ~520ms
- ASR 语义纠错：~380–700ms，偶发 1s 超时

两项合计每题 ~0.9–1.3s 的云端往返。本地 qwen3.5-2b（llama.cpp）在
8080 常驻后，这两类短任务（输出 ≤100 token）本地只需 ~120–200ms。

## 实现

网关新增 `MED_AUX_LOCAL=1`：ASR 纠错 + RAG 扩展优先调本地 llama.cpp
（OpenAI 兼容接口 + `chat_template_kwargs` 关思考），本地超时/报错自动
回退云端 qwen-flash；意图 judge 仍保持云端（准确率优先）。

```bash
# 本地推理服务（必须带 CUDA 库，否则静默退回 CPU）
CUDA_LIB=/home/artrc/tools/llama.cpp-b12428/cudart-llama-b12428-bin-ubuntu-cuda-12.8-x64
LD_LIBRARY_PATH=$CUDA_LIB llama-server \
  -m ~/models/qwen3.5-2b-glm-deepseek/qwen3.5-2b-glm-deepseek-Q6_K.gguf \
  --mmproj ~/models/qwen3.5-2b-glm-deepseek/mmproj-f16.gguf \
  -ngl 99 -fa on -c 8192 --host 127.0.0.1 --port 8080 -a qwen-local

# 语音链（答题仍走云端 qwen-plus）
cd ~/medical_ws/Medical_Embodied_dev
MED_AUX_LOCAL=1 ./voice_start.sh --no-bt
```

单项开关：`MED_ASR_CORRECT_LOCAL=1` / `MED_RAG_EXPAND_LOCAL=1`
（默认跟随 `MED_AUX_LOCAL`）。本地服务地址/模型可用
`MED_LLM_LOCAL_BASE_URL` / `MED_LLM_LOCAL_MODEL` 覆盖。

## 实测结果（23 题同口径，答题模型 qwen-plus-latest）

| 指标 | 全云端辅助 | 本地辅助 | 变化 |
| --- | ---: | ---: | ---: |
| RAG 检索（含扩展） | 558ms | **168ms** | -390ms |
| LLM 首包（注入→首 delta） | 2689ms | **1202ms** | **-1487ms（-55%）** |
| LLM 生成 | 915ms | 793ms | 同量级（抖动） |
| 首回答音频起播 | 3660ms | **2126ms** | **-1534ms（-42%）** |
| 端到端（注入→播完） | 18496ms | **15152ms** | -3344ms |
| 23 题总耗时 | 425s | **348s** | -18% |
| 辅助调用本地占比 | — | **45/45，0 回退** | 本地中位 125ms |

质量（同 23 题、同 judge）：

| 指标 | 全云端辅助 | 本地辅助 |
| --- | ---: | ---: |
| 要点覆盖 中位 | 67% | **67%** |
| 要点覆盖 均值 | 64.2% | 63.1% |
| 编造（judge） | 0 | **0** |
| 安全话术 | 2/2 | **2/2** |
| 覆盖率 100% 题数 | 5/23 | 6/23 |

个别语义题（睡不着/耳穴）的覆盖率波动来自路由与评分措辞，非辅助调用回退；
整体均值差异在采样噪声内。

## 注意事项

1. **CUDA 库**：llama.cpp 的 `libggml-cuda.so` 依赖 `libcudart/cublas 12`，
   未设置 `LD_LIBRARY_PATH` 时不会报错，会静默用 CPU（辅助调用 500–900ms）。
   启动后检查 `nvidia-smi` 是否出现 llama-server（约 2.7GB）。
2. **本地服务不可用**：自动回退云端，链路不受影响，只是恢复原来的 ~0.5s/次；
   `aux_llm_error` 埋点会记录 `provider=local, Connection error`。
3. 本地服务单槽（`-np 1`）；纠错与扩展串行调用，不会互相排队。
4. 该优化不改变回答模型：医学问答仍是 qwen-plus，质量风险仅在预处理。

## 复现

```bash
./voice_start.sh --no-bt              # 全云端辅助基线
MED_AUX_LOCAL=1 ./voice_start.sh --no-bt   # 本地辅助
python3 tools/voice_regression/measure_100q.py --limit 23
python3 tools/voice_regression/analyze_100q.py
python3 tools/voice_regression/quality_eval.py
```

产物：`tools/voice_regression/out/auxlocal_measure_*`、`auxlocal_quality_*`。
