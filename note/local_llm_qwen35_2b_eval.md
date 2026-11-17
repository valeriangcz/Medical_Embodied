# 本地 qwen3.5-2b-glm-deepseek 接入与效果对比（2026-11-16）

## 一、接入方式

用户本地模型：`~/models/qwen3.5-2b-glm-deepseek/`
- `qwen3.5-2b-glm-deepseek-Q6_K.gguf`（1.56GB，本次评测用）
- `qwen3.5-2b-glm-deepseek-Q4_K_M.gguf`（1.27GB，更省显存）
- `mmproj-f16.gguf`（视觉投影，语音链纯文本不用）

推理服务：llama.cpp `b12428` 的 `llama-server`（用户已在 8080 启动）：

```bash
llama-server -m ~/models/qwen3.5-2b-glm-deepseek/qwen3.5-2b-glm-deepseek-Q6_K.gguf \
  --mmproj ~/models/qwen3.5-2b-glm-deepseek/mmproj-f16.gguf \
  -ngl 99 -fa on -c 8192 --host 127.0.0.1 --port 8080 -a qwen-local
```

网关接入（OpenAI 兼容路径，无需改 llama.cpp 服务）：

```bash
export MED_LLM_PROVIDER=openai
export MED_LLM_BASE_URL=http://127.0.0.1:8080/v1
export MED_LLM_MODEL=qwen-local
export MED_LLM_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'
```

> 关键点：llama-server 不认云端顶层 `enable_thinking=false`，思考会照常输出到
> `reasoning_content`，正文为空；必须在请求里带 `chat_template_kwargs`
> （网关已加 `MED_LLM_CHAT_TEMPLATE_KWARGS` 支持）。启动脚本：
> `./voice_start_local.sh`（自动检查 8080 并导出上述变量）。

## 二、效果对比（同一天、同一 23 题、同一 RAG/prompt/judge）

| 指标 | 云端 qwen-plus-latest | 本地 qwen3.5-2b Q6_K |
| --- | ---: | ---: |
| 要点覆盖率 中位 | **67%** | 33% |
| 要点覆盖率 均值 | **64.2%** | 40.6% |
| 覆盖率最低 | 25% | 0% |
| 覆盖率 100% 题数 | 5/23 | 4/23 |
| 安全话术 | **2/2** | **2/2** |
| LLM judge 判编造 | **0 题** | 2 题 |
| 回答含"资料里没有" | 0/23 | 21/23 |
| 回答长度 中位/最长 | 55 / 84 字 | 67 / 115 字 |
| 超过 80 字（违反播报约束） | 1/23 | 7/23 |

**典型差异**：本地模型过度触发"资料里没有"兜底（21/23），即使检索片段里有相关内容
（如"针眼 24 小时"）也不会用自己的话归纳；云端会归纳。另有 2 题给固定数值（饮水量
500ml、抽筋归因）被判编造。安全话术两者都正确。

## 三、延迟对比（同 23 题、同栈；单位 ms）

| 阶段 | 云端 | 本地 | 结论 |
| --- | ---: | ---: | --- |
| RAG（含云端查询扩展） | 558 | 508 | 持平 |
| LLM 首包（注入→首 delta） | 2689 | **1624** | 本地快约 1.1s（-40%） |
| LLM 生成（首 delta→DONE） | 915 | **395** | 本地快 2.3 倍 |
| 首回答音频起播 | 3660 | 3662 | **基本持平**（TTS 首句合成 ~2s 抵消了 LLM 增益） |
| 合成总耗时（每单元合计） | 5570 | 7865 | 本地更久（回答更长、单元更多） |
| 端到端（注入→播报完成） | **18496** | 23791 | 本地反而慢 29% |
| 23 题总耗时 | 425s | 633s | 本地慢 49% |

结论：本地模型把"首字"提前了约 1.1s，但首句音频起播由 TTS 合成主导，实际听感提升
约等于 0；由于回答更长、更容易违反 80 字约束，整段播放反而更久。

## 四、资源占用

- 本地 llama-server（Q6_K + mmproj，8k ctx）：**显存 2766MiB**，RSS 3.48GB（含 mmap）
- 云端 qwen-plus：本地显存/RAM 0，但依赖公网（~3% 服务端流式停顿）
- TTS/ASR 不受影响（同一套 CosyVoice2 TRT）

## 五、结论与建议

1. **默认仍用 qwen-plus-latest**：医疗问答的要点覆盖、资料归纳、防编造明显更好。
2. 本地 2B 只建议作为**断网降级**或**闲聊通道**，不建议承担医学问答主模型。
3. 若必须全本地，优先做提示词精简/微调（当前系统提示对 2B 偏长，且"资料里没有"约束
   被过度执行），或换 7B~8B 级本地模型；仅靠换 2B 无法达到云端质量。
4. 本地延迟的真正瓶颈已不是 LLM，而是 TTS 首句合成（~2s）与回答长度控制。

## 六、复现

```bash
# 终端 A：llama-server（略，见上）
# 终端 B：
cd ~/medical_ws/Medical_Embodied_dev
./voice_start_local.sh --headless --no-bt
LATENCY_PROBE=1          # 已由 voice_start_local.sh 之前的栈导出亦可

# 质量（栈运行时）
python3 tools/voice_regression/quality_eval.py
# 延迟
python3 tools/voice_regression/measure_100q.py --limit 23
python3 tools/voice_regression/analyze_100q.py
```

产物：`tools/voice_regression/out/` 下 `local_quality_summary.md`、
`local_measure_*.{md,csv,json,jsonl}`、`cloud_today_quality_*`、`cloud_today_measure_*`。
