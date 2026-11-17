# LLM 接入方式切换：dashscope SDK + MaaS 专有部署（2026-10-30）

## 一、结论

| 项目 | 切换前 | 切换后 |
| --- | --- | --- |
| 接入方式 | `openai` SDK 直连 dashscope 兼容接口 | `dashscope` SDK 原生接口（`MultiModalConversation.call`） |
| 服务地址 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `https://llm-wd6fnilu0sk3blx6.cn-beijing.maas.aliyuncs.com/api/v1`（专有 MaaS 部署） |
| 模型 | qwen-plus-latest | deepseek-v4.1-flash-1017 |
| 首块 TTFT 中位 | 728 ms（kb50，n=50） | 681 ms（ds2_kb10，n=10） |
| 问句→首音 中位 | 1503 ms（kb50） | 1746 ms（ds2_kb10） |
| 首块→结束 中位 | 660 ms（kb50） | 328 ms（生成明显更快） |
| 拒答 | 10/50（6 条资料确实没有，4 条检索漏） | 0/10（放宽规则后） |

一句话：**接入方式、模型都换了，首块速度基本持平、生成更快；新模型更"守规矩"，需要放宽拒答规则，否则会把"资料没有逐字回答"当成没资料。**

## 二、地址与协议实测

切换到专有 MaaS 地址后，先确认哪些路径可用：

| 请求 | 结果 |
| --- | --- |
| `POST {host}/api/v1/services/aigc/text-generation/generation` | 400 `InvalidParameter: url error, please check url！` |
| `POST {host}/api/v1/services/aigc/multimodal-generation/generation` | **200**（原生多模态路径可用 → `MultiModalConversation` 可行） |
| `POST {host}/compatible-mode/v1/chat/completions` | **200**（OpenAI 兼容可用，作为回退方案） |
| `{host}/v1/chat/completions`、`{host}/api/v1/chat/completions` | 404 |

主机名注意：必须是 `llm-wd6fnilu0sk3blx6`（第七位是**数字 0**）。写成字母 o 的 `llm-wd6fniluosk3blx6` 会返回 403。

## 三、思考模型的关键开关

`deepseek-v4.1-flash-1017` 是思考模型，默认先吐 `reasoning_content`，正文要等推理结束才出来（80 token 内可能一个字正文都没有）。语音链路等不起。

- `enable_thinking=False` **生效**：首正文 465 ms、总 759 ms（实测「我是DeepSeek…」）。
- `thinking={"type":"disabled"}` **无效**（仍会推理）。

节点默认 `MED_LLM_ENABLE_THINKING=0`，即关闭思考；需要推理能力时设 1，但首句时延会明显变差。

## 四、实现要点

文件：`src/llm_node_py/llm_node_py/ros2_node_llm_medical.py`

1. `MED_LLM_PROVIDER=dashscope|openai` 分发：
   - `dashscope` → `_stream_dashscope()`，用 `MultiModalConversation.call(..., stream=True, incremental_output=True, enable_thinking=...)`；
   - `openai` → `_stream_openai()`（原 qwen-plus 实现，保留做回退）。
2. `dashscope.base_http_api_url = MED_LLM_BASE_URL`；消息转成 `{"role":..,"content":[{"text":..}]}`。
3. 流式消费放在 worker 线程，结果经 `queue.Queue` 回到主循环：首块最多等 `MED_LLM_TIMEOUT`（默认 10 s）、之后每块 5 s，超时/报错播兜底话术并正常 `[DONE]`，不会卡死整轮。
4. 只发正文：`reasoning_content` 丢弃；表情符号与中文引号继续过滤。
5. judge（闲聊复核）与主模型解耦：`MED_INTENT_BASE_URL` 默认仍指向公网兼容接口，`qwen-flash` 不受 MaaS 地址影响。
6. 旧百炼应用路径（`MED_LLM_BACKEND=app`）在调用前临时把 `base_http_api_url` 切回公网，调用完还原。

## 五、实测数据

链路：`dashscope` provider、`top_k=3`、`chunk 300`、judge `qwen-flash`、闲聊快速通道、`LATENCY_PROBE=1`。

### 5.1 知识库组（kb10，10 轮全 ok，单位 ms）

| 阶段 | mean | median | p90 |
| --- | --- | --- | --- |
| RAG 检索 | 13 | 12 | 16 |
| LLM 首块 TTFT | 837 | 681 | 1305 |
| 首块→[DONE] | 378 | 328 | 481 |
| 首块→首段文本 | 78 | 71 | 92 |
| TTS 首句合成 | 807 | 758 | 1202 |
| 合成完成→播放 | 44 | 48 | 79 |
| **问句→首音** | **1767** | **1746** | **2182** |
| 会话总时长 | 14085 | 14275 | 16361 |

对比 qwen-plus 的 kb50（n=50）：TTFT 859/728、问句→首音 1685/1503。首块持平，TTS 首句略慢（回答更长），生成段快一倍。

### 5.2 闲聊组（chat10，10 轮全 ok）

| 阶段 | mean | median | p90 |
| --- | --- | --- | --- |
| LLM 首块 TTFT | 855 | 784 | 1144 |
| TTS 首句合成 | 591 | 550 | 723 |
| **问句→首音** | **1600** | **1648** | **1830** |

10 轮全部走闲聊快速通道（`source=chitchat`），没有调 judge、没有检索。

## 六、新模型的"过度拒答"与修复

首轮 ds_kb10 拒答 4/10，且检索 top1 都是相关内容：

| 问题 | 检索 top1 | 修复后回答 |
| --- | --- | --- |
| 我下周第一次透析，去之前要准备什么？ | 首次透析要注意什么 | 「放轻松就好。先带齐水杯、纸巾、医疗卡、透析卡、小点心和止血带…」 |
| 第一次透析会很疼吗？ | 首次透析要注意什么 | 「不用太担心。首次透析会用短时间、慢血流量慢慢适应，一般不会很疼…」 |
| 透析的时候能吃东西吗？ | 透析患者该如何运动 | 「尽量别在透析中吃东西。这样能减少低血压、抽筋等不适…」 |
| 干体重是什么意思？ | 控水文档 | 仍拒答（资料里没有定义，属知识库缺口） |

原因：新模型严格照「资料里没有就说没有」，把"资料没有逐字回答"当成没资料。修复两处：

1. `MED_SYSTEM_PROMPT` 规则 1 改为：资料里有相关内容（哪怕不是逐字回答）就用自己的话讲清楚、给可执行建议；只有完全没提到这个话题才拒答；不编造具体数值/剂量。
2. `tools/rag/retriever.py` 的 `COLLOQUIAL_MAP` 补口语别名：第一次→首次、准备/去之前→透析前 准备 注意事项 带好物品、吃东西/饿了→进食 透析中进食。

复测（ds_fix5 + ds2_kb10）：拒答 4/10 → **0/10**；「透析患者能喝咖啡吗」仍按规则拒答（资料确实没有咖啡内容，符合"医学问题只依据资料"的设计）。

代价：回答更完整，音频从 8.8 s 变 12.3 s、会话从 10.5 s 变 14.1 s——是"答全"和"答快"的取舍，需要更短可调 `MED_LLM_MAX_TOKENS` 或把 prompt 的"1-3 句 80 字"再收紧。

## 七、环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `MED_LLM_PROVIDER` | `dashscope` | `dashscope`=SDK 原生（MaaS）｜`openai`=兼容接口 |
| `MED_LLM_MODEL` | `deepseek-v4.1-flash-1017` | dashscope 模式默认值 |
| `MED_LLM_BASE_URL` | MaaS 地址 | 专有部署地址（`.../api/v1`） |
| `MED_LLM_ENABLE_THINKING` | `0` | 思考开关；语音链路必须 0 |
| `MED_LLM_TIMEOUT` | `10` | 首块等待上限（秒），超时播兜底话术 |
| `MED_LLM_MAX_TOKENS` | `400` | 单轮输出上限 |
| `MED_INTENT_BASE_URL` | 公网 compatible-mode | judge 用的地址（与主模型解耦） |
| `MED_INTENT_MODEL` | `qwen-flash` | 意图复核模型 |

## 八、回退与复测

```bash
# 回退到 qwen-plus（OpenAI 兼容直连）
MED_LLM_PROVIDER=openai MED_LLM_MODEL=qwen-plus-latest \
MED_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1 \
bash tools/latency_probe/start_stack.sh

# 复测（10 条知识库问题）
source /opt/ros/jazzy/setup.bash && source /home/artrc/medical_ws/setup_env.sh
$PY tools/latency_probe/latency_probe.py --inject-file /tmp/bench/kb10.txt \
  --tag ds2_kb10 --repeat 0 --interval 2.0 --timeout 45 --warmup 3.0
```

记录在 `tools/latency_probe/records/ds_kb10_*`、`ds_chat10_*`、`ds_fix5_*`、`ds2_kb10_*`。
