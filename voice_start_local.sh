#!/usr/bin/env bash
# =====================================================================
# 本地大模型（llama.cpp OpenAI 兼容服务）启动语音链
#
#   ./voice_start_local.sh              # = voice_start.sh 全部能力
#   ./voice_start_local.sh --no-bt      # 纯语音链
#
# 前置：先启动 llama-server（示例，Q6_K + GPU 全量卸载）：
#   CUDA_LIB=/home/artrc/tools/llama.cpp-b12428/cudart-llama-b12428-bin-ubuntu-cuda-12.8-x64
#   LD_LIBRARY_PATH=$CUDA_LIB llama-server \
#     -m ~/models/qwen3.5-2b-glm-deepseek/qwen3.5-2b-glm-deepseek-Q6_K.gguf \
#     -ngl 99 -fa on -c 8192 --host 127.0.0.1 --port 8080 -a qwen-local
# 注意：不设 LD_LIBRARY_PATH 时 llama.cpp 会静默退回 CPU（辅助调用会慢 4~6 倍）。
#
# 可用环境变量覆盖：
#   LOCAL_LLM_URL   默认 http://127.0.0.1:8080/v1
#   LOCAL_LLM_MODEL 默认 qwen-local（llama-server 不校验名称）
# =====================================================================
set -e
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LLM_URL="${LOCAL_LLM_URL:-http://127.0.0.1:8080/v1}"
LLM_URL="${LLM_URL%/}"
LLM_HEALTH="${LLM_URL%/v1}/health"
MODEL="${LOCAL_LLM_MODEL:-qwen-local}"

if ! curl -sf -m 2 "$LLM_HEALTH" >/dev/null 2>&1; then
    echo "[local-llm] 未检测到本地推理服务：$LLM_HEALTH"
    echo "[local-llm] 请先启动 llama-server，例如："
    echo "  llama-server -m ~/models/qwen3.5-2b-glm-deepseek/qwen3.5-2b-glm-deepseek-Q6_K.gguf \\"
    echo "    -ngl 99 -fa on -c 8192 --host 127.0.0.1 --port 8080 -a qwen-local &"
    exit 1
fi

# llama.cpp 不认云端顶层 enable_thinking，必须在 chat_template_kwargs 里关思考，
# 否则流里只有 reasoning_content，TTS 要等整段推理完才出声。
export MED_LLM_PROVIDER=openai
export MED_LLM_BASE_URL="$LLM_URL"
export MED_LLM_MODEL="$MODEL"
export MED_LLM_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'

echo "[local-llm] 使用本地模型 $MODEL @ $LLM_URL（thinking=off）"
exec "$BASE/voice_start.sh" "$@"
