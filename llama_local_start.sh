#!/usr/bin/env bash
# =====================================================================
# 启动本地 llama.cpp 推理服务（qwen3.5-2b-glm-deepseek）
#
#   ./llama_local_start.sh              # 默认 127.0.0.1:8080，Q6_K
#   LOCAL_LLM_GGUF=.../Q4_K_M.gguf ./llama_local_start.sh   # 省显存
#
# 安装包布局：<根>/models/qwen3.5-2b-glm-deepseek
#             <根>/tools/llama.cpp-b12428/{llama-b12428,cudart-...}
# 源码机兼容：模型找不到时回退 ~/models，llama.cpp 回退 ~/tools。
# 已启动（8080 health OK）时直接退出，可安全重复执行。
# =====================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

DEFAULT_MODEL="$ROOT/models/qwen3.5-2b-glm-deepseek"
[ -d "$DEFAULT_MODEL" ] || DEFAULT_MODEL="$HOME/models/qwen3.5-2b-glm-deepseek"
DEFAULT_LLAMA="$ROOT/tools/llama.cpp-b12428"
[ -d "$DEFAULT_LLAMA" ] || DEFAULT_LLAMA="$HOME/tools/llama.cpp-b12428"

MODEL_DIR="${LOCAL_LLM_DIR:-$DEFAULT_MODEL}"
LLAMA_ROOT="${LLAMA_CPP_ROOT:-$DEFAULT_LLAMA}"
LLAMA_BIN="${LLAMA_CPP_DIR:-$LLAMA_ROOT/llama-b12428}/llama-server"
CUDA_LIB="${LLAMA_CUDA_LIB:-$LLAMA_ROOT/cudart-llama-b12428-bin-ubuntu-cuda-12.8-x64}"
GGUF="${LOCAL_LLM_GGUF:-$MODEL_DIR/qwen3.5-2b-glm-deepseek-Q6_K.gguf}"
MMPROJ="${LOCAL_LLM_MMPROJ:-$MODEL_DIR/mmproj-f16.gguf}"
PORT="${LOCAL_LLM_PORT:-8080}"
ALIAS="${LOCAL_LLM_ALIAS:-qwen-local}"
LOG="${LOCAL_LLM_LOG:-$ROOT/logs/llama_local.log}"

info() { echo "[llama-local] $*"; }
die() { echo "[llama-local][错误] $*" >&2; exit 1; }

if curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    info "已有服务在 127.0.0.1:$PORT 运行，跳过启动"
    exit 0
fi
[ -x "$LLAMA_BIN" ] || die "找不到 llama-server: $LLAMA_BIN（可设 LLAMA_CPP_DIR）"
[ -f "$GGUF" ] || die "找不到模型: $GGUF（可设 LOCAL_LLM_GGUF）"
[ -f "$MMPROJ" ] || die "找不到 mmproj: $MMPROJ（可设 LOCAL_LLM_MMPROJ）"
[ -d "$CUDA_LIB" ] || die "找不到 CUDA 运行库: $CUDA_LIB（可设 LLAMA_CUDA_LIB）"

mkdir -p "$(dirname "$LOG")"
info "启动: $GGUF"
info "CUDA: $CUDA_LIB"
setsid nohup env LD_LIBRARY_PATH="$CUDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$LLAMA_BIN" -m "$GGUF" --mmproj "$MMPROJ" \
    -ngl 99 -fa on -c "${LOCAL_LLM_CTX:-8192}" \
    --host 127.0.0.1 --port "$PORT" -a "$ALIAS" \
    > "$LOG" 2>&1 < /dev/null &
PID=$!
info "PID=$PID，日志: $LOG"

for _ in $(seq 1 60); do
    if curl -sf -m 1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        info "服务就绪: http://127.0.0.1:$PORT/v1 (model=$ALIAS)"
        if ! nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
                | grep -qw "$PID"; then
            info "[警告] nvidia-smi 未见该进程：可能退回 CPU（检查 CUDA 库）"
        fi
        exit 0
    fi
    sleep 1
done
die "启动超时，请查看 $LOG"
