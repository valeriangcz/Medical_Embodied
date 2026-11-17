#!/bin/bash
set -e

# ================================
# 获取当前脚本所在目录
# ================================
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[INFO] Base directory: $BASE_DIR"

# ================================
# 单实例保护：与 voice_terminal.py 共用 /tmp/voice_stack.lock
# （双栈同时运行会导致每个问题被回答两次）
# ================================
exec 9>/tmp/voice_stack.lock
if ! flock -n 9; then
    echo "[ERROR] 已有语音链路在运行（voice_terminal 或另一套 tmux 栈），拒绝重复启动。"
    exit 1
fi

# 清理可能残留的节点（与语音终端行为一致）
pkill -f "ros2_node[_]" 2>/dev/null || true
pkill -f "keyword_as[r]" 2>/dev/null || true
pkill -f "bt_hci_interfac[e]" 2>/dev/null || true
sleep 0.5

# ================================
# 参数检查
# ================================
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <tmux_session> <node1> <node2> ..."
    exit 1
fi

# ================================
# tmux session
# ================================
TMUX_SESSION="$1"
shift

# ================================
# conda 环境
# ================================
CONDA_ENV_NAME="qwen_tts_online"
PYTHON_EXEC="$(conda run -n "$CONDA_ENV_NAME" which python)"

if [[ ! -x "$PYTHON_EXEC" ]]; then
    echo "[ERROR] Cannot find python in env: $CONDA_ENV_NAME"
    exit 1
fi
echo "[INFO] Using python: $PYTHON_EXEC"

# ================================
# 运行环境（直接前缀到每个节点的命令行，避免 tmux 服务器环境不继承）
# ================================
# PYTHONNOUSERSITE=1 必须：~/.local 的旧 numpy 会遮蔽 conda 环境的 2.4.2，
# 导致 transformers/CosyVoice2 导入即崩（2026-10-30 起）。
export PYTHONNOUSERSITE=1
# sherpa ASR/KWS/标点模型根目录（开发机与部署机均为 <ws>/../models/sherpa 布局）
MODELS_DIR="$(cd "$BASE_DIR/../.." && pwd)/models"
SHERPA_MODEL_ROOT="${SHERPA_MODEL_ROOT:-$MODELS_DIR/sherpa}"
PUNCT_MODEL_PATH="${PUNCT_MODEL_PATH:-$SHERPA_MODEL_ROOT/sherpa-onnx-punct-model/model.onnx}"
echo "[INFO] SHERPA_MODEL_ROOT: $SHERPA_MODEL_ROOT"
PY_ENV_PREFIX="PYTHONNOUSERSITE=1"
CPP_ENV_PREFIX="SHERPA_MODEL_ROOT=$SHERPA_MODEL_ROOT PUNCT_MODEL_PATH=$PUNCT_MODEL_PATH"

# ================================
# 如果 session 已存在
# ================================
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "[WARN] Killing existing session: $TMUX_SESSION"
    tmux kill-session -t "$TMUX_SESSION"
fi

# ================================
# 创建 session
# ================================
tmux new-session -d -s "$TMUX_SESSION"

WINDOW_INDEX=0

# ================================
# 启动节点
# ================================
for NODE in "$@"
do
    # -------------------------------
    # 转换成绝对路径（基于脚本目录）
    # -------------------------------
    if [[ "$NODE" == /* ]]; then
        # 已经是绝对路径
        ABS_NODE="$NODE"
    else
        ABS_NODE="$BASE_DIR/$NODE"
    fi

    if [[ "$WINDOW_INDEX" -eq 0 ]]; then
        TARGET="$TMUX_SESSION"
    else
        tmux new-window -t "$TMUX_SESSION"
        TARGET="$TMUX_SESSION:$WINDOW_INDEX"
    fi

    echo "[INFO] Starting: $ABS_NODE"

    # -------------------------------
    # 判断 Python / C++ 执行方式
    # -------------------------------
    if [[ "$ABS_NODE" == *.py ]]; then
        tmux send-keys -t "$TARGET" "$PY_ENV_PREFIX $PYTHON_EXEC $ABS_NODE" C-m
    else
        tmux send-keys -t "$TARGET" "$CPP_ENV_PREFIX $ABS_NODE" C-m
    fi

    WINDOW_INDEX=$((WINDOW_INDEX + 1))
done

# ================================
# 提示信息
# ================================
echo
echo "[INFO] All nodes started in tmux session: $TMUX_SESSION"
echo "[INFO] Attach: tmux attach -t $TMUX_SESSION"
echo "[INFO] Detach: Ctrl+b then d"