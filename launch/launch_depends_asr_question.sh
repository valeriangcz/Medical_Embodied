#!/bin/bash
set -e

# ================================
# tmux session 名
# ================================
TMUX_SESSION="depends_asr_question"


# ================================
# Python 节点列表
# ================================
PYTHON_NODES=(
    # 是否需要呼叫护士
    "../src/llm_node_py/llm_node_py/ros2_node_nurse_alert.py"
    # 唤醒桥：/call_signal -> /start_asr。精简栈没有行为树，没有它真人唤醒后
    # ASR 永远处于 pause（2026-11-09 排障确认）。真机有行为树时可注释掉本行。
    "../src/llm_node_py/llm_node_py/ros2_node_wake_bridge.py"
    # 大模型网关（已并入 router 纠错/分流 + dialog_history + check_end_dialog，
    # 直接消费 /question_asr，2026-11 架构精简）
    "../src/llm_node_py/llm_node_py/ros2_node_llm_medical.py"
    # 以下四个节点已并入 llm_medical 网关，保留注释便于回退
    # "../src/llm_node_py/llm_node_py/ros2_node_llm_network.py"
    # "../src/llm_node_py/llm_node_py/ros2_node_question_router.py"
    # "../src/llm_node_py/llm_node_py/ros2_node_dialog_history.py"
)


# ================================
# C++ 节点列表
# ================================
CPP_NODES=(
    # ASR 节点
    "../install/llm_node_cpp/lib/llm_node_cpp/ros_node_keyword_asr"
    # ASR, KWS 的管理节点
    # "../../../devel/lib/llm_node/ros_node_keyword_asr"
)


# ================================
# launch 脚本
# ================================
LAUNCH_SCRIPT="./launch_ros_nodes.sh"

# ================================
# 获取当前脚本所在目录
# ================================
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[INFO] Base directory: $BASE_DIR"

# ================================
# launch 脚本路径（绝对路径）
# ================================
LAUNCH_SCRIPT="$BASE_DIR/launch_ros_nodes.sh"


# ================================
# 检查 launch 脚本
# ================================
if [[ ! -f "$LAUNCH_SCRIPT" ]]; then
    echo "[ERROR] launch_ros_nodes.sh not found"
    exit 1
fi


echo "[INFO] Starting dialog system..."
echo "[INFO] tmux session: $TMUX_SESSION"


# ================================
# 合并节点列表
# ================================
ALL_NODES=("${PYTHON_NODES[@]}" "${CPP_NODES[@]}")


# ================================
# 调用 launch
# ================================
$LAUNCH_SCRIPT "$TMUX_SESSION" "${ALL_NODES[@]}"


echo
echo "[INFO] Dialog system started"
echo "[INFO] Attach tmux:"
echo "tmux attach -t $TMUX_SESSION"

