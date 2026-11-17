#!/bin/bash
set -e

# ================================
# tmux session 名
# ================================
TMUX_SESSION="depends_ui_question"


PYTHON_NODES=(
    # 实时的 tts
    # "../src/llm_node_py/llm_node_py/ros2_node_tts_realtime.py"
    
    # CosyVoice2 TTS（权重 fp16 + 填充词 + 首单元流式 + 内嵌断句器；占用 1.6GB，回退换回下一行）
    "../src/llm_node_py/llm_node_py/ros2_node_tts_cosyvoice.py"
    # "../src/llm_node_py/llm_node_py/ros2_node_tts_local.py"

    # 以下节点已并入 llm_medical 网关 / 随 UI 通道移除（2026-11 架构精简），保留注释便于回退
    # "../src/llm_node_py/llm_node_py/ros2_node_check_end_session.py"
    # "../src/llm_node_py/llm_node_py/ros2_node_stream_text.py"
    # "../src/llm_node_py/llm_node_py/ros2_node_question_manager.py"
    # "../src/llm_node_py/llm_node_py/ros2_node_llm_ui_stream_bridge.py"
    # "../src/llm_node_py/llm_node_py/ros2_node_llm_ui.py"
)

# ================================
# 1️⃣B 定义要启动的 C++ 节点可执行文件
# ================================
CPP_NODES=(

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
