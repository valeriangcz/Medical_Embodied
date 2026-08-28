#!/bin/bash
# conda 环境激活包装脚本（用于 launch 文件中通过 prefix 启动 ROS2 Python 节点）
#
# 用法（launch 中 Node prefix）:
#   prefix=[<本脚本路径>, '_']
# 生成的命令: <本脚本> _ ros2 run monitor bed_detection_server ...
# 脚本激活 conda 环境后 exec 原始命令。
#
# 可用环境变量覆盖默认值:
#   MONITOR_CONDA_HOME: conda 安装根目录（默认 /home/medical/miniconda3）
#   MONITOR_CONDA_ENV:  目标 conda 环境名（默认 monitor）

CONDA_HOME="${MONITOR_CONDA_HOME:-/home/medical/miniconda3}"
CONDA_ENV="${MONITOR_CONDA_ENV:-monitor}"

if [ ! -f "${CONDA_HOME}/etc/profile.d/conda.sh" ]; then
    echo "[conda_activate] 未找到 conda.sh: ${CONDA_HOME}/etc/profile.d/conda.sh" >&2
    echo "[conda_activate] 将直接执行命令（不激活 conda 环境）" >&2
    if [ "$1" = "_" ]; then
        shift
    fi
    exec "$@"
fi

# shellcheck disable=SC1091
source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
echo "[conda_activate] 已激活 conda 环境: ${CONDA_ENV}"

# launch 的 prefix 机制会在命令前插入 '_' 作为 $0 占位, 跳过它
if [ "$1" = "_" ]; then
    shift
fi
exec "$@"
