#!/usr/bin/env bash
# =====================================================================
# 语音链路一键启动（前台状态终端）：ASR → 网关(LLM) → TTS 全链路
#
#   ./voice_start.sh            # 完整链路（含行为树两节点）
#   ./voice_start.sh --no-bt    # 纯语音链（不起 dialog_manager/bt_hci）
#
# - 所有节点是本脚本所启仪表盘的子进程
# - 关闭本终端 或 Ctrl+C = 退出并关闭全部节点
# - 启动前自动清理上次可能遗留的节点进程
# - 唤醒词: 小医小医；BT 主动对话: tools/voice_regression/bt_flow_sim_client.py
# =====================================================================
set -e
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE"

# 清理残留（防上次异常退出遗留节点抢麦克风/GPU）
pkill -f "ros2_node[_]" 2>/dev/null || true
pkill -f "keyword_as[r]" 2>/dev/null || true
pkill -f "bt_hci_interfac[e]" 2>/dev/null || true
sleep 0.5

# exec：仪表盘进程替换本 shell，关闭行为直达全链路
exec python3 voice_terminal.py "$@"
