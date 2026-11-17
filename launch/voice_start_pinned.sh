#!/usr/bin/env bash
# =====================================================================
# 本机专用：语音链路 CPU/GPU 绑定启动（新增文件，不属于机器人同步树）
#
#   ./launch/voice_start_pinned.sh --no-bt     # 其余参数原样透传
#
# - GPU: CUDA_VISIBLE_DEVICES 锁定语音合成用卡（多卡时防止跑错）
# - CPU: TTS（语音合成）独占 2 个物理 P 核（逻辑 0,1,6,7）；
#        ASR 绑到其余核（2-5,8-15），避免流式推理干扰合成喂卡
# - 看护循环每 10s 重申绑定，覆盖节点崩溃被 watchdog 重启后掉亲和力的场景；
#   voice_terminal.py 退出后看护自动结束
# - 核位可通过环境变量覆盖: TTS_CORES=0-1,6-7 ASR_CORES=2-5,8-15
# =====================================================================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
TTS_CORES="${TTS_CORES:-0-1,6-7}"
ASR_CORES="${ASR_CORES:-2-5,8-15}"
export TTS_CORES ASR_CORES

# 绑定看护（自守护：链路退出即退出；重复启动无害，taskset 幂等）
nohup bash -c '
  while pgrep -f voice_terminal.py >/dev/null; do
    for p in $(pgrep -f ros2_node_tts_cosyvoice); do taskset -apc "$TTS_CORES" "$p" >/dev/null 2>&1; done
    for p in $(pgrep -f ros_node_keyword_asr);    do taskset -apc "$ASR_CORES" "$p" >/dev/null 2>&1; done
    sleep 10
  done
' >/dev/null 2>&1 & disown

cd "$(dirname "$0")/.."
exec ./voice_start.sh "$@"
