#!/usr/bin/env bash
# Rich optical-flow video features for the held-out barn (Room 6).
# GoPro timestamps come from embedded metadata (ffprobe creation_time + SMPTE
# timecode) — camera-consistent across rooms, so no per-recorder change needed.
#
# Usage (in tmux):
#   tmux new -s room6vid
#   bash crossbarn/run_video.sh "/mnt/<video-drive>/<...>/Room6"
#     arg1 = parent dir CONTAINING the dated GoPro session subfolders for Room 6
set -euo pipefail
cd "$(dirname "$0")/.."

PARENT="${1:?usage: run_video.sh <room6-video-parent-dir>}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
LOG="crossbarn/Room6_video_$(date +%Y%m%d_%H%M).log"
echo "logging to $LOG"

python crossbarn/cross_barn_video.py --all-folders \
  --video-parent-dir "$PARENT" \
  --room-label Room6 \
  --output-dir features/rich_video_optical_features \
  --workers 4 2>&1 | tee "$LOG"

echo "DONE. Output: features/rich_video_optical_features/video_rich_features_hourly_Room6.csv"
