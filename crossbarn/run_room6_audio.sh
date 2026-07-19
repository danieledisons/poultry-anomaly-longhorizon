#!/usr/bin/env bash
# Overnight rich-audio extraction for the held-out barn (Room 6, Zoom F6).
# Run inside tmux:   tmux new -s room6 ;  bash crossbarn/run_room6_audio.sh
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root

# keep BLAS single-threaded so the process pool doesn't oversubscribe cores
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

LOG="crossbarn/room6_audio_$(date +%Y%m%d_%H%M).log"
echo "logging to $LOG"

python crossbarn/extract_rich_audio_crossbarn.py \
  --input-root "/mnt/crucial_x10/Poultry Multimodal Data/Audio Data/Room 6" \
  --input-root "/mnt/drive_bf/Poultry_Multimodal_SeptDec/Audio data/Room6" \
  --room-label Room6 --month-tag all \
  --output-dir features/rich_audio_features \
  --workers 12 2>&1 | tee "$LOG"

echo "DONE. Output: features/rich_audio_features/audio_rich_features_hourly_Room6_all.csv"
