#!/bin/bash
# Diagnostic: check whether GoPro chapter files in VIDEO_DIR really share one
# creation_time, or whether other metadata/mtime reveals real per-file timing.
#
# Usage: bash inspect_timestamps.sh "/path/to/Room 2 (17, 18, 19 Aug)"

VIDEO_DIR="$1"
if [ -z "$VIDEO_DIR" ]; then
    echo "Usage: bash inspect_timestamps.sh /path/to/video/folder"
    exit 1
fi

echo "=================================================================="
echo "STEP 1: Filesystem modification time (mtime) per file"
echo "=================================================================="
echo "If mtime varies meaningfully across files (in file order), it's a"
echo "strong signal these ARE sequential chapters recorded over time,"
echo "even though creation_time in the container is identical/wrong."
echo ""
ls -l --time-style=full-iso "$VIDEO_DIR"/*.MP4 2>/dev/null | awk '{print $6, $7, $NF}'
echo ""

echo "=================================================================="
echo "STEP 2: Full ffprobe metadata dump on first 3 files (sorted)"
echo "=================================================================="
echo "Looking for per-chapter timing beyond format.tags.creation_time --"
echo "GoPro sometimes stores this in a timecode stream, GPS9 stream, or"
echo "per-stream tags rather than the top-level format tags."
echo ""

count=0
for f in "$VIDEO_DIR"/*.MP4; do
    count=$((count+1))
    if [ $count -gt 3 ]; then
        break
    fi
    echo "---- $f ----"
    ffprobe -v quiet -print_format json -show_format -show_streams "$f" | \
        python3 -c "
import json, sys
info = json.load(sys.stdin)
fmt = info.get('format', {})
print('FORMAT tags:', fmt.get('tags', {}))
for s in info.get('streams', []):
    tags = s.get('tags', {})
    if tags:
        print(f\"  STREAM {s.get('index')} ({s.get('codec_type')}, {s.get('codec_name')}): {tags}\")
"
    echo ""
done

echo "=================================================================="
echo "STEP 3: Check for GPMF/timecode stream specifically"
echo "=================================================================="
echo "GoPro embeds a timecode track (tmcd) with the true start timecode."
echo "This is often more reliable than format-level creation_time."
echo ""
first_file=$(ls "$VIDEO_DIR"/*.MP4 2>/dev/null | head -1)
if [ -n "$first_file" ]; then
    echo "Checking: $first_file"
    ffprobe -v quiet -show_entries stream=index,codec_type,codec_name,codec_tag_string -show_entries stream_tags=timecode "$first_file"
fi