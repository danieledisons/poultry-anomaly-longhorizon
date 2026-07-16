# Video Feature Extraction

Extracts hourly-aggregated optical-flow and occupancy features from GoPro
poultry-barn recordings for the video modality of a multimodal anomaly-detection
pipeline.

## Setup

```bash
pip install -r requirements.txt
```

`ffmpeg`/`ffprobe` must be on `PATH` (used for metadata extraction):

```bash
conda install -c conda-forge ffmpeg -y      # or: brew install ffmpeg / apt install ffmpeg
```

## Usage

**1. Audit first.** Before any full batch run, sweep the dataset for unreadable
files or timestamp anomalies (fast, metadata-only, no optical flow):

```bash
python extract_video_features.py --audit-only \
    --video-parent-dir "/path/to/Room2_Video" \
    --output-dir results/
```

Review `audit_summary_by_folder.csv` / `audit_sessions.csv` for flagged folders
before proceeding.

**2. Single folder** (e.g. a validation run on one date range):

```bash
python extract_video_features.py \
    --video-dir "/path/to/Room 2 (17, 18, 19 Aug)" \
    --output-dir results/
```

**3. Full batch across every subfolder**, combined into one continuous output:

```bash
python extract_video_features.py --all-folders \
    --video-parent-dir "/path/to/Room2_Video" \
    --output-dir results/
```

Recommended to run under `tmux`/`screen`/`nohup` since a full multi-week batch
can take hours.

## Dev/validation tools

Before trusting `--motion-threshold` or brightness settings on a new dataset,
validate visually rather than guessing:

```bash
# Compare original vs. brightened frames from a random (or specific) clip
python extract_video_features.py --check-brightness \
    --video-dir "/path/to/folder" --output-dir results/ [--file GX010031.MP4]

# Overlay exactly which pixels count as "in motion" at a given timestamp
python extract_video_features.py --flow-overlay \
    --video-dir "/path/to/folder" --output-dir results/ \
    --file GX210031.MP4 --offset 300
```

Outputs land in `results/brightness_check/`.

## Key design notes

- **GoPro chapter-timestamp bug**: the container's `creation_time` tag is frozen
  at session start and does not advance per chapter. The `timecode` stream tag
  does advance correctly and is used (combined with the calendar date) to build
  a real `chapter_start` per file. See `resolve_chapter_starts()`.
- **Darkness gating**: frames below `--dark-mean-threshold` mean intensity are
  flagged (`is_dark=True`, zeroed features) rather than processed or dropped,
  so overnight lights-off periods don't consume compute or get misread as "low
  activity." Check `dark_fraction` in the hourly output before interpreting a
  low/NaN flow value.
- **Motion threshold**: tune per-dataset using `--flow-overlay`, don't reuse a
  value across rooms/cameras without re-validating.
- **Parallelism**: `--workers` defaults to a value tuned for I/O-bound behavior
  (many large files read off one drive), not raw CPU core count. Each worker
  caps its own OpenCV thread pool to 1 to avoid oversubscription.
- **Behavior-state columns** (`frac_idle`, `frac_feeding`, etc.) are written as
  `NaN` placeholders, reserved for a future detection/tracking/classification
  pipeline (e.g. YOLO + ByteTrack + a temporal classifier). Optical flow alone
  cannot produce these and they are intentionally not faked.

## Output files

| File | Produced by | Contents |
|---|---|---|
| `hourly_features.csv` | single-folder run | hourly flow/occupancy stats |
| `frame_level_features.csv` | single-folder run | per-sampled-frame detail |
| `file_metadata_log.csv` | single-folder run | per-file ffprobe metadata + resolved timestamps |
| `hourly_features_all_folders.csv` | `--all-folders` | combined hourly stats across the full date range |
| `frame_level_features_all_folders.csv` | `--all-folders` | combined frame-level detail |
| `file_metadata_log_all_folders.csv` | `--all-folders` | combined metadata, tagged by source folder |
| `batch_run_summary_by_folder.csv` | `--all-folders` | per-folder file/record counts |
| `audit_all_files.csv` / `audit_sessions.csv` / `audit_summary_by_folder.csv` | `--audit-only` | pre-batch validation report |