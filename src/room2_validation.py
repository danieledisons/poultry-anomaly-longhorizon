"""
Room 2 Video Feature Extraction — Phase A: Validation Run
============================================================
Validation week: Room 2, 31 July - 3 Aug (post brooding-circle, birds unconfined).

Pipeline:
  1. Discover video files in the validation-week folder
  2. Pull real timestamps via ffprobe (exif-style creation_time metadata)
     -- do NOT trust filenames alone; GoPro embeds creation_time in the mp4 container
  3. Brighten each video with a FIXED, uniform transform (locked params -> reused in Phase B)
  4. Sample frames, compute dense optical flow (Farneback)
  5. Derive per-frame features: flow energy (mean/var of magnitude), occupancy proxy
     (fraction of pixels with motion above threshold)
  6. Aggregate frame-level features to hourly (mean, std, percentiles)
  7. Write hourly_features.csv + a per-file metadata log (for the missing-data audit)

Environment
-----------
Works identically on your MacBook (miniconda) or the server -- only EXTERNAL_DRIVE_PATH
and OUTPUT_DIR need to change. Server is recommended for speed once the hard drive is attached.

    conda create -n poultry-video python=3.11 -y
    conda activate poultry-video
    pip install opencv-python-headless numpy pandas tqdm

    # ffmpeg/ffprobe must be on PATH:
    conda install -c conda-forge ffmpeg -y      # or: brew install ffmpeg / apt install ffmpeg

Usage
-----
    python room2_validation_pipeline.py
"""

import os
import sys
import glob
import json
import random
import subprocess
import datetime as dt
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ----------------------------------------------------------------------------
# CONFIG -- the only section you should need to edit
# ----------------------------------------------------------------------------

# Folder containing the validation week's raw videos, e.g.:
#   ".../Room 2 (31 July - 3 Aug)/"
VIDEO_DIR = os.path.expanduser("~/workspace/projects/longhorizon/data/raw_room2/video/Room 2 (17, 18, 19 Aug)")

# Parent folder containing MULTIPLE date-range subfolders (e.g. all of Room 2's
# July+Aug folders side by side). Used only by --audit-only, which sweeps every
# subfolder here without doing any optical flow -- fast metadata-only pass.
VIDEO_PARENT_DIR = os.path.expanduser("~/workspace/projects/longhorizon/data/Room2_Video")

# Where CSVs and logs get written
OUTPUT_DIR = os.path.expanduser("~/workspace/projects/longhorizon/results")

# Where the brightness-check clip/stills get written (see --check-brightness)
BRIGHTNESS_CHECK_DIR = os.path.join(OUTPUT_DIR, "brightness_check")
BRIGHTNESS_CHECK_CLIP_SECONDS = 10

# Video file extensions to pick up
VIDEO_EXTS = (".mp4", ".MP4", ".mov", ".MOV")

# Frame sampling: process 1 frame every SAMPLE_EVERY_N_SEC seconds of video
# (avoids full fps optical flow over a whole week -- tune up/down after Phase A)
SAMPLE_EVERY_N_SEC = 2.0

# Brightening transform -- FIXED here, must stay identical in Phase B (full batch)
BRIGHTNESS_ALPHA = 1.3   # contrast multiplier
BRIGHTNESS_BETA = 25     # brightness offset (added to pixel values)

# Occupancy proxy: fraction of pixels with flow magnitude above this threshold
# counts as "in motion". LOCKED at 1.2 after visual validation on GX210031.MP4
# via --flow-overlay: 1.5 missed subtler walking/shuffling near walls, 1.0 was
# tested as the other end; 1.2 chosen as the preferred middle point. This value
# must stay fixed for the full Phase B batch -- do not retune mid-run.
MOTION_THRESHOLD = 1.2

# Resize frames before flow computation for speed (None = keep native resolution)
RESIZE_WIDTH = 640  # None to disable

# Darkness gate: frames with mean pixel intensity (0-255, raw/pre-brighten) below this
# are treated as "lights out" -- skipped entirely (no resize/brighten/flow compute).
# Tune by running --check-brightness on a lights-out clip and reading the printed mean.
DARK_MEAN_THRESHOLD = 15.0

# Number of videos to process in parallel during the full batch run (Phase B).
# Each file is processed independently, so this parallelizes across files, not
# within one file. On a high-core-count server (e.g. 64c/128t Threadripper),
# the bottleneck shifts from CPU to disk I/O (reading 54 large mp4s off the
# same drive at once) and to OpenCV's internal per-process thread pool fighting
# the process pool for cores -- so this is deliberately NOT set to os.cpu_count().
# Rule of thumb: 16-24 workers saturates most single external drives; going
# higher rarely helps and can slow things down via I/O contention. Tune by
# watching `iostat` / `htop` during a run -- if CPU sits idle and disk is maxed,
# LOWER this; if CPU is maxed and disk has headroom, you can raise it.
N_WORKERS = min(24, os.cpu_count() or 1)

# Each worker process sets OpenCV's internal thread count to this value (see
# cv2.setNumThreads in process_video). Without this, every one of the N_WORKERS
# processes independently tries to multi-thread its own OpenCV calls across ALL
# cores, causing massive oversubscription (N_WORKERS x cores worth of threads
# competing). Setting this to 1 makes each worker single-threaded internally,
# so parallelism comes cleanly from N_WORKERS processes instead.
CV2_THREADS_PER_WORKER = 1


# ----------------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------------

@dataclass
class FrameRecord:
    filepath: str
    timestamp: dt.datetime
    flow_mean: float
    flow_var: float
    occupancy: float
    is_dark: bool = False


@dataclass
class FileMetadata:
    filepath: str
    creation_time: dt.datetime | None   # session-start UTC value from format tags -- NOT
                                         # per-chapter (see probe_metadata docstring). Kept
                                         # for the date component and as a QA reference.
    timecode_raw: str | None            # e.g. "14:43:16;17" -- per-chapter LOCAL start time
    duration_sec: float | None
    fps: float | None
    width: int | None
    height: int | None
    readable: bool
    error: str = ""
    chapter_start: dt.datetime | None = None  # the ACTUAL usable per-chapter timestamp,
                                               # computed in resolve_chapter_starts()


# ----------------------------------------------------------------------------
# Step 1-2: discovery + timestamp/metadata extraction via ffprobe
# ----------------------------------------------------------------------------

def discover_videos(video_dir: str) -> list[str]:
    files = []
    for ext in VIDEO_EXTS:
        files.extend(glob.glob(os.path.join(video_dir, f"**/*{ext}"), recursive=True))
    return sorted(set(files))


def probe_metadata(filepath: str) -> FileMetadata:
    """Use ffprobe to pull timing, duration, fps, resolution from the container.

    IMPORTANT (found via inspect_timestamps.sh diagnostic on Room 2, 17-19 Aug):
    GoPro's format.tags.creation_time is the SESSION-start time and is identical
    across every chapter file of a multi-chapter recording -- it does NOT advance
    per chapter, so it cannot be used alone to place chapters in time.

    The video stream's `timecode` tag (e.g. "14:43:16;17") DOES advance correctly
    per chapter and is the reliable source for chapter start time. It is in LOCAL
    time (observed ~3h behind the UTC creation_time) and uses drop-frame SMPTE
    notation (HH:MM:SS;FF -- FF is a frame count, not decimal seconds).

    We combine: the calendar DATE from creation_time (reliable) with the time-of-day
    from `timecode` (reliable), producing chapter_start in resolve_chapter_starts()
    once all files are probed (needed to handle midnight rollover across chapters).
    """
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", filepath,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True)
        info = json.loads(result.stdout)

        fmt = info.get("format", {})
        tags = fmt.get("tags", {})
        creation_raw = tags.get("creation_time")
        creation_time = None
        if creation_raw:
            try:
                creation_time = dt.datetime.fromisoformat(creation_raw.replace("Z", "+00:00"))
            except ValueError:
                creation_time = None

        video_stream = next(
            (s for s in info.get("streams", []) if s.get("codec_type") == "video"), {}
        )
        fps = None
        if video_stream.get("avg_frame_rate"):
            num, den = video_stream["avg_frame_rate"].split("/")
            den = float(den) if float(den) != 0 else 1.0
            fps = float(num) / den

        timecode_raw = video_stream.get("tags", {}).get("timecode")

        return FileMetadata(
            filepath=filepath,
            creation_time=creation_time,
            timecode_raw=timecode_raw,
            duration_sec=float(fmt.get("duration", 0)) or None,
            fps=fps,
            width=video_stream.get("width"),
            height=video_stream.get("height"),
            readable=True,
        )
    except Exception as e:
        return FileMetadata(
            filepath=filepath, creation_time=None, timecode_raw=None, duration_sec=None,
            fps=None, width=None, height=None, readable=False, error=str(e),
        )


def _parse_smpte_timecode(timecode_raw: str, fps: float) -> dt.time | None:
    """Parse 'HH:MM:SS;FF' (drop-frame) or 'HH:MM:SS:FF' into a time-of-day,
    treating the frame count as a sub-second fraction (FF / fps)."""
    if not timecode_raw:
        return None
    try:
        sep = ";" if ";" in timecode_raw else ":"
        parts = timecode_raw.replace(";", ":").split(":")
        if len(parts) != 4:
            return None
        hh, mm, ss, ff = (int(p) for p in parts)
        microsec = int(round((ff / fps) * 1_000_000)) if fps else 0
        return dt.time(hour=hh % 24, minute=mm, second=ss, microsecond=microsec)
    except (ValueError, ZeroDivisionError):
        return None


def resolve_chapter_starts(metas: list["FileMetadata"]) -> list["FileMetadata"]:
    """Combine each file's calendar date (from creation_time) with its per-chapter
    time-of-day (from the timecode tag) to produce a real, advancing chapter_start.

    Files are expected sorted by name (GX01, GX02, ... = chapter order). Detects
    midnight rollover: if a chapter's time-of-day is earlier than the previous
    chapter's, it must have crossed into the next calendar day.
    """
    base_date = None
    for m in metas:
        if m.creation_time is not None:
            base_date = m.creation_time.date()
            break

    if base_date is None:
        return metas  # nothing to anchor to -- leave chapter_start as None

    current_date = base_date
    prev_time_of_day = None

    for m in metas:
        tod = _parse_smpte_timecode(m.timecode_raw, m.fps) if m.timecode_raw and m.fps else None
        if tod is None:
            m.chapter_start = None
            continue

        if prev_time_of_day is not None and tod < prev_time_of_day:
            current_date = current_date + dt.timedelta(days=1)

        m.chapter_start = dt.datetime.combine(current_date, tod)
        prev_time_of_day = tod

    return metas


# ----------------------------------------------------------------------------
# Step 3: brightening (fixed transform)
# ----------------------------------------------------------------------------

def brighten(frame: np.ndarray) -> np.ndarray:
    """Uniform brightness/contrast adjustment. Keep BRIGHTNESS_ALPHA/BETA identical
    between Phase A and Phase B so brightness itself never becomes a confound."""
    return cv2.convertScaleAbs(frame, alpha=BRIGHTNESS_ALPHA, beta=BRIGHTNESS_BETA)


def is_dark_frame(frame: np.ndarray) -> bool:
    """Cheap lights-out check on the RAW frame, before resize/brighten/flow.
    Downsampling to a small patch keeps this near-instant even on 4K frames."""
    small = cv2.resize(frame, (64, 36))
    return float(np.mean(small)) < DARK_MEAN_THRESHOLD


# ----------------------------------------------------------------------------
# Step 4-5: optical flow + feature derivation
# ----------------------------------------------------------------------------

def process_video(meta: FileMetadata) -> list[FrameRecord]:
    # Cap OpenCV's internal thread pool PER WORKER PROCESS. Without this, each
    # of the N_WORKERS processes spawned by ProcessPoolExecutor tries to use
    # every core for its own cv2 calls, causing N_WORKERS x cores of threads to
    # fight each other. This must be called inside the worker (not just once in
    # main) because each process gets its own OpenCV thread pool on spawn.
    cv2.setNumThreads(CV2_THREADS_PER_WORKER)

    if not meta.readable or meta.chapter_start is None or meta.fps is None:
        return []

    cap = cv2.VideoCapture(meta.filepath)
    if not cap.isOpened():
        return []

    frame_interval = max(1, int(round(meta.fps * SAMPLE_EVERY_N_SEC)))
    records: list[FrameRecord] = []

    prev_gray = None
    frame_idx = 0
    ok, frame = cap.read()

    while ok:
        if frame_idx % frame_interval == 0:
            elapsed_sec = frame_idx / meta.fps
            timestamp = meta.chapter_start + dt.timedelta(seconds=elapsed_sec)

            # Darkness gate FIRST, on the raw frame -- skip all resize/brighten/flow
            # work entirely for lights-out frames. This is the main compute saver:
            # overnight/dark stretches are common and flow-on-noise is meaningless anyway.
            if is_dark_frame(frame):
                records.append(FrameRecord(
                    filepath=meta.filepath, timestamp=timestamp,
                    flow_mean=0.0, flow_var=0.0, occupancy=0.0, is_dark=True,
                ))
                prev_gray = None  # break the flow chain across a dark gap
                frame_idx += 1
                ok, frame = cap.read()
                continue

            if RESIZE_WIDTH is not None:
                h, w = frame.shape[:2]
                scale = RESIZE_WIDTH / w
                frame_resized = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)))
            else:
                frame_resized = frame

            bright = brighten(frame_resized)
            gray = cv2.cvtColor(bright, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                )
                magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

                flow_mean = float(np.mean(magnitude))
                flow_var = float(np.var(magnitude))
                occupancy = float(np.mean(magnitude > MOTION_THRESHOLD))

                records.append(FrameRecord(
                    filepath=meta.filepath, timestamp=timestamp,
                    flow_mean=flow_mean, flow_var=flow_var, occupancy=occupancy,
                    is_dark=False,
                ))

            prev_gray = gray

        frame_idx += 1
        ok, frame = cap.read()

    cap.release()
    return records


# ----------------------------------------------------------------------------
# Step 6: hourly aggregation
# ----------------------------------------------------------------------------

def aggregate_hourly(records: list[FrameRecord]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame([r.__dict__ for r in records])
    df["hour"] = df["timestamp"].dt.floor("h")

    # dark_fraction + file_count computed over ALL sampled frames (dark + lit)
    dark_frac = df.groupby("hour")["is_dark"].mean().rename("dark_fraction")
    file_count = df.groupby("hour")["filepath"].nunique().rename("file_count")

    # flow/occupancy stats computed only over LIT frames, so a dark stretch
    # doesn't drag the hourly mean toward zero and get misread as "low activity"
    lit = df[~df["is_dark"]]
    hourly = lit.groupby("hour").agg(
        flow_mean_avg=("flow_mean", "mean"),
        flow_mean_std=("flow_mean", "std"),
        flow_mean_min=("flow_mean", "min"),
        flow_mean_max=("flow_mean", "max"),
        flow_var_avg=("flow_var", "mean"),
        occupancy_avg=("occupancy", "mean"),
        occupancy_std=("occupancy", "std"),
        occupancy_p50=("occupancy", lambda x: x.quantile(0.5)),
        occupancy_p90=("occupancy", lambda x: x.quantile(0.9)),
        n_frames_lit=("flow_mean", "count"),
    )

    hourly = hourly.join(dark_frac, how="outer").join(file_count, how="outer").reset_index()

    # Placeholder columns for Phase 2 (YOLO + ByteTrack + TimeSformer behavior-state
    # composition). Left as NaN here deliberately -- these require a trained detector/
    # tracker/classifier, not optical flow, and should not be faked from flow-derived
    # proxies. Filled in once the full behavioral pipeline is built and validated.
    for col in ["frac_idle", "frac_active", "frac_drinking", "frac_feeding",
                "frac_preening", "frac_perching", "frac_wing_flapping"]:
        hourly[col] = np.nan

    return hourly


# ----------------------------------------------------------------------------
# Step 7: missing-data audit log
# ----------------------------------------------------------------------------

def write_metadata_log(metas: list[FileMetadata], output_dir: str):
    rows = [m.__dict__ for m in metas]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "file_metadata_log.csv"), index=False)

    unreadable = df[~df["readable"]]
    no_chapter_start = df[df["readable"] & df["chapter_start"].isna()]

    # Flag if creation_time is suspiciously identical across many files -- this is
    # exactly the bug found in Room 2 (17-19 Aug): GoPro copies session-start time
    # into every chapter instead of advancing it. chapter_start (derived from the
    # timecode tag) is what should actually be used downstream, not creation_time.
    n_unique_creation_times = df.loc[df["readable"], "creation_time"].nunique()

    print(f"\n--- Missing-data audit ---")
    print(f"Total files found:            {len(df)}")
    print(f"Unreadable files:             {len(unreadable)}")
    print(f"Readable but no chapter_start:{len(no_chapter_start)}")
    print(f"Unique creation_time values:  {n_unique_creation_times} "
          f"(out of {len(df[df['readable']])} readable files)")
    if n_unique_creation_times == 1 and len(df[df["readable"]]) > 1:
        print("  -> creation_time is IDENTICAL across all files (known GoPro chapter "
              "bug). chapter_start (from timecode tag) is used for actual timestamps.")
    if len(unreadable) or len(no_chapter_start):
        print("See file_metadata_log.csv for details -- these need manual review "
              "before Phase B (full 3-month batch).")


# ----------------------------------------------------------------------------
# Brightness check -- pick 1 random video, snip a 10s clip, save original vs. brightened
# ----------------------------------------------------------------------------

def check_brightness(target_filename: str | None = None):
    os.makedirs(BRIGHTNESS_CHECK_DIR, exist_ok=True)

    video_files = discover_videos(VIDEO_DIR)
    if not video_files:
        print("No videos found -- check VIDEO_DIR in the config section.")
        return

    if target_filename is not None:
        matches = [f for f in video_files if os.path.basename(f) == target_filename]
        if not matches:
            print(f"No file named '{target_filename}' found under {VIDEO_DIR}")
            return
        video_path = matches[0]
    else:
        video_path = random.choice(video_files)
    print(f"Selected: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_frames = int(fps * BRIGHTNESS_CHECK_CLIP_SECONDS)

    start_frame = 0 if total_frames <= clip_frames else random.randint(0, total_frames - clip_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    orig_writer = cv2.VideoWriter(os.path.join(BRIGHTNESS_CHECK_DIR, "original_clip.mp4"), fourcc, fps, (w, h))
    bright_writer = cv2.VideoWriter(os.path.join(BRIGHTNESS_CHECK_DIR, "brightened_clip.mp4"), fourcc, fps, (w, h))

    saved_still = False
    for i in range(clip_frames):
        ok, frame = cap.read()
        if not ok:
            break

        bright_frame = brighten(frame)
        orig_writer.write(frame)
        bright_writer.write(bright_frame)

        if not saved_still and i == clip_frames // 2:
            cv2.imwrite(os.path.join(BRIGHTNESS_CHECK_DIR, "original_frame.png"), frame)
            cv2.imwrite(os.path.join(BRIGHTNESS_CHECK_DIR, "brightened_frame.png"), bright_frame)
            mean_intensity = float(np.mean(cv2.resize(frame, (64, 36))))
            print(f"Raw mean intensity (this frame): {mean_intensity:.1f}  "
                  f"(current DARK_MEAN_THRESHOLD = {DARK_MEAN_THRESHOLD})")
            print("  -> if this clip IS lights-out and mean_intensity is above threshold,")
            print("     raise DARK_MEAN_THRESHOLD. If it's lit but flagged dark, lower it.")
            saved_still = True

    cap.release()
    orig_writer.release()
    bright_writer.release()

    print(f"Clip starts at frame {start_frame} (~{start_frame/fps:.1f}s into the video)")
    print(f"Saved to: {BRIGHTNESS_CHECK_DIR}/")
    print("  original_clip.mp4, brightened_clip.mp4")
    print("  original_frame.png, brightened_frame.png  <- open these two side by side first")


# ----------------------------------------------------------------------------
# Flow overlay -- visualize exactly which pixels count as "in motion" at a
# given moment, so occupancy_avg numbers can be checked against what you see.
# ----------------------------------------------------------------------------

def flow_overlay(target_filename: str, offset_sec: float):
    """Grab the frame at offset_sec and the next SAMPLE_EVERY_N_SEC-later frame
    (matching how the main pipeline pairs frames for flow), then save:
      - raw_frame.png            (brightened, what the eye would judge motion against)
      - flow_heatmap.png         (flow magnitude, continuous, brighter = more motion)
      - motion_mask_overlay.png  (red = pixels exceeding MOTION_THRESHOLD, i.e.
                                   exactly what occupancy_avg counts)
    Prints the resulting occupancy fraction so you can compare it to the CSV row
    for the hour this offset falls into.
    """
    os.makedirs(BRIGHTNESS_CHECK_DIR, exist_ok=True)

    video_files = discover_videos(VIDEO_DIR)
    matches = [f for f in video_files if os.path.basename(f) == target_filename]
    if not matches:
        print(f"No file named '{target_filename}' found under {VIDEO_DIR}")
        return
    video_path = matches[0]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    start_frame_idx = int(offset_sec * fps)
    gap_frames = max(1, int(round(fps * SAMPLE_EVERY_N_SEC)))  # same pairing as process_video

    def read_frame(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        return frame if ok else None

    frame_a = read_frame(start_frame_idx)
    frame_b = read_frame(start_frame_idx + gap_frames)
    cap.release()

    if frame_a is None or frame_b is None:
        print("Could not read one or both frames -- try a smaller offset_sec.")
        return

    def prep(frame):
        if RESIZE_WIDTH is not None:
            h, w = frame.shape[:2]
            scale = RESIZE_WIDTH / w
            frame = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)))
        return brighten(frame)

    bright_a = prep(frame_a)
    bright_b = prep(frame_b)
    gray_a = cv2.cvtColor(bright_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(bright_b, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        gray_a, gray_b, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    occupancy = float(np.mean(magnitude > MOTION_THRESHOLD))

    # Heatmap: normalize magnitude to 0-255 for visualization only
    mag_norm = np.clip(magnitude / (magnitude.max() + 1e-6) * 255, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(mag_norm, cv2.COLORMAP_JET)

    # Motion mask overlay: red where magnitude > MOTION_THRESHOLD, laid over bright_b
    overlay = bright_b.copy()
    mask = magnitude > MOTION_THRESHOLD
    overlay[mask] = [0, 0, 255]  # BGR red

    raw_path = os.path.join(BRIGHTNESS_CHECK_DIR, "raw_frame.png")
    heat_path = os.path.join(BRIGHTNESS_CHECK_DIR, "flow_heatmap.png")
    overlay_path = os.path.join(BRIGHTNESS_CHECK_DIR, "motion_mask_overlay.png")

    cv2.imwrite(raw_path, bright_b)
    cv2.imwrite(heat_path, heatmap)
    cv2.imwrite(overlay_path, overlay)

    print(f"File: {video_path}")
    print(f"Frame pair: {start_frame_idx} -> {start_frame_idx + gap_frames} "
          f"(~{offset_sec:.1f}s -> ~{offset_sec + SAMPLE_EVERY_N_SEC:.1f}s)")
    print(f"MOTION_THRESHOLD = {MOTION_THRESHOLD}")
    print(f"Occupancy for this single frame pair: {occupancy:.4f} ({occupancy*100:.2f}%)")
    print(f"Saved to: {BRIGHTNESS_CHECK_DIR}/")
    print("  raw_frame.png             <- what the eye judges motion against")
    print("  flow_heatmap.png          <- continuous motion intensity")
    print("  motion_mask_overlay.png   <- red = counted as 'in motion' (compare to raw_frame.png)")


# ----------------------------------------------------------------------------
# Audit-only -- metadata/timestamp sweep across ALL subfolders under
# VIDEO_PARENT_DIR, no optical flow. Fast (~seconds per file) since it's just
# ffprobe calls. Answers two questions before committing to the full batch:
#   1. Are there unreadable files / missing timecodes anywhere in July+Aug?
#   2. Does the creation_time-frozen / timecode-advancing GoPro bug (found in
#      "Room 2 (17, 18, 19 Aug)") hold consistently in every other folder, or
#      does it vary (e.g. different firmware/settings on a different session)?
# ----------------------------------------------------------------------------

def _split_into_sessions(metas: list[FileMetadata]) -> list[list[FileMetadata]]:
    """Group files into sessions: a session is a contiguous run of files (in sorted
    filename order) sharing the same creation_time. A folder can contain more than
    one session (e.g. SD card swap, camera restart, or genuinely separate recording
    runs) -- this is expected and NOT itself a problem. Files with no creation_time
    each form their own singleton session."""
    sessions: list[list[FileMetadata]] = []
    current: list[FileMetadata] = []
    current_ct = object()  # sentinel that won't equal any real creation_time

    for m in metas:
        if m.creation_time == current_ct and current:
            current.append(m)
        else:
            if current:
                sessions.append(current)
            current = [m]
            current_ct = m.creation_time

    if current:
        sessions.append(current)

    return sessions


def _check_session(session: list[FileMetadata]) -> dict:
    """Per-session anomaly checks: does the known creation_time-frozen bug pattern
    hold, do chapter_start values advance monotonically with sane gaps, any dups?"""
    readable = [m for m in session if m.readable]
    unique_ct = len({m.creation_time for m in readable if m.creation_time})
    bug_pattern_holds = (unique_ct == 1 and len(readable) > 1) or len(readable) <= 1

    starts = [m.chapter_start for m in readable if m.chapter_start is not None]
    monotonic = all(starts[i] < starts[i + 1] for i in range(len(starts) - 1))
    duplicates = len(starts) != len(set(starts))

    gaps_min = []
    for i in range(len(starts) - 1):
        gap = (starts[i + 1] - starts[i]).total_seconds() / 60.0
        gaps_min.append(gap)
    # Flag gaps that are wildly larger than a typical ~50min chapter (e.g. >90min
    # mid-session would be unusual -- could indicate a missing file or clock jump)
    irregular_gaps = [g for g in gaps_min if g > 90]

    anomaly = (not bug_pattern_holds) or (not monotonic) or duplicates or bool(irregular_gaps)

    return {
        "n_files": len(session),
        "n_readable": len(readable),
        "unique_creation_time": unique_ct,
        "bug_pattern_holds": bug_pattern_holds,
        "chapter_start_monotonic": monotonic,
        "duplicate_chapter_starts": duplicates,
        "n_irregular_gaps_over_90min": len(irregular_gaps),
        "max_gap_min": max(gaps_min) if gaps_min else None,
        "session_start": starts[0] if starts else None,
        "session_end": starts[-1] if starts else None,
        "anomaly": anomaly,
    }


def audit_only():
    if not os.path.isdir(VIDEO_PARENT_DIR):
        print(f"VIDEO_PARENT_DIR not found: {VIDEO_PARENT_DIR}")
        return

    subfolders = sorted(
        os.path.join(VIDEO_PARENT_DIR, d)
        for d in os.listdir(VIDEO_PARENT_DIR)
        if os.path.isdir(os.path.join(VIDEO_PARENT_DIR, d))
    )
    print(f"Found {len(subfolders)} subfolders under {VIDEO_PARENT_DIR}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_rows = []
    session_rows = []
    folder_rows = []

    for folder in subfolders:
        folder_name = os.path.basename(folder)
        video_files = discover_videos(folder)
        if not video_files:
            print(f"[{folder_name}] no video files found -- skipping")
            continue

        metas = [probe_metadata(f) for f in video_files]
        metas.sort(key=lambda m: m.filepath)
        metas = resolve_chapter_starts(metas)

        unreadable = [m for m in metas if not m.readable]
        no_timecode = [m for m in metas if m.readable and not m.timecode_raw]
        no_chapter_start = [m for m in metas if m.readable and m.chapter_start is None]

        sessions = _split_into_sessions(metas)
        session_checks = [_check_session(s) for s in sessions]
        n_anomalous_sessions = sum(1 for c in session_checks if c["anomaly"])

        print(f"[{folder_name}] {len(metas)} files | {len(sessions)} session(s) | "
              f"unreadable={len(unreadable)} | no_timecode={len(no_timecode)} | "
              f"no_chapter_start={len(no_chapter_start)} | "
              f"anomalous_sessions={n_anomalous_sessions}/{len(sessions)}")

        for i, (session, check) in enumerate(zip(sessions, session_checks)):
            flag = "ANOMALY" if check["anomaly"] else "ok"
            print(f"    session {i+1}: {check['n_files']} files, "
                  f"{check['session_start']} -> {check['session_end']}, "
                  f"max_gap={check['max_gap_min']:.1f}min  [{flag}]"
                  if check["max_gap_min"] is not None else
                  f"    session {i+1}: {check['n_files']} files  [{flag}]")
            if check["anomaly"]:
                reasons = []
                if not check["bug_pattern_holds"]:
                    reasons.append("creation_time varies WITHIN session (unexpected)")
                if not check["chapter_start_monotonic"]:
                    reasons.append("chapter_start NOT monotonic (out of order)")
                if check["duplicate_chapter_starts"]:
                    reasons.append("duplicate chapter_start values")
                if check["n_irregular_gaps_over_90min"] > 0:
                    reasons.append(f"{check['n_irregular_gaps_over_90min']} gap(s) > 90min mid-session")
                print(f"      -> {'; '.join(reasons)}")

            session_rows.append({
                "folder": folder_name, "session_index": i + 1, **check,
            })

        for m in metas:
            row = m.__dict__.copy()
            row["folder"] = folder_name
            all_rows.append(row)

        all_starts = [m.chapter_start for m in metas if m.chapter_start]
        folder_rows.append({
            "folder": folder_name,
            "n_files": len(metas),
            "n_sessions": len(sessions),
            "n_anomalous_sessions": n_anomalous_sessions,
            "unreadable": len(unreadable),
            "no_timecode": len(no_timecode),
            "no_chapter_start": len(no_chapter_start),
            "earliest_chapter_start": min(all_starts, default=None),
            "latest_chapter_start": max(all_starts, default=None),
        })

    all_df = pd.DataFrame(all_rows)
    session_df = pd.DataFrame(session_rows)
    folder_df = pd.DataFrame(folder_rows)

    all_path = os.path.join(OUTPUT_DIR, "audit_all_files.csv")
    session_path = os.path.join(OUTPUT_DIR, "audit_sessions.csv")
    folder_path = os.path.join(OUTPUT_DIR, "audit_summary_by_folder.csv")
    all_df.to_csv(all_path, index=False)
    session_df.to_csv(session_path, index=False)
    folder_df.to_csv(folder_path, index=False)

    print(f"\nPer-file audit written to: {all_path}")
    print(f"Per-session audit written to: {session_path}")
    print(f"Per-folder summary written to: {folder_path}")

    problem_folders = folder_df[
        (folder_df["unreadable"] > 0) |
        (folder_df["no_timecode"] > 0) |
        (folder_df["no_chapter_start"] > 0) |
        (folder_df["n_anomalous_sessions"] > 0)
    ]
    if len(problem_folders) > 0:
        print(f"\n{len(problem_folders)} folder(s) need manual review before batch processing:")
        print(problem_folders[["folder", "unreadable", "no_timecode", "no_chapter_start", "n_anomalous_sessions"]])
    else:
        print("\nAll folders/sessions look consistent -- safe to proceed to full batch processing.")

    print(f"\n--- Overall date coverage ---")
    all_starts = [r["earliest_chapter_start"] for r in folder_rows if r["earliest_chapter_start"]]
    all_ends = [r["latest_chapter_start"] for r in folder_rows if r["latest_chapter_start"]]
    if all_starts and all_ends:
        print(f"Earliest recording: {min(all_starts)}")
        print(f"Latest recording:   {max(all_ends)}")
    print("Per-folder date ranges:")
    for r in sorted(folder_rows, key=lambda r: r["earliest_chapter_start"] or dt.datetime.min):
        print(f"  {r['folder']}: {r['earliest_chapter_start']} -> {r['latest_chapter_start']}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main(target_filename: str | None = None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Discovering videos in: {VIDEO_DIR}")
    video_files = discover_videos(VIDEO_DIR)
    print(f"Found {len(video_files)} video files")

    if not video_files:
        print("No videos found -- check VIDEO_DIR and VIDEO_EXTS in the config section.")
        return

    if target_filename is not None:
        video_files = [f for f in video_files if os.path.basename(f) == target_filename]
        if not video_files:
            print(f"No file named '{target_filename}' found under {VIDEO_DIR}")
            return
        print(f"--file given: restricting run to {video_files[0]}")

    print("Probing metadata (ffprobe)...")
    metas = [probe_metadata(f) for f in tqdm(video_files)]

    # Sort by filename (GX01, GX02, ...) BEFORE resolving chapter starts -- chapter
    # order must match filename order for midnight-rollover detection to work.
    metas.sort(key=lambda m: m.filepath)
    metas = resolve_chapter_starts(metas)

    write_metadata_log(metas, OUTPUT_DIR)

    usable_metas = [m for m in metas if m.readable and m.chapter_start and m.fps]
    print(f"\n{len(usable_metas)}/{len(metas)} files usable for feature extraction "
          f"(readable + valid chapter_start + valid fps)")

    all_records: list[FrameRecord] = []
    print(f"Running optical flow extraction across {N_WORKERS} worker process(es)...")

    if N_WORKERS <= 1:
        for meta in tqdm(usable_metas):
            all_records.extend(process_video(meta))
    else:
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {executor.submit(process_video, meta): meta for meta in usable_metas}
            for future in tqdm(as_completed(futures), total=len(futures)):
                meta = futures[future]
                try:
                    all_records.extend(future.result())
                except Exception as e:
                    print(f"\nFAILED on {meta.filepath}: {e}")

    print(f"Extracted {len(all_records)} frame-level feature records")

    hourly_df = aggregate_hourly(all_records)
    hourly_path = os.path.join(OUTPUT_DIR, "hourly_features.csv")
    hourly_df.to_csv(hourly_path, index=False)
    print(f"\nHourly features written to: {hourly_path}")
    print(hourly_df.head(10))

    # Frame-level output too, useful for the visual spot-check step
    frame_df = pd.DataFrame([r.__dict__ for r in all_records])
    frame_path = os.path.join(OUTPUT_DIR, "frame_level_features.csv")
    frame_df.to_csv(frame_path, index=False)
    print(f"Frame-level features written to: {frame_path}")


if __name__ == "__main__":
    target = None
    if "--file" in sys.argv:
        idx = sys.argv.index("--file")
        if idx + 1 < len(sys.argv):
            target = sys.argv[idx + 1]

    if "--flow-overlay" in sys.argv:
        offset = 0.0
        if "--offset" in sys.argv:
            idx = sys.argv.index("--offset")
            if idx + 1 < len(sys.argv):
                offset = float(sys.argv[idx + 1])
        if target is None:
            print("--flow-overlay requires --file <filename.MP4>")
        else:
            flow_overlay(target_filename=target, offset_sec=offset)
    elif "--check-brightness" in sys.argv:
        check_brightness(target_filename=target)
    elif "--audit-only" in sys.argv:
        audit_only()
    else:
        main(target_filename=target)