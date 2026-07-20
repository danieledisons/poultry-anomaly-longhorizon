#!/usr/bin/env python3
"""Scalar video features (optical flow + occupancy) with GoPro timestamp handling and a few audit modes.

Run: python src/extraction/extract_video_features.py --all-folders --video-parent-dir <dir>
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import random
import subprocess
import sys
import datetime as dt
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger("extract_video_features")


# =============================================================================
# Config
# =============================================================================

@dataclass
class Config:
    video_dir: str | None = None
    video_parent_dir: str | None = None
    output_dir: str = "results"
    video_exts: tuple[str, ...] = (".mp4", ".MP4", ".mov", ".MOV")

    sample_every_n_sec: float = 2.0
    brightness_alpha: float = 1.3
    brightness_beta: int = 25
    motion_threshold: float = 1.2
    resize_width: int | None = 640
    dark_mean_threshold: float = 15.0

    n_workers: int = 24
    cv2_threads_per_worker: int = 1

    @property
    def brightness_check_dir(self) -> str:
        return os.path.join(self.output_dir, "brightness_check")


CFG = Config()  # populated from CLI args in main(); module-level so worker
                 # processes (spawned by ProcessPoolExecutor) can access it.


# =============================================================================
# Data structures
# =============================================================================

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
    creation_time: dt.datetime | None   # session-start value from container tags --
                                         # NOT per-chapter, see probe_metadata().
    timecode_raw: str | None            # e.g. "14:43:16;17" -- per-chapter LOCAL start
    duration_sec: float | None
    fps: float | None
    width: int | None
    height: int | None
    readable: bool
    error: str = ""
    chapter_start: dt.datetime | None = None  # resolved, usable per-chapter timestamp


# =============================================================================
# Discovery + metadata extraction
# =============================================================================

def discover_videos(video_dir: str) -> list[str]:
    files: list[str] = []
    for ext in CFG.video_exts:
        files.extend(glob.glob(os.path.join(video_dir, f"**/*{ext}"), recursive=True))
    return sorted(set(files))


def probe_metadata(filepath: str) -> FileMetadata:
    """Extract timing, duration, fps, resolution via ffprobe.

    See module docstring point 1 for why chapter_start (not creation_time) is the
    field to use downstream.
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
        parts = timecode_raw.replace(";", ":").split(":")
        if len(parts) != 4:
            return None
        hh, mm, ss, ff = (int(p) for p in parts)
        microsec = int(round((ff / fps) * 1_000_000)) if fps else 0
        return dt.time(hour=hh % 24, minute=mm, second=ss, microsecond=microsec)
    except (ValueError, ZeroDivisionError):
        return None


def resolve_chapter_starts(metas: list[FileMetadata]) -> list[FileMetadata]:
    """Combine each file's calendar date (from creation_time) with its per-chapter
    time-of-day (from the timecode tag) to produce a real, advancing chapter_start.

    Files must be pre-sorted by filename (chapter order) before calling this.
    Detects midnight rollover: if a chapter's time-of-day is earlier than the
    previous chapter's, it must have crossed into the next calendar day.
    """
    base_date = next((m.creation_time.date() for m in metas if m.creation_time), None)
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
            current_date += dt.timedelta(days=1)

        m.chapter_start = dt.datetime.combine(current_date, tod)
        prev_time_of_day = tod

    return metas


def probe_and_resolve(video_dir: str, target_filename: str | None = None) -> list[FileMetadata]:
    """Discover + probe + resolve chapter_start for one folder."""
    logger.info("Discovering videos in: %s", video_dir)
    video_files = discover_videos(video_dir)
    logger.info("Found %d video files", len(video_files))

    if not video_files:
        logger.warning("No videos found in %s -- check the path and video_exts config.", video_dir)
        return []

    if target_filename is not None:
        video_files = [f for f in video_files if os.path.basename(f) == target_filename]
        if not video_files:
            logger.error("No file named '%s' found under %s", target_filename, video_dir)
            return []
        logger.info("Restricting run to single file: %s", video_files[0])

    logger.info("Probing metadata (ffprobe)...")
    metas = [probe_metadata(f) for f in tqdm(video_files, desc="ffprobe")]

    metas.sort(key=lambda m: m.filepath)
    metas = resolve_chapter_starts(metas)
    return metas


# =============================================================================
# Frame preprocessing
# =============================================================================

def brighten(frame: np.ndarray) -> np.ndarray:
    """Uniform brightness/contrast adjustment. Must stay identical across an
    entire dataset -- changing it mid-project makes weeks/rooms incomparable."""
    return cv2.convertScaleAbs(frame, alpha=CFG.brightness_alpha, beta=CFG.brightness_beta)


def is_dark_frame(frame: np.ndarray) -> bool:
    """Cheap lights-out check on the RAW frame, before resize/brighten/flow."""
    small = cv2.resize(frame, (64, 36))
    return float(np.mean(small)) < CFG.dark_mean_threshold


# =============================================================================
# Optical flow + feature extraction
# =============================================================================

def process_video(meta: FileMetadata) -> list[FrameRecord]:
    """Extract flow-energy and occupancy features from one video file.

    Runs inside a worker process (see extract_records) -- must cap OpenCV's
    internal thread pool here, not just once in the parent, since each spawned
    process gets its own OpenCV thread pool.
    """
    cv2.setNumThreads(CFG.cv2_threads_per_worker)

    if not meta.readable or meta.chapter_start is None or meta.fps is None:
        return []

    cap = cv2.VideoCapture(meta.filepath)
    if not cap.isOpened():
        return []

    frame_interval = max(1, int(round(meta.fps * CFG.sample_every_n_sec)))
    records: list[FrameRecord] = []

    prev_gray = None
    frame_idx = 0
    ok, frame = cap.read()

    while ok:
        if frame_idx % frame_interval == 0:
            elapsed_sec = frame_idx / meta.fps
            timestamp = meta.chapter_start + dt.timedelta(seconds=elapsed_sec)

            if is_dark_frame(frame):
                records.append(FrameRecord(
                    filepath=meta.filepath, timestamp=timestamp,
                    flow_mean=0.0, flow_var=0.0, occupancy=0.0, is_dark=True,
                ))
                prev_gray = None  # break the flow chain across a dark gap
                frame_idx += 1
                ok, frame = cap.read()
                continue

            if CFG.resize_width is not None:
                h, w = frame.shape[:2]
                scale = CFG.resize_width / w
                frame_resized = cv2.resize(frame, (CFG.resize_width, int(h * scale)))
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

                records.append(FrameRecord(
                    filepath=meta.filepath, timestamp=timestamp,
                    flow_mean=float(np.mean(magnitude)),
                    flow_var=float(np.var(magnitude)),
                    occupancy=float(np.mean(magnitude > CFG.motion_threshold)),
                    is_dark=False,
                ))

            prev_gray = gray

        frame_idx += 1
        ok, frame = cap.read()

    cap.release()
    return records


def extract_records(usable_metas: list[FileMetadata]) -> list[FrameRecord]:
    """Run process_video across usable_metas, parallelized across CFG.n_workers
    processes."""
    all_records: list[FrameRecord] = []
    logger.info("Running optical flow extraction across %d worker process(es)...", CFG.n_workers)

    if CFG.n_workers <= 1:
        for meta in tqdm(usable_metas, desc="optical flow"):
            all_records.extend(process_video(meta))
    else:
        with ProcessPoolExecutor(max_workers=CFG.n_workers) as executor:
            futures = {executor.submit(process_video, meta): meta for meta in usable_metas}
            for future in tqdm(as_completed(futures), total=len(futures), desc="optical flow"):
                meta = futures[future]
                try:
                    all_records.extend(future.result())
                except Exception as e:
                    logger.error("FAILED on %s: %s", meta.filepath, e)

    return all_records


# =============================================================================
# Hourly aggregation
# =============================================================================

def aggregate_hourly(records: list[FrameRecord]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame([r.__dict__ for r in records])
    df["hour"] = df["timestamp"].dt.floor("h")

    # dark_fraction + file_count computed over ALL sampled frames (dark + lit)
    dark_frac = df.groupby("hour")["is_dark"].mean().rename("dark_fraction")
    file_count = df.groupby("hour")["filepath"].nunique().rename("file_count")

    # flow/occupancy stats computed only over LIT frames, so a dark stretch
    # doesn't drag the hourly mean toward zero and get misread as "low activity".
    # A fully-dark hour will have no lit rows and so will show NaN here (correctly
    # distinct from "measured zero motion") -- check dark_fraction to interpret.
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

    # Placeholder columns for the planned detection/tracking/classification pipeline
    # (e.g. YOLO + ByteTrack + a temporal classifier). Left as NaN deliberately --
    # optical flow cannot substitute for per-behavior-state composition.
    for col in ["frac_idle", "frac_active", "frac_drinking", "frac_feeding",
                "frac_preening", "frac_perching", "frac_wing_flapping"]:
        hourly[col] = np.nan

    return hourly


def write_metadata_log(metas: list[FileMetadata], output_dir: str, filename: str = "file_metadata_log.csv"):
    df = pd.DataFrame([m.__dict__ for m in metas])
    df.to_csv(os.path.join(output_dir, filename), index=False)

    unreadable = df[~df["readable"]]
    no_chapter_start = df[df["readable"] & df["chapter_start"].isna()]
    n_unique_creation_times = df.loc[df["readable"], "creation_time"].nunique()

    logger.info("--- Missing-data audit ---")
    logger.info("Total files found:             %d", len(df))
    logger.info("Unreadable files:              %d", len(unreadable))
    logger.info("Readable but no chapter_start: %d", len(no_chapter_start))
    logger.info("Unique creation_time values:   %d (of %d readable)",
                n_unique_creation_times, len(df[df["readable"]]))
    if n_unique_creation_times == 1 and len(df[df["readable"]]) > 1:
        logger.info("  -> creation_time identical across all files (known GoPro chapter "
                    "bug). chapter_start (from timecode tag) used for actual timestamps.")
    if len(unreadable) or len(no_chapter_start):
        logger.warning("See %s for details -- review before batch processing.", filename)


# =============================================================================
# Audit-only mode (metadata sweep, no optical flow)
# =============================================================================

def _split_into_sessions(metas: list[FileMetadata]) -> list[list[FileMetadata]]:
    """Group files into sessions: a contiguous run (in sorted filename order)
    sharing the same creation_time. A folder can legitimately contain multiple
    sessions (SD card swap, camera restart, separate recording runs) -- this is
    expected and not itself a problem; anomaly checks happen per-session."""
    sessions: list[list[FileMetadata]] = []
    current: list[FileMetadata] = []
    current_ct = object()

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
    readable = [m for m in session if m.readable]
    unique_ct = len({m.creation_time for m in readable if m.creation_time})
    bug_pattern_holds = (unique_ct == 1 and len(readable) > 1) or len(readable) <= 1

    starts = [m.chapter_start for m in readable if m.chapter_start is not None]
    monotonic = all(starts[i] < starts[i + 1] for i in range(len(starts) - 1))
    duplicates = len(starts) != len(set(starts))

    gaps_min = [(starts[i + 1] - starts[i]).total_seconds() / 60.0 for i in range(len(starts) - 1)]
    irregular_gaps = [g for g in gaps_min if g > 90]  # >90min mid-session is unusual

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


def audit_only(video_parent_dir: str, output_dir: str):
    if not os.path.isdir(video_parent_dir):
        logger.error("video_parent_dir not found: %s", video_parent_dir)
        return

    os.makedirs(output_dir, exist_ok=True)
    subfolders = sorted(
        os.path.join(video_parent_dir, d)
        for d in os.listdir(video_parent_dir)
        if os.path.isdir(os.path.join(video_parent_dir, d))
    )
    logger.info("Found %d subfolders under %s", len(subfolders), video_parent_dir)

    all_rows, session_rows, folder_rows = [], [], []

    for folder in subfolders:
        folder_name = os.path.basename(folder)
        video_files = discover_videos(folder)
        if not video_files:
            logger.info("[%s] no video files found -- skipping", folder_name)
            continue

        metas = [probe_metadata(f) for f in video_files]
        metas.sort(key=lambda m: m.filepath)
        metas = resolve_chapter_starts(metas)

        unreadable = [m for m in metas if not m.readable]
        no_timecode = [m for m in metas if m.readable and not m.timecode_raw]
        no_chapter_start = [m for m in metas if m.readable and m.chapter_start is None]

        sessions = _split_into_sessions(metas)
        session_checks = [_check_session(s) for s in sessions]
        n_anomalous = sum(1 for c in session_checks if c["anomaly"])

        logger.info("[%s] %d files | %d session(s) | unreadable=%d | no_timecode=%d | "
                    "no_chapter_start=%d | anomalous_sessions=%d/%d",
                    folder_name, len(metas), len(sessions), len(unreadable),
                    len(no_timecode), len(no_chapter_start), n_anomalous, len(sessions))

        for i, check in enumerate(session_checks):
            flag = "ANOMALY" if check["anomaly"] else "ok"
            logger.info("    session %d: %d files, %s -> %s  [%s]",
                        i + 1, check["n_files"], check["session_start"],
                        check["session_end"], flag)

        for m in metas:
            row = m.__dict__.copy()
            row["folder"] = folder_name
            all_rows.append(row)

        for i, check in enumerate(session_checks):
            session_rows.append({"folder": folder_name, "session_index": i + 1, **check})

        all_starts = [m.chapter_start for m in metas if m.chapter_start]
        folder_rows.append({
            "folder": folder_name,
            "n_files": len(metas),
            "n_sessions": len(sessions),
            "n_anomalous_sessions": n_anomalous,
            "unreadable": len(unreadable),
            "no_timecode": len(no_timecode),
            "no_chapter_start": len(no_chapter_start),
            "earliest_chapter_start": min(all_starts, default=None),
            "latest_chapter_start": max(all_starts, default=None),
        })

    pd.DataFrame(all_rows).to_csv(os.path.join(output_dir, "audit_all_files.csv"), index=False)
    pd.DataFrame(session_rows).to_csv(os.path.join(output_dir, "audit_sessions.csv"), index=False)
    folder_df = pd.DataFrame(folder_rows)
    folder_df.to_csv(os.path.join(output_dir, "audit_summary_by_folder.csv"), index=False)

    logger.info("Audit CSVs written to %s", output_dir)

    problems = folder_df[
        (folder_df["unreadable"] > 0) | (folder_df["no_timecode"] > 0) |
        (folder_df["no_chapter_start"] > 0) | (folder_df["n_anomalous_sessions"] > 0)
    ]
    if len(problems):
        logger.warning("%d folder(s) need manual review before batch processing:\n%s",
                        len(problems), problems[["folder", "unreadable", "no_timecode",
                                                  "no_chapter_start", "n_anomalous_sessions"]])
    else:
        logger.info("All folders/sessions look consistent -- safe to proceed to full batch.")


# =============================================================================
# Single-folder run
# =============================================================================

def run_single_folder(video_dir: str, output_dir: str, target_filename: str | None = None):
    os.makedirs(output_dir, exist_ok=True)

    metas = probe_and_resolve(video_dir, target_filename=target_filename)
    if not metas:
        return

    write_metadata_log(metas, output_dir)

    usable_metas = [m for m in metas if m.readable and m.chapter_start and m.fps]
    logger.info("%d/%d files usable for feature extraction", len(usable_metas), len(metas))

    records = extract_records(usable_metas)
    logger.info("Extracted %d frame-level feature records", len(records))

    hourly_df = aggregate_hourly(records)
    hourly_path = os.path.join(output_dir, "hourly_features.csv")
    hourly_df.to_csv(hourly_path, index=False)
    logger.info("Hourly features written to: %s", hourly_path)

    frame_df = pd.DataFrame([r.__dict__ for r in records])
    frame_path = os.path.join(output_dir, "frame_level_features.csv")
    frame_df.to_csv(frame_path, index=False)
    logger.info("Frame-level features written to: %s", frame_path)


# =============================================================================
# Multi-folder batch run
# =============================================================================

def run_all_folders(video_parent_dir: str, output_dir: str):
    if not os.path.isdir(video_parent_dir):
        logger.error("video_parent_dir not found: %s", video_parent_dir)
        return

    os.makedirs(output_dir, exist_ok=True)
    subfolders = sorted(
        os.path.join(video_parent_dir, d)
        for d in os.listdir(video_parent_dir)
        if os.path.isdir(os.path.join(video_parent_dir, d))
    )
    logger.info("Found %d subfolders under %s", len(subfolders), video_parent_dir)

    all_metas: list[FileMetadata] = []
    all_records: list[FrameRecord] = []
    folder_summaries = []

    for i, folder in enumerate(subfolders):
        folder_name = os.path.basename(folder)
        logger.info("=== [%d/%d] %s ===", i + 1, len(subfolders), folder_name)

        metas = probe_and_resolve(folder)
        if not metas:
            continue
        all_metas.extend(metas)

        usable_metas = [m for m in metas if m.readable and m.chapter_start and m.fps]
        logger.info("%d/%d files usable for feature extraction", len(usable_metas), len(metas))

        records = extract_records(usable_metas)
        all_records.extend(records)

        folder_summaries.append({
            "folder": folder_name, "n_files": len(metas),
            "n_usable": len(usable_metas), "n_records": len(records),
        })
        logger.info("[%s] extracted %d records (running total: %d)",
                    folder_name, len(records), len(all_records))

    metadata_rows = []
    for m in all_metas:
        row = m.__dict__.copy()
        row["folder"] = os.path.basename(os.path.dirname(m.filepath))
        metadata_rows.append(row)
    pd.DataFrame(metadata_rows).to_csv(
        os.path.join(output_dir, "file_metadata_log_all_folders.csv"), index=False)

    logger.info("Total frame-level records across all folders: %d", len(all_records))

    hourly_df = aggregate_hourly(all_records)
    hourly_path = os.path.join(output_dir, "hourly_features_all_folders.csv")
    hourly_df.to_csv(hourly_path, index=False)
    if len(hourly_df):
        logger.info("Combined hourly features written to: %s (%d rows, %s -> %s)",
                    hourly_path, len(hourly_df), hourly_df["hour"].min(), hourly_df["hour"].max())

    frame_df = pd.DataFrame([r.__dict__ for r in all_records])
    frame_df.to_csv(os.path.join(output_dir, "frame_level_features_all_folders.csv"), index=False)

    summary_df = pd.DataFrame(folder_summaries)
    summary_df.to_csv(os.path.join(output_dir, "batch_run_summary_by_folder.csv"), index=False)
    logger.info("Per-folder run summary:\n%s", summary_df)


# =============================================================================
# Dev/validation tools
# =============================================================================

def check_brightness(video_dir: str, output_dir: str, target_filename: str | None = None):
    """Grab a 10s clip (random or specified file), save original vs. brightened
    versions side by side for visual validation of BRIGHTNESS_ALPHA/BETA."""
    check_dir = os.path.join(output_dir, "brightness_check")
    os.makedirs(check_dir, exist_ok=True)

    video_files = discover_videos(video_dir)
    if not video_files:
        logger.error("No videos found in %s", video_dir)
        return

    if target_filename is not None:
        matches = [f for f in video_files if os.path.basename(f) == target_filename]
        if not matches:
            logger.error("No file named '%s' found under %s", target_filename, video_dir)
            return
        video_path = matches[0]
    else:
        video_path = random.choice(video_files)
    logger.info("Selected: %s", video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Could not open %s", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_frames = int(fps * 10)
    start_frame = 0 if total_frames <= clip_frames else random.randint(0, total_frames - clip_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    orig_writer = cv2.VideoWriter(os.path.join(check_dir, "original_clip.mp4"), fourcc, fps, (w, h))
    bright_writer = cv2.VideoWriter(os.path.join(check_dir, "brightened_clip.mp4"), fourcc, fps, (w, h))

    saved_still = False
    for i in range(clip_frames):
        ok, frame = cap.read()
        if not ok:
            break
        bright_frame = brighten(frame)
        orig_writer.write(frame)
        bright_writer.write(bright_frame)

        if not saved_still and i == clip_frames // 2:
            cv2.imwrite(os.path.join(check_dir, "original_frame.png"), frame)
            cv2.imwrite(os.path.join(check_dir, "brightened_frame.png"), bright_frame)
            mean_intensity = float(np.mean(cv2.resize(frame, (64, 36))))
            logger.info("Raw mean intensity: %.1f (dark_mean_threshold=%.1f)",
                        mean_intensity, CFG.dark_mean_threshold)
            saved_still = True

    cap.release()
    orig_writer.release()
    bright_writer.release()
    logger.info("Saved brightness check clips/stills to: %s", check_dir)


def flow_overlay(video_dir: str, output_dir: str, target_filename: str, offset_sec: float):
    """Visualize exactly which pixels count toward occupancy at a given moment --
    used to validate MOTION_THRESHOLD against real footage."""
    check_dir = os.path.join(output_dir, "brightness_check")
    os.makedirs(check_dir, exist_ok=True)

    video_files = discover_videos(video_dir)
    matches = [f for f in video_files if os.path.basename(f) == target_filename]
    if not matches:
        logger.error("No file named '%s' found under %s", target_filename, video_dir)
        return
    video_path = matches[0]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Could not open %s", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    start_idx = int(offset_sec * fps)
    gap_frames = max(1, int(round(fps * CFG.sample_every_n_sec)))

    def read_frame(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        return frame if ok else None

    frame_a, frame_b = read_frame(start_idx), read_frame(start_idx + gap_frames)
    cap.release()
    if frame_a is None or frame_b is None:
        logger.error("Could not read one or both frames -- try a smaller offset.")
        return

    def prep(frame):
        if CFG.resize_width is not None:
            h, w = frame.shape[:2]
            scale = CFG.resize_width / w
            frame = cv2.resize(frame, (CFG.resize_width, int(h * scale)))
        return brighten(frame)

    gray_a = cv2.cvtColor(prep(frame_a), cv2.COLOR_BGR2GRAY)
    bright_b = prep(frame_b)
    gray_b = cv2.cvtColor(bright_b, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        gray_a, gray_b, None, pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    occupancy = float(np.mean(magnitude > CFG.motion_threshold))

    mag_norm = np.clip(magnitude / (magnitude.max() + 1e-6) * 255, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(mag_norm, cv2.COLORMAP_JET)

    overlay = bright_b.copy()
    overlay[magnitude > CFG.motion_threshold] = [0, 0, 255]

    cv2.imwrite(os.path.join(check_dir, "raw_frame.png"), bright_b)
    cv2.imwrite(os.path.join(check_dir, "flow_heatmap.png"), heatmap)
    cv2.imwrite(os.path.join(check_dir, "motion_mask_overlay.png"), overlay)

    logger.info("File: %s | frames %d->%d | motion_threshold=%.2f | occupancy=%.4f",
                video_path, start_idx, start_idx + gap_frames, CFG.motion_threshold, occupancy)
    logger.info("Saved overlay images to: %s", check_dir)


# =============================================================================
# CLI
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract hourly video behavioral features (optical flow + "
                    "occupancy) from GoPro poultry-barn recordings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--audit-only", action="store_true",
                       help="Metadata-only sweep across --video-parent-dir (no optical "
                            "flow). Run this before a full batch.")
    mode.add_argument("--all-folders", action="store_true",
                       help="Process every subfolder under --video-parent-dir, combining "
                            "into one output.")
    mode.add_argument("--check-brightness", action="store_true",
                       help="Dev tool: save a 10s clip original vs. brightened for visual "
                            "validation.")
    mode.add_argument("--flow-overlay", action="store_true",
                       help="Dev tool: visualize which pixels count as motion at a given "
                            "offset (requires --file and --offset).")

    p.add_argument("--video-dir", type=str, default=None,
                   help="Single folder of video files to process.")
    p.add_argument("--video-parent-dir", type=str, default=None,
                   help="Parent folder containing multiple date-range subfolders "
                        "(used with --audit-only / --all-folders).")
    p.add_argument("--output-dir", type=str, required=True,
                   help="Directory to write CSV outputs and logs.")
    p.add_argument("--file", type=str, default=None,
                   help="Restrict processing to a single filename (basename match).")
    p.add_argument("--offset", type=float, default=0.0,
                   help="Offset in seconds into the file, for --flow-overlay.")

    p.add_argument("--sample-every-n-sec", type=float, default=CFG.sample_every_n_sec,
                   help="Sample one frame every N seconds of video.")
    p.add_argument("--brightness-alpha", type=float, default=CFG.brightness_alpha,
                   help="Contrast multiplier applied before flow computation.")
    p.add_argument("--brightness-beta", type=int, default=CFG.brightness_beta,
                   help="Brightness offset applied before flow computation.")
    p.add_argument("--motion-threshold", type=float, default=CFG.motion_threshold,
                   help="Flow magnitude threshold for a pixel to count as 'in motion'. "
                        "Validate with --flow-overlay before changing for a new dataset.")
    p.add_argument("--resize-width", type=int, default=CFG.resize_width,
                   help="Resize frame width before flow computation (0 to disable).")
    p.add_argument("--dark-mean-threshold", type=float, default=CFG.dark_mean_threshold,
                   help="Mean pixel intensity (0-255) below which a frame is 'dark'.")
    p.add_argument("--workers", type=int, default=CFG.n_workers,
                   help="Parallel worker processes for optical flow extraction. Tuned "
                        "for I/O-bound workloads, not raw core count -- see module docstring.")
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return p


def main(argv: list[str] | None = None):
    args = build_arg_parser().parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    CFG.output_dir = args.output_dir
    CFG.sample_every_n_sec = args.sample_every_n_sec
    CFG.brightness_alpha = args.brightness_alpha
    CFG.brightness_beta = args.brightness_beta
    CFG.motion_threshold = args.motion_threshold
    CFG.resize_width = args.resize_width if args.resize_width > 0 else None
    CFG.dark_mean_threshold = args.dark_mean_threshold
    CFG.n_workers = args.workers

    os.makedirs(CFG.output_dir, exist_ok=True)

    if args.audit_only:
        if not args.video_parent_dir:
            logger.error("--audit-only requires --video-parent-dir")
            sys.exit(1)
        audit_only(args.video_parent_dir, CFG.output_dir)

    elif args.all_folders:
        if not args.video_parent_dir:
            logger.error("--all-folders requires --video-parent-dir")
            sys.exit(1)
        run_all_folders(args.video_parent_dir, CFG.output_dir)

    elif args.check_brightness:
        if not args.video_dir:
            logger.error("--check-brightness requires --video-dir")
            sys.exit(1)
        check_brightness(args.video_dir, CFG.output_dir, target_filename=args.file)

    elif args.flow_overlay:
        if not args.video_dir or not args.file:
            logger.error("--flow-overlay requires --video-dir and --file")
            sys.exit(1)
        flow_overlay(args.video_dir, CFG.output_dir, args.file, args.offset)

    else:
        if not args.video_dir:
            logger.error("Provide --video-dir (single folder), or use --audit-only / "
                        "--all-folders with --video-parent-dir.")
            sys.exit(1)
        run_single_folder(args.video_dir, CFG.output_dir, target_filename=args.file)


if __name__ == "__main__":
    main()