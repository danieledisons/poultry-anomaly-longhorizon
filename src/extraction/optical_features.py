#!/usr/bin/env python3
"""Optical-flow features per room-month.

Run: python src/extraction/optical_features.py --room 'Room 2' --month July --in <dir> --out <dir>
"""

import argparse
import logging
import re
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import cv2

# ============================================================================
# FLOW / SAMPLING CONFIG  —  SINGLE SOURCE OF TRUTH. Identical for every room.
# Change here only; never per-room, or cross-room comparison breaks.
# ============================================================================
SAMPLE_FPS     = 5.0        # frames/sec sampled from each video before flow.
                            #   Fixed sampling => framerate-independent motion.
RESIZE_WIDTH   = 320        # downscale frames to this width (keep aspect) before
                            #   flow. Fixed size => flow magnitude comparable across
                            #   files. Speed + denoise. Log it (magnitude scales w/ res).
BIN_MINUTES    = 15         # Tier-1 temporal grain. 15 min => 96 bins/day: resolves
                            #   feeding peaks / lights on-off / disturbances without bloat.
ACTIVITY_THR   = 2.0        # px/sec: a frame-pair above this counts as "active".
                            #   Feeds vid_activity_frac (fraction of active frames).
                            #   Tune once on a sample day; keep fixed thereafter.

# Farnebäck dense optical flow params (OpenCV). Global, explainable, no GPU/model.
# (To swap in RAFT/TV-L1 later, replace compute_global_motion + record it in config.)
FLOW_ALGO      = "farneback"
FB_PARAMS      = dict(pyr_scale=0.5, levels=3, winsize=15,
                      iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

# Data folders are named by MONTH NAME (June, July, ...). Study year fixed here.
STUDY_YEAR     = 2025
MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

# Config fingerprint written into outputs so every artifact is traceable.
FLOW_CONFIG = dict(
    sample_fps=SAMPLE_FPS, resize_width=RESIZE_WIDTH, bin_minutes=BIN_MINUTES,
    activity_thr=ACTIVITY_THR, flow_algo=FLOW_ALGO, fb_params=FB_PARAMS,
    motion_units="pixels_per_second",
)


def resolve_month(name: str):
    """'July' -> (7, '2025-07'). Accepts any case; also accepts '2025-07' directly."""
    key = name.strip().lower()
    if key in MONTH_NAMES:
        mnum = MONTH_NAMES[key]
        return mnum, f"{STUDY_YEAR}-{mnum:02d}"
    m = re.fullmatch(r"(\d{4})-(\d{2})", key)
    if m:
        return int(m.group(2)), key
    raise ValueError(f"Unrecognised --month '{name}'. Use a month name (e.g. July) "
                     f"or YYYY-MM.")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(logfile: Path):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(logfile), logging.StreamHandler(sys.stdout)],
    )


# ---------------------------------------------------------------------------
# Timestamp parsing  —  VIDEO NEEDS START DATETIME (date + time-of-day),
# because each frame's wall-clock time = file_start + frame_index / SAMPLE_FPS,
# and that wall-clock time is what assigns the frame to a 15-min bin.
#   Trail/security cam:   20250711_083412  -> 2025-07-11 08:34:12
#   Extend the regexes here for other rooms/recorders.
# ---------------------------------------------------------------------------
FNAME_DT_PATTERNS = [
    (re.compile(r"(\d{8}_\d{6})"), "%Y%m%d_%H%M%S"),   # 20250711_083412
    (re.compile(r"(\d{8}T\d{6})"), "%Y%m%dT%H%M%S"),   # 20250711T083412
    (re.compile(r"(\d{6}_\d{6})"), "%y%m%d_%H%M%S"),   # 250711_083412
]

def parse_start_dt(path: Path):
    """Return a datetime (date + time-of-day) for the file's start, or None."""
    stem = path.stem
    for pat, fmt in FNAME_DT_PATTERNS:
        m = pat.search(stem)
        if m:
            try:
                return datetime.strptime(m.group(1), fmt)
            except ValueError:
                continue
    return None


# ---------------------------------------------------------------------------
# Global optical-flow motion per frame-pair
# ---------------------------------------------------------------------------
def _prep_gray(frame):
    h, w = frame.shape[:2]
    if w != RESIZE_WIDTH:
        new_h = int(round(h * (RESIZE_WIDTH / w)))
        frame = cv2.resize(frame, (RESIZE_WIDTH, new_h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def compute_global_motion(prev_gray, curr_gray):
    """One scalar: spatial-mean flow magnitude, in pixels/second."""
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, **FB_PARAMS)
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)      # px per sampled step
    return float(mag.mean()) * SAMPLE_FPS                     # -> px per second


def extract_file_motion(path: Path, start_dt: datetime):
    """
    Sample a video at SAMPLE_FPS, compute global motion per frame-pair.
    Returns (df[timestamp, motion], info_dict). df may be empty for unreadable files.
    Never raises for ordinary decode issues — logs via info and returns what it has.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return pd.DataFrame(columns=["timestamp", "motion"]), {"issue": "unreadable"}

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if src_fps <= 0:
        src_fps = 25.0                                        # sane fallback
    step = max(int(round(src_fps / SAMPLE_FPS)), 1)           # frames to skip

    times, motions = [], []
    prev_gray = None
    sample_idx = 0
    fidx = 0
    read_frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fidx % step == 0:
            read_frames += 1
            gray = _prep_gray(frame)
            if prev_gray is not None:
                t = start_dt + timedelta(seconds=sample_idx / SAMPLE_FPS)
                times.append(t)
                motions.append(compute_global_motion(prev_gray, gray))
                sample_idx += 1
            else:
                sample_idx += 1   # first sampled frame seeds prev; no pair yet
            prev_gray = gray
        fidx += 1
    cap.release()

    df = pd.DataFrame({"timestamp": times, "motion": motions})
    info = {"src_fps": round(src_fps, 3), "frames_read": read_frames,
            "pairs": len(df), "issue": None if len(df) else "no_frame_pairs"}
    return df, info


# ---------------------------------------------------------------------------
# Binning helpers
# ---------------------------------------------------------------------------
def assign_bins(df):
    """Add date + time_bin (floor to BIN_MINUTES) columns from timestamp."""
    ts = pd.to_datetime(df["timestamp"])
    floored = ts.dt.floor(f"{BIN_MINUTES}min")
    out = df.copy()
    out["date"] = floored.dt.strftime("%Y-%m-%d")
    out["time_bin"] = floored.dt.strftime("%Y-%m-%d %H:%M")
    return out

def _pct(series, q):
    return float(np.percentile(series, q)) if len(series) else np.nan

# Expected sampled frame-PAIRS in a fully-covered bin (for coverage fraction).
EXPECTED_PAIRS_PER_BIN = SAMPLE_FPS * BIN_MINUTES * 60


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", required=True)
    ap.add_argument("--month", required=True, help="month name e.g. July (or YYYY-MM)")
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--out", required=True,
                    help="output stem; _15min.parquet and _daily.parquet are appended")
    ap.add_argument("--logdir", default="results/logs")
    ap.add_argument("--exts", default="mp4,MP4,avi,AVI,mov,MOV,mkv,MKV")
    ap.add_argument("--keep-checkpoints", action="store_true",
                    help="do not delete the per-file checkpoint dir after success")
    args = ap.parse_args()

    month_num, month_iso = resolve_month(args.month)
    month_name = args.month.strip().capitalize()

    room_tag = args.room.replace(" ", "")
    setup_logging(Path(args.logdir) / f"{room_tag}_{month_name}_video.log")
    logging.info(f"START room={args.room} month={month_name} ({month_iso})")
    logging.info(f"flowcfg={json.dumps(FLOW_CONFIG)}")

    out_stem     = Path(args.out)
    tier1_path   = Path(f"{out_stem}_15min.parquet")
    tier2_path   = Path(f"{out_stem}_daily.parquet")
    quality_path = Path(f"{out_stem}_quality.csv")
    ckpt_dir     = Path(f"{out_stem}_ckpt")          # per-file checkpoints live here
    manifest_path = ckpt_dir / "_manifest.json"

    # ---- resumability level 1: skip whole month if final outputs already exist ----
    if tier1_path.exists() and tier2_path.exists():
        logging.info(f"OUTPUTS EXIST, skipping month: {tier1_path}, {tier2_path}")
        return

    # ---- resumability level 2: load manifest of already-processed files ----
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        logging.info(f"RESUME: {len(manifest['done'])} file(s) already checkpointed")
    else:
        manifest = {"done": {}}   # stem -> quality dict

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2))

    exts = tuple("." + e.lstrip(".") for e in args.exts.split(","))
    files = sorted(p for p in Path(args.indir).rglob("*") if p.suffix in exts)
    logging.info(f"found {len(files)} video files under {args.indir}")

    # ---- per-file processing with checkpointing ----
    for i, f in enumerate(files, 1):
        if f.stem in manifest["done"]:
            logging.info(f"[{i}/{len(files)}] SKIP (checkpointed): {f.name}")
            continue
        try:
            start_dt = parse_start_dt(f)
            if start_dt is None:
                logging.warning(f"NO TIMESTAMP, skip: {f.name}")
                manifest["done"][f.stem] = dict(file=f.name, issue="no_timestamp")
                save_manifest()
                continue
            if start_dt.month != month_num:
                logging.warning(f"MONTH MISMATCH: {f.name} starts {start_dt.date()} "
                                f"but folder is {month_name}")

            df, info = extract_file_motion(f, start_dt)
            if df.empty:
                logging.warning(f"NO FRAME PAIRS ({info.get('issue')}): {f.name}")
                manifest["done"][f.stem] = dict(file=f.name, date=str(start_dt.date()),
                                                issue=info.get("issue", "empty"))
                save_manifest()
                continue

            # write this file's per-frame motion to a checkpoint parquet, then record it.
            # (raw per-frame values kept so cross-file bin boundaries aggregate exactly,
            #  including percentiles, in the final pass.)
            df = assign_bins(df)
            df.insert(0, "src_file", f.name)
            df.to_parquet(ckpt_dir / f"{f.stem}.parquet")

            manifest["done"][f.stem] = dict(
                file=f.name, date=str(start_dt.date()),
                src_fps=info["src_fps"], pairs=info["pairs"], issue=None,
            )
            save_manifest()      # <-- durable after EVERY file: crash-safe resume point
            logging.info(f"[{i}/{len(files)}] ok pairs={info['pairs']}: {f.name}")

        except Exception as e:
            # one bad file never kills the month; it simply isn't marked done, so a
            # re-run retries it.
            logging.error(f"FAILED {f.name}: {e}")

    # ---- write per-file quality CSV (feeds inventory 'missing data' column) ----
    quality_rows = list(manifest["done"].values())
    pd.DataFrame(quality_rows).to_csv(quality_path, index=False)
    logging.info(f"WROTE {quality_path} (per-file quality)")

    # ---- aggregate all checkpoints -> Tier 1 (15-min) then Tier 2 (daily) ----
    ckpt_files = sorted(p for p in ckpt_dir.glob("*.parquet"))
    frames = []
    for p in ckpt_files:
        try:
            part = pd.read_parquet(p)
            if len(part):
                frames.append(part)
        except Exception as e:
            logging.error(f"CKPT READ FAILED {p.name}: {e}")

    if not frames:
        logging.error("NO DATA AGGREGATED — writing nothing. Check inputs/timestamps.")
        sys.exit(1)     # non-zero => driver keeps raw video, stops before delete

    allf = pd.concat(frames, ignore_index=True)

    # Tier 1: one row per (date, time_bin)
    def bin_stats(g):
        m = g["motion"].values
        n = len(m)
        return pd.Series({
            "vid_flow_mag_mean": float(np.mean(m)),
            "vid_flow_mag_std":  float(np.std(m)),
            "vid_flow_mag_max":  float(np.max(m)),
            "vid_flow_mag_p95":  _pct(m, 95),
            "vid_flow_mag_p05":  _pct(m, 5),
            "vid_activity_frac": float(np.mean(m > ACTIVITY_THR)),
            "vid_n_frames":      int(n),
            "vid_coverage":      min(n / EXPECTED_PAIRS_PER_BIN, 1.0),
        })

    tier1 = (allf.groupby(["date", "time_bin"], sort=True)
                 .apply(bin_stats).reset_index())
    tier1.insert(1, "room", args.room)
    tier1["month"] = month_name
    tier1["month_iso"] = month_iso
    tier1.attrs["flowcfg"] = FLOW_CONFIG
    tier1_path.parent.mkdir(parents=True, exist_ok=True)
    tier1.to_parquet(tier1_path)
    logging.info(f"WROTE {tier1_path}  bins={len(tier1)}")

    # Tier 2: roll Tier-1 bins up to one row per (date). Recompute frame-weighted
    # stats from the raw frames for correctness (not an average-of-averages).
    expected_day = SAMPLE_FPS * 86400   # sampled pairs in a perfectly-covered day

    def day_stats(g):
        m = g["motion"].values
        n = len(m)
        return pd.Series({
            "vid_flow_mag_mean": float(np.mean(m)),
            "vid_flow_mag_std":  float(np.std(m)),
            "vid_flow_mag_max":  float(np.max(m)),
            "vid_flow_mag_p95":  _pct(m, 95),
            "vid_flow_mag_p05":  _pct(m, 5),
            "vid_activity_frac": float(np.mean(m > ACTIVITY_THR)),
            "vid_n_frames":      int(n),
            "vid_coverage":      min(n / expected_day, 1.0),
        })

    tier2 = allf.groupby("date", sort=True).apply(day_stats).reset_index()

    # number of populated 15-min bins per day (diurnal completeness signal)
    bins_per_day = tier1.groupby("date")["time_bin"].nunique().rename("vid_n_bins")
    tier2 = tier2.merge(bins_per_day, on="date", how="left")
    tier2.insert(1, "room", args.room)
    tier2["month"] = month_name
    tier2["month_iso"] = month_iso
    tier2.attrs["flowcfg"] = FLOW_CONFIG
    tier2.to_parquet(tier2_path)
    logging.info(f"WROTE {tier2_path}  days={len(tier2)}")

    # ---- cleanup checkpoints on full success (unless asked to keep) ----
    if not args.keep_checkpoints:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        logging.info(f"removed checkpoint dir {ckpt_dir}")
    else:
        logging.info(f"kept checkpoint dir {ckpt_dir}")

    logging.info("DONE")


if __name__ == "__main__":
    main()