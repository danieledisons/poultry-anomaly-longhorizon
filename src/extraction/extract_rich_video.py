#!/usr/bin/env python3
"""Hourly rich video features (flow-magnitude histogram + 4x4 motion grid) from GoPro clips.

Run: python src/extraction/extract_rich_video.py --all-folders --video-parent-dir <dir>
"""
from __future__ import annotations
import argparse, glob, json, os, subprocess, sys
import datetime as dt
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import cv2

# --- rich-feature params ---
HIST_BINS = 32
HIST_MAX = 20.0
GRID = 4
# --- preprocessing params REPLICATED from extract_video_features.py ---
SAMPLE_EVERY_N_SEC = 2.0
BRIGHT_ALPHA, BRIGHT_BETA = 1.3, 25
MOTION_THRESHOLD = 1.2
RESIZE_WIDTH = 640
DARK_MEAN_THRESHOLD = 15.0
CV2_THREADS = 1
VIDEO_EXTS = (".mp4", ".MP4", ".mov", ".MOV")


@dataclass
class FileMetadata:
    filepath: str
    creation_time: "dt.datetime | None"
    timecode_raw: "str | None"
    fps: "float | None"
    readable: bool
    error: str = ""
    chapter_start: "dt.datetime | None" = None


# ---- discovery + GoPro timestamp resolution (copied from main pipeline) ----
def discover_videos(video_dir):
    files = []
    for ext in VIDEO_EXTS:
        files.extend(glob.glob(os.path.join(video_dir, f"**/*{ext}"), recursive=True))
    return sorted(set(files))


def probe_metadata(filepath):
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_format", "-show_streams", filepath]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True)
        info = json.loads(result.stdout)
        fmt = info.get("format", {}); tags = fmt.get("tags", {})
        creation_raw = tags.get("creation_time"); creation_time = None
        if creation_raw:
            try:
                creation_time = dt.datetime.fromisoformat(creation_raw.replace("Z", "+00:00"))
            except ValueError:
                creation_time = None
        vs = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
        fps = None
        if vs.get("avg_frame_rate"):
            num, den = vs["avg_frame_rate"].split("/"); den = float(den) if float(den) != 0 else 1.0
            fps = float(num) / den
        return FileMetadata(filepath, creation_time, vs.get("tags", {}).get("timecode"), fps, True)
    except Exception as e:
        return FileMetadata(filepath, None, None, None, False, str(e))


def _parse_smpte_timecode(timecode_raw, fps):
    if not timecode_raw:
        return None
    try:
        parts = timecode_raw.replace(";", ":").split(":")
        if len(parts) != 4:
            return None
        hh, mm, ss, ff = (int(p) for p in parts)
        micro = int(round((ff / fps) * 1_000_000)) if fps else 0
        return dt.time(hour=hh % 24, minute=mm, second=ss, microsecond=micro)
    except (ValueError, ZeroDivisionError):
        return None


def resolve_chapter_starts(metas):
    base_date = next((m.creation_time.date() for m in metas if m.creation_time), None)
    if base_date is None:
        return metas
    current_date = base_date; prev_tod = None
    for m in metas:
        tod = _parse_smpte_timecode(m.timecode_raw, m.fps) if (m.timecode_raw and m.fps) else None
        if tod is None:
            m.chapter_start = None; continue
        if prev_tod is not None and tod < prev_tod:
            current_date += dt.timedelta(days=1)
        m.chapter_start = dt.datetime.combine(current_date, tod); prev_tod = tod
    return metas


def probe_and_resolve(video_dir):
    metas = [probe_metadata(f) for f in discover_videos(video_dir)]
    metas.sort(key=lambda m: m.filepath)
    return resolve_chapter_starts(metas)


# ---- preprocessing (copied) ----
def brighten(frame):
    return cv2.convertScaleAbs(frame, alpha=BRIGHT_ALPHA, beta=BRIGHT_BETA)


def is_dark_frame(frame):
    return float(np.mean(cv2.resize(frame, (64, 36)))) < DARK_MEAN_THRESHOLD


# ---- rich flow features ----
def flow_feats(prev_gray, gray):
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    hist, _ = np.histogram(np.clip(mag, 0, HIST_MAX), bins=HIST_BINS, range=(0, HIST_MAX))
    hist = hist.astype(np.float64); hist /= max(hist.sum(), 1)
    H, W = mag.shape; gh, gw = H // GRID, W // GRID
    gm = np.zeros(GRID * GRID); gs = np.zeros(GRID * GRID)
    for i in range(GRID):
        for j in range(GRID):
            cell = mag[i*gh:(i+1)*gh, j*gw:(j+1)*gw]
            gm[i*GRID+j] = cell.mean(); gs[i*GRID+j] = cell.std()
    return dict(hist=hist, gm=gm, gs=gs, flow_mean=float(mag.mean()),
                moving=float((mag > MOTION_THRESHOLD).mean()))


def process_video(meta):
    cv2.setNumThreads(CV2_THREADS)
    if not meta.readable or meta.chapter_start is None or meta.fps is None:
        return []
    cap = cv2.VideoCapture(meta.filepath)
    if not cap.isOpened():
        return []
    interval = max(1, int(round(meta.fps * SAMPLE_EVERY_N_SEC)))
    out = []; prev_gray = None; idx = 0
    ok, frame = cap.read()
    while ok:
        if idx % interval == 0:
            ts = meta.chapter_start + dt.timedelta(seconds=idx / meta.fps)
            if is_dark_frame(frame):
                out.append((ts, None)); prev_gray = None            # dark marker
            else:
                if RESIZE_WIDTH:
                    h, w = frame.shape[:2]; frame_r = cv2.resize(frame, (RESIZE_WIDTH, int(h*RESIZE_WIDTH/w)))
                else:
                    frame_r = frame
                gray = cv2.cvtColor(brighten(frame_r), cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    out.append((ts, flow_feats(prev_gray, gray)))
                prev_gray = gray
        idx += 1; ok, frame = cap.read()
    cap.release()
    return out


def aggregate_hourly(records):
    """records: list of (timestamp, feat_dict_or_None). None = dark frame."""
    if not records:
        return pd.DataFrame()
    rows = []
    df = pd.DataFrame({"ts": [r[0] for r in records], "feat": [r[1] for r in records]})
    df["hour"] = pd.to_datetime(df["ts"]).dt.floor("h")
    for hr, sub in df.groupby("hour"):
        feats = [f for f in sub["feat"] if f is not None]
        dark_frac = float(sub["feat"].isna().mean())
        row = {"time": hr, "n_pairs": len(feats), "dark_fraction": dark_frac}
        if feats:
            row["flow_mean_avg"] = float(np.mean([f["flow_mean"] for f in feats]))
            row["moving_frac_avg"] = float(np.mean([f["moving"] for f in feats]))
            hist = np.mean([f["hist"] for f in feats], axis=0)
            gm = np.mean([f["gm"] for f in feats], axis=0)
            gs = np.mean([f["gs"] for f in feats], axis=0)
            row.update({f"flowhist{b:02d}": hist[b] for b in range(HIST_BINS)})
            row.update({f"gridmean{c:02d}": gm[c] for c in range(GRID*GRID)})
            row.update({f"gridstd{c:02d}": gs[c] for c in range(GRID*GRID)})
        else:
            row["flow_mean_avg"] = np.nan; row["moving_frac_avg"] = np.nan
            for b in range(HIST_BINS): row[f"flowhist{b:02d}"] = np.nan
            for c in range(GRID*GRID): row[f"gridmean{c:02d}"] = np.nan; row[f"gridstd{c:02d}"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("time")


def run_folders(folders, output_dir, workers, out_name):
    import time
    all_records = []
    # pre-scan so we can show global progress (N of TOTAL videos).
    # Print per folder so this ffprobe phase isn't silent on large batches.
    print(f"[scan] probing metadata for {len(folders)} folder(s) (ffprobe)...", flush=True)
    folder_metas = []
    total_usable = 0
    for fi, folder in enumerate(folders, 1):
        metas = probe_and_resolve(folder)
        usable = [m for m in metas if m.readable and m.chapter_start and m.fps]
        folder_metas.append((folder, metas, usable))
        total_usable += len(usable)
        print(f"  [scan {fi}/{len(folders)}] {os.path.basename(folder)}: "
              f"{len(usable)}/{len(metas)} usable (running total {total_usable})", flush=True)
    print(f"[plan] {len(folders)} folder(s), {total_usable} usable videos to process", flush=True)
    done = 0; t_start = time.time()
    for k, (folder, metas, usable) in enumerate(folder_metas, 1):
        print(f"[{k}/{len(folders)}] {os.path.basename(folder)}: {len(usable)}/{len(metas)} usable")
        if not usable:
            continue
        if workers and workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(process_video, m): m for m in usable}
                for fut in as_completed(futs):
                    try:
                        all_records.extend(fut.result())
                    except Exception as e:
                        print("  [WARN]", futs[fut].filepath, e)
                    done += 1
                    rate = done / max(time.time() - t_start, 1e-6)
                    eta = (total_usable - done) / rate / 60 if rate > 0 else float("nan")
                    print(f"  [{done}/{total_usable}] videos done  "
                          f"({rate*60:.1f}/min, ETA ~{eta:.0f} min, records so far {len(all_records)})",
                          flush=True)
        else:
            for m in usable:
                all_records.extend(process_video(m)); done += 1
                print(f"  [{done}/{total_usable}] videos done (records {len(all_records)})", flush=True)
    hourly = aggregate_hourly(all_records)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, out_name)
    hourly.to_csv(out_path, index=False)
    if len(hourly):
        print(f"[write] {out_path}  ({len(hourly)} hours x {hourly.shape[1]} cols, "
              f"{hourly['time'].min()} -> {hourly['time'].max()})")
    else:
        print(f"[write] {out_path}  (EMPTY - check timestamps/ffprobe)")


def self_test():
    rng = np.random.default_rng(0)
    a = rng.normal(80, 5, (120, 160)).astype(np.uint8)
    b = np.roll(a, 3, axis=1)
    f = flow_feats(a, b)
    ncols = HIST_BINS + 2*GRID*GRID
    print(f"[self-test] hist={len(f['hist'])}(={HIST_BINS}) grid={len(f['gm'])}+{len(f['gs'])}(=2x{GRID*GRID})")
    print(f"[self-test] flow_mean={f['flow_mean']:.2f}px moving={f['moving']:.2f}  per-hour feat dim={ncols}")
    assert len(f['hist']) == HIST_BINS and abs(f['hist'].sum()-1) < 1e-6
    # aggregation incl a dark marker
    recs = [(dt.datetime(2025,8,15,9,0,0), f), (dt.datetime(2025,8,15,9,1,0), None),
            (dt.datetime(2025,8,15,9,2,0), f)]
    h = aggregate_hourly(recs)
    print(f"[self-test] hourly rows={len(h)} cols={h.shape[1]} dark_fraction={h['dark_fraction'].iloc[0]:.2f}")
    assert h['dark_fraction'].iloc[0] > 0
    print("[self-test] PASS")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--all-folders", action="store_true")
    ap.add_argument("--video-dir"); ap.add_argument("--video-parent-dir")
    ap.add_argument("--output-dir", default="./features_room2/video")
    ap.add_argument("--out-name", default="video_rich_features_hourly_Room2.csv")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test: self_test(); return
    if a.all_folders:
        if not a.video_parent_dir: ap.error("--all-folders requires --video-parent-dir")
        subs = sorted(os.path.join(a.video_parent_dir, d) for d in os.listdir(a.video_parent_dir)
                      if os.path.isdir(os.path.join(a.video_parent_dir, d)))
        run_folders(subs, a.output_dir, a.workers, a.out_name)
    else:
        if not a.video_dir: ap.error("provide --video-dir, or --all-folders with --video-parent-dir")
        run_folders([a.video_dir], a.output_dir, a.workers, a.out_name)


if __name__ == "__main__":
    main()