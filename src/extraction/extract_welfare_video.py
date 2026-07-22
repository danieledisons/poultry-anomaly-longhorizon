#!/usr/bin/env python3
"""
WELFARE-RELEVANT VIDEO FEATURE EXTRACTION  (poultry behaviour & spatial welfare).

Raw optical-flow energy is camera- and lighting-dependent and carries little direct
welfare meaning. This extractor produces NORMALISED, COMPOSITIONAL, SPATIAL features
that are far more room-invariant and welfare-interpretable, per hour.

Two paths:
  --mode occupancy   (default, NO model, no labels): background-subtraction +
     foreground geometry. Gives occupancy, spatial distribution, dispersion /
     huddling index, wall-vs-centre spread (thermal comfort), zone occupancy, and
     activity NORMALISED BY OCCUPANCY (motion per bird-area, not total motion).
  --mode detector    (optional, needs ultralytics + a bird/poultry model): per-bird
     boxes -> count, nearest-neighbour distance distribution (huddling), zone dwell
     (feeder/drinker if a zones JSON is given), and per-bird speed via simple
     centroid tracking.

Welfare rationale (why these features):
  huddling/dispersion  -> cold vs heat stress, illness clustering
  wall-vs-centre       -> heat stress pushes birds to cooler perimeter
  zone occupancy/dwell -> feeding & drinking behaviour; a drop in drinker use is an
                          early disease sign
  activity per bird    -> lethargy / hyperactivity, gain-invariant
  spatial entropy      -> even use of the house vs clumping

Dark-hour handling: mean luminance below --dark_lum marks the frame UNLIT and all
features are NaN for that hour fraction (video is undefined in darkness; we never
impute it as "no activity").

Sampling: one frame every --sample_s seconds is enough for hourly aggregates and
keeps the job tractable overnight.

Writes to a NEW filename; never overwrites existing features.

Dependencies: numpy pandas opencv-python  (+ ultralytics for --mode detector)

Usage (model-free, recommended first):
  python src/extraction/extract_welfare_video.py --room "Room 2" \
      --in "/mnt/.../Video data/Room2" \
      --out features/video_welfare_spatial_Room2.csv --mode occupancy

Usage (detector):
  ... --mode detector --weights yolov8n.pt --zones zones_room2.json
"""
from __future__ import annotations
import argparse, logging, re, sys, json, subprocess
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import cv2

SAMPLE_S   = 2.0        # analyse one frame every SAMPLE_S seconds
DARK_LUM   = 25         # mean luminance (0-255) below which the frame is "unlit"
GRID       = 6          # GRID x GRID spatial occupancy grid for entropy/zone stats
FG_MIN_AREA = 30        # min foreground blob area (px) to count as a bird cluster
PERCENTILES = [10, 50, 90]


def setup_logging(logfile):
    Path(logfile).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(logfile), logging.StreamHandler(sys.stdout)])


FNAME_DT_PATTERNS = [
    (re.compile(r"(\d{8})_(\d{6})"), "%Y%m%d%H%M%S"),
    (re.compile(r"(\d{6})_(\d{6})"), "%y%m%d%H%M%S"),
]
STUDY_YEAR = 2025
MONTHS = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,"aug":8,
          "sep":9,"oct":10,"nov":11,"dec":12}


def _ffprobe_creation_time(path: Path):
    """GoPro/most cameras write recording start into MP4 metadata `creation_time`.
    This is the RELIABLE timestamp (filenames like GX010028.MP4 have none)."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_entries", "format_tags=creation_time:stream_tags=creation_time",
             str(path)], capture_output=True, text=True, timeout=30).stdout
        j = json.loads(out)
        ct = (j.get("format", {}).get("tags", {}) or {}).get("creation_time")
        if not ct:
            for s in j.get("streams", []):
                ct = (s.get("tags", {}) or {}).get("creation_time")
                if ct:
                    break
        if ct:
            return datetime.fromisoformat(ct.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        pass
    return None


def _ffprobe_duration(path: Path):
    try:
        out = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries",
             "format=duration", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30).stdout.strip()
        return float(out) if out else None
    except Exception:
        return None


# GoPro filename: G[X|H] <chapter:2> <group:4> .MP4  e.g. GX02 0028 -> chapter 2, group 0028
GOPRO_RE = re.compile(r"G[XH](\d{2})(\d{4})$", re.IGNORECASE)


def build_start_times(files):
    """Resolve a correct start datetime per file. GoPro splits one continuous
    recording into chapters that ALL share the same metadata creation_time, so we
    group by (folder, group-id), order by chapter, and offset each chapter by the
    cumulative duration of the earlier chapters. Non-GoPro files use the normal
    single-file resolver."""
    groups = {}; singles = []
    for f in files:
        m = GOPRO_RE.search(f.stem)
        if m:
            groups.setdefault((str(f.parent), m.group(2)), []).append((int(m.group(1)), f))
        else:
            singles.append(f)
    starts = {}
    for key, lst in groups.items():
        lst.sort()
        base = None; cum = 0.0
        for ch, f in lst:
            ct = _ffprobe_creation_time(f)
            if base is None:
                base = ct or _folder_date(f) or datetime.fromtimestamp(f.stat().st_mtime)
            starts[f] = base + timedelta(seconds=cum)   # chapter offset by prior durations
            cum += (_ffprobe_duration(f) or 0.0)
    for f in singles:
        starts[f] = file_start_datetime(f)
    return starts


def _folder_date(path: Path):
    """Parse the first date from a folder name like 'Room 2 (10, 11, 12, 13 Aug)'."""
    for parent in [path.parent.name, path.parent.parent.name]:
        m = re.search(r"(\d{1,2})[^)]*?\b([A-Za-z]{3})", parent)
        if m:
            mon = MONTHS.get(m.group(2).lower()[:3])
            if mon:
                try:
                    return datetime(STUDY_YEAR, mon, int(m.group(1)))
                except ValueError:
                    pass
    return None


def file_start_datetime(path: Path):
    # 1) filename with explicit datetime (AudioMoth-style, if ever present)
    for pat, fmt in FNAME_DT_PATTERNS:
        m = pat.search(path.stem)
        if m:
            try:
                return datetime.strptime(m.group(1) + m.group(2), fmt)
            except ValueError:
                pass
    # 2) MP4 metadata creation_time (the reliable source for GoPro)
    ct = _ffprobe_creation_time(path)
    if ct is not None:
        return ct
    # 3) folder-name date (GoPro filenames carry no date) — day granularity only
    fd = _folder_date(path)
    if fd is not None:
        logging.warning(f"{path.name}: no metadata time -> using folder date {fd.date()} "
                        f"(hourly precision degraded)")
        return fd
    # 4) last resort: file mtime (may be copy time, not recording time)
    try:
        logging.warning(f"{path.name}: falling back to mtime (may be wrong)")
        return datetime.fromtimestamp(path.stat().st_mtime)
    except Exception:
        return None


def spatial_stats(points, w, h):
    """points: Nx2 array of foreground/bird centroids. Returns welfare geometry."""
    out = dict(occupancy_centroids=len(points),
               disp_mean_nn=np.nan, disp_std_nn=np.nan, huddle_index=np.nan,
               spread_x=np.nan, spread_y=np.nan, wall_frac=np.nan,
               spatial_entropy=np.nan, center_frac=np.nan)
    if len(points) == 0:
        return out
    p = np.asarray(points, float)
    out["spread_x"] = float(p[:, 0].std() / w); out["spread_y"] = float(p[:, 1].std() / h)
    # nearest-neighbour distances (huddling: small NN distances => clustered)
    if len(p) >= 2:
        d = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        nn = d.min(1) / np.hypot(w, h)          # normalized by frame diagonal
        out["disp_mean_nn"] = float(np.mean(nn)); out["disp_std_nn"] = float(np.std(nn))
        out["huddle_index"] = float(1.0 / (np.mean(nn) + 1e-6))
    # wall vs centre (heat stress -> perimeter)
    margin_x, margin_y = 0.15 * w, 0.15 * h
    wall = ((p[:, 0] < margin_x) | (p[:, 0] > w - margin_x) |
            (p[:, 1] < margin_y) | (p[:, 1] > h - margin_y))
    out["wall_frac"] = float(wall.mean()); out["center_frac"] = float(1 - wall.mean())
    # grid occupancy entropy (even use vs clumping)
    gx = np.clip((p[:, 0] / w * GRID).astype(int), 0, GRID - 1)
    gy = np.clip((p[:, 1] / h * GRID).astype(int), 0, GRID - 1)
    hist = np.zeros(GRID * GRID); np.add.at(hist, gy * GRID + gx, 1)
    pr = hist / (hist.sum() + 1e-9)
    out["spatial_entropy"] = float(-(pr * np.log(pr + 1e-12)).sum() / np.log(GRID * GRID))
    return out


def occupancy_frame(fg, gray):
    """Model-free: foreground mask -> cluster centroids + occupancy fraction."""
    h, w = fg.shape
    n = cv2.connectedComponentsWithStats((fg > 0).astype(np.uint8), 8)
    _, _, stats, cents = n
    pts = [cents[i] for i in range(1, len(cents)) if stats[i, cv2.CC_STAT_AREA] >= FG_MIN_AREA]
    occ_frac = float((fg > 0).mean())
    return pts, occ_frac


def process_video_occupancy(path, start_dt):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(int(SAMPLE_S * fps), 1)
    bg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)
    rows = []; prev_gray = None; idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lum = float(gray.mean())
            t = start_dt + timedelta(seconds=idx / fps)
            if lum < DARK_LUM:                       # unlit: undefined, record NaN row
                rows.append(dict(time=t, lit=0)); prev_gray = gray; idx += 1; continue
            fg = bg.apply(frame)
            h, w = gray.shape
            pts, occ = occupancy_frame(fg, gray)
            st = spatial_stats(pts, w, h)
            # activity normalised by occupancy (motion per occupied area)
            act = np.nan; flow = {}
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray).mean()
                act = float(diff / (occ + 1e-3))
                # dense optical-flow MOMENTS (welfare-relevant: activity, unrest,
                # heterogeneity, fear response). Cheap: one Farneback per sample.
                fl = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                        0.5, 3, 15, 3, 5, 1.2, 0)
                mag = np.sqrt(fl[..., 0] ** 2 + fl[..., 1] ** 2)
                m = mag.ravel(); mu = m.mean() + 1e-9
                flow = dict(flow_mean=float(m.mean()), flow_var=float(m.var()),
                            flow_skew=float(((m - m.mean()) ** 3).mean() / (m.std() ** 3 + 1e-9)),
                            flow_kurt=float(((m - m.mean()) ** 4).mean() / (m.var() ** 2 + 1e-9)),
                            moving_frac=float((m > 0.5).mean()),
                            flow_p90=float(np.percentile(m, 90)))
            rows.append(dict(time=t, lit=1, occupancy_frac=occ,
                             activity_per_occ=act, **st, **flow))
            prev_gray = gray
        idx += 1
    cap.release()
    return rows


def process_video_detector(path, start_dt, model, zones):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(int(SAMPLE_S * fps), 1)
    rows = []; idx = 0; prev_c = None
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY); lum = float(gray.mean())
            t = start_dt + timedelta(seconds=idx / fps)
            if lum < DARK_LUM:
                rows.append(dict(time=t, lit=0)); idx += 1; continue
            h, w = gray.shape
            res = model(frame, verbose=False)[0]
            boxes = res.boxes.xywh.cpu().numpy() if res.boxes is not None else np.empty((0, 4))
            cents = boxes[:, :2] if len(boxes) else np.empty((0, 2))
            st = spatial_stats(list(cents), w, h)
            row = dict(time=t, lit=1, bird_count=len(cents), **st)
            # zone dwell (feeder/drinker) if zones provided (normalized polygons)
            for zname, poly in (zones or {}).items():
                pol = (np.array(poly) * [w, h]).astype(np.int32)
                inside = sum(cv2.pointPolygonTest(pol, (float(x), float(y)), False) >= 0
                             for x, y in cents)
                row[f"zone_{zname}_count"] = inside
                row[f"zone_{zname}_frac"] = inside / max(len(cents), 1)
            # mean per-bird speed via nearest-centroid matching to previous frame
            if prev_c is not None and len(cents) and len(prev_c):
                d = np.sqrt(((cents[:, None, :] - prev_c[None, :, :]) ** 2).sum(-1))
                row["mean_speed"] = float((d.min(1) / np.hypot(w, h)).mean() / SAMPLE_S)
            prev_c = cents
            rows.append(row)
        idx += 1
    cap.release()
    return rows


def aggregate_hourly(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"]); df["hour"] = df["time"].dt.floor("h")
    lit = df.groupby("hour")["lit"].mean().rename("lit_fraction")
    litdf = df[df["lit"] == 1].drop(columns=["lit"])
    cont = [c for c in litdf.columns if c not in ("time", "hour")]
    agg = {}
    for c in cont:
        g = litdf.groupby("hour")[c]
        agg[f"{c}_mean"] = g.mean(); agg[f"{c}_std"] = g.std()
        for p in PERCENTILES:
            agg[f"{c}_p{p}"] = g.quantile(p / 100)
    out = pd.DataFrame(agg).join(lit).reset_index().rename(columns={"hour": "time"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", required=True)
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["occupancy", "detector"], default="occupancy")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--zones", default=None, help="JSON: {name:[[x,y],...normalized]}")
    ap.add_argument("--exts", default="mp4,MP4,avi,mkv,mov")
    ap.add_argument("--log", default="src/welfare_video.log")
    args = ap.parse_args()
    setup_logging(args.log)

    out_path = Path(args.out)
    if out_path.exists():
        logging.info(f"OUTPUT EXISTS, not overwriting: {out_path}"); return

    model = zones = None
    if args.mode == "detector":
        from ultralytics import YOLO
        model = YOLO(args.weights)
        if args.zones:
            zones = json.load(open(args.zones))
        logging.info(f"detector mode: {args.weights}")

    exts = tuple("." + e.lstrip(".") for e in args.exts.split(","))
    files = sorted(p for p in Path(args.indir).rglob("*") if p.suffix in exts)
    logging.info(f"{len(files)} videos under {args.indir}")
    logging.info("resolving GoPro chapter start times (ffprobe)...")
    start_times = build_start_times(files)

    all_rows = []; ok = 0
    for i, f in enumerate(files, 1):
        try:
            sdt = start_times.get(f)
            if sdt is None:
                logging.warning(f"NO TIMESTAMP skip {f.name}"); continue
            rows = (process_video_detector(f, sdt, model, zones) if args.mode == "detector"
                    else process_video_occupancy(f, sdt))
            all_rows.extend(rows); ok += 1
            if i % 10 == 0 or i == len(files):
                logging.info(f"[{i}/{len(files)}] ok={ok} rows={len(all_rows)}")
        except Exception as e:
            logging.error(f"FAILED {f.name}: {e}")

    if not all_rows:
        logging.error("NO DATA — nothing written."); sys.exit(1)
    hourly = aggregate_hourly(all_rows).sort_values("time").reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(out_path, index=False)
    logging.info(f"WROTE {out_path}  hours={len(hourly)}  cols={hourly.shape[1]}  files_ok={ok}/{len(files)}")


if __name__ == "__main__":
    main()
