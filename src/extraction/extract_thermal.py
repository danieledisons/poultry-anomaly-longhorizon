#!/usr/bin/env python3
"""
extract_thermal.py — batch thermal feature extraction (barn-level).

Walks a folder tree of FLIR images, extracts whole-frame radiometric summary
statistics on Otsu-masked BIRD pixels, and writes two CSVs:

  1. <out>/thermal_frame_features.csv   one row per image (timestamped)
  2. <out>/thermal_daily_features.csv   one row per calendar day (aggregates)

Design (matches the project's flock-level, no-ROI decision):
  - bird pixels isolated by an Otsu temperature threshold (birds warmer than litter)
  - features: mean / p10 / p90 bird surface temp, hotspot_frac (heads/feet proxy),
    spatial heterogeneity, bird_frac (occupancy proxy), ambient_est (barn temp proxy)
  - timestamp from EXIF (camera_metadata.date_time); falls back to file mtime
  - NO room attribution (thermal is barn-level); a `source_path` column is kept so a
    room key can be re-derived later if folder structure ever encodes one.

Usage:
    pip install flyr numpy pandas
    python extract_thermal.py --root /path/to/thermal_root --out-dir ./outputs
    # options:
    #   --hotspot-c 34.0     absolute C threshold for "hotspot" (heads/feet); default 34
    #   --glob "*.jpg"       file pattern (default *.jpg, case-insensitive)
    #   --self-test          run a synthetic sanity check and exit (no images needed)
"""
from __future__ import annotations
import argparse, os, sys, glob, re
import numpy as np
import pandas as pd

_MONTHS = {m.lower(): i for i, m in enumerate(
    ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}
_MONTHS.update({m.lower(): i for i, m in enumerate(
    ["", "January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"])})


def parse_folder_date(path):
    """Best-effort date from messy day-folder names, using the month-year ancestor
    folder for the year. Handles 'Aug 19', '19 sep', '24Sep', '11 sep', '3 Aug'.
    Returns a pd.Timestamp or None. Only used as a fallback when EXIF time is absent."""
    parts = [p for p in path.split(os.sep) if p]
    # find a 'Month YYYY' ancestor for the year
    year = None
    for p in parts:
        m = re.match(r"([A-Za-z]+)\s+(\d{4})$", p.strip())
        if m and m.group(1).lower() in _MONTHS:
            year = int(m.group(2))
    # find a 'day + month' leaf (either order), scanning from the deepest folder up
    for p in reversed(parts[:-1]):            # skip the filename itself
        s = p.strip()
        m = (re.match(r"(\d{1,2})\s*([A-Za-z]+)$", s) or
             re.match(r"([A-Za-z]+)\s*(\d{1,2})$", s))
        if not m:
            continue
        a, b = m.group(1), m.group(2)
        if a.isdigit():
            day, mon = int(a), _MONTHS.get(b.lower())
        else:
            mon, day = _MONTHS.get(a.lower()), int(b)
        if mon and 1 <= day <= 31:
            try:
                return pd.Timestamp(year=year or 2025, month=mon, day=day)
            except Exception:
                return None
    return None


# ----------------------------------------------------------------------
def otsu_threshold(x):
    """Otsu threshold on a 1-D array (no OpenCV)."""
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    hist, edges = np.histogram(x, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    w = hist.astype(float); wsum = w.sum()
    if wsum == 0:
        return float(np.median(x))
    p = w / wsum
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    denom = omega * (1 - omega)
    denom[denom == 0] = 1e-12
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    return float(centers[np.nanargmax(sigma_b2)])


def frame_features(celsius, hotspot_c=34.0):
    """Compute whole-frame bird-masked radiometric features from a 2-D C matrix."""
    c = np.asarray(celsius, float)
    thr = otsu_threshold(c.ravel())
    mask = c > thr
    n_total = c.size
    n_bird = int(mask.sum())
    feat = {
        "otsu_thr_c": thr,
        "bird_frac": n_bird / n_total if n_total else np.nan,
        "n_pixels": n_total,
        "frame_min_c": float(np.min(c)),
        "frame_max_c": float(np.max(c)),
        "frame_mean_c": float(np.mean(c)),
    }
    if n_bird >= 10:                       # need a few bird pixels to be meaningful
        b = c[mask]
        feat.update({
            "t_mean_c": float(b.mean()),
            "t_p10_c": float(np.percentile(b, 10)),
            "t_p90_c": float(np.percentile(b, 90)),
            "t_max_c": float(b.max()),
            "t_spatial_std_c": float(b.std()),
            "hotspot_frac": float((b > hotspot_c).mean()),   # heads/feet proxy
        })
    else:
        for k in ["t_mean_c", "t_p10_c", "t_p90_c", "t_max_c", "t_spatial_std_c", "hotspot_frac"]:
            feat[k] = np.nan
    nonbird = c[~mask]
    feat["ambient_est_c"] = float(np.median(nonbird)) if nonbird.size else np.nan
    feat["mask_ok"] = bool(0.02 < feat["bird_frac"] < 0.95)   # flag degenerate masks
    return feat


# ----------------------------------------------------------------------
def get_timestamp(thermogram, path):
    """Capture time: EXIF first, then messy folder-date, then file mtime."""
    try:
        dt = thermogram.camera_metadata.date_time
        if dt is not None:
            return pd.Timestamp(dt), "exif"
    except Exception:
        pass
    fd = parse_folder_date(path)
    if fd is not None:
        return fd, "folder"
    return pd.Timestamp(os.path.getmtime(path), unit="s"), "mtime"


def iter_files(root, pattern):
    for dirpath, _, _ in os.walk(root):
        for f in glob.glob(os.path.join(dirpath, pattern)):
            yield f
        # also match uppercase extension
        for f in glob.glob(os.path.join(dirpath, pattern.upper())):
            yield f


def run(root, out_dir, hotspot_c, pattern):
    import flyr
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(set(iter_files(root, pattern)))
    print(f"[scan] {len(files)} candidate files under {root}")
    rows, rejects = [], []
    for i, path in enumerate(files, 1):
        try:
            tg = flyr.unpack(path)
            c = tg.celsius
            ts, ts_src = get_timestamp(tg, path)
            session = os.path.basename(os.path.dirname(path))     # day/session folder name
            fdate = parse_folder_date(path)
            # sanity: does the EXIF date agree with the folder name?
            folder_matches_exif = (fdate is not None and ts_src == "exif"
                                   and fdate.date() == ts.date())
            feat = frame_features(c, hotspot_c=hotspot_c)
            feat.update({"timestamp": ts, "ts_source": ts_src,
                         "session_folder": session,
                         "folder_date_hint": fdate,
                         "folder_matches_exif": folder_matches_exif,
                         "make": getattr(tg.camera_metadata, "make", None),
                         "model": getattr(tg.camera_metadata, "model", None),
                         "source_path": path})
            rows.append(feat)
        except Exception as e:
            rejects.append({"source_path": path, "error": repr(e)})
        if i % 25 == 0:
            print(f"  ...{i}/{len(files)}")

    if not rows:
        print("[error] no readable thermal frames; nothing written.")
        if rejects:
            pd.DataFrame(rejects).to_csv(os.path.join(out_dir, "thermal_rejects.csv"), index=False)
        return

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    lead = ["timestamp", "ts_source", "session_folder", "folder_date_hint",
            "folder_matches_exif", "t_mean_c", "t_p10_c", "t_p90_c", "t_max_c",
            "hotspot_frac", "t_spatial_std_c", "bird_frac", "ambient_est_c",
            "otsu_thr_c", "mask_ok", "frame_min_c", "frame_max_c", "frame_mean_c",
            "n_pixels", "make", "model", "source_path"]
    df = df[[c for c in lead if c in df.columns]]
    per_img = os.path.join(out_dir, "thermal_frame_features.csv")
    df.to_csv(per_img, index=False)
    print(f"[write] {per_img}  ({len(df)} frames, {df['timestamp'].min()} -> {df['timestamp'].max()})")

    # ----- daily aggregate (average across all birds/frames that day) -----
    d = df.copy()
    d["date"] = d["timestamp"].dt.floor("D")
    agg_cols = ["t_mean_c", "t_p10_c", "t_p90_c", "t_max_c", "hotspot_frac",
                "t_spatial_std_c", "bird_frac", "ambient_est_c"]
    daily = d.groupby("date").agg(
        n_frames=("timestamp", "size"),
        **{f"{c}_mean": (c, "mean") for c in agg_cols},
        **{f"{c}_std": (c, "std") for c in agg_cols},
    ).reset_index()
    daily["thermal_minus_ambient_c"] = daily["t_mean_c_mean"] - daily["ambient_est_c_mean"]
    daily_path = os.path.join(out_dir, "thermal_daily_features.csv")
    daily.to_csv(daily_path, index=False)
    print(f"[write] {daily_path}  ({len(daily)} days)")

    if rejects:
        rp = os.path.join(out_dir, "thermal_rejects.csv")
        pd.DataFrame(rejects).to_csv(rp, index=False)
        print(f"[write] {rp}  ({len(rejects)} unreadable files)")

    bad = (~df["mask_ok"]).sum()
    print(f"[qa] frames flagged mask_ok=False (degenerate mask): {bad}/{len(df)}")


# ----------------------------------------------------------------------
def self_test():
    """Synthetic sanity check: litter background + warm bird blobs."""
    rng = np.random.default_rng(0)
    frame = rng.normal(22, 1.0, (120, 160))
    for _ in range(8):
        y, x = rng.integers(15, 105), rng.integers(15, 145)
        frame[y-10:y+10, x-10:x+10] = rng.normal(33, 1.5, (20, 20))
    # inject a few very hot pixels (heads/feet)
    frame[30:33, 40:43] = 37.0
    f = frame_features(frame, hotspot_c=34.0)
    print("[self-test] frame_features on synthetic frame:")
    for k, v in f.items():
        print(f"    {k:18}: {v}")
    assert 24 < f["otsu_thr_c"] < 30, "Otsu threshold off"
    assert f["t_mean_c"] > f["ambient_est_c"], "birds should be warmer than ambient"
    assert 0 < f["hotspot_frac"] < 1, "hotspot_frac out of range"
    print("[self-test] PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", help="root folder to walk for thermal images")
    ap.add_argument("--out-dir", default="./outputs")
    ap.add_argument("--hotspot-c", type=float, default=34.0)
    ap.add_argument("--glob", default="*.jpg")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return
    if not args.root:
        ap.error("--root is required (or use --self-test)")
    run(args.root, args.out_dir, args.hotspot_c, args.glob)


if __name__ == "__main__":
    main()