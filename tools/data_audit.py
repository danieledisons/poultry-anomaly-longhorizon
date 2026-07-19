#!/usr/bin/env python3
"""
data_audit.py — coverage audit across the multi-drive raw dataset.

The recording date lives in the FOLDER name (files are just STE-000.wav /
GX0001.MP4), and folder naming is inconsistent across drives:

    Nov28-Dec1            Oct29-Nov3           Sep 11-13
    Room 8 (13, 14, 15 Aug)   Room 8 - 3, 4, 5, 6 July   June 6, 7, 8
    Room 6 (31 July - 3 Aug)  Room 6 (31 Aug, 1 Sep)

So we parse dates out of each session folder's name, attach a room (from the
path), classify modality (audio/video/thermal by path + extension), and report
per room × modality coverage: earliest date, latest date, #session-days,
#files, total size.

Usage
-----
    python tools/data_audit.py \
        --root "/mnt/drive_bf/Poultry_Multimodal_SeptDec" \
        --root "/mnt/crucial_x10/Poultry Multimodal Data" \
        --out data_audit.csv --year 2025

Roots that don't exist (e.g. the video drive when it's unplugged) are skipped
with a warning, so you can run it with whatever is mounted.
"""
from __future__ import annotations

import argparse
import calendar
import csv
import datetime as dt
import os
import re
from collections import defaultdict

# ---- month lookup (abbrev + full name) -----------------------------------
_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}
_MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_name) if m})

AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".aif", ".aiff"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mts"}
THERMAL_EXT = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".seq", ".csq"}


def extract_dates(folder_name: str, year: int) -> list[dt.date]:
    """Pull all (year, month, day) dates implied by a folder name.

    Strategy: drop room/modality words, find month tokens and number tokens,
    and assign each number to its nearest month by character position. Handles
    'Oct18-20', '13, 14, 15 Aug', 'Nov28-Dec1', '31 July - 2 Aug', etc.
    """
    s = folder_name.lower()
    s = re.sub(r"room\s*\d+", " ", s)                       # strip "Room 8"
    s = re.sub(r"\b(audio|video|thermal|data)\b", " ", s)   # strip modality words

    month_hits = [
        (m.start(), _MONTHS[m.group()])
        for m in re.finditer(r"[a-z]+", s) if m.group() in _MONTHS
    ]
    if not month_hits:
        return []

    dates: list[dt.date] = []
    for m in re.finditer(r"\d+", s):
        val = int(m.group())
        if not (1 <= val <= 31):
            continue
        month = min(month_hits, key=lambda mh: abs(mh[0] - m.start()))[1]
        try:
            dates.append(dt.date(year, month, val))
        except ValueError:
            pass  # e.g. day 31 wrongly attached to a 30-day month — skip
    return dates


def detect_modality(path: str) -> str:
    p = path.lower()
    if "video" in p:
        return "video"
    if "thermal" in p:
        return "thermal"
    if "audio" in p:
        return "audio"
    return "other"


def detect_room(path: str) -> str:
    m = re.search(r"room\s*(\d+)", path, re.IGNORECASE)
    return f"Room{int(m.group(1))}" if m else "Room?"


def ext_bucket(fname: str) -> str | None:
    e = os.path.splitext(fname)[1].lower()
    if e in AUDIO_EXT:
        return "audio"
    if e in VIDEO_EXT:
        return "video"
    if e in THERMAL_EXT:
        return "thermal"
    return None


def audit(roots: list[str], year: int, with_size: bool):
    sessions = []  # one row per leaf folder that directly holds media
    for root in roots:
        if not os.path.isdir(root):
            print(f"[skip] not mounted / not found: {root}")
            continue
        print(f"[scan] {root}")
        for dirpath, _dirs, files in os.walk(root):
            media = [f for f in files if ext_bucket(f)]
            if not media:
                continue

            modality = detect_modality(dirpath)
            room = detect_room(dirpath)
            leaf = os.path.basename(dirpath)

            # parse dates from leaf; if none, climb parents until we find some
            probe, dates = dirpath, []
            while probe and os.path.commonpath([probe, root]) == os.path.normpath(root):
                dates = extract_dates(os.path.basename(probe), year)
                if dates:
                    break
                parent = os.path.dirname(probe)
                if parent == probe:
                    break
                probe = parent

            size_mb = 0.0
            if with_size:
                for f in media:
                    try:
                        size_mb += os.path.getsize(os.path.join(dirpath, f)) / 1048576
                    except OSError:
                        pass

            sessions.append({
                "drive": os.path.basename(os.path.normpath(root)) or root,
                "room": room,
                "modality": modality,
                "session": leaf,
                "start_date": min(dates).isoformat() if dates else "",
                "end_date": max(dates).isoformat() if dates else "",
                "n_files": len(media),
                "size_mb": round(size_mb, 1),
                "dated": bool(dates),
                "path": dirpath,
            })
    return sessions


def summarize(sessions):
    """Per room × modality: earliest, latest, #session-days, #files, size."""
    agg = defaultdict(lambda: {"days": set(), "files": 0, "size": 0.0,
                               "sessions": 0, "undated": 0})
    for s in sessions:
        k = (s["room"], s["modality"])
        a = agg[k]
        a["sessions"] += 1
        a["files"] += s["n_files"]
        a["size"] += s["size_mb"]
        if s["dated"]:
            d0 = dt.date.fromisoformat(s["start_date"])
            d1 = dt.date.fromisoformat(s["end_date"])
            for n in range((d1 - d0).days + 1):
                a["days"].add(d0 + dt.timedelta(days=n))
        else:
            a["undated"] += 1
    rows = []
    for (room, modality), a in sorted(agg.items()):
        days = sorted(a["days"])
        rows.append({
            "room": room, "modality": modality,
            "earliest": days[0].isoformat() if days else "",
            "latest": days[-1].isoformat() if days else "",
            "days_covered": len(days),
            "sessions": a["sessions"],
            "files": a["files"],
            "size_gb": round(a["size"] / 1024, 2),
            "undated_sessions": a["undated"],
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", required=True,
                    help="A dataset root to scan (repeat for multiple drives).")
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--out", default="data_audit.csv",
                    help="Per-session CSV output path.")
    ap.add_argument("--no-size", action="store_true",
                    help="Skip file-size totals (much faster on slow drives).")
    args = ap.parse_args()

    sessions = audit(args.root, args.year, with_size=not args.no_size)
    if not sessions:
        print("No media found. Are the drives mounted?")
        return

    # write per-session detail
    fields = ["drive", "room", "modality", "session", "start_date", "end_date",
              "n_files", "size_mb", "dated", "path"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(sessions)

    summary = summarize(sessions)
    summary_path = os.path.splitext(args.out)[0] + "_summary.csv"
    with open(summary_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    # printed coverage table
    print("\n=== COVERAGE BY ROOM x MODALITY ===")
    hdr = f"{'room':7} {'modality':8} {'earliest':11} {'latest':11} " \
          f"{'days':>5} {'sess':>5} {'files':>8} {'size_gb':>8} {'undated':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in summary:
        print(f"{r['room']:7} {r['modality']:8} {r['earliest']:11} {r['latest']:11} "
              f"{r['days_covered']:5d} {r['sessions']:5d} {r['files']:8d} "
              f"{r['size_gb']:8.2f} {r['undated_sessions']:7d}")

    undated = [s for s in sessions if not s["dated"]]
    if undated:
        print(f"\n[!] {len(undated)} session folder(s) had no parseable date "
              f"(listed in {args.out} with dated=False) — check these by hand:")
        for s in undated[:15]:
            print(f"    {s['room']:7} {s['modality']:8} {s['session']}")

    print(f"\nWrote {len(sessions)} sessions -> {args.out}")
    print(f"Wrote summary          -> {summary_path}")


if __name__ == "__main__":
    main()