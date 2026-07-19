#!/usr/bin/env python3
"""
build_spine.py — align rich audio + rich video + (daily) env onto ONE hourly
index, and tag each hour with its modality-coverage state.

Why a spine: you can't correlate audio against video, or run fusion, until both
sit on the same hourly clock. This also encodes the key structural fact about
this dataset — video is a DAYTIME-ONLY modality (grid/flow features exist only
for lit hours; night rows are dark by construction, not missing data) — so
nothing downstream accidentally imputes video across the night.

Coverage state per hour:
    both_lit      audio present AND video lit  -> the true fusion hours
    audio_only    audio present, video dark/absent (mostly night)
    video_only    video lit, audio missing (rare)
    gap           neither modality present

Columns are namespaced: aud_* / vid_* / env_*.  Env (daily) is broadcast onto
every hour of its date.

Usage
-----
    python src/pipeline/build_spine.py                       # defaults from .env
    python src/pipeline/build_spine.py --start 2025-07-03 --end 2025-08-31
    python src/pipeline/build_spine.py --env data/env_features_Room2.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import FEATURES_DIR, RESULTS_DIR

# ---- default inputs (override via CLI) ------------------------------------
RICH_AUDIO = FEATURES_DIR / "rich_audio_features" / "audio_rich_features_hourly_Room2_all.csv"
RICH_VIDEO = FEATURES_DIR / "rich_video_optical_features" / "video_rich_features_hourly_Room2.csv"
ENV_CSV    = FEATURES_DIR / "env_features_Room2.csv"


def _prefix(df: pd.DataFrame, prefix: str, keep: str = "time") -> pd.DataFrame:
    return df.rename(columns={c: f"{prefix}{c}" for c in df.columns if c != keep})


def build_spine(rich_audio, rich_video, env_csv, start=None, end=None):
    # --- load ---
    a = pd.read_csv(rich_audio, parse_dates=["time"]).sort_values("time")
    v = pd.read_csv(rich_video, parse_dates=["time"]).sort_values("time")
    env = pd.read_csv(env_csv, parse_dates=["date"]).sort_values("date")

    # --- lit / present flags BEFORE prefixing (need raw column names) ---
    # video is "lit" when the spatial-grid features exist (they are NaN at night)
    a_present = a.drop(columns=["time"]).filter(like="mel").notna().any(axis=1)
    v_lit = v["gridmean00"].notna() if "gridmean00" in v else v.drop(columns=["time"]).notna().any(axis=1)
    a = a.assign(_aud_present=a_present.values)
    v = v.assign(_vid_present=True, _vid_lit=v_lit.values)

    a = _prefix(a, "aud_"); v = _prefix(v, "vid_")

    # --- hourly spine spanning the union (or the requested window) ---
    lo = min(a["time"].min(), v["time"].min())
    hi = max(a["time"].max(), v["time"].max())
    if start:
        lo = max(lo, pd.Timestamp(start))
    if end:
        hi = min(hi, pd.Timestamp(end) + pd.Timedelta(hours=23))
    spine = pd.DataFrame({"time": pd.date_range(lo.floor("h"), hi.ceil("h"), freq="h")})

    m = spine.merge(a, on="time", how="left").merge(v, on="time", how="left")

    # --- env: daily -> broadcast onto each hour of that calendar date ---
    env = _prefix(env, "env_", keep="date")
    m["date"] = m["time"].dt.floor("D")
    env_daily = (env.set_index("date")
                    .reindex(pd.date_range(env["date"].min(), env["date"].max(), freq="D"))
                    .ffill())
    env_daily.index.name = "date"
    m = m.merge(env_daily.reset_index(), on="date", how="left").drop(columns="date")

    # --- coverage flags ---
    has_audio = m["aud__aud_present"].eq(True)
    has_vid_row = m["vid__vid_present"].eq(True)
    has_vid_lit = m["vid__vid_lit"].eq(True)
    m = m.drop(columns=["aud__aud_present", "vid__vid_present", "vid__vid_lit"])

    m["has_audio"] = has_audio
    m["has_video_row"] = has_vid_row
    m["has_video_lit"] = has_vid_lit
    m["hour_of_day"] = m["time"].dt.hour

    state = np.select(
        [has_audio & has_vid_lit,
         has_audio & ~has_vid_lit,
         ~has_audio & has_vid_lit],
        ["both_lit", "audio_only", "video_only"],
        default="gap",
    )
    m["coverage_state"] = state
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rich-audio", default=str(RICH_AUDIO))
    ap.add_argument("--rich-video", default=str(RICH_VIDEO))
    ap.add_argument("--env", default=str(ENV_CSV))
    ap.add_argument("--start", default=None, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--out", default=str(RESULTS_DIR / "spine_room2_rich.csv"))
    args = ap.parse_args()

    m = build_spine(args.rich_audio, args.rich_video, args.env, args.start, args.end)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(args.out, index=False)

    # --- summary ---
    n_aud = int(m.columns.str.startswith("aud_").sum())
    n_vid = int(m.columns.str.startswith("vid_").sum())
    n_env = int(m.columns.str.startswith("env_").sum())
    print(f"Spine: {len(m)} hourly rows  {m['time'].min()} -> {m['time'].max()}")
    print(f"Features: audio={n_aud}  video={n_vid}  env={n_env}")
    print("\nCoverage state counts:")
    print(m["coverage_state"].value_counts().to_string())
    both = (m["coverage_state"] == "both_lit").sum()
    ao = (m["coverage_state"] == "audio_only").sum()
    print(f"\nFUSION hours (both_lit): {both}   |   audio-only (mostly night): {ao}")
    print(f"Wrote -> {args.out}")


if __name__ == "__main__":
    main()
