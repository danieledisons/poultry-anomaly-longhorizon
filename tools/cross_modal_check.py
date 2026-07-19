#!/usr/bin/env python3
"""
cross_modal_check.py
======================
First cross-modal sanity check for Room 2: does video activity (optical flow /
occupancy) track audio activity (vocalization/energy) on the same shared hourly
cadence? This is a validation step, not a modeling step -- the goal is to catch
a broken pipeline (either modality) before building anything on top of both.

Approach
--------
1. Load video hourly features (hourly_features_all_folders.csv) and audio hourly
   features (audio_features_hourly_Room2_*.csv, one file per month -- concatenated).
2. Align both on the shared 'hour' timestamp column (inner join -- only hours
   present in BOTH modalities are compared; hours where one modality has a gap,
   e.g. video's nightly camera-off periods, are correctly excluded rather than
   silently filled).
3. Report: how many hours overlap, correlation between video activity proxies
   (flow_mean_avg, occupancy_avg) and audio activity proxies (auto-detected from
   column names -- e.g. columns containing 'activity', 'rms', 'vocalization').
4. Save a merged CSV and a simple time-series comparison plot for visual review.

IMPORTANT: this script auto-detects likely audio activity columns by name
matching, since the exact column names in your audio feature CSVs weren't known
in advance. Check the printed "audio columns detected" list -- if it picked the
wrong column, override AUDIO_ACTIVITY_COL_OVERRIDE below.

Usage
-----
    python cross_modal_check.py \\
        --video-hourly /path/to/hourly_features_all_folders.csv \\
        --audio-hourly /path/to/audio_features_hourly_Room2_2025-07.csv /path/to/audio_features_hourly_Room2_2025-08.csv \\
        --output-dir /path/to/results
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

# If auto-detection picks the wrong audio column, set this explicitly, e.g.:
# AUDIO_ACTIVITY_COL_OVERRIDE = "vocalization_activity_index_mean"
AUDIO_ACTIVITY_COL_OVERRIDE: str | None = None

VIDEO_ACTIVITY_COLS = ["flow_mean_avg", "occupancy_avg"]

# Keywords used to auto-detect likely audio "activity" columns by name, in
# priority order -- first match wins per candidate search.
AUDIO_ACTIVITY_KEYWORDS = [
    "vocalization_activity", "activity_index", "activity",
    "rms_energy", "rms", "vocal",
]


def load_video_hourly(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["hour"])
    return df


def load_audio_hourly(paths: list[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        frames.append(df)
    audio = pd.concat(frames, ignore_index=True)

    # Find the timestamp column -- name may differ from video's 'hour'
    ts_col_candidates = [c for c in audio.columns if c.lower() in
                          ("hour", "timestamp", "datetime", "time", "date_hour")]
    if not ts_col_candidates:
        raise ValueError(
            f"Could not find a timestamp column in audio data. Columns found: "
            f"{list(audio.columns)}. Please check the file and adjust this script."
        )
    ts_col = ts_col_candidates[0]
    audio = audio.rename(columns={ts_col: "hour"})
    audio["hour"] = pd.to_datetime(audio["hour"])
    return audio


def detect_audio_activity_column(audio_df: pd.DataFrame) -> str:
    if AUDIO_ACTIVITY_COL_OVERRIDE:
        if AUDIO_ACTIVITY_COL_OVERRIDE not in audio_df.columns:
            raise ValueError(
                f"AUDIO_ACTIVITY_COL_OVERRIDE='{AUDIO_ACTIVITY_COL_OVERRIDE}' not "
                f"found in audio columns: {list(audio_df.columns)}"
            )
        return AUDIO_ACTIVITY_COL_OVERRIDE

    numeric_cols = audio_df.select_dtypes(include=[np.number]).columns.tolist()
    for keyword in AUDIO_ACTIVITY_KEYWORDS:
        matches = [c for c in numeric_cols if keyword in c.lower()]
        if matches:
            return matches[0]

    raise ValueError(
        f"Could not auto-detect an audio activity column. Numeric columns "
        f"available: {numeric_cols}. Set AUDIO_ACTIVITY_COL_OVERRIDE in this "
        f"script to the correct column name and re-run."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video-hourly", required=True, help="Path to hourly_features_all_folders.csv")
    parser.add_argument("--audio-hourly", required=True, nargs="+",
                        help="Path(s) to audio_features_hourly_Room2_*.csv (one or more months)")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading video hourly features from: {args.video_hourly}")
    video = load_video_hourly(args.video_hourly)
    print(f"  {len(video)} video hourly rows, {video['hour'].min()} -> {video['hour'].max()}")

    print(f"Loading audio hourly features from: {args.audio_hourly}")
    audio = load_audio_hourly(args.audio_hourly)
    print(f"  {len(audio)} audio hourly rows, {audio['hour'].min()} -> {audio['hour'].max()}")
    print(f"  Audio columns: {list(audio.columns)}")

    audio_activity_col = detect_audio_activity_column(audio)
    print(f"\nAuto-detected audio activity column: '{audio_activity_col}'")
    print("(If this looks wrong, set AUDIO_ACTIVITY_COL_OVERRIDE at the top of this script and re-run.)")

    # Inner join: only compare hours present in BOTH modalities. Hours missing
    # from either side (e.g. video's nightly camera-off gaps) are correctly
    # excluded here rather than being filled with a misleading zero/NaN guess.
    merged = pd.merge(
        video[["hour"] + VIDEO_ACTIVITY_COLS],
        audio[["hour", audio_activity_col]],
        on="hour", how="inner",
    )
    print(f"\nOverlapping hours (present in both modalities): {len(merged)} "
          f"(video had {len(video)}, audio had {len(audio)})")

    if len(merged) < 10:
        print("WARNING: very few overlapping hours -- check that both files cover "
              "the same date range and that timestamp formats/timezones match.")

    print("\n--- Correlation: video activity vs audio activity ---")
    for video_col in VIDEO_ACTIVITY_COLS:
        valid = merged[[video_col, audio_activity_col]].dropna()
        if len(valid) < 3:
            print(f"{video_col} vs {audio_activity_col}: not enough overlapping non-null data")
            continue
        corr = valid[video_col].corr(valid[audio_activity_col])
        print(f"{video_col} vs {audio_activity_col}: r = {corr:.3f}  (n={len(valid)})")

    merged_path = os.path.join(args.output_dir, "cross_modal_video_audio_merged.csv")
    merged.to_csv(merged_path, index=False)
    print(f"\nMerged comparison data written to: {merged_path}")

    # Time-series plots -- ONE PNG PER video-vs-audio COMPARISON, with clean
    # "Jul 18" style date labels, so each chart can be viewed/shared on its own.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        date_formatter = mdates.DateFormatter("%b %d")

        def format_date_axis(ax):
            ax.xaxis.set_major_formatter(date_formatter)
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=8, maxticks=14))
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha("right")

        for video_col in VIDEO_ACTIVITY_COLS:
            fig, ax = plt.subplots(figsize=(13, 5))
            ax2 = ax.twinx()
            ax.plot(merged["hour"], merged[video_col], color="tab:blue", label=video_col, linewidth=0.9)
            ax2.plot(merged["hour"], merged[audio_activity_col], color="tab:orange",
                     label=audio_activity_col, linewidth=0.9, alpha=0.75)
            ax.set_ylabel(video_col, color="tab:blue")
            ax2.set_ylabel(audio_activity_col, color="tab:orange")
            ax.set_title(f"{video_col} (video) vs. {audio_activity_col} (audio)")
            ax.set_xlabel("Date")
            format_date_axis(ax)

            fig.tight_layout()
            plot_path = os.path.join(args.output_dir, f"cross_modal_{video_col}_vs_{audio_activity_col}.png")
            fig.savefig(plot_path, dpi=600)
            plt.close(fig)
            print(f"Chart written to: {plot_path}")
    except ImportError:
        print("matplotlib not available -- skipping plots. "
              "Install with: pip install matplotlib")

    print("\nDone. Review the correlation values and plot before drawing conclusions --")
    print("a moderate positive correlation is a good sign; near-zero or negative")
    print("warrants checking both pipelines before trusting either modality further.")


if __name__ == "__main__":
    main()