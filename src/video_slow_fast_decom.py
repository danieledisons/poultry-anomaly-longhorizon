#!/usr/bin/env python3
"""
slow_fast_decomposition.py
============================
Decomposes video hourly features into a SLOW band (trend + diurnal recurrence
across weeks) and a FAST band (sub-hourly/hourly deviations from that trend),
per the project's stated modeling convention:

    Slow band = trend + diurnal recurrence across weeks
    Fast band = sub-hourly deviations

Approach
--------
1. Reindex the hourly series onto a COMPLETE hourly grid spanning the full date
   range (filling gaps with NaN, not interpolating over them silently) -- this
   makes gaps explicit rather than letting missing hours quietly compress the
   rolling windows.
2. Slow band = a long rolling median/mean (default: 7-day window) computed with
   min_periods set low enough to survive real gaps (e.g. nightly camera-off
   stretches) without collapsing to NaN for weeks at a time.
3. Diurnal component = average value by hour-of-day, computed on the
   trend-removed residual, to capture the recurring daily rhythm (feeding
   times, lights on/off, etc.) separately from the slower week-to-week drift.
4. Fast band = original signal minus (slow trend + diurnal component) --
   i.e. what's left after removing the slow-moving and recurring-daily parts.
5. Output: one CSV with all components (original, slow_trend, diurnal, fast,
   reconstructed) for each configured column, plus a plot for visual review.

This treats video's real gaps (nightly camera-off, multi-day outages) as
missing data (NaN), not zero -- a gap contributes no information to the
rolling trend rather than dragging it toward zero.

Usage
-----
    python slow_fast_decomposition.py \\
        --hourly-csv /path/to/hourly_features_all_folders.csv \\
        --output-dir /path/to/results \\
        --columns flow_mean_avg occupancy_avg \\
        --slow-window-days 7
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def build_complete_hourly_index(df: pd.DataFrame, hour_col: str = "hour") -> pd.DatetimeIndex:
    """Full hourly range from min to max timestamp -- makes gaps explicit as
    NaN rows rather than leaving them as absent rows (which would silently
    compress rolling windows across a gap as if it were shorter than it is)."""
    return pd.date_range(df[hour_col].min(), df[hour_col].max(), freq="h")


def _smooth_circular(profile: pd.Series, window: int = 3) -> pd.Series:
    """Smooth a 24-value hour-of-day profile with a circular (wrap-around)
    rolling mean, so hour 23 and hour 0 are treated as adjacent. Without this,
    diurnal profiles estimated from sparse per-hour sample counts tend to jump
    noisily between adjacent hours instead of forming a smooth daily curve."""
    tripled = pd.concat([profile, profile, profile])
    smoothed = tripled.rolling(window=window, center=True, min_periods=1).mean()
    return smoothed.iloc[len(profile):2 * len(profile)]


def decompose_column(
    series: pd.Series,
    slow_window_hours: int,
    min_periods_frac: float = 0.2,
    diurnal_smooth_window: int = 3,
) -> pd.DataFrame:
    """Decompose one hourly series (indexed by a complete hourly DatetimeIndex,
    with NaN for missing/gap hours) into slow trend + diurnal + fast residual.
    """
    min_periods = max(3, int(slow_window_hours * min_periods_frac))

    # Slow trend: long rolling median (robust to spikes) over real elapsed time
    # (not just row count), so gaps don't distort the window width.
    slow_trend = series.rolling(
        window=f"{slow_window_hours}h", min_periods=min_periods, center=True
    ).median()

    detrended = series - slow_trend

    # Diurnal component: mean detrended value per hour-of-day, SMOOTHED across
    # adjacent hours (circular, so hour 23 connects to hour 0) to produce an
    # actual daily rhythm curve rather than noisy jumps from sparse per-hour
    # sample counts, then mapped back onto the full index.
    hour_of_day = series.index.hour
    diurnal_by_hod = pd.Series(detrended.values, index=hour_of_day).groupby(level=0).mean()
    diurnal_by_hod = diurnal_by_hod.reindex(range(24))  # ensure all 24 hours present, in order
    diurnal_by_hod = _smooth_circular(diurnal_by_hod, window=diurnal_smooth_window)
    diurnal = pd.Series(hour_of_day, index=series.index).map(diurnal_by_hod)

    fast = series - slow_trend - diurnal
    reconstructed = slow_trend + diurnal + fast

    return pd.DataFrame({
        "original": series,
        "slow_trend": slow_trend,
        "diurnal": diurnal,
        "fast_residual": fast,
        "reconstructed": reconstructed,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hourly-csv", required=True, help="Path to hourly_features_all_folders.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--columns", nargs="+", default=["flow_mean_avg", "occupancy_avg"],
                        help="Which hourly columns to decompose.")
    parser.add_argument("--slow-window-days", type=float, default=7.0,
                        help="Rolling window size (in days) for the slow trend.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    slow_window_hours = int(args.slow_window_days * 24)

    print(f"Loading: {args.hourly_csv}")
    df = pd.read_csv(args.hourly_csv, parse_dates=["hour"])
    print(f"  {len(df)} rows, {df['hour'].min()} -> {df['hour'].max()}")

    full_index = build_complete_hourly_index(df)
    print(f"Complete hourly grid: {len(full_index)} hours "
          f"({len(full_index) - len(df)} gap hours will show as NaN)")

    df_indexed = df.set_index("hour").reindex(full_index)
    df_indexed.index.name = "hour"

    all_results = {}
    for col in args.columns:
        if col not in df_indexed.columns:
            print(f"WARNING: column '{col}' not found, skipping. "
                  f"Available: {list(df_indexed.columns)}")
            continue

        print(f"\nDecomposing '{col}' (slow window = {args.slow_window_days} days)...")
        result = decompose_column(df_indexed[col], slow_window_hours)
        all_results[col] = result

        n_valid = result["original"].notna().sum()
        n_slow_valid = result["slow_trend"].notna().sum()
        print(f"  {n_valid} hours with real data, {n_slow_valid} hours with a computable slow trend")

        # Basic residual check: fast band should have ~zero mean if the
        # decomposition is behaving (slow + diurnal captured the systematic parts)
        fast_mean = result["fast_residual"].mean()
        fast_std = result["fast_residual"].std()
        print(f"  fast_residual: mean={fast_mean:.4f} (should be near 0), std={fast_std:.4f}")

    # Combine into one wide output CSV
    combined = pd.DataFrame(index=full_index)
    combined.index.name = "hour"
    for col, result in all_results.items():
        for component in ["original", "slow_trend", "diurnal", "fast_residual"]:
            combined[f"{col}__{component}"] = result[component]

    out_path = os.path.join(args.output_dir, "slow_fast_decomposition.csv")
    combined.reset_index().to_csv(out_path, index=False)
    print(f"\nDecomposition written to: {out_path}")

    # Plots -- ONE PNG PER PANEL PER COLUMN, so each chart can be viewed/shared
    # independently rather than as a cramped multi-panel grid.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        # "Jul 18" style date ticks on every chart, applied consistently.
        date_formatter = mdates.DateFormatter("%b %d")

        def format_date_axis(ax):
            ax.xaxis.set_major_formatter(date_formatter)
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=8, maxticks=14))
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha("right")

        # Identify real gap spans (>=6 consecutive missing hours) to shade on
        # every chart -- otherwise a straight connecting line across a 4-day
        # outage looks like real, smooth data instead of an absence of data.
        def find_gap_spans(missing_series: pd.Series, min_gap_hours: int = 6) -> list[tuple]:
            spans = []
            in_gap = False
            gap_start = None
            run_len = 0
            for ts, missing in missing_series.items():
                if missing:
                    if not in_gap:
                        in_gap = True
                        gap_start = ts
                        run_len = 1
                    else:
                        run_len += 1
                else:
                    if in_gap and run_len >= min_gap_hours:
                        spans.append((gap_start, ts))
                    in_gap = False
            if in_gap and run_len >= min_gap_hours:
                spans.append((gap_start, missing_series.index[-1]))
            return spans

        def shade_gaps(ax, spans):
            for gs, ge in spans:
                ax.axvspan(gs, ge, color="lightgray", alpha=0.4, zorder=0)

        def save_chart(fig, ax, filename, gap_spans):
            shade_gaps(ax, gap_spans)
            format_date_axis(ax)
            ax.set_xlabel("Date")
            fig.tight_layout()
            path = os.path.join(args.output_dir, filename)
            fig.savefig(path, dpi=600)
            plt.close(fig)
            print(f"  Saved: {path}")

        for col, result in all_results.items():
            gap_spans = find_gap_spans(result["original"].isna())
            safe_col = col.replace(" ", "_")

            # Chart 1: original + slow trend
            fig, ax = plt.subplots(figsize=(13, 5))
            ax.scatter(result.index, result["original"], color="gray", s=5, label="Original (hourly)", zorder=2)
            ax.plot(result.index, result["slow_trend"], color="tab:blue", linewidth=2, label="Slow trend (7-day rolling median)", zorder=3)
            ax.set_title(f"{col}: Original Signal and Slow Trend")
            ax.set_ylabel(col)
            ax.legend(loc="upper left", fontsize=9)
            save_chart(fig, ax, f"{safe_col}_01_original_and_trend.png", gap_spans)

            # Chart 2: diurnal component
            fig, ax = plt.subplots(figsize=(13, 5))
            ax.plot(result.index, result["diurnal"], color="tab:green", linewidth=1.2)
            ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
            ax.set_title(f"{col}: Diurnal Component (Smoothed Daily Rhythm)")
            ax.set_ylabel(f"{col} (deviation from trend)")
            save_chart(fig, ax, f"{safe_col}_02_diurnal.png", gap_spans)

            # Chart 3: fast residual
            fig, ax = plt.subplots(figsize=(13, 5))
            ax.scatter(result.index, result["fast_residual"], color="tab:red", s=4)
            ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
            ax.set_title(f"{col}: Fast Residual (Sub-Daily Deviations)")
            ax.set_ylabel(f"{col} (residual)")
            save_chart(fig, ax, f"{safe_col}_03_fast_residual.png", gap_spans)

            # Chart 4: original vs reconstructed fit check
            fig, ax = plt.subplots(figsize=(13, 5))
            ax.scatter(result.index, result["original"], color="gray", s=5, label="Original", zorder=2)
            ax.plot(result.index, result["reconstructed"], color="tab:purple", linewidth=1.4,
                    label="Reconstructed (trend + diurnal + residual)", alpha=0.85, zorder=3)
            ax.set_title(f"{col}: Original vs. Reconstructed Signal")
            ax.set_ylabel(col)
            ax.legend(loc="upper left", fontsize=9)
            save_chart(fig, ax, f"{safe_col}_04_reconstructed_fit.png", gap_spans)

        print(f"\n(Gray shaded bands = real gaps of 6+ consecutive missing hours, "
              f"e.g. nightly camera-off periods or multi-day outages)")
    except ImportError:
        print("matplotlib not available -- skipping plots. Install with: pip install matplotlib")

    print("\nDone. Check that slow_trend follows the visible long-term drift, diurnal")
    print("shows a sensible daily pattern, and fast_residual looks like noise centered")
    print("near zero (not still showing obvious trend/daily structure it should have absorbed).")


if __name__ == "__main__":
    main()