#!/usr/bin/env python3
"""
build_env_features_room6.py — derive env_features_Room6.csv in the SAME schema as
env_features_Room2.csv, from the monthly Temp/RH Excel sheets.

Raw sheets come in two header layouts (both map to the same 6 fields, in order:
date, temp_am_min, temp_am_max, temp_pm, rh_am, rh_pm):
  A) two-row header: row0 "Date|Temp-AM|_|Temp-PM|RH|_", row1 "_|Min|Max|PM|AM|PM"
  B) one-row header: "Date|Temp_AM_Min|Temp_AM_Max|Temp_PM|RH_AM|RH_PM"

Derived columns replicate Room 2 EXACTLY (verified against env_features_Room2.csv):
  temp_day_mean_c   = mean(am_min, am_max, pm)
  temp_am_range_c   = am_max - am_min
  temp_am_pm_swing_c= pm - mean(am_min, am_max)
  rh_day_mean_pct   = mean(rh_am, rh_pm)
  rh_am_pm_change_pct = rh_pm - rh_am
  temp_rate_c_per_day = day-to-day diff of temp_day_mean
  rh_rate_pct_per_day = day-to-day diff of rh_day_mean
  temp_roll_mean_c  = 7-day centered rolling mean of temp_day_mean (min_periods=1)
  rh_roll_mean_pct  = 7-day centered rolling mean of rh_day_mean
  temp_roll_slope_c_per_day = 7-day centered rolling OLS slope of temp_day_mean
        (NOTE: the ONLY env column whose original Room 2 formula was unrecoverable;
         defined cleanly here. The locked AV model uses env only via day_index, so
         this does not affect cross-barn results.)
  day_index = days since the first recorded date (flock age; day 0 = 2025-06-05,
              same as Room 2).

Usage:
    python crossbarn/build_env_features_room6.py \
        --xlsx-dir data/raw_room6/env --out features/env_features_Room6.csv
    # or pass files explicitly with repeated --xlsx
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np
import pandas as pd

COLS6 = ["date", "temp_am_min_c", "temp_am_max_c", "temp_pm_c", "rh_am_pct", "rh_pm_pct"]


def read_sheet(path):
    raw = pd.read_excel(path, header=None)
    # detect 2-row header (row1 contains 'Min'/'Max' sub-labels)
    skip = 2 if "min" in str(raw.iloc[1, 1]).lower() else 1
    data = raw.iloc[skip:, :6].copy()
    data.columns = COLS6
    data["date"] = pd.to_datetime(data["date"], errors="coerce", dayfirst=True)
    for c in COLS6[1:]:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna(subset=["date"]).reset_index(drop=True)
    data["source_file"] = os.path.basename(path)
    return data


def derive(df):
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    am_mean = df[["temp_am_min_c", "temp_am_max_c"]].mean(axis=1)
    tdm = df[["temp_am_min_c", "temp_am_max_c", "temp_pm_c"]].mean(axis=1)
    rdm = df[["rh_am_pct", "rh_pm_pct"]].mean(axis=1)

    df["day_index"] = (df["date"] - df["date"].min()).dt.days
    df["temp_day_mean_c"] = tdm
    df["temp_am_range_c"] = df["temp_am_max_c"] - df["temp_am_min_c"]
    df["temp_am_pm_swing_c"] = df["temp_pm_c"] - am_mean
    df["rh_day_mean_pct"] = rdm
    df["rh_am_pm_change_pct"] = df["rh_pm_pct"] - df["rh_am_pct"]
    df["temp_rate_c_per_day"] = tdm.diff()
    df["rh_rate_pct_per_day"] = rdm.diff()
    df["temp_roll_mean_c"] = tdm.rolling(7, center=True, min_periods=1).mean()
    df["rh_roll_mean_pct"] = rdm.rolling(7, center=True, min_periods=1).mean()

    def slope(x):
        x = np.asarray(x, float); x = x[~np.isnan(x)]
        return np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) >= 2 else np.nan
    df["temp_roll_slope_c_per_day"] = tdm.rolling(7, center=True, min_periods=2).apply(slope, raw=True)

    order = ["date", "temp_am_min_c", "temp_am_max_c", "temp_pm_c", "rh_am_pct", "rh_pm_pct",
             "source_file", "day_index", "temp_day_mean_c", "temp_am_range_c",
             "temp_am_pm_swing_c", "rh_day_mean_pct", "rh_am_pm_change_pct",
             "temp_rate_c_per_day", "rh_rate_pct_per_day", "temp_roll_mean_c",
             "rh_roll_mean_pct", "temp_roll_slope_c_per_day"]
    return df[order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx-dir", default=None, help="folder of monthly Temp/RH .xlsx")
    ap.add_argument("--xlsx", action="append", default=[], help="explicit .xlsx file(s)")
    ap.add_argument("--out", default="features/env_features_Room6.csv")
    a = ap.parse_args()
    files = list(a.xlsx)
    if a.xlsx_dir:
        files += glob.glob(os.path.join(a.xlsx_dir, "*.xlsx"))
    files = sorted(set(files))
    if not files:
        ap.error("no .xlsx given (use --xlsx-dir or --xlsx)")
    parts = [read_sheet(f) for f in files]
    env = derive(pd.concat(parts, ignore_index=True))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    env["date"] = env["date"].dt.strftime("%Y-%m-%d")
    env.to_csv(a.out, index=False)
    print(f"[write] {a.out}  ({len(env)} days, {env.shape[1]} cols, "
          f"{env['date'].min()} -> {env['date'].max()})")
    print(f"[days] day_index 0..{env['day_index'].max()}  from {len(files)} sheet(s)")


if __name__ == "__main__":
    main()
