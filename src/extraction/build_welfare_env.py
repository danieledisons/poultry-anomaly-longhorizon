#!/usr/bin/env python3
"""
WELFARE-RELEVANT ENVIRONMENT FEATURES (physiological, age-referenced).

Raw temperature/RH are near-deterministic (managed set-point schedule), so the
informative quantity is the DEVIATION from what the schedule prescribes for the
bird's age, plus physiological indices that map to bird EXPERIENCE.

Derived from the existing env feature CSV (no raw re-logging needed). Writes to a
NEW filename; never overwrites existing features.

Features:
  thi                  temperature-humidity index (heat-stress proxy)
  temp_setpoint        age-based broiler target temperature (schedule)
  temp_dev_setpoint    actual - target  (the informative residual)
  temp_dev_abs         |deviation|
  rh_dev               RH deviation from a comfort band (45-65%)
  thi_dev              THI above the heat-stress threshold (>~70)
  temp_roll_slope      multi-day rate of change (carried through if present)
  cold_stress_flag     below target by > COLD_MARGIN
  heat_stress_flag     THI above HEAT_THI

Broiler target-temperature schedule (edit to your management guide):
  day 0-3: 33C, then decline ~ -0.5C/day to a floor of 21C by ~day 24, held after.

Usage:
  python src/extraction/build_welfare_env.py \
      --in features/env_features_Room2.csv --room 2 \
      --out features/env_welfare_Room2.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

COMFORT_RH = (45, 65)
HEAT_THI = 70.0
COLD_MARGIN = 3.0


def target_temp(age_days):
    """Age-based broiler set-point (deg C). 33C brooding -> 21C floor by ~day24."""
    return np.clip(33.0 - 0.5 * np.maximum(age_days - 3, 0), 21.0, 33.0)


def thi_index(temp_c, rh_pct):
    """Temperature-humidity index (poultry). THI = 0.8*T + RH/100*(T-14.4) + 46.4."""
    return 0.8 * temp_c + (rh_pct / 100.0) * (temp_c - 14.4) + 46.4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--room", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        print(f"OUTPUT EXISTS, not overwriting: {out}"); return

    df = pd.read_csv(args.infile)
    # find the relevant columns flexibly
    def pick(*cands):
        for c in cands:
            if c in df.columns:
                return c
        return None
    t_mean = pick("temp_day_mean_c", "env_temp_day_mean_c", "temp_mean_c")
    rh_mean = pick("rh_day_mean_pct", "env_rh_day_mean_pct", "rh_mean_pct")
    age = pick("day_index", "env_day_index", "age_days")
    time = pick("time", "date")
    if t_mean is None or rh_mean is None:
        raise SystemExit(f"could not find temp/RH columns in {args.infile}: {list(df.columns)[:20]}")

    o = pd.DataFrame()
    if time:
        o["time"] = df[time]
    o["temp_c"] = df[t_mean]; o["rh_pct"] = df[rh_mean]
    o["thi"] = thi_index(df[t_mean], df[rh_mean])
    if age:
        o["age_days"] = df[age]
        o["temp_setpoint"] = target_temp(df[age].astype(float))
        o["temp_dev_setpoint"] = o["temp_c"] - o["temp_setpoint"]
        o["temp_dev_abs"] = o["temp_dev_setpoint"].abs()
        o["cold_stress_flag"] = (o["temp_dev_setpoint"] < -COLD_MARGIN).astype(int)
    o["rh_dev"] = np.where(o["rh_pct"] < COMFORT_RH[0], o["rh_pct"] - COMFORT_RH[0],
                   np.where(o["rh_pct"] > COMFORT_RH[1], o["rh_pct"] - COMFORT_RH[1], 0.0))
    o["thi_dev"] = np.maximum(o["thi"] - HEAT_THI, 0.0)
    o["heat_stress_flag"] = (o["thi"] > HEAT_THI).astype(int)
    for c in ["temp_rate_c_per_day", "env_temp_roll_slope_c_per_day", "temp_roll_slope_c_per_day"]:
        if c in df.columns:
            o["temp_roll_slope"] = df[c]; break

    out.parent.mkdir(parents=True, exist_ok=True)
    o.to_csv(out, index=False)
    print(f"WROTE {out}  rows={len(o)}  cols={o.shape[1]}")
    print(o.head().to_string(index=False))


if __name__ == "__main__":
    main()
