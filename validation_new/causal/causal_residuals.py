#!/usr/bin/env python3
"""
Strictly-causal (past-only) preprocessing for the residual gate.

Purpose (July-17 review, point 2): the reported pipeline used operations that
peek at the future -- centred rolling smoothing, bidirectional interpolation,
and a full-series diurnal mean. Those are invalid for an online detection
claim. This script re-derives every per-signal residual using ONLY past
observations and writes both versions so the leakage can be shown directly.

For each of the three classical-fusion signals it computes a residual two ways:

  NON-CAUSAL (leaky, what was reported):
    - centred rolling median trend        (uses future rows)
    - full-series per-hour-of-day mean     (uses future rows)
    - bidirectional (interpolate) fill      (uses future rows)

  CAUSAL (past-only, valid online):
    - trailing rolling median trend         (t uses rows <= t)
    - per-hour-of-day EWMA, pre-update       (prediction at t uses rows < t only)
    - forward-fill only                      (never reaches back from the future)

Outputs
  causal_residuals_room2.csv   both residual streams per signal, aligned on time
  fig_causal_vs_leaky.png      before/after comparison

Run:
  python validation_new/causal/causal_residuals.py
Deterministic: no randomness, no seed needed.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

# The three representative classical-fusion signals (report S3.1).
SIGNALS = {
    "video_activity":     "vid_flow_mean_avg",
    "audio_vocalization": "aud_voc_frac_mean",
    "env_temperature":    "env_temp_day_mean_c",
}
TREND_WIN = 25   # hours; odd so the centred window is symmetric
EWMA_ALPHA = 0.2  # per hour-of-day occurrence; ~5-day memory, tracks growth


def robust_scale(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = 1.4826 * mad if mad > 1e-9 else 1.0
    return (x - med) / scale


# ----------------------------------------------------------------------------
# NON-CAUSAL (leaky) path -- reproduces the future-peeking operations.
# ----------------------------------------------------------------------------
def leaky_residual(s: pd.Series, hod: np.ndarray) -> np.ndarray:
    # bidirectional fill (reaches into the future)
    s = s.interpolate(limit_direction="both")
    # centred rolling median trend (window straddles the current point)
    trend = s.rolling(TREND_WIN, center=True, min_periods=1).median()
    detr = s - trend
    # full-series per-hour-of-day mean (computed from ALL rows, incl. future)
    diur = pd.Series(detr).groupby(hod).transform("mean")
    return robust_scale((detr - diur).to_numpy(float))


# ----------------------------------------------------------------------------
# CAUSAL (past-only) path -- valid for an online claim.
# ----------------------------------------------------------------------------
def causal_residual(s: pd.Series, hod: np.ndarray) -> np.ndarray:
    # forward-fill only: a missing value is filled from the past, never the future
    s = s.ffill()
    x = s.to_numpy(float)
    n = len(x)
    # trailing rolling median trend: row t uses only rows <= t
    trend = s.rolling(TREND_WIN, center=False, min_periods=1).median().to_numpy(float)
    detr = x - trend
    # per-hour-of-day causal EWMA: prediction at t is the state BEFORE seeing t
    state = {h: None for h in range(24)}
    diur = np.full(n, np.nan)
    for t in range(n):
        h = int(hod[t])
        v = detr[t]
        if state[h] is not None:
            diur[t] = state[h]          # causal prediction (pre-update)
        if np.isfinite(v):
            state[h] = v if state[h] is None else (1 - EWMA_ALPHA) * state[h] + EWMA_ALPHA * v
    diur = np.where(np.isnan(diur), 0.0, diur)
    return robust_scale(detr - diur)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", default=str(DATA / "room2_merged_hourly.csv"))
    ap.add_argument("--outdir", default=str(HERE))
    args = ap.parse_args()

    df = pd.read_csv(args.merged).sort_values("time").reset_index(drop=True)
    t = pd.to_datetime(df["time"])
    hod = t.dt.hour.to_numpy()

    out = pd.DataFrame({"time": df["time"]})
    for name, col in SIGNALS.items():
        s = df[col]
        out[f"{name}__resid_leaky"] = leaky_residual(s.copy(), hod)
        out[f"{name}__resid_causal"] = causal_residual(s.copy(), hod)

    outdir = Path(args.outdir)
    csv_path = outdir / "causal_residuals_room2.csv"
    out.to_csv(csv_path, index=False)

    # ---- before/after figure -------------------------------------------------
    fig, axes = plt.subplots(len(SIGNALS), 1, figsize=(12, 9), sharex=True)
    for ax, name in zip(axes, SIGNALS):
        ax.plot(t, out[f"{name}__resid_leaky"], lw=0.8, alpha=0.6,
                label="leaky (centred / bidirectional / full-series)")
        ax.plot(t, out[f"{name}__resid_causal"], lw=0.8,
                label="causal (past-only)")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel(name, fontsize=9)
        ax.legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("time")
    fig.suptitle("Residuals: leaky (future-peeking) vs strictly-causal (past-only)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig_path = outdir / "fig_causal_vs_leaky.png"
    fig.savefig(fig_path, dpi=140)

    # ---- leakage summary -----------------------------------------------------
    print(f"wrote {csv_path}")
    print(f"wrote {fig_path}\n")
    print(f"{'signal':20s} {'corr(leaky,causal)':>18s} {'std_leaky':>10s} {'std_causal':>11s}")
    for name in SIGNALS:
        a = out[f"{name}__resid_leaky"]; b = out[f"{name}__resid_causal"]
        m = a.notna() & b.notna()
        r = np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 2 else np.nan
        print(f"{name:20s} {r:18.3f} {a.std():10.3f} {b.std():11.3f}")


if __name__ == "__main__":
    main()
