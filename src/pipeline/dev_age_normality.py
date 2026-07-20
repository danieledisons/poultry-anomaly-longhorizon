#!/usr/bin/env python3
"""
dev_age_normality.py — CAUSAL, development-conditioned normality (component #1 of
the dev-conditioned drift-vs-anomaly model).

Prerequisite for the whole novelty programme: every detector in the factorized
benchmark (CUSUM / Page-Hinkley / EWMA / ADWIN / full method) must run on the
SAME causal residual, or the comparison is unfair and the "leakage" critique bites.

Model:  x_{t,j} = m_{t,j}(age, time-of-day, barn) + r_{t,j}
where m is estimated ONLINE using only the past — no future leakage. Concretely,
m is a per-(hour-of-day) exponentially-weighted running mean updated across days;
the prediction for hour t uses the state BEFORE seeing x_t, so it is strictly causal
and naturally tracks the slow growth + diurnal drift as the flock ages.

Outputs (RESULTS_DIR/dev_conditioned/model/):
    residuals_room{R}.csv         time + causal residual per feature + coverage flags
    fig_causal_vs_global.png      one feature: raw, causal m, causal residual vs the
                                  old (non-causal, full-record) detrend residual

This is a validation of the FOUNDATION, not a detector yet.

Usage
-----
    python src/pipeline/dev_age_normality.py --spine results/spine_room2_rich.csv --room 2
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import RESULTS_DIR

OUT = RESULTS_DIR / "dev_conditioned" / "model"


def behavioural_features(cols, feature_csv):
    lean = pd.read_csv(feature_csv)["feature"].tolist()
    return [f for f in lean if (f.startswith("aud_") or f.startswith("vid_")) and f in cols]


def robust_standardize(X):
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    scale = np.where(mad < 1e-9, 1.0, 1.4826 * mad)
    return np.clip((X - med) / scale, -5, 5)   # clip heavy tails so variance is meaningful


def causal_age_tod_residual(df, feats, alpha=0.2):
    """Per-(hour-of-day) causal EWMA across days. Prediction at t uses state built
    from strictly earlier rows only. alpha = update rate per day-occurrence of an
    hour (0.2 => ~5-occurrence memory, tracks growth without leaking the future).
    Features are robustly standardized first so the residual is well-scaled."""
    df = df.sort_values("time").reset_index(drop=True)
    hod = pd.to_datetime(df["time"]).dt.hour.to_numpy()
    X = robust_standardize(df[feats].to_numpy(float))
    n, d = X.shape
    state = {h: None for h in range(24)}      # per-hour EWMA vector
    m = np.full((n, d), np.nan)
    for t in range(n):
        h = hod[t]
        if state[h] is not None:
            m[t] = state[h]                    # causal prediction (pre-update)
        x = X[t]
        if state[h] is None:
            state[h] = x.copy()
        else:
            state[h] = (1 - alpha) * state[h] + alpha * x
    resid = X - m
    return resid, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spine", default=str(RESULTS_DIR / "spine_room2_rich.csv"))
    ap.add_argument("--features", default=str(RESULTS_DIR / "recommended_features.csv"))
    ap.add_argument("--room", default="2")
    ap.add_argument("--alpha", type=float, default=0.2)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    m_df = pd.read_csv(a.spine, parse_dates=["time"])
    feats = behavioural_features(m_df.columns, a.features)
    fus = m_df[m_df["coverage_state"] == "both_lit"].sort_values("time").dropna(subset=feats).reset_index(drop=True)

    resid, mhat = causal_age_tod_residual(fus, feats, a.alpha)

    out = pd.DataFrame({"time": fus["time"], "hour_of_day": pd.to_datetime(fus["time"]).dt.hour,
                        "day_index": fus["env_day_index"]})
    for j, f in enumerate(feats):
        out[f"resid_{f}"] = resid[:, j]
    out.to_csv(OUT / f"residuals_room{a.room}.csv", index=False)

    # diagnostics: causal residual variance vs raw variance (should shrink -> m explains structure)
    Xs = robust_standardize(fus[feats].to_numpy(float))
    valid = ~np.isnan(resid).any(axis=1)
    raw_var = np.nanvar(Xs[valid], axis=0).mean()
    res_var = np.nanvar(resid[valid], axis=0).mean()
    print(f"Room {a.room}: {int(valid.sum())}/{len(fus)} hours with causal prediction")
    print(f"mean feature variance: raw={raw_var:.3f}  causal-residual={res_var:.3f}  "
          f"(explained {100*(1-res_var/raw_var):.1f}%)")

    # figure: one representative feature
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                             "axes.grid": True, "grid.alpha": 0.25})
        f = "vid_flow_mean_avg" if "vid_flow_mean_avg" in feats else feats[0]
        j = feats.index(f)
        t = pd.to_datetime(fus["time"])
        x = Xs[:, j]                                       # standardized raw
        # old non-causal global detrend (centered full-record) for contrast
        glob = pd.Series(x, index=t).rolling(24 * 7, center=True, min_periods=24).mean().to_numpy()
        fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        ax[0].plot(t, x, ".", ms=2, color="#b8b8b8", label="raw")
        ax[0].plot(t, mhat[:, j], color="#1565c0", lw=1.6, label="causal m(age, time-of-day)")
        ax[0].plot(t, glob, color="#c62828", lw=1.2, ls="--", label="non-causal full-record trend (old)")
        ax[0].set_title(f"Causal development-conditioned normality — {f}", fontweight="bold")
        ax[0].legend(fontsize=8)
        ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax[1].axhline(0, color="k", lw=.6)
        ax[1].plot(t, resid[:, j], color="#2e7d32", lw=.6)
        ax[1].fill_between(t, 0, resid[:, j], color="#2e7d32", alpha=.2)
        ax[1].set_title("Causal residual (fast band, no future leakage) — detector input")
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        fig.tight_layout(); fig.savefig(OUT / "fig_causal_vs_global.png", dpi=600); plt.close(fig)
        print(f"[write] residuals_room{a.room}.csv, fig_causal_vs_global.png in {OUT}")
    except ImportError:
        print("(matplotlib missing — CSV written, figure skipped)")


if __name__ == "__main__":
    main()
