#!/usr/bin/env python3
"""Factorized benchmark: run the classical sequential detectors (CUSUM, Page-Hinkley,
EWMA, ADWIN) and our alpha_t persistence gate on the SAME causal residual, all
calibrated to the same false-alarm rate, and compare detection recall and delay.
This is the test that decides whether the gate / coupling beats the classics.

Run: python src/pipeline/dev_benchmark.py --spine results/spine_room2_rich.csv --trials 400
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import RESULTS_DIR
from src.pipeline.dev_age_normality import behavioural_features, causal_age_tod_residual

OUT = RESULTS_DIR / "dev_conditioned" / "benchmark"
GATE = dict(deadband=1.0, decay=0.85, close=6.0, open=2.0, cap=2.0)


# ---------------------------------------------------------------- residual stream
def residual_energy(spine, features_csv):
    """Scalar per-hour anomaly signal = magnitude of the causal residual, robustly
    z-scored on the clean distribution (the shared input every detector sees)."""
    m = pd.read_csv(spine, parse_dates=["time"])
    feats = behavioural_features(m.columns, features_csv)
    fus = m[m["coverage_state"] == "both_lit"].sort_values("time").dropna(subset=feats).reset_index(drop=True)
    resid, _ = causal_age_tod_residual(fus, feats)
    e = np.linalg.norm(np.nan_to_num(resid), axis=1)
    e = e[~np.isnan(resid).any(axis=1)]                      # drop warm-up rows
    med = np.median(e); mad = np.median(np.abs(e - med)) * 1.4826 + 1e-9
    return (e - med) / mad                                   # z-scored clean pool


# ---------------------------------------------------------------- detectors
# Each returns an ALARM-ONSET stream (True only at the hour an alarm *starts*), so
# a running detector that stays triggered counts as one event, making the matched
# false-alarm calibration meaningful. Accumulating detectors RESET when they fire.
def _rising(level):
    out = np.zeros(len(level), bool)
    out[1:] = level[1:] & ~level[:-1]; out[0] = level[0]
    return out

def d_static(z, h):
    return _rising(z > h)

def d_ewma(z, h, lam=0.3):
    g = 0.0; out = np.zeros(len(z), bool)
    for t, v in enumerate(z):
        g = (1 - lam) * g + lam * v
        if g > h:
            out[t] = True; g = 0.0                          # fire + reset
    return out

def d_cusum(z, h, k=0.5):
    S = 0.0; out = np.zeros(len(z), bool)
    for t, v in enumerate(z):
        S = max(0.0, S + v - k)
        if S > h:
            out[t] = True; S = 0.0
    return out

def d_ph(z, h, delta=0.1):
    mean = 0.0; cum = 0.0; mn = 0.0; nt = 0; out = np.zeros(len(z), bool)
    for t, v in enumerate(z):
        nt += 1; mean += (v - mean) / nt
        cum += v - mean - delta; mn = min(mn, cum)
        if (cum - mn) > h:
            out[t] = True; cum = 0.0; mn = 0.0; mean = 0.0; nt = 0
    return out

def d_adwin(z, h, w=24):
    """Lightweight ADWIN-style: recent-window mean exceeds the older reference mean
    by more than h std units (onset = rising edge of that condition)."""
    level = np.zeros(len(z), bool)
    for t in range(2 * w, len(z)):
        recent = z[t - w:t]; older = z[t - 2 * w:t - w]
        s = np.sqrt(recent.var() / w + older.var() / w) + 1e-9
        level[t] = (recent.mean() - older.mean()) / s > h
    return _rising(level)

def d_gate(z, h):
    """alpha_t persistence gate; h is the close threshold (calibration knob).
    Onset = the hour the gate latches closed."""
    P = 0.0; latched = False; level = np.zeros(len(z), bool)
    for t, v in enumerate(z):
        e = min(max(0.0, v - GATE["deadband"]), GATE["cap"])
        P = GATE["decay"] * P + e
        latched = (P > GATE["open"]) if latched else (P >= h)
        level[t] = latched
    return _rising(level)

DETECTORS = {"Static": d_static, "EWMA": d_ewma, "CUSUM": d_cusum,
             "Page-Hinkley": d_ph, "ADWIN": d_adwin, "alpha_t gate": d_gate}


# ---------------------------------------------------------------- eval
def make_timeline(pool, rng, timeline, inject, mag=3.0, dur=(12, 30), spike_mag=6.0, n_spike=4):
    z = rng.choice(pool, size=timeline).astype(float)
    span, spikes = None, []
    if inject:
        a0 = rng.integers(40, timeline - 40); L = int(rng.integers(*dur))
        z[a0:a0 + L] += mag; span = (a0, a0 + L)
        for s in rng.integers(20, timeline - 4, size=n_spike):
            z[s] += spike_mag; spikes.append(int(s))
    return z, span, spikes


def calibrate(fn, pool, rng, target_fa, timeline=240, n=200):
    """Find threshold h giving ~target_fa alarms per clean timeline."""
    clean = [make_timeline(pool, rng, timeline, inject=False)[0] for _ in range(n)]
    grid = np.linspace(0.5, 15, 60)
    best, bestgap = grid[-1], 1e9
    for h in grid:
        fa = np.mean([fn(z, h).sum() for z in clean])
        if abs(fa - target_fa) < bestgap:
            bestgap, best = abs(fa - target_fa), h
    return best


def evaluate(fn, h, pool, rng, trials, timeline=240):
    det, delays, spike_fp, clean_fa = 0, [], 0, 0
    for _ in range(trials):
        z, span, spikes = make_timeline(pool, rng, timeline, inject=True)
        al = fn(z, h)
        a0, a1 = span
        win = al[a0:a1]
        if win.any():
            det += 1; delays.append(int(np.argmax(win)))
        for s in spikes:
            if al[s:s + 2].any():
                spike_fp += 1
        # clean-region false alarms (outside event + spikes)
        mask = np.ones(timeline, bool); mask[a0:a1] = False
        for s in spikes:
            mask[s:s + 2] = False
        clean_fa += int(al[mask].sum())
    return dict(recall=det / trials,
                median_delay_h=float(np.median(delays)) if delays else np.nan,
                spike_fp_per_trial=spike_fp / trials,
                clean_fa_per_trial=clean_fa / trials)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spine", default=str(RESULTS_DIR / "spine_room2_rich.csv"))
    ap.add_argument("--features", default=str(RESULTS_DIR / "recommended_features.csv"))
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--target-fa", type=float, default=0.3, help="alarms per clean 240h timeline")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    pool = residual_energy(a.spine, a.features)
    rng = np.random.default_rng(0)
    rows = []
    for name, fn in DETECTORS.items():
        h = calibrate(fn, pool, rng, a.target_fa)
        r = evaluate(fn, h, pool, rng, a.trials)
        rows.append({"detector": name, "threshold": round(float(h), 2), **{k: round(v, 3) for k, v in r.items()}})
        print(f"[done] {name}")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "benchmark_results.csv", index=False)
    print("\n=== Matched false-alarm benchmark on the causal residual "
          f"(target FA ~{a.target_fa}/timeline) ===")
    print(res.to_string(index=False))

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                             "axes.grid": True, "grid.alpha": 0.3})
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
        order = res["detector"].tolist()
        colors = ["#9aa0a6"] * len(order)
        colors[order.index("alpha_t gate")] = "#2e7d32"
        ax[0].bar(order, res["recall"], color=colors)
        ax[0].set_ylabel("sustained-anomaly recall"); ax[0].set_ylim(0, 1.05)
        ax[0].set_title("Detection at matched false-alarm rate")
        for lab in ax[0].get_xticklabels(): lab.set_rotation(30); lab.set_ha("right")
        ax[1].bar(order, res["spike_fp_per_trial"], color=colors)
        ax[1].set_ylabel("benign-spike false alarms / trial")
        ax[1].set_title("Robustness to brief spikes")
        for lab in ax[1].get_xticklabels(): lab.set_rotation(30); lab.set_ha("right")
        fig.suptitle("Factorized benchmark on the shared causal residual", fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(OUT / "fig_benchmark.png", dpi=600); plt.close(fig)
        print(f"\nWrote benchmark_results.csv, fig_benchmark.png in {OUT}")
    except ImportError:
        print("(matplotlib missing — CSV only)")


if __name__ == "__main__":
    main()
