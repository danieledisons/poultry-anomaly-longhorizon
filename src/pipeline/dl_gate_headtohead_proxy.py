#!/usr/bin/env python3
"""
dl_gate_headtohead_proxy.py — PROXY (no GPU needed). PCA reconstruction stands in
for the DL model so the harness runs anywhere. For real numbers use the PyTorch
version: src/pipeline/dl_gate_pytorch.py (identical evaluation, trained AE/GRU).

The core thesis claim, quantified:

    DL + alpha_t gate  >  DL alone

Scenario: a realistic timeline contains BOTH
  * brief spikes      — benign transients (a bird crosses the mic, a 1-frame glare):
                        NOT anomalies we care about, but they blow past a magnitude
                        threshold.
  * sustained shifts  — the real anomalies (a developing problem over many hours).

DL alone = threshold the per-hour reconstruction error. It cannot tell a benign
spike from the onset of a real event, so it false-alarms on spikes.
DL + gate = feed the same DL error stream through the alpha_t persistence gate,
which only "closes" under SUSTAINED unexplained energy — so spikes are ignored
while sustained events are still caught.

We measure event-level precision / recall / F1 over many randomized trials.

The DL scorer here is a low-rank PCA reconstruction (fast, sandbox-friendly).
On the GPU server, swap `dl_scores()` for a trained autoencoder/GRU — the gate
comparison harness is identical.

Usage
-----
    python src/pipeline/dl_gate_headtohead.py --variant DETREND --trials 300
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import RESULTS_DIR
from src.models.alpha_gate import AlphaGate, robust_scale
from src.pipeline.fusion_compare import robust_standardize, detrend_dayindex

GATE_PARAMS = dict(deadband=1.0, decay=0.85, close_threshold=6.0,
                   open_threshold=2.0, per_step_cap=2.0)


def load_features(spine, features_csv, variant):
    m = pd.read_csv(spine, parse_dates=["time"])
    feats = pd.read_csv(features_csv)["feature"].tolist()
    model_feats = [f for f in feats if f != "env_day_index"]
    fus = (m[m["coverage_state"] == "both_lit"]
           .sort_values("time").dropna(subset=model_feats).reset_index(drop=True))
    X = fus[model_feats].to_numpy(float)
    if variant == "DETREND":
        X = detrend_dayindex(X, fus["env_day_index"].to_numpy(float))
    return robust_standardize(X)


def evaluate(X, trials=300, seed=0,
             n_sustained=3, sustained_len=(12, 30), sustained_mag=2.5,
             n_spikes=5, spike_mag=10.0, timeline=240):
    """Randomized event-level P/R/F1 for DL-alone vs DL+gate."""
    rng = np.random.default_rng(seed)
    n = len(X)
    tp_dl = fp_dl = fn_dl = 0
    tp_g = fp_g = fn_g = 0

    for _ in range(trials):
        # sample a clean timeline of feature rows
        idx = rng.integers(0, n, size=timeline)
        T = X[idx].copy()

        # fit DL (PCA) on a separate clean draw; threshold on its clean errors
        fit = X[rng.integers(0, n, size=min(n, 400))]
        pca = PCA(n_components=0.95, svd_solver="full").fit(fit)
        clean_err = ((fit - pca.inverse_transform(pca.transform(fit))) ** 2).sum(1)
        thr = np.quantile(clean_err, 0.99)
        scale = robust_scale(np.sqrt(clean_err) - np.median(np.sqrt(clean_err)))

        # place events on non-overlapping windows
        slots = rng.permutation(np.arange(20, timeline - 40, 8))
        sustained_spans, spike_pts = [], []
        for s in slots[:n_sustained]:
            L = int(rng.integers(*sustained_len))
            T[s:s + L] += sustained_mag * rng.choice([-1., 1.], size=(1, X.shape[1]))
            sustained_spans.append((s, s + L))
        for s in slots[n_sustained:n_sustained + n_spikes]:
            T[s] += spike_mag * rng.choice([-1., 1.], size=X.shape[1])
            spike_pts.append(s)

        # DL score stream over the timeline
        err = ((T - pca.inverse_transform(pca.transform(T))) ** 2).sum(1)
        dl_flag = err > thr

        # gate on the same stream (as a residual)
        resid = np.sqrt(err) - np.median(np.sqrt(err))
        closed = AlphaGate(scale=scale, **GATE_PARAMS).run(resid)["closed"]

        # event-level scoring
        def hit(flag, span):
            return flag[span[0]:span[1]].any()

        for span in sustained_spans:
            if hit(dl_flag, span): tp_dl += 1
            else: fn_dl += 1
            if hit(closed, span): tp_g += 1
            else: fn_g += 1
        for pt in spike_pts:
            if dl_flag[pt:pt + 2].any(): fp_dl += 1
            if closed[pt:pt + 2].any(): fp_g += 1

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return round(p, 3), round(r, 3), round(f, 3)

    dl = prf(tp_dl, fp_dl, fn_dl)
    g = prf(tp_g, fp_g, fn_g)
    return pd.DataFrame([
        {"method": "DL alone",   "precision": dl[0], "recall": dl[1], "f1": dl[2],
         "sustained_TP": tp_dl, "spike_FP": fp_dl, "sustained_FN": fn_dl},
        {"method": "DL + gate",  "precision": g[0],  "recall": g[1],  "f1": g[2],
         "sustained_TP": tp_g,  "spike_FP": fp_g,  "sustained_FN": fn_g},
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spine", default=str(RESULTS_DIR / "spine_room2_rich.csv"))
    ap.add_argument("--features", default=str(RESULTS_DIR / "recommended_features.csv"))
    ap.add_argument("--variant", default="DETREND", choices=["RAW", "DETREND"])
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--out", default=str(RESULTS_DIR / "dl_vs_gate.csv"))
    args = ap.parse_args()

    X = load_features(args.spine, args.features, args.variant)
    print(f"Features: {X.shape[1]}   fusion hours: {len(X)}   variant: {args.variant}")
    res = evaluate(X, trials=args.trials)
    res.to_csv(args.out, index=False)

    print("\n=== DL alone vs DL + alpha_t gate (event-level, "
          f"{args.trials} trials) ===")
    print(res.to_string(index=False))
    dlf, gf = res.loc[0, "f1"], res.loc[1, "f1"]
    print(f"\nF1: DL={dlf}  ->  DL+gate={gf}  "
          f"({'+' if gf>=dlf else ''}{round(gf-dlf,3)})")
    print("Spike false-alarms: "
          f"DL={res.loc[0,'spike_FP']}  ->  DL+gate={res.loc[1,'spike_FP']}")
    print(f"Wrote -> {args.out}")


if __name__ == "__main__":
    main()
