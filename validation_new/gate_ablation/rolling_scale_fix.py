#!/usr/bin/env python3
"""
Stop-gap for the §2.1 late-cycle false-alarm problem: trailing rolling robust scale.

#1 calibrated the gate's scale/threshold once, on the early-growth reference. As the
flock matures the operating Mahalanobis level rises, so a fixed threshold over-fires
late in the cycle (adaptation-lag trap). Here we recalibrate the threshold ONLINE
against a trailing (past-only) rolling median/MAD of the operating score, so a
departure is judged relative to the flock's RECENT normal variability rather than
its week-1 variability.

This is a deliberate stop-gap: it corrects the scale non-stationarity, not the
geometry rotation (§6). We quantify the improvement by comparing fixed vs rolling
late-cycle false-closure, and confirm injection detection is preserved.

Caveat enforced: the rolling window must be much longer than a plausible sustained
event, or it would adapt to (absorb) the anomaly. We use 1 week (168 h) >> the 72 h
event, and report the window as an explicit parameter.

Usage:
  python validation_new/gate_ablation/rolling_scale_fix.py --spine <csv> --room 2
"""
from __future__ import annotations
import argparse
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
spec = importlib.util.spec_from_file_location("ra", HERE / "gate_ablation_richaudio.py")
ra = importlib.util.module_from_spec(spec); spec.loader.exec_module(ra)
TREND_WIN = ra.TREND_WIN

K_GAIN = 0.10; GAMMA = 0.85; C_CAP = 4.0; THETA_CLOSE = 6.0; THETA_OPEN = 2.0
B_DEAD = 1.0; ROLL_WIN = 168; INJ_BANDS = list(range(20, 41)); INJ_LEN = 72; REF_FRAC = 0.35


def run(Z, mu0, Sinv, calib, gated):
    """calib(t, score_hist) -> (med_t, sig_t, det_t). If None, fixed calibration is
    applied by the caller via a closure. Returns score, alpha, det_thresh series."""
    n = len(Z); mu = mu0.copy(); P = 0.0
    score = np.zeros(n); alpha = np.ones(n); det = np.zeros(n)
    for t in range(n):
        d = Z[t] - mu; score[t] = np.sqrt(max(d @ Sinv @ d, 0.0))
        med_t, sig_t, det_t = calib(t, score)
        det[t] = det_t
        if gated:
            e = min(max((score[t] - med_t) / sig_t - B_DEAD, 0.0), C_CAP)
            P = GAMMA * P + e
            a = 0.0 if P >= THETA_CLOSE else (1.0 if P <= THETA_OPEN else
                 (THETA_CLOSE - P) / (THETA_CLOSE - THETA_OPEN))
        else:
            a = 1.0
        alpha[t] = a; mu = mu + K_GAIN * a * (Z[t] - mu)
    return score, alpha, det


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spine", default=str(DATA / "spine_room2_rich.csv"))
    ap.add_argument("--room", default="2"); ap.add_argument("--outdir", default=str(HERE))
    a = ap.parse_args(); R = a.room

    df = pd.read_csv(a.spine).sort_values("time").reset_index(drop=True)
    mel = [c for c in df.columns if c.startswith("aud_mel") and c.endswith("_mean")]
    df = df[df[mel].notna().all(axis=1)].reset_index(drop=True); n = len(df)
    Z = ra.causal_slow_band(df, mel)
    ref_end = int(REF_FRAC * n); ref = Z[:ref_end]
    mu0 = ref[:TREND_WIN].mean(0); Sigma, lam = ra.ledoit_wolf(ref); Sinv = np.linalg.inv(Sigma)

    # fixed reference operating stats
    op0, _, _ = run(ref, mu0, Sinv, lambda t, s: (0.0, 1.0, 1e9), gated=False)
    op0 = op0[TREND_WIN:]
    fmed = np.median(op0); fsig = 1.4826 * np.median(np.abs(op0 - fmed)) or op0.std()
    fdet = np.quantile(op0, 0.99)
    q = (fdet - fmed) / fsig                        # detection quantile in sigma, kept constant

    def fixed_calib(t, s):
        return fmed, fsig, fdet

    def rolling_calib(t, s):
        lo = max(0, t - ROLL_WIN)
        w = s[lo:t]
        if len(w) < 24:
            return fmed, fsig, fdet
        m = np.median(w); sg = 1.4826 * np.median(np.abs(w - m)) or fsig
        return m, sg, m + q * sg                     # threshold tracks recent normal

    # injection after reference
    start = ref_end + 150; end = min(start + INJ_LEN, n)
    bidx = [mel.index(f"aud_mel{b:02d}_mean") for b in INJ_BANDS if f"aud_mel{b:02d}_mean" in mel]
    bmad = 1.4826 * np.median(np.abs(Z[:ref_end][:, bidx] - np.median(Z[:ref_end][:, bidx], 0)), 0)
    Zin = Z.copy(); Zin[start:end][:, bidx] += 4.0 * bmad

    out = {}
    for tag, calib in [("fixed", fixed_calib), ("rolling", rolling_calib)]:
        sc_g, al_g, det = run(Zin, mu0, Sinv, calib, gated=True)
        sc_u, _, _ = run(Zin, mu0, Sinv, calib, gated=False)
        # metrics
        retention = float((sc_g[start:end] >= det[start:end]).mean())
        below = np.where(sc_g[start:end] < det[start:end])[0]
        gated_lat = int(below[0]) if len(below) else np.inf
        # false-closure on clean series (no injection)
        scc, alc, _ = run(Z, mu0, Sinv, calib, gated=True)
        late = np.ones(n, bool); late[:ref_end] = False
        false_late = float((alc[late] < 0.5).mean())
        out[tag] = dict(retention=round(retention, 3), gated_latency=gated_lat,
                        late_false_closure=round(false_late, 3),
                        sc_g=sc_g, det=det, al_g=al_g)

    summ = pd.DataFrame([{"room": R, "roll_win_hr": ROLL_WIN, "calib": k,
        "gated_retention": v["retention"], "gated_latency_hr": v["gated_latency"],
        "late_cycle_false_closure": v["late_false_closure"]} for k, v in out.items()])
    summ.to_csv(Path(a.outdir) / f"rolling_room{R}_summary.csv", index=False)

    x = np.arange(n)
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for tag, col in [("fixed", "tab:red"), ("rolling", "tab:green")]:
        ax[0].plot(x, out[tag]["det"], color=col, lw=1.0, ls="--", label=f"{tag} threshold")
    ax[0].plot(x, out["rolling"]["sc_g"], color="0.4", lw=0.8, label="gated Mahalanobis (rolling)")
    ax[0].axvspan(start, end, color="orange", alpha=0.12, label="injection"); ax[0].axvspan(0, ref_end, color="blue", alpha=0.05)
    ax[0].set_ylabel("joint Mahalanobis"); ax[0].legend(fontsize=7); ax[0].set_ylim(0, np.percentile(out["rolling"]["sc_g"], 99.5))
    ax[0].set_title(f"Room {R}: fixed vs trailing-rolling threshold (win={ROLL_WIN}h)")
    ax[1].plot(x, out["fixed"]["al_g"], color="tab:red", lw=1.0, label="alpha fixed")
    ax[1].plot(x, out["rolling"]["al_g"], color="tab:green", lw=1.0, label="alpha rolling")
    ax[1].axvspan(start, end, color="orange", alpha=0.12); ax[1].set_ylabel("alpha_t"); ax[1].set_xlabel("hours"); ax[1].set_ylim(-0.05, 1.05); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(Path(a.outdir) / f"fig_rolling_room{R}.png", dpi=140)

    print(f"=== Room {R}: fixed vs rolling ({ROLL_WIN}h) ===")
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
