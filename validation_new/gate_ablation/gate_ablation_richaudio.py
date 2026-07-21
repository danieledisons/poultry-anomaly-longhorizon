#!/usr/bin/env python3
"""
#1: Joint Mahalanobis slow-state gate ablation on RICH AUDIO (welfare-relevant).

Replaces the env demonstration with a high-dimensional, welfare-meaningful one.
The slow-state is the multivariate mean vector of the rich mel-spectrum; the
anomaly score is the Mahalanobis distance of the current causal slow-band level
from that mean. A sustained, COHERENT vocalization-band departure lights up many
mel bands at once (persistent, high joint energy) while idiosyncratic per-band
noise averages out -- exactly the multimodal advantage a single scalar lacks.

Scientific-integrity commitments (enforced in code, see comments):
  * Strictly causal: slow band, reference mean, and shrinkage covariance are all
    estimated from PAST-ONLY rows before the injection window. No future leakage.
  * Principled injection: we specify a PHYSICAL perturbation (offset across a set
    of mel bands, each scaled to that band's own causal MAD) and MEASURE the
    resulting Mahalanobis magnitude. The Mahalanobis level is an output, not a
    tuned input.
  * Data-driven thresholds: deadband and detection threshold come from the
    reference (pre-injection) Mahalanobis distribution's robust quantiles.
  * Honest reporting: we also report the clean-data false-closure rate.

Usage:
  python validation_new/gate_ablation/gate_ablation_richaudio.py --spine <csv> --room 2
Outputs (per room) into this directory:
  richaudio_room{R}_trace.csv, richaudio_room{R}_summary.csv, fig_richaudio_room{R}.png
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

# --- causal slow-band extraction ---
TREND_WIN = 25          # trailing median window (hours)
DIURNAL_ALPHA = 0.2     # per hour-of-day causal EWMA rate
# --- online slow-state assimilation ---
K_GAIN = 0.10           # baseline forgetting rate per hour
# --- gate persistence (report S3.3); thresholds set from reference below ---
GAMMA = 0.85
C_CAP = 4.0
# --- injection (PHYSICAL spec, not a Mahalanobis target) ---
INJ_BANDS = list(range(20, 41))   # contiguous mid mel-band block (vocalization-ish)
INJ_PER_BAND_SIGMA = 4.0          # illustrative trace: 4 x each band's causal MAD
INJ_SWEEP = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]  # magnitude sweep (detection boundary)
INJ_LENGTH = 72                   # hours
REF_FRAC = 0.35                   # fraction of series (from start) used as causal reference


def ledoit_wolf(X):
    """Ledoit-Wolf shrinkage covariance toward a scaled identity. Closed form,
    no fitting. X: (n, d) centered or not (we center here)."""
    n, d = X.shape
    Xc = X - X.mean(0)
    S = (Xc.T @ Xc) / n
    mu = np.trace(S) / d
    F = mu * np.eye(d)                      # shrinkage target
    # optimal shrinkage intensity
    d2 = np.sum((S - F) ** 2)
    b2 = 0.0
    for i in range(n):
        xi = Xc[i][:, None]
        b2 += np.sum((xi @ xi.T - S) ** 2)
    b2 = b2 / (n ** 2)
    b2 = min(b2, d2)
    lam = b2 / d2 if d2 > 0 else 0.0
    return (1 - lam) * S + lam * F, lam


def causal_slow_band(df, feats):
    """Per-feature strictly-causal slow band: trailing median trend + per
    hour-of-day causal EWMA. Row t uses only rows <= t."""
    t = pd.to_datetime(df["time"])
    hod = t.dt.hour.to_numpy()
    Z = np.zeros((len(df), len(feats)))
    for j, f in enumerate(feats):
        s = df[f].ffill()
        trend = s.rolling(TREND_WIN, center=False, min_periods=1).median().to_numpy(float)
        detr = s.to_numpy(float) - trend
        state = {h: None for h in range(24)}
        diur = np.zeros(len(df))
        for i in range(len(df)):
            h = int(hod[i]); v = detr[i]
            if state[h] is not None:
                diur[i] = state[h]
            if np.isfinite(v):
                state[h] = v if state[h] is None else (1 - DIURNAL_ALPHA) * state[h] + DIURNAL_ALPHA * v
        Z[:, j] = np.nan_to_num(detr - diur)
    return Z


def alpha_gate(scores, sigma, b_dead, theta_close, theta_open):
    """Runs the assimilation NOT here; this only maps a score series to alpha via
    persistence. (Kept separate for the clean-data false-closure measurement.)"""
    P = 0.0; alpha = np.ones(len(scores))
    for t, sc in enumerate(scores):
        e = min(max(abs(sc) / sigma - b_dead, 0.0), C_CAP)
        P = GAMMA * P + e
        alpha[t] = 0.0 if P >= theta_close else (1.0 if P <= theta_open else
                     (theta_close - P) / (theta_close - theta_open))
    return alpha


def run(Z, mu0, Sinv, med, sigma, b_dead, theta_close, theta_open, gated):
    """Online multivariate slow-state. mu is the ASSIMILATING mean vector; score
    is the Mahalanobis distance of z_t from mu (pre-update). The gate fires on the
    EXCESS of that distance over the normal operating level `med` (centering is
    essential: under bird growth the slow band drifts, so an un-centered distance
    would be chronically large and close the gate on clean data)."""
    n = len(Z); mu = mu0.copy()
    score = np.zeros(n); alpha = np.ones(n); P = 0.0
    for t in range(n):
        d = Z[t] - mu
        score[t] = np.sqrt(max(d @ Sinv @ d, 0.0))
        if gated:
            e = min(max((score[t] - med) / sigma - b_dead, 0.0), C_CAP)
            P = GAMMA * P + e
            a = 0.0 if P >= theta_close else (1.0 if P <= theta_open else
                 (theta_close - P) / (theta_close - theta_open))
        else:
            a = 1.0
        alpha[t] = a
        mu = mu + K_GAIN * a * (Z[t] - mu)     # gated assimilation of the mean
    return score, alpha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spine", default=str(DATA / "spine_room2_rich.csv"))
    ap.add_argument("--room", default="2")
    ap.add_argument("--outdir", default=str(HERE))
    args = ap.parse_args()

    df = pd.read_csv(args.spine).sort_values("time").reset_index(drop=True)
    mel = [c for c in df.columns if c.startswith("aud_mel") and c.endswith("_mean")]
    # keep hours with real audio
    df = df[df[mel].notna().all(axis=1)].reset_index(drop=True)
    n = len(df)
    feats = mel  # 64 mel-mean bands: welfare-relevant vocalization spectrum

    Z = causal_slow_band(df, feats)

    # ---- CAUSAL reference: first REF_FRAC of the series only ----
    ref_end = int(REF_FRAC * n)
    ref = Z[:ref_end]
    mu0 = ref[:TREND_WIN].mean(0)                    # initial mean from earliest rows
    Sigma, lam = ledoit_wolf(ref)
    Sinv = np.linalg.inv(Sigma)

    # Calibrate on the OPERATING-score distribution: run the ASSIMILATING slow
    # state (ungated) over the clean reference so the mean tracks legitimate
    # growth. The residual Mahalanobis distances that remain are the true null.
    b_dead = 1.0; theta_close = 6.0; theta_open = 2.0
    ref_op, _ = run(ref, mu0, Sinv, 0.0, 1.0, 0.0, theta_close, theta_open, gated=False)
    ref_op = ref_op[TREND_WIN:]                     # drop warm-up transient
    med = np.median(ref_op)
    sigma = 1.4826 * np.median(np.abs(ref_op - med))
    sigma = sigma if sigma > 1e-9 else ref_op.std()
    det_thresh = np.quantile(ref_op, 0.99)          # detection at ref 99th pct

    # ---- injection AFTER the reference window (no leakage into ref) ----
    start = ref_end + 150
    end = min(start + INJ_LENGTH, n)
    band_idx = [feats.index(f"aud_mel{b:02d}_mean") for b in INJ_BANDS if f"aud_mel{b:02d}_mean" in feats]
    band_mad = 1.4826 * np.median(np.abs(Z[:ref_end][:, band_idx] -
                                         np.median(Z[:ref_end][:, band_idx], 0)), 0)

    def retention(sc):
        seg = sc[start:end]
        return float((seg >= det_thresh).mean())

    def latency(sc):
        seg = sc[start:end]
        below = np.where(seg < det_thresh)[0]
        return int(below[0]) if len(below) else np.inf

    def evaluate(mag):
        Zin = Z.copy()
        Zin[start:end][:, band_idx] += mag * band_mad          # physical shift
        su, au = run(Zin, mu0, Sinv, med, sigma, b_dead, theta_close, theta_open, gated=False)
        sg, ag = run(Zin, mu0, Sinv, med, sigma, b_dead, theta_close, theta_open, gated=True)
        return su, au, sg, ag

    # --- magnitude sweep: honest detection boundary ---
    sweep_rows = []
    for mag in INJ_SWEEP:
        su, _, sg, _ = evaluate(mag)
        onset = np.median(sg[start:start + 3])
        sweep_rows.append({
            "mag_band_mad": mag, "onset_joint_maha": round(onset, 2),
            "onset_centered_sigma": round((onset - med) / sigma, 2),
            "above_natural_thresh": bool(onset > det_thresh),
            "ungated_retention": round(retention(su), 3),
            "gated_retention": round(retention(sg), 3),
            "ungated_latency_hr": latency(su), "gated_latency_hr": latency(sg),
        })
    sweep = pd.DataFrame(sweep_rows)

    # --- illustrative trace at INJ_PER_BAND_SIGMA ---
    sc_u, al_u, sc_g, al_g = evaluate(INJ_PER_BAND_SIGMA)
    inj_maha = np.median(sc_g[start:start + 3])

    # clean-data false-closure rate: gate on the reference (no injection)
    sc_ref, al_ref = run(Z[:ref_end], mu0, Sinv, med, sigma, b_dead, theta_close, theta_open, gated=True)
    false_closure = float((al_ref[TREND_WIN:] < 0.5).mean())

    R = args.room
    trace = pd.DataFrame({
        "time": df["time"], "maha_ungated": sc_u, "maha_gated": sc_g,
        "alpha_gated": al_g,
    })
    trace.to_csv(Path(args.outdir) / f"richaudio_room{R}_trace.csv", index=False)
    sweep.to_csv(Path(args.outdir) / f"richaudio_room{R}_sweep.csv", index=False)
    summary = pd.DataFrame([{
        "room": R, "n_hours": n, "n_features": len(feats), "shrinkage_lambda": round(lam, 4),
        "ref_hours": ref_end, "inj_bands": f"{INJ_BANDS[0]}-{INJ_BANDS[-1]}",
        "inj_per_band_sigma": INJ_PER_BAND_SIGMA, "measured_inj_maha_sigma": round(inj_maha / sigma, 2),
        "det_thresh_maha": round(det_thresh, 3), "sigma_maha": round(sigma, 3),
        "ungated_contam_latency_hr": latency(sc_u), "gated_contam_latency_hr": latency(sc_g),
        "ungated_retention": round(retention(sc_u), 3), "gated_retention": round(retention(sc_g), 3),
        "clean_false_closure_rate": round(false_closure, 3),
    }])
    summary.to_csv(Path(args.outdir) / f"richaudio_room{R}_summary.csv", index=False)

    # ---- figure ----
    x = np.arange(n)
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax[0].axhline(det_thresh, color="k", ls="--", lw=0.8, label="detect thresh (ref 99pct)")
    ax[0].plot(x, sc_u, color="tab:red", lw=1.0, label="Mahalanobis ungated (absorbs)")
    ax[0].plot(x, sc_g, color="tab:green", lw=1.0, label="Mahalanobis gated (holds)")
    ax[0].axvspan(start, end, color="orange", alpha=0.12, label="sustained voc-band departure")
    ax[0].axvspan(0, ref_end, color="blue", alpha=0.05, label="causal reference window")
    ax[0].set_ylabel("joint Mahalanobis"); ax[0].legend(fontsize=7, loc="upper left")
    ax[0].set_title(f"Room {R}: rich-audio joint slow-state gate ablation "
                    f"({len(feats)} mel bands, causal)")
    ax[1].plot(x, al_g, color="tab:green", lw=1.2, label="alpha_t (gated)")
    ax[1].axvspan(start, end, color="orange", alpha=0.12)
    ax[1].set_ylabel("trust alpha_t"); ax[1].set_xlabel("audio hours"); ax[1].set_ylim(-0.05, 1.05)
    ax[1].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(Path(args.outdir) / f"fig_richaudio_room{R}.png", dpi=140)

    print(f"=== Room {R} rich-audio gate ablation ===")
    print(f"features {len(feats)} mel bands | shrinkage lambda {lam:.3f} | ref hours {ref_end}/{n}")
    print(f"injection: {INJ_PER_BAND_SIGMA} band-MAD across mel {INJ_BANDS[0]}-{INJ_BANDS[-1]} "
          f"-> measured {inj_maha/sigma:.1f} sigma joint")
    print(f"ungated: latency {latency(sc_u)} h  retention {retention(sc_u):.2f}")
    print(f"gated:   latency {latency(sc_g)} h  retention {retention(sc_g):.2f}")
    print(f"clean-data false-closure rate: {false_closure:.3f}")


if __name__ == "__main__":
    main()
