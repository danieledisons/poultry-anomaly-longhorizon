#!/usr/bin/env python3
"""
Gated vs ungated online slow-state assimilation  (July-17 review, point 1).

The reviewer asked us to "demonstrate an online model whose state is updated
with and without the gate and show directly whether the gate prevents baseline
contamination." This does exactly that on the strictly-causal residual stream
produced by ../causal/causal_residuals.py.

Model (the minimal Approach-C mechanism, isolated):
  An online slow-state b_t tracks the signal and is the baseline against which
  anomaly is scored (score_t = x_t - b_t). It assimilates each new observation
  with gain k, but that gain is scaled by a trust weight alpha_t:

      b_t = b_{t-1} + k * alpha_t * (x_t - b_{t-1})

  UNGATED: alpha_t == 1 always            (pure online forgetting baseline)
  GATED:   alpha_t from the residual gate (report S3.3)

  alpha_t gate (per report):
      e_t = min(max(|score_t|/sigma - b_dead, 0), c_cap)   bounded excess energy
      P_t = gamma * P_{t-1} + e_t                           leaky accumulation
      alpha_t = 0 if P_t >= theta_close                     hysteresis close
                1 if P_t <= theta_open                       hysteresis open
                linear ramp between

We inject a SUSTAINED step departure into a clean segment. The prediction:
  - UNGATED baseline drifts up to absorb the step -> score decays to ~0 ->
    detection is lost after a "contamination latency".
  - GATED baseline freezes (alpha -> 0) -> score stays elevated -> detection
    is preserved indefinitely.

Outputs
  gate_ablation_trace.csv   full per-hour trace (both variants)
  gate_ablation_summary.csv contamination latency + detection retention
  fig_gate_ablation.png     the headline figure

Run:
  python validation_new/gate_ablation/gate_ablation.py
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
CAUSAL = HERE.parent / "causal" / "causal_residuals_room2.csv"

# --- online slow-state assimilation gain ---
K_GAIN = 0.10          # baseline forgetting rate (per hour)
# --- alpha_t gate parameters (fixed, reused across rooms/modalities) ---
B_DEAD = 1.0           # deadband in sigma units
C_CAP = 4.0            # per-step cap on excess energy
GAMMA = 0.85           # leak / memory of the persistence accumulator
THETA_CLOSE = 6.0      # close (stop trusting) above this persistence
THETA_OPEN = 2.0       # reopen below this persistence
# --- detection ---
DET_THRESH = 2.5       # |score| in sigma flagged as a detection


def alpha_from_persistence(P):
    if P >= THETA_CLOSE:
        return 0.0
    if P <= THETA_OPEN:
        return 1.0
    return (THETA_CLOSE - P) / (THETA_CLOSE - THETA_OPEN)


def run_assimilation(x, sigma, gated):
    """Online slow-state. Returns baseline, score, alpha traces."""
    n = len(x)
    b = np.zeros(n)      # slow-state baseline
    score = np.zeros(n)  # anomaly score = x - b (pre-update)
    alpha = np.ones(n)
    P = 0.0
    state = x[0]
    for t in range(n):
        score[t] = x[t] - state          # score against current baseline
        if gated:
            e = min(max(abs(score[t]) / sigma - B_DEAD, 0.0), C_CAP)
            P = GAMMA * P + e
            a = alpha_from_persistence(P)
        else:
            a = 1.0
        alpha[t] = a
        state = state + K_GAIN * a * (x[t] - state)   # gated assimilation
        b[t] = state
    return b, score, alpha


def inject_step(x, start, length, mag_sigma, sigma):
    y = x.copy()
    end = min(start + length, len(x))
    y[start:end] = y[start:end] + mag_sigma * sigma
    return y, start, end


def contamination_latency(score, sigma, start, end):
    """Hours after injection until |score| falls below detection threshold
    (i.e. the anomaly has been absorbed into the baseline). np.inf = never."""
    seg = np.abs(score[start:end]) / sigma
    below = np.where(seg < DET_THRESH)[0]
    return int(below[0]) if len(below) else np.inf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--causal", default=str(CAUSAL))
    ap.add_argument("--signal", default="env_temperature")
    ap.add_argument("--mag", type=float, default=3.0, help="step magnitude (sigma)")
    ap.add_argument("--length", type=int, default=72, help="step duration (hours)")
    ap.add_argument("--smooth_alpha", type=float, default=0.15,
                    help="causal EWMA rate for slow-band extraction (past-only)")
    ap.add_argument("--outdir", default=str(HERE))
    args = ap.parse_args()

    df = pd.read_csv(args.causal)
    col = f"{args.signal}__resid_causal"
    raw = pd.Series(df[col]).interpolate(limit_direction="forward").fillna(0).to_numpy(float)

    # Slow-state assimilation operates on the SLOW BAND (report S3.2), not the raw
    # fast residual. Extract the slow band with a strictly-causal EWMA (past-only).
    sb_alpha = args.smooth_alpha
    slow = np.zeros_like(raw)
    s = raw[0]
    for i in range(len(raw)):
        s = (1 - sb_alpha) * s + sb_alpha * raw[i]
        slow[i] = s

    # pick a clean, well-populated window; sigma from a robust scale of it
    x = slow[200:200 + 600]
    sigma = 1.4826 * np.median(np.abs(x - np.median(x)))
    sigma = sigma if sigma > 1e-6 else np.std(x)

    start = 250
    xin, s0, s1 = inject_step(x, start, args.length, args.mag, sigma)

    b_u, sc_u, al_u = run_assimilation(xin, sigma, gated=False)
    b_g, sc_g, al_g = run_assimilation(xin, sigma, gated=True)

    lat_u = contamination_latency(sc_u, sigma, s0, s1)
    lat_g = contamination_latency(sc_g, sigma, s0, s1)

    trace = pd.DataFrame({
        "t": np.arange(len(x)), "signal_injected": xin,
        "baseline_ungated": b_u, "score_ungated": sc_u,
        "baseline_gated": b_g, "score_gated": sc_g, "alpha_gated": al_g,
    })
    trace.to_csv(Path(args.outdir) / "gate_ablation_trace.csv", index=False)

    def retention(score):  # fraction of the sustained event still detected
        seg = np.abs(score[s0:s1]) / sigma
        return float((seg >= DET_THRESH).mean())

    summary = pd.DataFrame([{
        "signal": args.signal, "mag_sigma": args.mag, "length_hr": args.length,
        "sigma": round(sigma, 4), "gain_k": K_GAIN,
        "ungated_contam_latency_hr": lat_u, "gated_contam_latency_hr": lat_g,
        "ungated_detection_retention": round(retention(sc_u), 3),
        "gated_detection_retention": round(retention(sc_g), 3),
    }])
    summary.to_csv(Path(args.outdir) / "gate_ablation_summary.csv", index=False)

    # ---- figure --------------------------------------------------------------
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    ax[0].plot(xin, color="0.5", lw=0.8, label="injected signal")
    ax[0].plot(b_u, color="tab:red", lw=1.6, label="ungated baseline (absorbs)")
    ax[0].plot(b_g, color="tab:green", lw=1.6, label="gated baseline (holds)")
    ax[0].axvspan(s0, s1, color="orange", alpha=0.12, label="sustained departure")
    ax[0].set_ylabel("signal / baseline"); ax[0].legend(fontsize=8, loc="upper left")
    ax[0].set_title(f"Slow-state assimilation under a sustained {args.mag}sigma / {args.length}h departure "
                    f"({args.signal})")

    ax[1].axhline(DET_THRESH, color="k", ls="--", lw=0.8, label=f"detect thresh {DET_THRESH}sigma")
    ax[1].axhline(-DET_THRESH, color="k", ls="--", lw=0.8)
    ax[1].plot(sc_u / sigma, color="tab:red", lw=1.2, label="score ungated (decays -> contaminated)")
    ax[1].plot(sc_g / sigma, color="tab:green", lw=1.2, label="score gated (stays detected)")
    ax[1].axvspan(s0, s1, color="orange", alpha=0.12)
    ax[1].set_ylabel("anomaly score (sigma)"); ax[1].legend(fontsize=8, loc="upper left")

    ax[2].plot(al_g, color="tab:green", lw=1.4, label="alpha_t (gated)")
    ax[2].axvspan(s0, s1, color="orange", alpha=0.12)
    ax[2].set_ylabel("trust alpha_t"); ax[2].set_xlabel("hours"); ax[2].set_ylim(-0.05, 1.05)
    ax[2].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(Path(args.outdir) / "fig_gate_ablation.png", dpi=140)

    print("=== gate ablation ===")
    print(f"sigma={sigma:.3f}  injection: {args.mag}sigma for {args.length}h at t={start}")
    print(f"ungated contamination latency: {lat_u} h   retention {retention(sc_u):.2f}")
    print(f"gated   contamination latency: {lat_g} h   retention {retention(sc_g):.2f}")


if __name__ == "__main__":
    main()
