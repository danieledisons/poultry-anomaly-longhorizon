#!/usr/bin/env python3
"""
#1 (video): joint Mahalanobis slow-state gate ablation on RICH VIDEO.

Same mechanism as gate_ablation_richaudio.py, applied to the rich optical-flow
video features, to check whether the gate generalizes ACROSS MODALITIES (it may
work for audio yet fail for video). We reuse the audio script's causal slow-band,
Ledoit-Wolf shrinkage, and gated-assimilation functions unchanged.

Video is captured only in lit daytime hours, so we operate on the sub-sequence of
fully-populated video rows (sorted by time). Injection is a sustained, coherent
ACTIVITY departure across a block of spatial grid cells (welfare-plausible: birds
clustering in / avoiding a region), each cell scaled to its own causal MAD.

Scientific-integrity commitments identical to the audio script: strictly causal
reference/covariance, physically-specified injection whose Mahalanobis magnitude
is measured (not tuned), data-driven thresholds, honest false-closure reporting.

Usage:
  python validation_new/gate_ablation/gate_ablation_richvideo.py --spine <csv> --room 2
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

# reuse the audio module's shared functions (identical mechanism)
spec = importlib.util.spec_from_file_location("ra", HERE / "gate_ablation_richaudio.py")
ra = importlib.util.module_from_spec(spec); spec.loader.exec_module(ra)

TREND_WIN = ra.TREND_WIN
# injection: coherent activity shift across a block of spatial grid cells
INJ_CELLS = list(range(0, 8))          # grid-mean cells 00-07 (a barn region)
INJ_PER_CELL_SIGMA = 4.0               # illustrative trace magnitude
INJ_SWEEP = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
INJ_LENGTH = 72
REF_FRAC = 0.35
PLOT_YMAX = 60.0          # shared Mahalanobis y-limit so Room 2 & Room 6 are comparable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spine", default=str(DATA / "spine_room2_rich.csv"))
    ap.add_argument("--room", default="2")
    ap.add_argument("--outdir", default=str(HERE))
    args = ap.parse_args()

    df = pd.read_csv(args.spine).sort_values("time").reset_index(drop=True)
    fh = [c for c in df.columns if c.startswith("vid_flowhist")]
    gm = [c for c in df.columns if c.startswith("vid_gridmean")]
    gs = [c for c in df.columns if c.startswith("vid_gridstd")]
    feats = fh + gm + gs                                    # 64 rich video features
    # operate on the fully-populated (lit) video sub-sequence
    df = df[df[feats].notna().all(axis=1)].reset_index(drop=True)
    n = len(df)

    Z = ra.causal_slow_band(df, feats)

    ref_end = int(REF_FRAC * n)
    ref = Z[:ref_end]
    mu0 = ref[:TREND_WIN].mean(0)
    Sigma, lam = ra.ledoit_wolf(ref)
    Sinv = np.linalg.inv(Sigma)

    b_dead = 1.0; theta_close = 6.0; theta_open = 2.0
    ref_op, _ = ra.run(ref, mu0, Sinv, 0.0, 1.0, 0.0, theta_close, theta_open, gated=False)
    ref_op = ref_op[TREND_WIN:]
    med = np.median(ref_op)
    sigma = 1.4826 * np.median(np.abs(ref_op - med)); sigma = sigma if sigma > 1e-9 else ref_op.std()
    det_thresh = np.quantile(ref_op, 0.99)

    start = ref_end + 80
    end = min(start + INJ_LENGTH, n)
    cell_idx = [feats.index(f"vid_gridmean{c:02d}") for c in INJ_CELLS if f"vid_gridmean{c:02d}" in feats]
    cell_mad = 1.4826 * np.median(np.abs(Z[:ref_end][:, cell_idx] -
                                         np.median(Z[:ref_end][:, cell_idx], 0)), 0)

    def retention(sc):
        seg = sc[start:end]; return float((seg >= det_thresh).mean())

    def latency(sc):
        seg = sc[start:end]; below = np.where(seg < det_thresh)[0]
        return int(below[0]) if len(below) else np.inf

    def evaluate(mag):
        Zin = Z.copy(); Zin[start:end][:, cell_idx] += mag * cell_mad
        su, au = ra.run(Zin, mu0, Sinv, med, sigma, b_dead, theta_close, theta_open, gated=False)
        sg, ag = ra.run(Zin, mu0, Sinv, med, sigma, b_dead, theta_close, theta_open, gated=True)
        return su, au, sg, ag

    sweep_rows = []
    for mag in INJ_SWEEP:
        su, _, sg, _ = evaluate(mag); onset = np.median(sg[start:start + 3])
        sweep_rows.append({"mag_cell_mad": mag, "onset_joint_maha": round(onset, 2),
            "onset_centered_sigma": round((onset - med) / sigma, 2),
            "above_natural_thresh": bool(onset > det_thresh),
            "ungated_retention": round(retention(su), 3), "gated_retention": round(retention(sg), 3),
            "ungated_latency_hr": latency(su), "gated_latency_hr": latency(sg)})
    sweep = pd.DataFrame(sweep_rows)

    sc_u, al_u, sc_g, al_g = evaluate(INJ_PER_CELL_SIGMA)
    inj_maha = np.median(sc_g[start:start + 3])
    sc_ref, al_ref = ra.run(Z[:ref_end], mu0, Sinv, med, sigma, b_dead, theta_close, theta_open, gated=True)
    false_closure = float((al_ref[TREND_WIN:] < 0.5).mean())

    R = args.room
    pd.DataFrame({"time": df["time"], "maha_ungated": sc_u, "maha_gated": sc_g,
                  "alpha_gated": al_g}).to_csv(Path(args.outdir) / f"richvideo_room{R}_trace.csv", index=False)
    sweep.to_csv(Path(args.outdir) / f"richvideo_room{R}_sweep.csv", index=False)
    pd.DataFrame([{"room": R, "n_lit_hours": n, "n_features": len(feats), "shrinkage_lambda": round(lam, 4),
        "ref_hours": ref_end, "inj_cells": f"gridmean{INJ_CELLS[0]:02d}-{INJ_CELLS[-1]:02d}",
        "inj_per_cell_sigma": INJ_PER_CELL_SIGMA, "measured_inj_maha_sigma": round(inj_maha / sigma, 2),
        "det_thresh_maha": round(det_thresh, 3), "sigma_maha": round(sigma, 3),
        "ungated_contam_latency_hr": latency(sc_u), "gated_contam_latency_hr": latency(sc_g),
        "ungated_retention": round(retention(sc_u), 3), "gated_retention": round(retention(sc_g), 3),
        "clean_false_closure_rate": round(false_closure, 3)}]).to_csv(
        Path(args.outdir) / f"richvideo_room{R}_summary.csv", index=False)

    x = np.arange(n)
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax[0].axhline(det_thresh, color="k", ls="--", lw=0.8, label="detect thresh (ref 99pct)")
    ax[0].plot(x, sc_u, color="tab:red", lw=1.0, label="Mahalanobis ungated (absorbs)")
    ax[0].plot(x, sc_g, color="tab:green", lw=1.0, label="Mahalanobis gated (holds)")
    ax[0].axvspan(start, end, color="orange", alpha=0.12, label="sustained activity departure")
    ax[0].axvspan(0, ref_end, color="blue", alpha=0.05, label="causal reference window")
    # shared y-limit for cross-room comparability; annotate any off-scale onset transient
    ax[0].set_ylim(0, PLOT_YMAX)
    for i, (name, sc, col) in enumerate([("ungated", sc_u, "tab:red"), ("gated", sc_g, "tab:green")]):
        pk = sc[start:end].max(); pt = start + int(np.argmax(sc[start:end]))
        if pk > PLOT_YMAX:
            ytxt = PLOT_YMAX * (0.90 - 0.14 * i)      # stagger so the two labels don't collide
            ax[0].annotate(f"{name} onset peak {pk:.0f}", xy=(pt, PLOT_YMAX * 0.985),
                           xytext=(pt + 35, ytxt), fontsize=7, color=col,
                           arrowprops=dict(arrowstyle="->", color=col, lw=0.8))
    ax[0].set_ylabel("joint Mahalanobis"); ax[0].legend(fontsize=7, loc="upper right")
    ax[0].set_title(f"Room {R}: rich-VIDEO joint slow-state gate ablation ({len(feats)} features, causal)")
    ax[1].plot(x, al_g, color="tab:green", lw=1.2, label="alpha_t (gated)")
    ax[1].axvspan(start, end, color="orange", alpha=0.12)
    ax[1].set_ylabel("trust alpha_t"); ax[1].set_xlabel("lit video hours"); ax[1].set_ylim(-0.05, 1.05)
    ax[1].legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(Path(args.outdir) / f"fig_richvideo_room{R}.png", dpi=140)

    print(f"=== Room {R} rich-VIDEO gate ablation ===")
    print(f"features {len(feats)} | lit hours {n} | shrinkage lambda {lam:.3f} | ref {ref_end}")
    print(f"injection {INJ_PER_CELL_SIGMA} cell-MAD across gridmean {INJ_CELLS[0]}-{INJ_CELLS[-1]}"
          f" -> measured {inj_maha/sigma:.1f} sigma joint")
    print(f"ungated: latency {latency(sc_u)} h  retention {retention(sc_u):.2f}")
    print(f"gated:   latency {latency(sc_g)} h  retention {retention(sc_g):.2f}")
    print(f"clean-data false-closure rate: {false_closure:.3f}")


if __name__ == "__main__":
    main()
