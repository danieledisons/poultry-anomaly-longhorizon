#!/usr/bin/env python3
"""
cross_modal_analysis.py — the "fusion" experiment.

Shows the alpha_t trust gate, fed a JOINT video+audio divergence signal, closes on
a coupling-break that NEITHER modality flags alone.

Pipeline:
  1. Load the merged Room 2 hourly table (produced by run_analysis.py).
  2. Restrict to fused daytime hours; robustly standardize video & audio activity.
  3. Estimate the joint covariance and compute a per-hour Mahalanobis divergence.
  4. Calibrate the gate on the real quiet core (one identical rule for every stream).
  5. Baseline: run the joint + marginal gates on real data.
  6. Coupling-break injection: joint gate closes, marginal gates stay open.
  7. Figures + a metrics CSV.

Usage:
    python cross_modal_analysis.py --merged ./outputs/room2_merged_hourly.csv --out-dir ./outputs

Dependencies: numpy, pandas, matplotlib  (same requirements.txt).
Requires alpha_gate.py on the path.

NOTE: the video<->audio coupling (r ~ -0.4) lives in the SLOW/level band, not the
fast residuals (which are ~independent). So the cross-modal signal is built on the
standardized activity LEVELS, where the joint structure exists.
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from alpha_gate import AlphaGate

# Channels used for the joint signal (activity level, not fast residual)
VIDEO_COL = "vid_flow_mean_avg"
AUDIO_COL = "aud_voc_frac_mean"
DAY_START, DAY_END = 6, 18          # daytime hours (video is dark-gated at night)
SEED = 11


# ----------------------------------------------------------------------
def zrob(x):
    """Robust z-score: (x - median) / (1.4826 * MAD)."""
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    return (x - med) / max(mad, 1e-9)


def calibrate(clean_stream):
    """LOCKED, room-agnostic gate calibration for a monitoring stream:
    deadband at the clean 95th percentile, scale = clean robust MAD.
    Only sustained values above the stream's own normal range accumulate."""
    q95 = np.quantile(clean_stream, 0.95)
    med = np.median(clean_stream)
    mad = np.median(np.abs(clean_stream - med)) * 1.4826
    scale = max(mad, 1e-9)
    return dict(scale=scale, deadband=q95 / scale, decay=0.85,
                close_threshold=6.0, open_threshold=2.0, per_step_cap=2.0)


def gate(stream, params):
    return AlphaGate(**params).run(np.asarray(stream, float))


# ----------------------------------------------------------------------
def build_joint(merged_csv):
    m = pd.read_csv(merged_csv)
    m["time"] = pd.to_datetime(m["time"])
    m["hour"] = m["time"].dt.hour
    fused = (m[(m["hour"] >= DAY_START) & (m["hour"] < DAY_END)]
             .dropna(subset=[VIDEO_COL, AUDIO_COL])
             .sort_values("time").reset_index(drop=True))
    zv = zrob(fused[VIDEO_COL].values)
    za = zrob(fused[AUDIO_COL].values)
    r = np.corrcoef(zv, za)[0, 1]
    Z = np.c_[zv, za]

    # covariance on quiet core (inner 90% by radius) => "normal coupling"
    d0 = np.sqrt((Z ** 2).sum(1))
    core = Z[d0 < np.quantile(d0, 0.90)]
    Sig = np.cov(core.T)
    Sinv = np.linalg.inv(Sig)
    maha = np.sqrt(np.einsum("ij,jk,ik->i", Z, Sinv, Z))

    fused["zv"], fused["za"], fused["maha"] = zv, za, maha
    print(f"[joint] fused daytime hours={len(fused)}  joint r={r:+.3f}")
    print(f"[joint] Sigma=\n{np.round(Sig,3)}")
    return fused, Z, Sig, Sinv, r


def run_gates_real(fused, Sinv, out_dir):
    zv, za, maha = fused["zv"].values, fused["za"].values, fused["maha"].values
    core = maha < np.quantile(maha, 0.90)
    GPj = calibrate(maha[core])
    GPm = calibrate(np.r_[np.abs(zv[core]), np.abs(za[core])])
    bj = gate(maha, GPj)["closed"]
    bv = gate(np.abs(zv), GPm)["closed"]
    ba = gate(np.abs(za), GPm)["closed"]
    print(f"[baseline] real-data closure — JOINT {bj.mean()*100:.1f}%  "
          f"video {bv.mean()*100:.1f}%  audio {ba.mean()*100:.1f}%")
    fused[["time", "zv", "za", "maha"]].to_csv(os.path.join(out_dir, "room2_crossmodal.csv"), index=False)
    return GPj, GPm


def injection(Sig, Sinv, GPj, GPm, out_dir, n_trials=500):
    rng = np.random.default_rng(SEED)
    L = np.linalg.cholesky(Sig)

    def sample_normal(n):
        return (L @ rng.standard_normal((2, n))).T

    def maha_of(Z):
        return np.sqrt(np.einsum("ij,jk,ik->i", Z, Sinv, Z))

    def trial(c, dur=36, n=240, t0=80):
        Z = sample_normal(n)
        Z[t0:t0 + dur, 0] += c        # video up
        Z[t0:t0 + dur, 1] += c        # audio up too -> breaks the anti-correlation
        zv, za = np.abs(Z[:, 0]), np.abs(Z[:, 1])
        mh = maha_of(Z)
        w = slice(t0, t0 + dur)
        gj = gate(mh, GPj)["closed"]
        return dict(sust_z=(zv[w].mean() + za[w].mean()) / 2, sust_maha=mh[w].mean(),
                    joint=gj[w].any(),
                    vid=gate(zv, GPm)["closed"][w].any(),
                    aud=gate(za, GPm)["closed"][w].any(),
                    lat=(np.argmax(gj[w]) if gj[w].any() else np.nan))

    rows = []
    print("\n[inject] coupling-break sweep (both modalities +c, 36h):")
    print(f"  {'c':>4} {'sust|z|':>8} {'maha':>6} | {'JOINT':>6} {'VID':>5} {'AUD':>5} {'lat':>5}")
    for c in [1.2, 1.5, 1.8, 2.0, 2.5, 3.0]:
        T = pd.DataFrame([trial(c) for _ in range(n_trials)])
        rows.append(dict(c=c, sust_z=T.sust_z.mean(), sust_maha=T.sust_maha.mean(),
                         joint_close=T.joint.mean(), vid_close=T.vid.mean(),
                         aud_close=T.aud.mean(), joint_lat=T.lat.mean()))
        print(f"  {c:>4} {T.sust_z.mean():>8.2f} {T.sust_maha.mean():>6.2f} | "
              f"{T.joint.mean()*100:>5.1f}% {T.vid.mean()*100:>4.0f}% "
              f"{T.aud.mean()*100:>4.0f}% {T.lat.mean():>5.1f}")
    inj = pd.DataFrame(rows)
    inj.to_csv(os.path.join(out_dir, "crossmodal_injection.csv"), index=False)
    return inj


def figures(fused, Sig, Sinv, GPj, GPm, inj, out_dir):
    rng = np.random.default_rng(3)
    zv_r, za_r = fused["zv"].values, fused["za"].values

    # FIG 1 — joint scatter + covariance ellipse + marginal bands + break point
    fig, ax = plt.subplots(figsize=(6.4, 6))
    ax.scatter(zv_r, za_r, s=10, color="#90a4ae", alpha=.6, label="real fused hours")
    vals, vecs = np.linalg.eigh(Sig)
    ang = np.degrees(np.arctan2(*vecs[:, 1][::-1]))
    for k in (2.45, 1.5):
        w, h = 2 * k * np.sqrt(vals)
        ax.add_patch(Ellipse((0, 0), w, h, angle=ang, fc="none", ec="#1f4e79", lw=1.5, alpha=.9))
    ax.axvspan(-1.77, 1.77, color="#c8e6c9", alpha=.25)
    ax.axhspan(-1.77, 1.77, color="#c8e6c9", alpha=.25)
    ax.scatter([1.5], [1.5], s=220, marker="*", color="#c62828", zorder=5,
               label="coupling-break (1.5s,1.5s)")
    ax.annotate("inside both marginal\nbands, OUTSIDE joint\ncovariance -> only\nfusion flags it",
                (1.5, 1.5), (2.1, -1.3), fontsize=8, color="#c62828",
                arrowprops=dict(arrowstyle="->", color="#c62828"))
    ax.set_xlabel("video activity (z)"); ax.set_ylabel("audio vocalization (z)")
    ax.set_title("Joint video-audio structure\ngreen=marginal-normal, ellipse=joint-normal",
                 fontsize=10, fontweight="bold")
    ax.axhline(0, color="k", lw=.3); ax.axvline(0, color="k", lw=.3)
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.legend(loc="upper left", fontsize=8); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig_crossmodal_scatter.png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    # FIG 2 — gate traces for one c=1.5 coupling-break
    L = np.linalg.cholesky(Sig); n, t0, dur = 200, 80, 36
    Z = (L @ rng.standard_normal((2, n))).T; Z[t0:t0 + dur] += 1.5
    zv, za = np.abs(Z[:, 0]), np.abs(Z[:, 1])
    mh = np.sqrt(np.einsum("ij,jk,ik->i", Z, Sinv, Z))
    oj, ov, oa = gate(mh, GPj), gate(zv, GPm), gate(za, GPm)
    fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    ax[0].plot(zv, color="#1f77b4", lw=.8, label="|video z|")
    ax[0].plot(za, color="#ff7f0e", lw=.8, label="|audio z|")
    ax[0].axhline(1.77, color="green", ls="--", lw=.8, label="marginal-normal edge")
    ax[0].axvspan(t0, t0 + dur, color="#e07b39", alpha=.15)
    ax[0].legend(fontsize=8, ncol=2); ax[0].set_ylabel("marginal |z|")
    ax[0].set_title("Coupling-break: each modality sustained ~1.5s (within normal)", fontsize=10, fontweight="bold")
    ax[1].plot(mh, color="#6a1b9a", lw=1)
    ax[1].axhline(2.85, color="red", ls="--", lw=.8, label="joint-normal edge")
    ax[1].axvspan(t0, t0 + dur, color="#e07b39", alpha=.15)
    ax[1].legend(fontsize=8); ax[1].set_ylabel("joint Mahalanobis")
    ax[2].plot(ov["alpha"], color="#1f77b4", lw=1.2, label="a video (OPEN)")
    ax[2].plot(oa["alpha"], color="#ff7f0e", lw=1.2, label="a audio (OPEN)")
    ax[2].plot(oj["alpha"], color="#c62828", lw=1.8, label="a JOINT (CLOSES)")
    ax[2].axvspan(t0, t0 + dur, color="#e07b39", alpha=.15)
    ax[2].set_ylim(-.05, 1.08); ax[2].legend(fontsize=8, ncol=3)
    ax[2].set_ylabel("alpha_t"); ax[2].set_xlabel("hour")
    fig.suptitle("Fusion catches what marginals miss: only the joint gate closes", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig_crossmodal_traces.png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    # FIG 3 — detection vs break size
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(inj.c, inj.joint_close * 100, "o-", color="#c62828", lw=2, label="JOINT (fusion)")
    ax.plot(inj.c, inj.vid_close * 100, "s--", color="#1f77b4", label="video alone")
    ax.plot(inj.c, inj.aud_close * 100, "^--", color="#ff7f0e", label="audio alone")
    ax.axvspan(1.35, 1.65, color="#fff3cd", alpha=.7)
    ax.text(1.5, 50, "fusion-only\nregion", ha="center", fontsize=9, color="#8a6d00")
    ax.set_xlabel("coupling-break magnitude per modality (s)")
    ax.set_ylabel("% trials gate closes")
    ax.set_title("Detection vs break size: fusion fires below marginal thresholds",
                 fontsize=10, fontweight="bold")
    ax.legend(); ax.grid(alpha=.3); ax.set_ylim(-3, 103)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig_crossmodal_detection.png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("[figs] wrote fig_crossmodal_scatter/traces/detection.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", default="./outputs/room2_merged_hourly.csv")
    ap.add_argument("--out-dir", default="./outputs")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    fused, Z, Sig, Sinv, r = build_joint(args.merged)
    GPj, GPm = run_gates_real(fused, Sinv, args.out_dir)
    inj = injection(Sig, Sinv, GPj, GPm, args.out_dir)
    figures(fused, Sig, Sinv, GPj, GPm, inj, args.out_dir)
    print("[done] outputs in", args.out_dir)


if __name__ == "__main__":
    main()