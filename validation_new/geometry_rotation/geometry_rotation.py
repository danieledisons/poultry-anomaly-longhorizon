#!/usr/bin/env python3
"""
Visualising the core finding: the correlation GEOMETRY rotates over the growth
cycle, so no FIXED representation gives a stationary null. Shown for THREE views:
  * audio   : leading-K subspace of the rich-audio causal slow band (residual)
  * video   : leading-K subspace of the rich-video causal slow band (residual)
  * cross-modal coupling: the audio-video joint axis in the slow LEVEL (where the
    coupling lives). Its rotation is why the fusion advantage (#3) is data-dependent.

Two panels:
  (1) Rotation drift vs cycle week: principal angle of each block's geometry
      relative to the week-0 geometry. Rising = rotating (fixed basis goes stale).
  (2) The cross-modal audio-video coupling ellipse per block, in a FIXED week-0
      frame. Stationary geometry -> coincident ellipses; instead they rotate.

Descriptive (each block uses only its own data); evidence for the modelling
requirement, not a detector.

Usage: python validation_new/geometry_rotation/geometry_rotation.py --spine <csv> --room 2
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
from matplotlib.patches import Ellipse

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
spec = importlib.util.spec_from_file_location("ra", HERE.parent / "gate_ablation" / "gate_ablation_richaudio.py")
ra = importlib.util.module_from_spec(spec); spec.loader.exec_module(ra)
cpl_spec = importlib.util.spec_from_file_location("cpl", HERE.parent / "cross_modal_coupling" / "coupling_break.py")

K = 3
BLOCK_DAYS = 14


def leading_subspace(X, k):
    C = np.cov((X - X.mean(0)).T)
    return np.linalg.eigh(C)[1][:, ::-1][:, :k]


def principal_angle_deg(Q1, Q2):
    s = np.clip(np.linalg.svd(Q1.T @ Q2, compute_uv=False), -1, 1)
    return np.degrees(np.arccos(s.min()))


def causal_level_component(df, feats, ref_end):
    """Leading component of a modality's causal slow LEVEL (trend-carrying, not
    detrended) -- coupling lives here. Standardized on the past-only reference."""
    X = df[feats].ffill().to_numpy(float)
    med = np.median(X[:ref_end], 0); mad = 1.4826 * np.median(np.abs(X[:ref_end] - med), 0)
    Xs = (X - med) / np.where(mad < 1e-9, 1.0, mad)
    Lv = pd.DataFrame(Xs).rolling(25, min_periods=1).median().to_numpy()
    w = np.linalg.eigh(np.cov((Lv[:ref_end] - Lv[:ref_end].mean(0)).T))[1][:, -1]
    if w.sum() < 0:
        w = -w
    s = Lv @ w
    return (s - np.median(s[:ref_end])) / (1.4826 * np.median(np.abs(s[:ref_end] - np.median(s[:ref_end]))) + 1e-9)


def block_edges(day):
    return np.arange(0, day.max() + BLOCK_DAYS, BLOCK_DAYS)


def subspace_drift(df, feats, day):
    Z = ra.causal_slow_band(df, feats)
    med = np.median(Z, 0); mad = 1.4826 * np.median(np.abs(Z - med), 0)
    Zs = (Z - med) / np.where(mad < 1e-9, 1.0, mad)
    edges = block_edges(day); rows = []; ell = []; Q0 = None; F = None
    for i in range(len(edges) - 1):
        m = (day >= edges[i]) & (day < edges[i + 1])
        if m.sum() < 60:
            continue
        Qb = leading_subspace(Zs[m], K)
        if Q0 is None:
            Q0 = Qb; F = Qb[:, :2]                      # fixed week-0 2-D frame
        rows.append((edges[i] / 7, principal_angle_deg(Q0, Qb)))
        P = (Zs[m] - Zs[m].mean(0)) @ F                 # project onto fixed early frame
        C = np.cov(P.T); ev, evec = np.linalg.eigh(C)
        ell.append((edges[i] / 7, np.zeros(2), ev, evec))
    return rows, ell


def make_fig(name, drift, ellipses, color, title, outpath):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    if drift:
        xs, ys = zip(*drift); ax[0].plot(xs, ys, "-o", color=color, lw=2)
    ax[0].axhline(90, color="0.6", ls=":", lw=1); ax[0].text(0.1, 91, "orthogonal (90 deg)", fontsize=7, color="0.4")
    ax[0].set_xlabel("cycle week"); ax[0].set_ylabel("rotation vs week-0 geometry (deg)")
    ax[0].set_title(f"{title}: rotation over the cycle"); ax[0].set_ylim(0, 96)
    palette = plt.cm.tab10(np.linspace(0, 1, 10))
    for j, (wk, mean, ev, evec) in enumerate(ellipses):
        th = np.linspace(0, 2*np.pi, 100)
        pts = (mean[:, None] + evec @ (2*np.sqrt(np.clip(ev, 0, None))[:, None] *
               np.array([np.cos(th), np.sin(th)]))).T
        ax[1].plot(pts[:, 0], pts[:, 1], color=palette[j % 10], lw=1.8, label=f"wk {wk:.0f}")
    ax[1].axhline(0, color="k", lw=0.4); ax[1].axvline(0, color="k", lw=0.4)
    ax[1].set_xlabel("week-0 frame axis 1"); ax[1].set_ylabel("week-0 frame axis 2")
    ax[1].set_title("ellipse in a FIXED week-0 frame"); ax[1].legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(outpath, dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spine", default=str(DATA / "spine_room2_rich.csv"))
    ap.add_argument("--room", default="2")
    a = ap.parse_args(); R = a.room

    df0 = pd.read_csv(a.spine).sort_values("time").reset_index(drop=True)
    mel = [c for c in df0.columns if c.startswith("aud_mel") and c.endswith("_mean")]
    vfe = ([c for c in df0.columns if c.startswith("vid_flowhist")] +
           [c for c in df0.columns if c.startswith("vid_gridmean")] +
           [c for c in df0.columns if c.startswith("vid_gridstd")])

    # per-modality subspace drift + ellipses (own coverage)
    da = df0[df0[mel].notna().all(axis=1)].reset_index(drop=True)
    ta = pd.to_datetime(da["time"]); day_a = (ta - ta.min()).dt.total_seconds().to_numpy() / 86400
    audio_drift, audio_ell = subspace_drift(da, mel, day_a)

    dv = df0[df0[vfe].notna().all(axis=1)].reset_index(drop=True)
    tv = pd.to_datetime(dv["time"]); day_v = (tv - tv.min()).dt.total_seconds().to_numpy() / 86400
    video_drift, video_ell = subspace_drift(dv, vfe, day_v)

    # cross-modal coupling drift + ellipses (overlap, slow level), in week-0 frame
    dc = df0[df0[mel + vfe].notna().all(axis=1)].reset_index(drop=True)
    tc = pd.to_datetime(dc["time"]); day_c = (tc - tc.min()).dt.total_seconds().to_numpy() / 86400
    ref_end = int(0.35 * len(dc))
    A = causal_level_component(dc, mel, ref_end)
    V = causal_level_component(dc, vfe, ref_end)
    edges = block_edges(day_c)
    cross_drift = []; cross_ell = []; axis0 = None; Fr = None
    for i in range(len(edges) - 1):
        m = (day_c >= edges[i]) & (day_c < edges[i + 1])
        if m.sum() < 40:
            continue
        P = np.column_stack([A[m], V[m]])
        C = np.cov((P - P.mean(0)).T); ev, evec = np.linalg.eigh(C)
        axis = evec[:, -1]
        if axis0 is None:
            axis0 = axis; Fr = np.column_stack([axis0, np.array([-axis0[1], axis0[0]])])
        ang = np.degrees(np.arccos(np.clip(abs(axis0 @ axis), -1, 1)))
        cross_drift.append((edges[i] / 7, ang))
        Cf = Fr.T @ C @ Fr; evf, evecf = np.linalg.eigh(Cf)   # covariance in week-0 frame
        cross_ell.append((edges[i] / 7, np.zeros(2), evf, evecf))

    # ---- three separate figures (kept in the previous style) ----
    make_fig("audio", audio_drift, audio_ell, "#1f77b4",
             f"Room {R} AUDIO (64-band)", HERE / "figs" / f"fig_geometry_audio_room{R}.png")
    make_fig("video", video_drift, video_ell, "#d62728",
             f"Room {R} VIDEO (64-feat)", HERE / "figs" / f"fig_geometry_video_room{R}.png")
    make_fig("cross", cross_drift, cross_ell, "#2ca02c",
             f"Room {R} CROSS-MODAL coupling", HERE / "figs" / f"fig_geometry_crossmodal_room{R}.png")

    for tag, d in [("audio", audio_drift), ("video", video_drift), ("crossmodal", cross_drift)]:
        pd.DataFrame(d, columns=["cycle_week", "rotation_deg"]).to_csv(
            HERE / "csv" / f"geometry_drift_{tag}_room{R}.csv", index=False)

    print(f"=== Room {R} geometry rotation ===")
    print("audio:", [(round(w,1), round(d,0)) for w, d in audio_drift])
    print("video:", [(round(w,1), round(d,0)) for w, d in video_drift])
    print("cross-modal:", [(round(w,1), round(d,0)) for w, d in cross_drift])


if __name__ == "__main__":
    main()
