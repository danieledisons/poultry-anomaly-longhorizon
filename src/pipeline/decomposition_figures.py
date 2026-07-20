#!/usr/bin/env python3
"""Combined slow/fast decomposition figure across modalities, plus the diurnal profiles.

Run: python src/pipeline/decomposition_figures.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import RESULTS_DIR
from src.models.alpha_gate import slow_fast_decompose

OUT = RESULTS_DIR / "experiment"

PALETTE = {
    "audio": "#8e24aa", "video": "#1565c0", "env": "#ef6c00", "fused": "#2e7d32",
}
LABEL = {
    "audio": "Audio — mean log-mel energy",
    "video": "Video — optical-flow magnitude",
    "env":   "Environment — daily mean temp (°C)",
    "fused": "Fused — multimodal activity index",
}


def composites(spine_path):
    m = pd.read_csv(spine_path, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    mel = [c for c in m.columns if c.startswith("aud_mel") and c.endswith("_mean")]
    df = pd.DataFrame({"time": m["time"]})
    df["audio"] = m[mel].mean(axis=1) if mel else np.nan
    df["video"] = m["vid_flow_mean_avg"] if "vid_flow_mean_avg" in m else np.nan
    tcol = next((c for c in ["env_temp_day_mean_c", "env_temp_roll_mean_c"] if c in m), None)
    df["env"] = m[tcol] if tcol else np.nan

    # fused = z-scored mean of available composites per hour
    def z(s):
        s = s.astype(float)
        return (s - s.mean()) / (s.std() + 1e-9)
    df["fused"] = pd.concat([z(df["audio"]), z(df["video"]), z(df["env"])], axis=1).mean(axis=1)
    return df


def decompose_all(df, slow_days=7):
    out = {}
    for mod in ["audio", "video", "env", "fused"]:
        sub = df[["time", mod]].dropna()
        if len(sub) < 48:
            continue
        dec, diurnal = slow_fast_decompose(sub, "time", mod, slow_window_days=slow_days)
        out[mod] = (dec, diurnal)
    return out


def fig_decomposition(dec_all):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    mods = [m for m in ["audio", "video", "env", "fused"] if m in dec_all]
    fig, axes = plt.subplots(len(mods), 2, figsize=(14, 2.7 * len(mods)),
                             gridspec_kw={"width_ratios": [2.4, 1]})
    if len(mods) == 1:
        axes = axes[None, :]

    for i, mod in enumerate(mods):
        dec, _ = dec_all[mod]
        c = PALETTE[mod]
        t = pd.to_datetime(dec["time"])
        # left: raw + slow(trend+diurnal) + trend
        axL = axes[i, 0]
        axL.plot(t, dec[mod], ".", ms=2.2, color="#b0b0b0", label="raw (hourly)")
        axL.plot(t, dec["slow"], color=c, lw=1.0, alpha=.55, label="slow (trend+diurnal)")
        axL.plot(t, dec["trend"], color=c, lw=2.4, label="growth trend")
        axL.set_ylabel(LABEL[mod].split(" — ")[0], fontweight="bold", color=c)
        axL.legend(fontsize=7.5, loc="upper left", framealpha=.9)
        axL.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        if i == 0:
            axL.set_title("Slow band: growth trend + diurnal rhythm", fontweight="bold")
        # right: fast residual
        axR = axes[i, 1]
        axR.axhline(0, color="k", lw=.6)
        axR.plot(t, dec["fast_residual"], color=c, lw=.5)
        axR.fill_between(t, 0, dec["fast_residual"], color=c, alpha=.25)
        axR.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        if i == 0:
            axR.set_title("Fast band: residual (where anomalies live)", fontweight="bold")
        for ax in (axL, axR):
            for lab in ax.get_xticklabels():
                lab.set_rotation(0); lab.set_fontsize(8)
    axes[-1, 0].set_xlabel("date"); axes[-1, 1].set_xlabel("date")
    fig.suptitle("Slow/fast decomposition across modalities (Room 2)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(OUT / "fig_decomposition.png", dpi=600)
    plt.close(fig)


def fig_diurnal(dec_all):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for mod, (_, diurnal) in dec_all.items():
        d = pd.Series(diurnal.values, index=range(len(diurnal)))
        d = (d - d.mean()) / (d.std() + 1e-9)   # z-score so shapes overlay
        ax.plot(d.index, d.values, "o-", color=PALETTE[mod], label=LABEL[mod].split(" — ")[0], lw=1.6)
    ax.axhline(0, color="gray", lw=.6, ls="--")
    ax.set_xlabel("hour of day"); ax.set_ylabel("diurnal deviation (z)")
    ax.set_xticks(range(0, 24, 2)); ax.legend()
    ax.set_title("Diurnal rhythm by modality (normalized)", fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "fig_diurnal.png", dpi=600); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spine", default=str(RESULTS_DIR / "spine_room2_rich.csv"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    df = composites(args.spine)
    dec_all = decompose_all(df)
    fig_decomposition(dec_all)
    fig_diurnal(dec_all)
    print(f"Decomposed modalities: {list(dec_all)}")
    print(f"Wrote {OUT/'fig_decomposition.png'} and {OUT/'fig_diurnal.png'}")


if __name__ == "__main__":
    main()
