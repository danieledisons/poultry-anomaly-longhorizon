#!/usr/bin/env python3
"""Per-modality slow/fast decomposition figures, saved one file each.

Run: python src/pipeline/decomposition_separate.py
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import RESULTS_DIR
from src.models.alpha_gate import slow_fast_decompose

OUT = RESULTS_DIR / "decomposition"
STYLE = {
    "audio":     ("#8e24aa", "Audio — mean log-mel energy"),
    "video":     ("#1565c0", "Video — optical-flow magnitude"),
    "env":       ("#ef6c00", "Environment — daily mean temperature (°C)"),
    "fused_av":  ("#2e7d32", "Fused (audio + video) — behavioural activity index"),
    "fused_all": ("#00695c", "Fused (audio + video + environment) index"),
}


def composites(spine_path):
    m = pd.read_csv(spine_path, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    mel = [c for c in m.columns if c.startswith("aud_mel") and c.endswith("_mean")]
    df = pd.DataFrame({"time": m["time"]})
    df["audio"] = m[mel].mean(axis=1) if mel else np.nan
    df["video"] = m["vid_flow_mean_avg"] if "vid_flow_mean_avg" in m else np.nan
    tcol = next((c for c in ["env_temp_day_mean_c", "env_temp_roll_mean_c"] if c in m), None)
    df["env"] = m[tcol] if tcol else np.nan

    def z(s):
        s = s.astype(float); return (s - s.mean()) / (s.std() + 1e-9)
    df["fused_av"] = pd.concat([z(df["audio"]), z(df["video"])], axis=1).mean(axis=1)
    df["fused_all"] = pd.concat([z(df["audio"]), z(df["video"]), z(df["env"])], axis=1).mean(axis=1)
    return df


def one_fig(df, mod, plt):
    import matplotlib.dates as mdates
    color, label = STYLE[mod]
    sub = df[["time", mod]].dropna()
    if len(sub) < 48:
        print(f"[skip] {mod}: too few points"); return
    dec, _ = slow_fast_decompose(sub, "time", mod, slow_window_days=7)
    t = pd.to_datetime(dec["time"])

    fig, ax = plt.subplots(1, 2, figsize=(13, 3.6), gridspec_kw={"width_ratios": [2.5, 1]})
    # left: raw + slow + trend
    ax[0].plot(t, dec[mod], ".", ms=2.4, color="#b8b8b8", label="raw (hourly)")
    ax[0].plot(t, dec["slow"], color=color, lw=1.0, alpha=.55, label="slow (trend + diurnal)")
    ax[0].plot(t, dec["trend"], color=color, lw=2.6, label="growth trend")
    ax[0].set_ylabel(label.split(" — ")[0], fontweight="bold", color=color)
    ax[0].legend(fontsize=8, loc="best", framealpha=.9)
    ax[0].set_title(f"{label}", fontsize=11, fontweight="normal", loc="left")
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    # right: fast residual
    ax[1].axhline(0, color="k", lw=.6)
    ax[1].plot(t, dec["fast_residual"], color=color, lw=.5)
    ax[1].fill_between(t, 0, dec["fast_residual"], color=color, alpha=.25)
    ax[1].set_title("Fast band: residual (where anomalies live)", fontsize=10, fontweight="normal")
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    for a in ax:
        for lab in a.get_xticklabels():
            lab.set_fontsize(8)
    fig.tight_layout()
    fig.savefig(OUT / f"fig_decomp_{mod}.png", dpi=600); plt.close(fig)
    print(f"[write] fig_decomp_{mod}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spine", default=str(RESULTS_DIR / "spine_room2_rich.csv"))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": True})
    df = composites(a.spine)
    for mod in ["audio", "video", "env", "fused_av", "fused_all"]:
        one_fig(df, mod, plt)
    print(f"\nAll per-modality decomposition figures in {OUT}")


if __name__ == "__main__":
    main()
