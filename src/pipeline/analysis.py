#!/usr/bin/env python3
"""End-to-end Room 2 run: merge the modalities, decompose, run the gate, do the synthetic-injection test, and write the figures.

Run: python run_pipeline.py
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import FEATURES_DIR, RESULTS_DIR
from src.models.alpha_gate import slow_fast_decompose, AlphaGate, robust_scale

# ----------------------------------------------------------------------
FILES = {
    "v2": "hourly_features_all_folders_room_2.csv",
    "v6": "hourly_features_all_folders_room_6.csv",
    "a2": [
        "audio_features_hourly_Room2_2025-06.csv",
        "audio_features_hourly_Room2_2025-07.csv",
        "audio_features_hourly_Room2_2025-08.csv",
    ],
    "a6": "audio_features_hourly_Room6_2025-07.csv",
    "env2": "env_features_Room2.csv",
}

# LOCKED gate parameters — reuse UNCHANGED across rooms (do not retune per room)
GATE_PARAMS = dict(
    deadband=1.0, decay=0.85, close_threshold=6.0, open_threshold=2.0, per_step_cap=2.0
)

AUDIO_KEEP = [
    "time", "rms_db_mean", "centroid_hz_mean", "flux_mean", "zcr_mean",
    "flatness_mean", "voc_frac_mean", "voc_activity_frac", "transient_rate",
    "ste_db_mean", "n_frames", "gap_hours_since_prev",
]
VID_KEEP = [
    "flow_mean_avg", "flow_mean_std", "flow_var_avg", "occupancy_avg",
    "occupancy_std", "occupancy_p90", "n_frames_lit", "dark_fraction", "file_count",
]
ENV_KEEP = [
    "date", "temp_day_mean_c", "temp_am_min_c", "temp_am_max_c", "temp_pm_c",
    "temp_am_range_c", "temp_am_pm_swing_c", "rh_day_mean_pct", "rh_am_pm_change_pct",
    "temp_rate_c_per_day", "temp_roll_mean_c", "temp_roll_slope_c_per_day", "day_index",
]


def p(dd, name):
    return os.path.join(dd, name)


# ----------------------------------------------------------------------
def merge_room2(dd):
    au = pd.concat([pd.read_csv(p(dd, f)) for f in FILES["a2"]], ignore_index=True)
    au["time"] = pd.to_datetime(au["time"])
    before = len(au)
    au = au.sort_values("time").drop_duplicates("time", keep="first").reset_index(drop=True)
    print(f"[merge] audio concat={before} -> dedup={len(au)} (removed {before-len(au)} dup ts)")
    au = au[AUDIO_KEEP].rename(columns={c: "aud_" + c for c in AUDIO_KEEP if c != "time"})

    vid = pd.read_csv(p(dd, FILES["v2"]))
    vid["hour"] = pd.to_datetime(vid["hour"])
    vid = vid.rename(columns={"hour": "time"})
    vid["vid_row_present"] = True
    vid = vid[["time", "vid_row_present"] + VID_KEEP].rename(
        columns={c: "vid_" + c for c in VID_KEEP}
    )

    env = pd.read_csv(p(dd, FILES["env2"]))
    env["date"] = pd.to_datetime(env["date"])
    env = env[ENV_KEEP].rename(columns={c: "env_" + c for c in ENV_KEEP if c != "date"})

    start = min(vid["time"].min(), au["time"].min())
    end = max(vid["time"].max(), au["time"].max())
    spine = pd.DataFrame({"time": pd.date_range(start.floor("h"), end.ceil("h"), freq="h")})
    m = spine.merge(vid, on="time", how="left").merge(au, on="time", how="left")

    m["date"] = m["time"].dt.floor("D")
    env_daily = (
        env.set_index("date")
        .reindex(pd.date_range(env["date"].min(), env["date"].max(), freq="D"))
        .ffill()
    )
    env_daily.index.name = "date"
    m = m.merge(env_daily.reset_index(), on="date", how="left").drop(columns="date")

    m["vid_row_present"] = m["vid_row_present"].notna()
    m["has_video_lit"] = m["vid_flow_mean_avg"].notna()
    m["has_audio"] = m["aud_rms_db_mean"].notna()
    m["has_env"] = m["env_temp_day_mean_c"].notna()

    vlo, vhi = vid["time"].min(), vid["time"].max()
    w = m[(m["time"] >= vlo) & (m["time"] <= vhi)]
    print(f"[merge] video window {vlo}..{vhi}  slots={len(w)}")
    print(f"[merge]   video-lit={w['has_video_lit'].sum()}  audio={w['has_audio'].sum()}  "
          f"env={w['has_env'].sum()}  ALL3={(w['has_video_lit']&w['has_audio']&w['has_env']).sum()}")
    return m


def room6_check(dd):
    v6 = pd.read_csv(p(dd, FILES["v6"])); v6["hour"] = pd.to_datetime(v6["hour"])
    a6 = pd.read_csv(p(dd, FILES["a6"])); a6["time"] = pd.to_datetime(a6["time"])
    v6l = v6[v6["flow_mean_avg"].notna()][["hour"]].rename(columns={"hour": "time"})
    ov = v6l.merge(a6[["time"]], on="time", how="inner")
    print(f"[room6] video rows={len(v6)} lit={v6['flow_mean_avg'].notna().sum()}  "
          f"audio rows={len(a6)}  video-lit&audio overlap={len(ov)}")
    if len(ov) < 100:
        print("[room6] WARNING: audio too sparse for a cross-modal claim; use video-residual pilot only.")


def run_gate(residual, scale):
    return AlphaGate(scale=scale, **GATE_PARAMS).run(residual)


def injection_test(residual, scale, out_dir, seed=42, n=300):
    rng = np.random.default_rng(seed)
    pool = residual[~np.isnan(residual)]
    quiet = pool[np.abs(pool - np.median(pool)) < 3 * scale]

    def mk(nn=240):
        return rng.choice(quiet, size=nn, replace=True)

    def spike(b, t0, m, dur=1):
        r = b.copy(); r[t0:t0 + dur] += m * scale; return r

    def sust(b, t0, m, dur=36):
        r = b.copy(); r[t0:t0 + dur] += m * scale; return r

    T0 = 80
    rows = []
    for m in [4, 6, 8, 10, 15, 20, 30]:
        c = sum(run_gate(spike(mk(), T0, m), scale)["closed"][T0:T0 + 12].any() for _ in range(n)) / n
        rows.append(("spike", m, 1, c, np.nan))
    for m in [1.5, 2, 2.5, 3, 4, 5]:
        det, lat = 0, []
        for _ in range(n):
            cl = run_gate(sust(mk(), T0, m), scale)["closed"][T0:T0 + 36]
            if cl.any():
                det += 1; lat.append(int(np.argmax(cl)))
        rows.append(("sustained", m, 36, det / n, np.mean(lat) if lat else np.nan))
    res = pd.DataFrame(rows, columns=["type", "mag_scales", "dur_h", "rate", "mean_latency_h"])
    res.to_csv(p(out_dir, "injection_results.csv"), index=False)
    print("[inject] spike false-closure (max 30 sigma): "
          f"{res[res.type=='spike']['rate'].max()*100:.1f}%  | "
          f"sustained detect @2.5sigma: {res[(res.type=='sustained')&(res.mag_scales==2.5)]['rate'].values[0]*100:.0f}%")
    return res, quiet


def make_figures(m, res, residual, scale, quiet, out_dir):
    rng = np.random.default_rng(7)
    base = rng.choice(quiet, size=200, replace=True)
    A = base.copy(); A[80] += 12 * scale
    B = base.copy(); B[80:116] += 3 * scale
    oA, oB = run_gate(A, scale), run_gate(B, scale)

    fig, ax = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    for col, (r, o, title, sh) in enumerate([
        (A, oA, "A. Brief spike (12 sigma, 1h) - gate stays OPEN", (80, 81)),
        (B, oB, "B. Sustained departure (3 sigma, 36h) - gate CLOSES", (80, 116))]):
        a0, a1 = ax[0, col], ax[1, col]
        a0.plot(r / scale, color="#1f4e79", lw=0.9); a0.axhline(0, color="gray", lw=0.5)
        a0.axvspan(*sh, color="#e07b39", alpha=0.18)
        a0.set_title(title, fontsize=11, fontweight="bold"); a0.set_ylabel("residual (sigma)")
        a1.plot(o["alpha"], color="#2e7d32", lw=1.4, label="alpha_t")
        a1.fill_between(range(len(o["alpha"])), 0, o["closed"].astype(float),
                        color="#c62828", alpha=0.25, label="gate closed")
        a1.axvspan(*sh, color="#e07b39", alpha=0.18)
        a1.set_ylim(-0.05, 1.08); a1.set_ylabel("alpha_t"); a1.set_xlabel("hour")
        a1.legend(loc="center right", fontsize=8)
    fig.suptitle("alpha_t gate: responds to PERSISTENCE, not magnitude (Room 2 video residual)",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(p(out_dir, "fig_gate_traces.png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    sp, su = res[res.type == "spike"], res[res.type == "sustained"]
    fig2, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
    a.plot(sp.mag_scales, sp.rate * 100, "o-", color="#c62828"); a.set_ylim(-2, 100)
    a.set_title("False-closure rate - brief spikes"); a.set_xlabel("spike magnitude (sigma)")
    a.set_ylabel("% trials gate closed"); a.grid(alpha=.3)
    b.plot(su.mag_scales, su.rate * 100, "s-", color="#2e7d32"); b.set_ylim(0, 105)
    b.set_xlabel("sustained magnitude (sigma)"); b.set_ylabel("% detected", color="#2e7d32")
    bl = b.twinx(); bl.plot(su.mag_scales, su.mean_latency_h, "^--", color="#1565c0")
    bl.set_ylabel("closure latency (h)", color="#1565c0")
    b.set_title("Sustained departures - detection & latency"); b.grid(alpha=.3)
    fig2.tight_layout(); fig2.savefig(p(out_dir, "fig_injection_metrics.png"), dpi=600, bbox_inches="tight")
    plt.close(fig2)

    dec, _ = slow_fast_decompose(m, "time", "vid_flow_mean_avg")
    dec.to_csv(p(out_dir, "room2_video_decomp.csv"), index=False)
    d = dec.dropna(subset=["vid_flow_mean_avg"])
    fig3, ax3 = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax3[0].plot(d["time"], d["vid_flow_mean_avg"], ".", ms=2, color="#9e9e9e", label="raw flow (lit)")
    ax3[0].plot(d["time"], d["trend"], color="#1f4e79", lw=2, label="slow trend")
    ax3[0].plot(d["time"], d["slow"], color="#e07b39", lw=0.8, alpha=.7, label="slow+diurnal")
    ax3[0].set_title("Room 2 video activity - slow/fast decomposition", fontweight="bold")
    ax3[0].legend(fontsize=8); ax3[0].set_ylabel("flow_mean")
    ax3[1].plot(d["time"], d["fast_residual"], color="#455a64", lw=0.6); ax3[1].axhline(0, color="r", lw=.5)
    ax3[1].set_title("Fast-band residual (gate input)"); ax3[1].set_ylabel("residual")
    fig3.tight_layout(); fig3.savefig(p(out_dir, "fig_decomposition.png"), dpi=600, bbox_inches="tight")
    plt.close(fig3)
    print("[figs] wrote fig_gate_traces.png, fig_injection_metrics.png, fig_decomposition.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(FEATURES_DIR),
                    help="Directory holding the hourly/audio/env feature CSVs "
                         "(default: FEATURES_DIR from .env).")
    ap.add_argument("--out-dir", default=str(RESULTS_DIR),
                    help="Where to write merged CSV, metrics and figures "
                         "(default: RESULTS_DIR from .env).")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    m = merge_room2(args.data_dir)
    m.to_csv(p(args.out_dir, "room2_merged_hourly.csv"), index=False)
    room6_check(args.data_dir)

    dec, _ = slow_fast_decompose(m, "time", "vid_flow_mean_avg")
    residual = dec["fast_residual"].values
    scale = robust_scale(residual)
    base = run_gate(residual, scale)
    print(f"[gate] baseline closed {base['closed'].sum()}/{len(residual)} "
          f"({base['closed'].mean()*100:.2f}%)  robust_scale={scale:.4f}")

    res, quiet = injection_test(residual, scale, args.out_dir)
    make_figures(m, res, residual, scale, quiet, args.out_dir)
    print("[done] all outputs in", args.out_dir)


if __name__ == "__main__":
    main()