#!/usr/bin/env python3
"""
#3: Cross-modal coupling-break detection  (the multimodal contribution).

A coupling break is a sustained departure that keeps EACH modality inside its own
normal marginal band, yet violates the normal JOINT relationship between them
(their usual audio-video co-movement). By construction it is invisible to any
single-modality detector and detectable only by a joint detector. This is the
capability single-modality, magnitude-based detectors structurally cannot have.

Design (audio + video; env/thermal to be added later for true >2-modality fusion):
  * Each modality -> its leading causal slow-band component (dominant coherent
    signal), standardized on a PAST-ONLY reference. 2-D joint vector (a, v).
  * Learn the normal joint mean + 2x2 covariance on the causal reference.
  * Inject a SUSTAINED coupling break: move a and v together along the direction
    that breaks their normal correlation, each kept within its marginal band, so
    only the joint Mahalanobis grows.
  * Three GATED detectors compared: audio-only, video-only, joint. Detection rate
    vs break magnitude is estimated over many random injection positions (seeds),
    with Wilson confidence intervals -> a "fusion-only" detection window.
  * Also: gated vs ungated JOINT contamination on one illustrative trace.
  * Room 2 (tuned) and Room 6 (held-out, parameters unchanged).

Integrity: causal reference/covariance, physically-specified break kept inside
marginal bands (verified and reported), data-driven thresholds, repeated seeds
with CIs, honest reporting of any single-modality leakage.

Usage:
  python validation_new/cross_modal_coupling/coupling_break.py --spine <csv> --room 2
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
GA = HERE.parent / "gate_ablation"
spec = importlib.util.spec_from_file_location("ra", GA / "gate_ablation_richaudio.py")
ra = importlib.util.module_from_spec(spec); spec.loader.exec_module(ra)
TREND_WIN = ra.TREND_WIN

REF_FRAC = 0.35
INJ_LENGTH = 60
MARGIN_CAP = 2.0          # break kept within +/- this many marginal-sigma (stays "in band")
MAG_SWEEP = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]   # break magnitude (marginal-sigma per modality)
N_SEEDS = 40
DET_FRAC = 0.5            # sustained detection = score over threshold for >= this fraction of window
# gate params (shared, fixed)
K_GAIN = 0.10; GAMMA = 0.85; C_CAP = 4.0; THETA_CLOSE = 6.0; THETA_OPEN = 2.0


def leading_component(df, feats, ref_end):
    """Leading component of a modality's causal slow LEVEL band (trend-carrying),
    standardized on the past-only reference. Coupling lives in the slow LEVEL, not
    the detrended residual (report S5): the level co-moves across modalities via the
    shared growth trajectory, so we must NOT detrend here. Level = past-only trailing
    median of each standardized feature; then leading reference-PCA component."""
    X = df[feats].ffill().to_numpy(float)
    med = np.median(X[:ref_end], 0); mad = 1.4826 * np.median(np.abs(X[:ref_end] - med), 0)
    mad = np.where(mad < 1e-9, 1.0, mad)
    Xs = (X - med) / mad
    # strictly-causal level: trailing median (row t uses rows <= t)
    Lv = pd.DataFrame(Xs).rolling(TREND_WIN, center=False, min_periods=1).median().to_numpy()
    mu = Lv[:ref_end].mean(0)
    C = np.cov((Lv[:ref_end] - mu).T)
    w = np.linalg.eigh(C)[1][:, -1]
    if w.sum() < 0:
        w = -w
    s = Lv @ w
    s_ref = s[:ref_end]
    return (s - np.median(s_ref)) / (1.4826 * np.median(np.abs(s_ref - np.median(s_ref))) + 1e-9)


def gate_scores(score, sigma, med, b_dead=1.0):
    """Map a score series to alpha via persistence (score already |dev|)."""
    P = 0.0; alpha = np.ones(len(score))
    for t, sc in enumerate(score):
        e = min(max((sc - med) / sigma - b_dead, 0.0), C_CAP)
        P = GAMMA * P + e
        alpha[t] = 0.0 if P >= THETA_CLOSE else (1.0 if P <= THETA_OPEN else
                    (THETA_CLOSE - P) / (THETA_CLOSE - THETA_OPEN))
    return alpha


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = p + z * z / (2 * n); half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - half) / d, (c + half) / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spine", default=str(DATA / "spine_room2_rich.csv"))
    ap.add_argument("--room", default="2")
    ap.add_argument("--outdir", default=str(HERE))
    a = ap.parse_args(); R = a.room
    rng = np.random.default_rng(0)

    df = pd.read_csv(a.spine).sort_values("time").reset_index(drop=True)
    aud = [c for c in df.columns if c.startswith("aud_mel") and c.endswith("_mean")]
    vid = [c for c in df.columns if c.startswith("vid_gridmean")]
    df = df[df[aud + vid].notna().all(axis=1)].reset_index(drop=True)
    n = len(df); ref_end = int(REF_FRAC * n)

    A = leading_component(df, aud, ref_end)     # standardized audio summary
    V = leading_component(df, vid, ref_end)     # standardized video summary
    XY = np.column_stack([A, V])

    # normal joint on causal reference
    mu = XY[:ref_end].mean(0)
    Cov = np.cov((XY[:ref_end] - mu).T)
    Sinv = np.linalg.inv(Cov)
    r = Cov[0, 1] / np.sqrt(Cov[0, 0] * Cov[1, 1])
    # break direction breaks the normal correlation: same sign if anti-correlated
    sign = -np.sign(r) if abs(r) > 1e-3 else 1.0
    bdir = np.array([1.0, sign])                # (audio, video) move; sign breaks corr

    def maha(P):  # joint Mahalanobis of rows in P (n,2) vs mu
        d = P - mu
        return np.sqrt(np.einsum("ij,jk,ik->i", d, Sinv, d))

    # Score each point against its LOCAL causal normal (trailing pre-window mean),
    # removing shared growth drift so all three detectors are compared fairly and
    # only the coupling structure remains. Marginal sigmas and the joint covariance
    # SHAPE come from local deviations over the reference region.
    LOCW = 48                                     # trailing local-normal window (hours)
    def local_dev(series):
        m = pd.Series(series).rolling(LOCW, min_periods=LOCW // 2).mean().to_numpy()
        return series - m, m
    dA, _ = local_dev(A); dV, _ = local_dev(V)
    valid = ~np.isnan(dA) & ~np.isnan(dV)
    refm = valid.copy(); refm[ref_end:] = False
    Cloc = np.cov(np.column_stack([dA[refm], dV[refm]]).T)   # local joint shape
    Sinv = np.linalg.inv(Cloc)
    r = Cloc[0, 1] / np.sqrt(Cloc[0, 0] * Cloc[1, 1])
    sign = -np.sign(r) if abs(r) > 1e-3 else 1.0
    bdir = np.array([1.0, sign])
    ma_s = np.sqrt(Cloc[0, 0]); mv_s = np.sqrt(Cloc[1, 1])

    def dstats(s):
        s = s[~np.isnan(s)]; return np.quantile(np.abs(s), 0.99)
    ma_t = dstats(dA[refm]); mv_t = dstats(dV[refm])
    mj_ref = np.sqrt(np.einsum("ij,jk,ik->i",
              np.column_stack([dA[refm], dV[refm]]), Sinv, np.column_stack([dA[refm], dV[refm]])))
    mj_t = np.quantile(mj_ref, 0.99)

    # ---- detection-rate sweep over seeds (break scored vs pre-window local normal) ----
    rows = []
    post = np.arange(ref_end + LOCW, n - INJ_LENGTH)
    for mag in MAG_SWEEP:
        det = {"audio": 0, "video": 0, "joint": 0}; marg_ok = 0
        for _ in range(N_SEEDS):
            s0 = int(rng.choice(post)); s1 = s0 + INJ_LENGTH
            # local normal from the window BEFORE the break (causal, pre-onset)
            mA = np.nanmean(A[s0 - LOCW:s0]); mV = np.nanmean(V[s0 - LOCW:s0])
            devA = (A[s0:s1] - mA) + mag * bdir[0] * ma_s   # break in local-sigma units
            devV = (V[s0:s1] - mV) + mag * bdir[1] * mv_s
            # marginal band check: each modality stays within +/- MARGIN_CAP local sigma
            in_a = np.abs(devA).mean() <= MARGIN_CAP * ma_s
            in_v = np.abs(devV).mean() <= MARGIN_CAP * mv_s
            marg_ok += int(in_a and in_v)
            D = np.column_stack([devA, devV])
            sj = np.sqrt(np.einsum("ij,jk,ik->i", D, Sinv, D))
            for name, s, thr in [("audio", np.abs(devA), ma_t),
                                 ("video", np.abs(devV), mv_t), ("joint", sj, mj_t)]:
                det[name] += int((s >= thr).mean() >= DET_FRAC)
        row = {"mag_sigma": mag, "marginal_in_band_frac": round(marg_ok / N_SEEDS, 3)}
        for name in ["audio", "video", "joint"]:
            p = det[name] / N_SEEDS; lo, hi = wilson(det[name], N_SEEDS)
            row[f"{name}_detect"] = round(p, 3); row[f"{name}_lo"] = round(lo, 3); row[f"{name}_hi"] = round(hi, 3)
        rows.append(row)
    sweep = pd.DataFrame(rows)
    mj_m = np.median(mj_ref); mj_s = 1.4826 * np.median(np.abs(mj_ref - mj_m)) or mj_ref.std()
    sweep.to_csv(HERE / "csv" / f"coupling_room{R}_detection_sweep.csv", index=False)

    # ---- illustrative gated vs ungated JOINT contamination (one placement) ----
    s0 = int(post[len(post) // 3]); s1 = s0 + INJ_LENGTH; mag = 2.0
    # contamination on the local-deviation representation (consistent with detection)
    DA = np.nan_to_num(dA).copy(); DV = np.nan_to_num(dV).copy()
    DA[s0:s1] += mag * bdir[0] * ma_s; DV[s0:s1] += mag * bdir[1] * mv_s
    DEV = np.column_stack([DA, DV])
    # joint slow-state m assimilates the deviation; ungated absorbs, gated freezes
    def run_joint(gated):
        m = np.zeros(2); P = 0.0; sc = np.zeros(n); al = np.ones(n)
        for t in range(n):
            d = DEV[t] - m; sc[t] = np.sqrt(d @ Sinv @ d)
            if gated:
                e = min(max((sc[t] - mj_m) / mj_s - 1.0, 0.0), C_CAP); P = GAMMA * P + e
                aph = 0.0 if P >= THETA_CLOSE else (1.0 if P <= THETA_OPEN else
                       (THETA_CLOSE - P) / (THETA_CLOSE - THETA_OPEN))
            else:
                aph = 1.0
            al[t] = aph; m = m + K_GAIN * aph * (DEV[t] - m)
        return sc, al
    sc_u, _ = run_joint(False); sc_g, al_g = run_joint(True)

    pd.DataFrame([{"room": R, "n_overlap_hours": n, "ref_hours": ref_end,
        "corr_audio_video": round(float(r), 3), "break_dir": f"({bdir[0]:.0f},{bdir[1]:.0f})",
        "joint_det_thresh": round(mj_t, 3), "audio_det_thresh": round(ma_t, 3),
        "video_det_thresh": round(mv_t, 3),
        "joint_detect_at_1.5sigma": float(sweep.loc[sweep.mag_sigma == 1.5, "joint_detect"].iloc[0]),
        "audio_detect_at_1.5sigma": float(sweep.loc[sweep.mag_sigma == 1.5, "audio_detect"].iloc[0]),
        "video_detect_at_1.5sigma": float(sweep.loc[sweep.mag_sigma == 1.5, "video_detect"].iloc[0]),
        }]).to_csv(HERE / "csv" / f"coupling_room{R}_summary.csv", index=False)

    # ---- figures ----
    # (a) detection-rate vs magnitude
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for name, col in [("audio", "tab:blue"), ("video", "tab:orange"), ("joint", "tab:green")]:
        ax[0].plot(sweep.mag_sigma, sweep[f"{name}_detect"], "-o", color=col, label=f"{name}-only" if name != "joint" else "joint (fusion)")
        ax[0].fill_between(sweep.mag_sigma, sweep[f"{name}_lo"], sweep[f"{name}_hi"], color=col, alpha=0.15)
    ax[0].set_xlabel("coupling-break magnitude (marginal sigma)"); ax[0].set_ylabel("detection rate")
    ax[0].set_title(f"Room {R}: coupling-break detection (corr a,v = {r:.2f})")
    ax[0].legend(fontsize=8); ax[0].set_ylim(-0.03, 1.03)
    # (b) slow-LEVEL joint structure (where coupling lives, report Fig 4). Break star
    # sits inside each marginal band but outside the joint ellipse.
    from matplotlib.patches import Ellipse
    Al = A[:ref_end] - mu[0]; Vl = V[:ref_end] - mu[1]
    r_lvl = Cov[0, 1] / np.sqrt(Cov[0, 0] * Cov[1, 1])
    sa, sv = np.sqrt(Cov[0, 0]), np.sqrt(Cov[1, 1])
    sgn = -np.sign(r_lvl) if abs(r_lvl) > 1e-3 else 1.0
    ax[1].scatter(Al, Vl, s=6, color="0.6", alpha=0.5, label="normal slow level (ref)")
    evals, evecs = np.linalg.eigh(Cov); ang = np.degrees(np.arctan2(*evecs[:, -1][::-1]))
    for k_ in (2, 3):
        ax[1].add_patch(Ellipse((0, 0), 2*k_*np.sqrt(evals[-1]), 2*k_*np.sqrt(evals[0]),
                        angle=ang, fill=False, color="tab:green", lw=1, alpha=0.7))
    bpt = 1.6 * np.array([sa, sgn * sv])
    ax[1].scatter(*bpt, color="red", marker="*", s=220, zorder=5, label="coupling break (1.6 sigma)")
    ax[1].axvspan(-MARGIN_CAP*sa, MARGIN_CAP*sa, color="tab:blue", alpha=0.06)
    ax[1].axhspan(-MARGIN_CAP*sv, MARGIN_CAP*sv, color="tab:orange", alpha=0.06)
    ax[1].set_xlabel("audio slow level"); ax[1].set_ylabel("video slow level")
    ax[1].set_title(f"Slow-level joint (r={r_lvl:.2f}): break inside marginals, off ellipse")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(HERE / "figs" / f"fig_coupling_room{R}_detection.png", dpi=140)

    # (c) contamination trace
    fig2, bx = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    bx[0].axhline(mj_t, color="k", ls="--", lw=0.8, label="joint detect thresh")
    bx[0].plot(sc_u, color="tab:red", lw=1.0, label="joint ungated (absorbs)")
    bx[0].plot(sc_g, color="tab:green", lw=1.0, label="joint gated (holds)")
    bx[0].axvspan(s0, s1, color="orange", alpha=0.12, label="sustained coupling break")
    bx[0].axvspan(0, ref_end, color="blue", alpha=0.05)
    bx[0].set_ylabel("joint Mahalanobis"); bx[0].legend(fontsize=7); bx[0].set_title(f"Room {R}: joint coupling-break contamination")
    bx[1].plot(al_g, color="tab:green", lw=1.2); bx[1].axvspan(s0, s1, color="orange", alpha=0.12)
    bx[1].set_ylabel("alpha_t"); bx[1].set_xlabel("overlap hours"); bx[1].set_ylim(-0.05, 1.05)
    fig2.tight_layout(); fig2.savefig(HERE / "figs" / f"fig_coupling_room{R}_contamination.png", dpi=140)

    print(f"=== Room {R} coupling break (audio+video) ===")
    print(f"overlap {n}h | ref {ref_end} | corr(a,v)={r:.3f} | break dir {bdir}")
    print(sweep.to_string(index=False))


if __name__ == "__main__":
    main()
