#!/usr/bin/env python3
"""
#2: Low-dimensional coherent projection (PCA slow manifold) gate ablation.

Progress toward the learned Approach-C slow state. Instead of scoring the full
Mahalanobis distance (#1), we learn a low-dimensional linear manifold of the
growth+diurnal dynamics from a PAST-ONLY reference, and define anomaly as the
OFF-MANIFOLD residual energy -- the part of the current slow-band vector that the
learned low-dim state cannot represent. This mirrors Approach C's "energy the
slow band refuses to absorb", in linear form.

Hypothesis (tested, not assumed): projecting out the top components removes the
growth-driven variance that made #1's fixed-covariance false-alarm rate climb
across the cycle, so the off-manifold residual should be MORE STATIONARY and give
a LOWER late-cycle false-closure rate than #1.

Mechanism / integrity:
  * PCA basis W and mean mu0 are fit on the reference window only (causal).
  * The full-dim mean mu assimilates online with gain k * alpha_t. Ungated it
    absorbs the (off-manifold) injection -> residual decays (contamination);
    gated it freezes -> residual persists.
  * Feature standardization, thresholds and gate parameters are all set from
    past-only reference statistics. Injection is the same physically-specified
    coherent band shift as #1, magnitude measured not tuned.

Usage:
  python validation_new/gate_ablation/gate_ablation_pca.py --spine <csv> --room 2 --modality audio
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

K_GAIN = 0.10
GAMMA = 0.85; C_CAP = 4.0
VAR_KEEP = 0.95            # keep enough components to explain 95% of reference variance
INJ_SWEEP = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
INJ_LENGTH = 72
REF_FRAC = 0.35


def fit_pca(ref_std, var_keep):
    """PCA via eigdecomposition of the reference covariance. Returns mean, basis
    W (d x k), and k chosen to reach var_keep cumulative variance. Closed form."""
    mu = ref_std.mean(0)
    C = np.cov((ref_std - mu).T)
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    evals = np.clip(evals, 0, None)
    cum = np.cumsum(evals) / evals.sum()
    k = int(np.searchsorted(cum, var_keep) + 1)
    return mu, evecs[:, :k], k, cum[k - 1]


def run_pca(Z, W, mu0, med, sigma, b_dead, theta_close, theta_open, gated):
    """Online: mu assimilates full-dim; score = off-manifold residual norm of
    (z - mu). Gate fires on excess residual over the normal operating level med."""
    n, d = Z.shape
    mu = mu0.copy(); WWt = W @ W.T
    score = np.zeros(n); alpha = np.ones(n); P = 0.0
    for t in range(n):
        r = Z[t] - mu
        resid = r - WWt @ r                     # component off the learned manifold
        score[t] = np.sqrt(resid @ resid)
        if gated:
            e = min(max((score[t] - med) / sigma - b_dead, 0.0), C_CAP)
            P = GAMMA * P + e
            a = 0.0 if P >= theta_close else (1.0 if P <= theta_open else
                 (theta_close - P) / (theta_close - theta_open))
        else:
            a = 1.0
        alpha[t] = a
        mu = mu + K_GAIN * a * (Z[t] - mu)
    return score, alpha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spine", default=str(DATA / "spine_room2_rich.csv"))
    ap.add_argument("--room", default="2")
    ap.add_argument("--modality", default="audio", choices=["audio", "video"])
    ap.add_argument("--outdir", default=str(HERE))
    args = ap.parse_args()
    R, MOD = args.room, args.modality

    df = pd.read_csv(args.spine).sort_values("time").reset_index(drop=True)
    if MOD == "audio":
        feats = [c for c in df.columns if c.startswith("aud_mel") and c.endswith("_mean")]
        inj_names = [f"aud_mel{b:02d}_mean" for b in range(20, 41)]
    else:
        feats = ([c for c in df.columns if c.startswith("vid_flowhist")] +
                 [c for c in df.columns if c.startswith("vid_gridmean")] +
                 [c for c in df.columns if c.startswith("vid_gridstd")])
        inj_names = [f"vid_gridmean{c:02d}" for c in range(0, 8)]
    df = df[df[feats].notna().all(axis=1)].reset_index(drop=True)
    n = len(df)

    Z = ra.causal_slow_band(df, feats)
    # standardize by PAST-ONLY reference robust scale so PCA is scale-fair
    ref_end = int(REF_FRAC * n)
    ref_med = np.median(Z[:ref_end], 0)
    ref_mad = 1.4826 * np.median(np.abs(Z[:ref_end] - ref_med), 0)
    ref_mad = np.where(ref_mad < 1e-9, 1.0, ref_mad)
    Zs = (Z - ref_med) / ref_mad

    mu0, W, k, var_expl = fit_pca(Zs[:ref_end], VAR_KEEP)

    b_dead = 1.0; theta_close = 6.0; theta_open = 2.0
    ref_op, _ = run_pca(Zs[:ref_end], W, mu0, 0.0, 1.0, 0.0, theta_close, theta_open, gated=False)
    ref_op = ref_op[TREND_WIN:]
    med = np.median(ref_op); sigma = 1.4826 * np.median(np.abs(ref_op - med))
    sigma = sigma if sigma > 1e-9 else ref_op.std()
    det_thresh = np.quantile(ref_op, 0.99)

    start = ref_end + (150 if MOD == "audio" else 80)
    end = min(start + INJ_LENGTH, n)
    inj_idx = [feats.index(name) for name in inj_names if name in feats]
    # inject in STANDARDIZED units (per-feature), physically specified
    def retention(sc): seg = sc[start:end]; return float((seg >= det_thresh).mean())
    def latency(sc):
        seg = sc[start:end]; below = np.where(seg < det_thresh)[0]
        return int(below[0]) if len(below) else np.inf

    def evaluate(mag):
        Zi = Zs.copy(); Zi[start:end][:, inj_idx] += mag          # +mag robust-SD per feature
        su, au = run_pca(Zi, W, mu0, med, sigma, b_dead, theta_close, theta_open, gated=False)
        sg, ag = run_pca(Zi, W, mu0, med, sigma, b_dead, theta_close, theta_open, gated=True)
        return su, au, sg, ag

    sweep = pd.DataFrame([dict(mag_sd=m,
        onset_offmanifold=round(np.median(evaluate(m)[2][start:start+3]), 2),
        above_thresh=bool(np.median(evaluate(m)[2][start:start+3]) > det_thresh),
        ungated_retention=round(retention(evaluate(m)[0]), 3),
        gated_retention=round(retention(evaluate(m)[2]), 3),
        ungated_latency_hr=latency(evaluate(m)[0]),
        gated_latency_hr=latency(evaluate(m)[2])) for m in INJ_SWEEP])

    ILL = 4.0
    sc_u, al_u, sc_g, al_g = evaluate(ILL)
    sc_ref, al_ref = run_pca(Zs[:ref_end], W, mu0, med, sigma, b_dead, theta_close, theta_open, gated=True)
    false_closure = float((al_ref[TREND_WIN:] < 0.5).mean())
    # late-cycle false-closure on gated run over full clean series (no injection)
    sc_full, al_full = run_pca(Zs, W, mu0, med, sigma, b_dead, theta_close, theta_open, gated=True)
    late_mask = np.ones(n, bool); late_mask[start:end] = False; late_mask[:ref_end] = False
    late_false_closure = float((al_full[late_mask] < 0.5).mean())

    pd.DataFrame({"time": df["time"], "offmanifold_ungated": sc_u,
                  "offmanifold_gated": sc_g, "alpha_gated": al_g}).to_csv(
        Path(args.outdir) / f"pca_{MOD}_room{R}_trace.csv", index=False)
    sweep.to_csv(Path(args.outdir) / f"pca_{MOD}_room{R}_sweep.csv", index=False)
    pd.DataFrame([{"room": R, "modality": MOD, "n_hours": n, "n_features": len(feats),
        "pca_k": k, "var_explained": round(float(var_expl), 3), "ref_hours": ref_end,
        "det_thresh": round(det_thresh, 3), "sigma": round(sigma, 3),
        "ungated_latency_hr": latency(sc_u), "gated_latency_hr": latency(sc_g),
        "ungated_retention": round(retention(sc_u), 3), "gated_retention": round(retention(sc_g), 3),
        "ref_false_closure": round(false_closure, 3),
        "late_cycle_false_closure": round(late_false_closure, 3)}]).to_csv(
        Path(args.outdir) / f"pca_{MOD}_room{R}_summary.csv", index=False)

    x = np.arange(n)
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax[0].axhline(det_thresh, color="k", ls="--", lw=0.8, label="detect thresh (ref 99pct)")
    ax[0].plot(x, sc_u, color="tab:red", lw=1.0, label="off-manifold ungated (absorbs)")
    ax[0].plot(x, sc_g, color="tab:green", lw=1.0, label="off-manifold gated (holds)")
    ax[0].axvspan(start, end, color="orange", alpha=0.12, label="sustained departure")
    ax[0].axvspan(0, ref_end, color="blue", alpha=0.05, label="causal reference window")
    ax[0].set_ylabel("off-manifold residual"); ax[0].legend(fontsize=7, loc="upper left")
    ax[0].set_title(f"Room {R} {MOD}: PCA slow-manifold gate ablation "
                    f"(k={k} comps, {var_expl*100:.0f}% var, causal)")
    ax[1].plot(x, al_g, color="tab:green", lw=1.2, label="alpha_t (gated)")
    ax[1].axvspan(start, end, color="orange", alpha=0.12)
    ax[1].set_ylabel("trust alpha_t"); ax[1].set_xlabel("hours"); ax[1].set_ylim(-0.05, 1.05)
    ax[1].legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(Path(args.outdir) / f"fig_pca_{MOD}_room{R}.png", dpi=140)

    print(f"=== Room {R} {MOD} PCA slow-manifold gate ablation ===")
    print(f"features {len(feats)} -> k={k} comps ({var_expl*100:.1f}% var) | ref {ref_end}/{n}")
    print(f"ungated: latency {latency(sc_u)} h  retention {retention(sc_u):.2f}")
    print(f"gated:   latency {latency(sc_g)} h  retention {retention(sc_g):.2f}")
    print(f"false-closure  ref: {false_closure:.3f}   late-cycle: {late_false_closure:.3f}")


if __name__ == "__main__":
    main()
