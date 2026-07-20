#!/usr/bin/env python3
"""
contamination_experiment.py — the standalone experiment that makes the alpha_t
gate (C3) a contribution in its own right, not an ablation of detection (C1) or
slow/fast modelling (C2).

THE CLAIM
---------
The gate is a TRUST / LEARNING-RATE controller. In any deployed detector that
adapts its notion of "normal" online (consolidation), a sustained anomaly that
is allowed to keep updating the model POISONS the baseline: the model learns the
anomaly as normal, and subsequent occurrences become undetectable. The alpha_t
gate freezes learning while unexplained energy persists, so the contaminated
period is never consolidated and future detection is preserved.

SETUP
-----
An online detector maintains a running "normal" estimate mu_t (EWMA over the
fused behavioural features). Anomaly score = standardized distance of the current
hour from mu. Two conditions differ ONLY in the learning rate:

    ungated:  mu update rate = lambda            (consolidates everything)
    gated:    mu update rate = lambda * alpha_t   (alpha_t -> 0 pauses learning
                                                   when a sustained anomaly persists)

Timeline: clean warm-up -> a long sustained CONTAMINATION event -> a post-event
period seeded with recurring PROBE anomalies (same signature as the contamination).
We measure how many post-event probes are still detected, and how far the baseline
drifted toward the anomaly.

Outputs (RESULTS_DIR/contamination/):
    contamination_results.csv   post-event recall/F1 + baseline drift, gated vs ungated
    fig_contamination.png       trace (score, baseline, gate) + post-event detection bars

Usage
-----
    python src/pipeline/contamination_experiment.py --spine results/spine_room2_rich.csv --trials 300
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import RESULTS_DIR

OUT = RESULTS_DIR / "contamination"
GREY, GREEN, RED = "#9aa0a6", "#2e7d32", "#c62828"

# gate params (locked, same family as the detection gate)
DEADBAND, DECAY, CLOSE, OPEN, CAP = 1.0, 0.85, 6.0, 2.0, 2.0


# ---------- feature prep (self-contained; behavioural fusion features) ----------
def robust_standardize(X):
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    scale = np.where(mad < 1e-9, 1.0, 1.4826 * mad)
    return (X - med) / scale


def detrend_dayindex(X, day):
    d = np.asarray(day, float); A = np.vstack([d, np.ones_like(d)]).T
    out = np.empty_like(X)
    for j in range(X.shape[1]):
        c, *_ = np.linalg.lstsq(A, X[:, j], rcond=None)
        out[:, j] = X[:, j] - A @ c
    return out


def load_features(spine, features_csv, n_comp=5):
    """Behavioural fusion features -> detrended, robustly standardized, clipped,
    and reduced to a compact WHITENED PCA space so that Euclidean distance is
    well-conditioned (the raw features have heavy-tailed outliers that otherwise
    dominate the norm)."""
    from sklearn.decomposition import PCA
    m = pd.read_csv(spine, parse_dates=["time"])
    lean = pd.read_csv(features_csv)["feature"].tolist()
    feats = [f for f in lean if (f.startswith("aud_") or f.startswith("vid_")) and f in m.columns]
    fus = m[m["coverage_state"] == "both_lit"].sort_values("time").dropna(subset=feats).reset_index(drop=True)
    X = detrend_dayindex(fus[feats].to_numpy(float), fus["env_day_index"].to_numpy(float))
    X = np.clip(robust_standardize(X), -5, 5)
    return PCA(n_components=n_comp, whiten=True, random_state=0).fit_transform(X)


# ---------- online detector with a gated learning rate ----------
def run_online(T, mu0, center, scale, lam, gated, direction, warm):
    """Stream T (timeline x d). Score = z-scored distance from the running normal
    mu, so anomalies are salient regardless of feature dimensionality."""
    mu = mu0.copy()
    P, latched = 0.0, False
    scores = np.zeros(len(T)); munorm = np.zeros(len(T)); closed = np.zeros(len(T), bool)
    for t in range(len(T)):
        x = T[t]
        s = (np.linalg.norm(x - mu) - center) / scale      # z-scored distance from "normal"
        scores[t] = s
        e = min(max(0.0, s - DEADBAND), CAP)               # persistence gate
        P = DECAY * P + e
        latched = (P > OPEN) if latched else (P >= CLOSE)
        closed[t] = latched
        g = (0.0 if latched else 1.0) if gated else 1.0    # gated learning rate
        if t >= warm:
            mu = mu + lam * g * (x - mu)                    # EWMA consolidation
        munorm[t] = mu @ direction
    return scores, munorm, closed


def one_trial(X, rng, timeline=320, warm=60, anomaly=(80, 300), mag=7.0,
              early_len=25, late_len=60, lam=0.06):
    """A LONG sustained anomaly. Without the gate the online baseline adapts to it
    (consolidation) and the detector habituates — goes blind to the ongoing fault.
    The gate freezes learning so the alarm stays alive."""
    d = X.shape[1]
    T = X[rng.integers(0, len(X), size=timeline)].copy()
    direction = rng.choice([-1., 1.], size=d); direction /= np.linalg.norm(direction)
    mu0 = X[rng.integers(0, len(X), size=200)].mean(0)
    dists = np.array([np.linalg.norm(X[i] - mu0) for i in rng.integers(0, len(X), 500)])
    center, scale = dists.mean(), dists.std() + 1e-9
    thr = 3.0

    a0, a1 = anomaly
    T[a0:a1] += mag * direction                               # long sustained anomaly
    early = (a0, a0 + early_len)                              # onset window
    late = (a1 - late_len, a1)                                # after adaptation would kick in

    out = {}
    for gated in (False, True):
        sc, mn, cl = run_online(T, mu0, center, scale, lam, gated, direction, warm)
        out[gated] = dict(
            det_early=float((sc[early[0]:early[1]] > thr).mean()),
            det_late=float((sc[late[0]:late[1]] > thr).mean()),
            drift=float(mn[a1 - 1] - (mu0 @ direction)),
            trace=(sc, cl, thr, (a0, a1), early, late))
    return out


def run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    X = load_features(args.spine, args.features)
    rng = np.random.default_rng(0)
    agg = {False: {"early": [], "late": [], "drift": []}, True: {"early": [], "late": [], "drift": []}}
    example = None
    for i in range(args.trials):
        o = one_trial(X, rng)
        for g in (False, True):
            agg[g]["early"].append(o[g]["det_early"]); agg[g]["late"].append(o[g]["det_late"])
            agg[g]["drift"].append(o[g]["drift"])
        if i == 0:
            example = o
    rows = []
    for g, name in [(False, "Ungated (adapts to anomaly)"), (True, "Gated (learning frozen)")]:
        e = np.array(agg[g]["early"]); l = np.array(agg[g]["late"]); dr = np.abs(np.array(agg[g]["drift"]))
        rows.append({"condition": name,
                     "onset_detection": round(e.mean(), 3),
                     "late_detection": round(l.mean(), 3),
                     "late_lo": round(np.percentile(l, 2.5), 3),
                     "late_hi": round(np.percentile(l, 97.5), 3),
                     "baseline_drift_abs": round(dr.mean(), 3)})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "contamination_results.csv", index=False)
    print("\n=== C3: habituation to a persistent anomaly (onset vs late detection) ===")
    print(res.to_string(index=False))
    make_fig(example, res)
    print(f"\nOutputs in {OUT}")


def make_fig(example, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "normal",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.3, "legend.frameon": False})
    fig = plt.figure(figsize=(13, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.3, 2.3, 1.1])

    for col, gated, title in [(0, False, "(a) Ungated: detector habituates"),
                              (1, True, "(b) Gated: alarm stays alive")]:
        ax = fig.add_subplot(gs[0, col])
        sc, cl, thr, (a0, a1), early, late = example[gated]["trace"]
        ax.plot(sc, color="#37474f", lw=1.0, label="anomaly score")
        ax.axhline(thr, color=RED, ls="--", lw=.8, label="detection threshold")
        ax.axvspan(a0, a1, color="#8e24aa", alpha=.12, label="persistent anomaly")
        ax.axvspan(*late, color=GREEN, alpha=.18, label="late window")
        ax.set_title(title); ax.set_xlabel("hour"); ax.set_ylim(bottom=0)
        if col == 0:
            ax.set_ylabel("anomaly score")
            h, l = ax.get_legend_handles_labels()
            ax.legend(h, l, fontsize=8, loc="upper right")

    ax = fig.add_subplot(gs[0, 2])
    vals = res["late_detection"].values
    lo = res["late_lo"].values; hi = res["late_hi"].values
    err = [vals - lo, hi - vals]
    ax.bar([0, 1], vals, 0.6, yerr=err, capsize=4, color=[GREY, GREEN])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Ungated", "Gated"])
    ax.set_ylim(0, 1.08); ax.set_ylabel("late-anomaly detection rate")
    ax.set_title("(c) Late detection")
    fig.suptitle("The gate prevents habituation: an online detector goes blind to a persistent anomaly unless learning is frozen",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "fig_contamination.png", dpi=600); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spine", default=str(RESULTS_DIR / "spine_room2_rich.csv"))
    ap.add_argument("--features", default=str(RESULTS_DIR / "recommended_features.csv"))
    ap.add_argument("--trials", type=int, default=300)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
