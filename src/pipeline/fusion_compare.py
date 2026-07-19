#!/usr/bin/env python3
"""
fusion_compare.py — compare anomaly-detection scorers on the lean rich features,
RAW vs day_index-detrended, evaluated by synthetic injection (no labels exist).

Design
------
Features: the recommended lean set MINUS env_day_index (day_index is the flock
growth clock — a slow-band covariate, not a fusion feature). Fusion hours only
(both modalities live).

Two feature variants:
    RAW       — robustly standardized features.
    DETREND   — each feature's linear day_index trend removed first (isolates the
                fast/hour-to-hour signal, per the slow/fast framing), then standardized.

Three scorers (all unsupervised, fit on the assumed-mostly-normal fusion hours):
    mahalanobis  — classical joint-distribution distance (LedoitWolf covariance).
    pca_recon    — reconstruction error from a low-rank PCA (DL-bridge proxy;
                   swap in a GRU/transformer autoencoder on the GPU server later).
    gate         — the alpha_t persistence gate run on the Mahalanobis residual
                   stream (your contribution): rewards SUSTAINED, not spiky, anomalies.

Evaluation
----------
Per-row scorers (mahalanobis, pca_recon): ROC-AUC separating held-out normal
hours from synthetically perturbed copies, swept over perturbation magnitude.
Gate: spike vs sustained injection on the time-ordered residual — false-closure
on brief spikes and detection rate on sustained departures.

Usage
-----
    python src/pipeline/fusion_compare.py --spine results/spine_room2_rich.csv \
        --features results/recommended_features.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import RESULTS_DIR
from src.models.alpha_gate import AlphaGate, robust_scale

GATE_PARAMS = dict(deadband=1.0, decay=0.85, close_threshold=6.0,
                   open_threshold=2.0, per_step_cap=2.0)


def robust_standardize(X):
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    scale = np.where(mad < 1e-9, 1.0, 1.4826 * mad)
    return (X - med) / scale


def detrend_dayindex(X, day_index):
    """Remove each feature's linear trend in day_index (the growth clock)."""
    d = np.asarray(day_index, dtype=float)
    A = np.vstack([d, np.ones_like(d)]).T
    out = np.empty_like(X)
    for j in range(X.shape[1]):
        coef, *_ = np.linalg.lstsq(A, X[:, j], rcond=None)
        out[:, j] = X[:, j] - A @ coef
    return out


def injection_auc(Xtrain, Xtest, mags, seed=0):
    """For each magnitude, perturb held-out normal rows and measure how well each
    scorer separates normal from perturbed (ROC-AUC)."""
    rng = np.random.default_rng(seed)
    cov = LedoitWolf().fit(Xtrain)
    pca = PCA(n_components=0.95, svd_solver="full").fit(Xtrain)

    def maha(X):
        return cov.mahalanobis(X)

    def recon(X):
        return ((X - pca.inverse_transform(pca.transform(X))) ** 2).sum(axis=1)

    rows = []
    for m in mags:
        # sustained departure: shift every feature by m (standardized units),
        # random sign per feature per row — an unusual but self-consistent state
        signs = rng.choice([-1.0, 1.0], size=Xtest.shape)
        Xpert = Xtest + m * signs
        y = np.r_[np.zeros(len(Xtest)), np.ones(len(Xpert))]
        for name, fn in [("mahalanobis", maha), ("pca_recon", recon)]:
            s = np.r_[fn(Xtest), fn(Xpert)]
            rows.append((name, m, roc_auc_score(y, s)))
    return pd.DataFrame(rows, columns=["scorer", "magnitude", "auc"]), cov


def gate_injection(residual, scale, seed=42, n=200):
    """Spike vs sustained on the 1-D residual stream (gate behaviour)."""
    rng = np.random.default_rng(seed)
    pool = residual[~np.isnan(residual)]
    quiet = pool[np.abs(pool - np.median(pool)) < 3 * scale]

    def run(r):
        return AlphaGate(scale=scale, **GATE_PARAMS).run(r)["closed"]

    def mk(nn=240):
        return rng.choice(quiet, size=nn, replace=True)

    T0 = 80
    out = {}
    # brief spike: should NOT close
    fc = np.mean([run(np.r_[mk()[:T0], mk()[:1] + 12 * scale, mk()[T0 + 1:]])[T0:T0 + 12].any()
                  for _ in range(n)])
    out["spike_false_closure_12sigma"] = float(fc)
    # sustained departure at 2.5 sigma for 36h: should close
    det = []
    for _ in range(n):
        b = mk(); b[T0:T0 + 36] += 2.5 * scale
        det.append(run(b)[T0:T0 + 36].any())
    out["sustained_detect_2.5sigma"] = float(np.mean(det))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spine", default=str(RESULTS_DIR / "spine_room2_rich.csv"))
    ap.add_argument("--features", default=str(RESULTS_DIR / "recommended_features.csv"))
    ap.add_argument("--out", default=str(RESULTS_DIR / "fusion_compare.csv"))
    args = ap.parse_args()

    m = pd.read_csv(args.spine, parse_dates=["time"])
    feats = pd.read_csv(args.features)["feature"].tolist()
    model_feats = [f for f in feats if f != "env_day_index"]

    fus = m[m["coverage_state"] == "both_lit"].sort_values("time").reset_index(drop=True)
    fus = fus.dropna(subset=model_feats)
    X = fus[model_feats].to_numpy(dtype=float)
    day_index = fus["env_day_index"].to_numpy(dtype=float)
    print(f"Fusion hours used: {len(fus)}   model features: {len(model_feats)}")

    variants = {
        "RAW": robust_standardize(X),
        "DETREND": robust_standardize(detrend_dayindex(X, day_index)),
    }

    mags = [1, 2, 3, 4, 6]
    n = len(X); cut = int(n * 0.7)
    all_auc = []
    gate_rows = []
    for vname, Xv in variants.items():
        Xtr, Xte = Xv[:cut], Xv[cut:]
        auc_df, cov = injection_auc(Xtr, Xte, mags)
        auc_df.insert(0, "variant", vname)
        all_auc.append(auc_df)

        # gate on the Mahalanobis residual stream (full ordered series)
        resid = np.sqrt(np.clip(cov.mahalanobis(Xv), 0, None))
        resid = resid - np.median(resid)
        g = gate_injection(resid, robust_scale(resid))
        gate_rows.append({"variant": vname, **g})

    auc = pd.concat(all_auc, ignore_index=True)
    auc.to_csv(args.out, index=False)
    gate = pd.DataFrame(gate_rows)
    gate.to_csv(str(Path(args.out).with_name("fusion_gate.csv")), index=False)

    # --- report ---
    print("\n=== ROC-AUC: normal vs injected (higher = better detection) ===")
    piv = auc.pivot_table(index=["scorer", "magnitude"], columns="variant", values="auc")
    print(piv.round(3).to_string())
    print("\n=== alpha_t gate on fused Mahalanobis residual ===")
    print(gate.round(3).to_string(index=False))

    # --- figure (600 dpi) ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        for ax, scorer in zip(axes, ["mahalanobis", "pca_recon"]):
            for vname in variants:
                d = auc[(auc.scorer == scorer) & (auc.variant == vname)]
                ax.plot(d.magnitude, d.auc, "o-", label=vname)
            ax.set_title(scorer); ax.set_xlabel("injection magnitude (sigma)")
            ax.set_ylim(0.45, 1.02); ax.grid(alpha=.3); ax.axhline(0.5, color="gray", lw=.6, ls="--")
        axes[0].set_ylabel("ROC-AUC (normal vs injected)")
        axes[0].legend(title="features")
        fig.suptitle("Fusion anomaly detection: RAW vs day_index-detrended", fontweight="bold")
        fig.tight_layout()
        png = str(Path(args.out).with_name("fusion_compare.png"))
        fig.savefig(png, dpi=600); plt.close(fig)
        print(f"\nWrote {args.out}, fusion_gate.csv, and {png}")
    except ImportError:
        print(f"\nWrote {args.out} and fusion_gate.csv (matplotlib missing — no figure)")


if __name__ == "__main__":
    main()
