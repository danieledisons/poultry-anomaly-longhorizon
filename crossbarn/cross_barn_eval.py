#!/usr/bin/env python3
"""Cross-barn check: fit the model on Room 2 and score the held-out Room 6 with the same settings.

Run: python crossbarn/cross_barn_eval.py --methods both --trials 400
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RESULTS_DIR
from src.pipeline.experiment import load, temporal_split, prf, bootstrap_prf
from src.pipeline.final_model import AVGatedDetector, eval_model, make_detector  # noqa

OUT = RESULTS_DIR / "crossbarn"
GREY, GREEN, BLUE = "#9aa0a6", "#2e7d32", "#1f5fa6"


def fit_room2(spine2, feats, kind, epochs, quick):
    _, order, X, _ = load(spine2, feats, "rich", "DETREND")
    Xtr, Xte = temporal_split(X)
    det = AVGatedDetector(kind, order, epochs, quick).fit(Xtr)
    return det, order, Xte


def pool(spine, feats, order_ref):
    _, order, X, _ = load(spine, feats, "rich", "DETREND")
    assert order == order_ref, "feature columns differ between barns"
    return X


def cell(det, Xpool, trials, use_gate=True):
    tp, fp, fn = eval_model(det, Xpool, trials, use_gate=use_gate)
    return bootstrap_prf(tp, fp, fn)


def sweep(det, Xpool, trials, mags=(0.5, 1, 1.5, 2, 2.5, 3)):
    rows = []
    for mg in mags:
        tp, fp, fn = eval_model(det, Xpool, trials, use_gate=True, sustained_mag=mg)
        _, r, f = prf(tp.sum(), fp.sum(), fn.sum())
        rows.append((mg, round(f, 3), round(r, 3)))
    return rows


def run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    families = ([("Classical", "classical")] if args.methods in ("classical", "both") else []) + \
               ([("Deep", "dl_late")] if args.methods in ("dl", "both") else [])
    res, swp = [], []
    for label, kind in families:
        det, order, X2te = fit_room2(args.spine2, args.features, kind, args.epochs, args.quick)
        X6 = pool(args.spine6, args.features, order)
        res.append({"model": label, "setting": "within-barn (Room 2)", **cell(det, X2te, args.trials)})
        res.append({"model": label, "setting": "cross-barn (Room 6)", **cell(det, X6, args.trials)})
        for mg, f, r in sweep(det, X2te, args.trials):
            swp.append({"model": label, "barn": "Room 2", "magnitude": mg, "f1": f, "recall": r})
        for mg, f, r in sweep(det, X6, args.trials):
            swp.append({"model": label, "barn": "Room 6", "magnitude": mg, "f1": f, "recall": r})
        print(f"[done] {label}")
    R = pd.DataFrame(res); S = pd.DataFrame(swp)
    R.to_csv(OUT / "crossbarn_results.csv", index=False)
    S.to_csv(OUT / "crossbarn_sweep.csv", index=False)
    print("\n=== Within-barn vs cross-barn (P/R/F1, 95% CI) ===")
    print(R[["model", "setting", "precision", "recall", "f1", "f1_lo", "f1_hi"]].to_string(index=False))
    make_figs(R, S, [f[0] for f in families])
    print(f"\nOutputs in {OUT}")


def make_figs(R, S, models):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "normal",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.3, "legend.frameon": False})

    # bars: within vs cross F1 per model
    fig, ax = plt.subplots(figsize=(1.8 + 1.8 * len(models), 4.4))
    x = np.arange(len(models)); w = 0.36
    for i, setting in enumerate(["within-barn (Room 2)", "cross-barn (Room 6)"]):
        vals = [R[(R.model == m) & (R.setting == setting)]["f1"].values[0] for m in models]
        lo = [R[(R.model == m) & (R.setting == setting)]["f1_lo"].values[0] for m in models]
        hi = [R[(R.model == m) & (R.setting == setting)]["f1_hi"].values[0] for m in models]
        err = [np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)]
        ax.bar(x + (i - 0.5) * w, vals, w, yerr=err, capsize=4,
               label=setting, color=[GREY, GREEN][i])
        for xi, v in zip(x + (i - 0.5) * w, vals):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylim(0, 1.08)
    ax.set_ylabel("F1 score"); ax.legend()
    ax.set_title("Cross-barn generalization (fit Room 2, test Room 6)")
    fig.tight_layout(); fig.savefig(OUT / "fig_crossbarn.png", dpi=600); plt.close(fig)

    # sweep: F1 vs magnitude, both barns (use proposed = deep if present else classical)
    prim = "Deep" if "Deep" in models else models[0]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for barn, c in [("Room 2", BLUE), ("Room 6", GREEN)]:
        d = S[(S.model == prim) & (S.barn == barn)].sort_values("magnitude")
        ax.plot(d.magnitude, d.f1, "o-", color=c, label=barn)
    ax.set_xlabel("sustained anomaly magnitude (σ)"); ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.02); ax.legend(title=prim)
    ax.set_title("Detection vs magnitude: held-out barn tracks the source barn")
    fig.tight_layout(); fig.savefig(OUT / "fig_crossbarn_sweep.png", dpi=600); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spine2", default=str(RESULTS_DIR / "spine_room2_rich.csv"))
    ap.add_argument("--spine6", default=str(RESULTS_DIR / "spine_room6_rich.csv"))
    ap.add_argument("--features", default=str(RESULTS_DIR / "recommended_features.csv"))
    ap.add_argument("--methods", default="both", choices=["classical", "dl", "both"])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--quick", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
