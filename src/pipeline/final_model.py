#!/usr/bin/env python3
"""
final_model.py — the locked anomaly-detection architecture and its justification.

MODEL (one architecture, deep and classical variants)
-----------------------------------------------------
Context-conditioned, detrended, dual-modality detector with a per-modality
persistence gate, OR-fused:

  * Environment = slow-band CONTEXT. The flock-age / growth trend is removed from
    the behavioural features (detrending), so detection operates on the fast band.
  * One reconstruction detector PER behavioural modality (audio, video):
        deep      -> autoencoder reconstruction error   (AVGatedDetector kind="dl")
        classical -> Mahalanobis distance               (kind="classical")
  * A per-modality alpha_t persistence gate on each modality's residual.
  * OR-fusion: an alarm fires if EITHER modality shows sustained, unexplained
    deviation. No dilution, no cross-modality masking.

The AVGatedDetector class is the deployable model and is reused unchanged for
cross-barn verification (fit on one barn, calibrate + score on another).

JUSTIFICATION (three questions, three panels)
---------------------------------------------
  (a) deep vs classical      -> why a learned detector
  (b) gate on vs off         -> why the persistence gate
  (c) detrend on vs off      -> why slow/fast (environmental context)

Evaluation: synthetic injection (no labels exist). Sustained departures are the
anomalies; brief spikes are benign transients. Event-level Precision/Recall/F1
with bootstrap 95% CIs.

Usage
-----
    python src/pipeline/final_model.py --methods both --epochs 300 --trials 400
    python src/pipeline/final_model.py --methods classical --trials 300   # sandbox
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import RESULTS_DIR
from src.models.alpha_gate import AlphaGate, robust_scale
from src.pipeline.experiment import (
    load, make_detector, fit_detector, prf, bootstrap_prf, temporal_split,
    GATE_PARAMS,
)

OUT = RESULTS_DIR / "final"
BEHAV = ["audio", "video"]          # behavioural detection modalities
BLUE, GREEN, GREY = "#1f5fa6", "#2e7d32", "#9aa0a6"


# ======================================================================
# The locked model
# ======================================================================
class AVGatedDetector:
    """Per-modality (audio, video) reconstruction detectors + per-modality
    alpha_t gate, OR-fused. Deep or classical backbone."""

    def __init__(self, kind, order, epochs=300, quick=False):
        self.kind, self.order, self.epochs, self.quick = kind, order, epochs, quick
        self.cols = {m: [i for i, f in enumerate(order) if f.startswith(p)]
                     for m, p in [("audio", "aud_"), ("video", "vid_")]}
        self.cols = {m: c for m, c in self.cols.items() if c}

    def fit(self, Xtr):
        self.dets = {}
        for m, c in self.cols.items():
            subdims = {k: sum(self.order[i].startswith(p) for i in c)
                       for k, p in [("aud", "aud_"), ("vid", "vid_")]}
            subdims = {k: v for k, v in subdims.items() if v > 0}
            self.dets[m] = fit_detector(
                make_detector(self.kind, subdims, self.epochs, self.quick), Xtr[:, c])
        return self

    def calibrate(self, Xref):
        """Set per-modality baselines from clean reference hours (of the barn
        being scored — this is what makes cross-barn transfer honest)."""
        self.base = {m: self.dets[m].score(Xref[:, c]) for m, c in self.cols.items()}
        self.thr = {m: np.quantile(self.base[m], 0.99) for m in self.cols}
        return self

    def _gate(self, scores, base):
        s = np.sqrt(np.clip(scores, 0, None)); b = np.sqrt(np.clip(base, 0, None))
        resid = s - np.median(b)
        return AlphaGate(scale=robust_scale(b - np.median(b)), **GATE_PARAMS).run(resid)["closed"]

    def alarm(self, X, use_gate=True):
        """Boolean per-hour alarm stream, OR-fused over modalities."""
        fused = np.zeros(len(X), dtype=bool)
        for m, c in self.cols.items():
            s = self.dets[m].score(X[:, c])
            fused |= (self._gate(s, self.base[m]) if use_gate else (s > self.thr[m]))
        return fused


# ======================================================================
# Synthetic-injection evaluation
# ======================================================================
def eval_model(model, Xpool, trials, use_gate=True, seed=0, timeline=240,
               n_sustained=3, sustained_len=(12, 30), sustained_mag=2.5,
               n_spikes=5, spike_mag=10.0):
    """Per-trial event-level counts. Each event is planted in a randomly chosen
    behavioural modality (a realistic partial-modality anomaly)."""
    rng = np.random.default_rng(seed)
    model.calibrate(Xpool)
    cols = model.cols
    n = len(Xpool)
    tp, fp, fn = [], [], []
    for _ in range(trials):
        T = Xpool[rng.integers(0, n, size=timeline)].copy()
        slots = rng.permutation(np.arange(20, timeline - 40, 8))
        spans, spikes = [], []
        for s in slots[:n_sustained]:
            c = cols[rng.choice(list(cols))]
            L = int(rng.integers(*sustained_len))
            T[s:s + L][:, c] += sustained_mag * rng.choice([-1., 1.], size=(1, len(c)))
            spans.append((s, s + L))
        for s in slots[n_sustained:n_sustained + n_spikes]:
            c = cols[rng.choice(list(cols))]
            T[s][c] += spike_mag * rng.choice([-1., 1.], size=len(c)); spikes.append(s)
        al = model.alarm(T, use_gate=use_gate)
        t = sum(al[a:b].any() for a, b in spans)
        tp.append(t); fn.append(len(spans) - t)
        fp.append(sum(al[p:p + 2].any() for p in spikes))
    return np.array(tp), np.array(fp), np.array(fn)


def cell(model, Xpool, trials, use_gate=True):
    tp, fp, fn = eval_model(model, Xpool, trials, use_gate=use_gate)
    return bootstrap_prf(tp, fp, fn)


# ======================================================================
# Journal figure style
# ======================================================================
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 12, "axes.titleweight": "normal",
        "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.35,
        "legend.frameon": False, "figure.dpi": 120,
    })
    return plt


def _grouped(ax, labels, series, colors, ylabel="F1 score"):
    """series: list of (name, values, err_lo, err_hi)."""
    x = np.arange(len(labels)); w = 0.8 / len(series)
    for i, (name, vals, lo, hi) in enumerate(series):
        err = [np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, yerr=err, capsize=3,
               label=name, color=colors[i])
        for xi, v in zip(x + i * w - 0.4 + w / 2, vals):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08); ax.set_ylabel(ylabel)


# ======================================================================
def load_model_data(spine, features, representation, kind, epochs, quick):
    _, order, X, _ = load(spine, features, "rich", representation)
    Xtr, Xpool = temporal_split(X)
    model = AVGatedDetector(kind, order, epochs, quick).fit(Xtr)
    return model, Xpool


def run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    do_c = args.methods in ("classical", "both")
    do_d = args.methods in ("dl", "both")
    rows = []

    def prf_row(question, condition, r):
        rows.append({"question": question, "condition": condition,
                     "precision": r["precision"], "recall": r["recall"], "f1": r["f1"],
                     "f1_lo": r["f1_lo"], "f1_hi": r["f1_hi"]})
        return r

    # ---------- primary model (deep, detrended, gated) ----------
    results = {}
    if do_d:
        m, pool = load_model_data(args.spine, args.features, "DETREND", "dl_late", args.epochs, args.quick)
        results["deep_detrend_gate"] = prf_row("model", "Deep (proposed)", cell(m, pool, args.trials, True))
        results["deep_detrend_nogate"] = prf_row("gate", "Gate off", cell(m, pool, args.trials, False))
        # detrend off
        mr, poolr = load_model_data(args.spine, args.features, "RAW", "dl_late", args.epochs, args.quick)
        results["deep_raw_gate"] = prf_row("detrend", "Trend retained", cell(mr, poolr, args.trials, True))
        prf_row("detrend", "Trend removed", results["deep_detrend_gate"])
        prf_row("gate", "Gate on", results["deep_detrend_gate"])
        print("[done] deep variants")
    if do_c:
        mc, poolc = load_model_data(args.spine, args.features, "DETREND", "classical", args.epochs, args.quick)
        results["classical_detrend_gate"] = prf_row("model", "Classical", cell(mc, poolc, args.trials, True))
        print("[done] classical variant")

    pd.DataFrame(rows).to_csv(OUT / "final_results.csv", index=False)
    make_figures(results, do_c, do_d)
    print(f"\nAll outputs in {OUT}")
    print(pd.DataFrame(rows).to_string(index=False))


def make_figures(R, do_c, do_d):
    plt = _style()

    # ---- Figure: three-panel design justification ----
    panels = []
    if do_d and do_c:
        panels.append(("a", "Detector backbone",
                       ["Classical\n+ gate", "Deep\n+ gate"],
                       [R["classical_detrend_gate"], R["deep_detrend_gate"]], [GREY, GREEN]))
    if do_d:
        panels.append(("b", "Persistence gate",
                       ["Deep,\nno gate", "Deep\n+ gate"],
                       [R["deep_detrend_nogate"], R["deep_detrend_gate"]], [GREY, GREEN]))
        panels.append(("c", "Environmental detrending",
                       ["Deep + gate,\ntrend kept", "Deep + gate,\ndetrended"],
                       [R["deep_raw_gate"], R["deep_detrend_gate"]], [GREY, GREEN]))

    if not panels:
        print("[figs] (design-justification panels need the deep variant; run --methods both)")
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.2), squeeze=False)
    for ax, (tag, title, labels, cells, colors) in zip(axes[0], panels):
        vals = [c["f1"] for c in cells]
        lo = [c["f1_lo"] for c in cells]; hi = [c["f1_hi"] for c in cells]
        x = np.arange(len(labels))
        err = [np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)]
        ax.bar(x, vals, 0.55, yerr=err, capsize=4, color=colors)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.08); ax.set_ylabel("F1 score" if tag == panels[0][0] else "")
        ax.set_title(f"({tag}) {title}")
    fig.tight_layout()
    fig.savefig(OUT / "fig_design_justification.png", dpi=600); plt.close(fig)

    # ---- Figure: full metric breakdown of the proposed model ----
    if do_d:
        r = R["deep_detrend_gate"]
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        mets = ["precision", "recall", "f1"]
        vals = [r[k] for k in mets]
        ax.bar(range(3), vals, 0.55, color=[BLUE, BLUE, GREEN])
        ax.errorbar([2], [r["f1"]], yerr=[[r["f1"] - r["f1_lo"]], [r["f1_hi"] - r["f1"]]],
                    fmt="none", ecolor="black", capsize=4)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9.5)
        ax.set_xticks(range(3)); ax.set_xticklabels(["Precision", "Recall", "F1"])
        ax.set_ylim(0, 1.08); ax.set_ylabel("Score")
        ax.set_title("Proposed model: detection performance")
        fig.tight_layout(); fig.savefig(OUT / "fig_proposed_model.png", dpi=600); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spine", default=str(RESULTS_DIR / "spine_room2_rich.csv"))
    ap.add_argument("--features", default=str(RESULTS_DIR / "recommended_features.csv"))
    ap.add_argument("--methods", default="both", choices=["classical", "dl", "both"])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
