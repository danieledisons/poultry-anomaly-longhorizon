#!/usr/bin/env python3
"""Modality-targeted injection and magnitude/duration sweeps; shows the OR-gate fusion catches an anomaly in any single modality where naive concatenation dilutes it.

Run: python src/pipeline/experiment_multimodal.py --methods both
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
    load, make_detector, fit_detector, prf, event_trials, temporal_split,
    GATE_PARAMS, OUT,
)

MODS = ["audio", "video", "env"]
# Detection design: audio+video are the behavioural detection pair; env is the
# slow-band CONTEXT (its growth/age trend normalizes the others via detrending),
# though env excursions are themselves detectable.
# Detector inputs: single modalities, naive concatenation, the primary
# audio+video OR-gate fusion, and the full OR-gate (adds env coverage).
SUBSETS = ["audio", "video", "env", "all(concat)", "AV(OR-gate)", "all(OR-gate)"]
OR_MODS = {"AV(OR-gate)": ["audio", "video"], "all(OR-gate)": ["audio", "video", "env"]}


def col_groups(order):
    g = {"audio": [i for i, f in enumerate(order) if f.startswith("aud_")],
         "video": [i for i, f in enumerate(order) if f.startswith("vid_")],
         "env":   [i for i, f in enumerate(order) if f.startswith("env_")]}
    g["all(concat)"] = list(range(len(order)))
    return g


def gate_closed_stream(scores, base):
    # clip: MaxFusion returns standardized scores that can be negative; the gate
    # consumes a non-negative "energy" residual.
    s = np.sqrt(np.clip(scores, 0, None))
    b = np.sqrt(np.clip(base, 0, None))
    resid = s - np.median(b)
    scale = robust_scale(b - np.median(b))
    return AlphaGate(scale=scale, **GATE_PARAMS).run(resid)["closed"]


def eval_targeted(det, Xtr_sub, Xpool_full, subset_cols, inject_cols, trials,
                  seed=0, timeline=240, n_sustained=3, sustained_len=(12, 30),
                  sustained_mag=2.5, n_spikes=5, spike_mag=10.0):
    """Inject the anomaly at inject_cols (a modality block); the detector only
    sees subset_cols. If those don't overlap, it structurally cannot detect."""
    rng = np.random.default_rng(seed)
    base = det.score(Xpool_full[:, subset_cols])
    n = len(Xpool_full)
    tp = fp = fn = 0
    for _ in range(trials):
        T = Xpool_full[rng.integers(0, n, size=timeline)].copy()
        slots = rng.permutation(np.arange(20, timeline - 40, 8))
        spans = []
        for s in slots[:n_sustained]:
            L = int(rng.integers(*sustained_len))
            T[s:s + L][:, inject_cols] += sustained_mag * rng.choice([-1., 1.], size=(1, len(inject_cols)))
            spans.append((s, s + L))
        for s in slots[n_sustained:n_sustained + n_spikes]:
            T[s][inject_cols] += spike_mag * rng.choice([-1., 1.], size=len(inject_cols))
        closed = gate_closed_stream(det.score(T[:, subset_cols]), base)
        spike_pts = slots[n_sustained:n_sustained + n_spikes]
        for a, b in spans:
            if closed[a:b].any(): tp += 1
            else: fn += 1
        for p in spike_pts:
            if closed[p:p + 2].any(): fp += 1
    _, r, f = prf(tp, fp, fn)
    return f, r


def eval_or_gate(mod_dets, groups, Xpool_full, inject_cols, trials,
                 seed=0, timeline=240, n_sustained=3, sustained_len=(12, 30),
                 sustained_mag=2.5, n_spikes=5, spike_mag=10.0):
    """Fused detection = OR of per-modality gate closures. Each modality is
    gated independently on its own clean baseline, so the anomalous modality
    is never masked by the others."""
    rng = np.random.default_rng(seed)
    bases = {m: mod_dets[m].score(Xpool_full[:, groups[m]]) for m in mod_dets}
    n = len(Xpool_full)
    tp = fp = fn = 0
    for _ in range(trials):
        T = Xpool_full[rng.integers(0, n, size=timeline)].copy()
        slots = rng.permutation(np.arange(20, timeline - 40, 8))
        spans = []
        for s in slots[:n_sustained]:
            L = int(rng.integers(*sustained_len))
            T[s:s + L][:, inject_cols] += sustained_mag * rng.choice([-1., 1.], size=(1, len(inject_cols)))
            spans.append((s, s + L))
        for s in slots[n_sustained:n_sustained + n_spikes]:
            T[s][inject_cols] += spike_mag * rng.choice([-1., 1.], size=len(inject_cols))
        closed = np.zeros(timeline, dtype=bool)
        for m, det in mod_dets.items():
            closed |= gate_closed_stream(det.score(T[:, groups[m]]), bases[m])
        for a, b in spans:
            if closed[a:b].any(): tp += 1
            else: fn += 1
        for p in slots[n_sustained:n_sustained + n_spikes]:
            if closed[p:p + 2].any(): fp += 1
    _, r, f = prf(tp, fp, fn)
    return f, r


def run_targeted(label, kind, which, args):
    _, order, Xall, dims = load(args.spine, args.features, which, "DETREND")
    Xtr, Xpool = temporal_split(Xall)
    groups = col_groups(order)
    # fit single-modality detectors + the naive concatenation detector
    dets = {}
    for sub in ["audio", "video", "env", "all(concat)"]:
        cols = groups[sub]
        if not cols:
            continue
        subdims = {k: sum(order[i].startswith(p) for i in cols)
                   for k, p in [("aud", "aud_"), ("vid", "vid_"), ("env", "env_")]}
        subdims = {k: v for k, v in subdims.items() if v > 0}
        dets[sub] = fit_detector(make_detector(kind, subdims, args.epochs, args.quick), Xtr[:, cols])
    mod_dets = {m: dets[m] for m in MODS if m in dets}   # reused by the OR-gate

    rows = []
    for sub in SUBSETS:
        for tgt in MODS:
            if not groups[tgt]:
                continue
            if sub in OR_MODS:
                use = {m: mod_dets[m] for m in OR_MODS[sub] if m in mod_dets}
                f, r = eval_or_gate(use, groups, Xpool, groups[tgt], max(120, args.trials // 2))
            else:
                f, r = eval_targeted(dets[sub], Xtr[:, groups[sub]], Xpool,
                                     groups[sub], groups[tgt], max(120, args.trials // 2))
            rows.append({"model": label, "detector_input": sub, "anomaly_target": tgt,
                         "f1_gate": round(f, 3), "recall_gate": round(r, 3)})
        print(f"[targeted] {label} detector={sub} done")
    return rows


def run_sweep(label, kind, which, args):
    _, order, Xall, dims = load(args.spine, args.features, which, "DETREND")
    Xtr, Xpool = temporal_split(Xall)
    det = fit_detector(make_detector(kind, dims, args.epochs, args.quick), Xtr)
    rows = []
    for mag in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        rec = event_trials(det, Xtr, Xpool, max(150, args.trials), sustained_mag=mag)
        _, ra, fa = prf(rec["tp_a"].sum(), rec["fp_a"].sum(), rec["fn_a"].sum())
        _, rg, fg = prf(rec["tp_g"].sum(), rec["fp_g"].sum(), rec["fn_g"].sum())
        rows.append({"model": label, "axis": "magnitude", "value": mag,
                     "f1_alone": round(fa, 3), "recall_alone": round(ra, 3),
                     "f1_gate": round(fg, 3), "recall_gate": round(rg, 3)})
    for dur in [(4, 6), (8, 12), (12, 18), (18, 26), (26, 36)]:
        rec = event_trials(det, Xtr, Xpool, max(150, args.trials),
                           sustained_mag=1.5, sustained_len=dur)
        _, ra, fa = prf(rec["tp_a"].sum(), rec["fp_a"].sum(), rec["fn_a"].sum())
        _, rg, fg = prf(rec["tp_g"].sum(), rec["fp_g"].sum(), rec["fn_g"].sum())
        rows.append({"model": label, "axis": "duration", "value": int(np.mean(dur)),
                     "f1_alone": round(fa, 3), "recall_alone": round(ra, 3),
                     "f1_gate": round(fg, 3), "recall_gate": round(rg, 3)})
    print(f"[sweep] {label} done")
    return rows


# ======================================================================
# Figures
# ======================================================================
def fig_targeted(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = list(df.model.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5.6 * len(models), 4.6), squeeze=False)
    for ax, mod in zip(axes[0], models):
        d = df[df.model == mod]
        M = d.pivot(index="detector_input", columns="anomaly_target", values="f1_gate")
        M = M.reindex(index=SUBSETS, columns=MODS)
        im = ax.imshow(M.values, cmap="Greens", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(MODS))); ax.set_xticklabels([f"anomaly in\n{m}" for m in MODS])
        ax.set_yticks(range(len(SUBSETS))); ax.set_yticklabels([f"{s} detector" for s in SUBSETS])
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="white" if v > 0.55 else "black", fontweight="bold")
        ax.set_title(f"{mod}: F1 by detector input × anomaly target", fontweight="bold", fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="F1 (with gate)")
    fig.suptitle("Modality-targeted anomalies: only the FUSED detector catches every target",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "fig_targeted.png", dpi=600); plt.close(fig)


def fig_sweep(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = {"Classical": "#1565c0", "DL": "#2e7d32"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, axis, xlabel in [(axes[0], "magnitude", "sustained magnitude (σ)"),
                             (axes[1], "duration", "sustained duration (h)")]:
        d = df[df.axis == axis]
        for mod in d.model.unique():
            dd = d[d.model == mod].sort_values("value")
            ax.plot(dd.value, dd.f1_alone, "o--", color=C.get(mod, "#333"), alpha=.55,
                    label=f"{mod} alone")
            ax.plot(dd.value, dd.f1_gate, "o-", color=C.get(mod, "#333"),
                    label=f"{mod} + gate")
        ax.set_xlabel(xlabel); ax.set_ylabel("F1"); ax.set_ylim(0, 1.02)
        ax.grid(alpha=.3); ax.legend(fontsize=8)
    axes[0].set_title("Sensitivity vs anomaly magnitude", fontweight="bold")
    axes[1].set_title("Sensitivity vs anomaly duration", fontweight="bold")
    fig.suptitle("Graduated difficulty sweep (recall comes off the ceiling)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "fig_sweep.png", dpi=600); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spine", default=str(RESULTS_DIR / "spine_room2_rich.csv"))
    ap.add_argument("--features", default=str(RESULTS_DIR / "recommended_features.csv"))
    ap.add_argument("--methods", default="both", choices=["classical", "dl", "both"])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    families = ([("Classical", "classical", "lean")] if args.methods in ("classical", "both") else []) + \
               ([("DL", "dl_late", "rich")] if args.methods in ("dl", "both") else [])

    tgt_rows, swp_rows = [], []
    for label, kind, which in families:
        tgt_rows += run_targeted(label, kind, which, args)
        swp_rows += run_sweep(label, kind, which, args)

    tgt = pd.DataFrame(tgt_rows); swp = pd.DataFrame(swp_rows)
    tgt.to_csv(OUT / "results_targeted.csv", index=False)
    swp.to_csv(OUT / "results_sweep.csv", index=False)
    fig_targeted(tgt); fig_sweep(swp)

    print("\n=== MODALITY-TARGETED F1 (with gate) ===")
    for mod in tgt.model.unique():
        print(f"\n{mod}:")
        print(tgt[tgt.model == mod].pivot(index="detector_input", columns="anomaly_target",
                                          values="f1_gate").reindex(index=SUBSETS, columns=MODS).to_string())
    print(f"\nWrote results_targeted.csv, results_sweep.csv, fig_targeted.png, fig_sweep.png in {OUT}")


if __name__ == "__main__":
    main()
