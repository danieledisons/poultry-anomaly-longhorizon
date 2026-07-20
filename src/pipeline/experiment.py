#!/usr/bin/env python3
"""Main experiment harness: classical vs DL, gate on/off, detrend on/off, modality and representation ablations, cross-room hook, and all the figures.

Run: python src/pipeline/experiment.py --methods both
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
from src.models.classical_detector import MahalanobisDetector
from src.pipeline.fusion_compare import robust_standardize, detrend_dayindex

GATE_PARAMS = dict(deadband=1.0, decay=0.85, close_threshold=6.0,
                   open_threshold=2.0, per_step_cap=2.0)
OUT = RESULTS_DIR / "experiment"


# ======================================================================
# Feature loading
# ======================================================================
def feature_list(cols, which, feature_csv):
    aud = [c for c in cols if c.startswith("aud_")]
    vid = [c for c in cols if c.startswith("vid_")]
    env = [c for c in cols if c.startswith("env_") and c != "env_day_index"]
    if which == "rich":
        return aud + vid + env
    lean = pd.read_csv(feature_csv)["feature"].tolist()
    return [f for f in lean if f != "env_day_index" and f in cols]


def load(spine_path, feature_csv, which, representation, modality="all"):
    m = pd.read_csv(spine_path, parse_dates=["time"])
    feats = feature_list(m.columns, which, feature_csv)
    feats = [f for f in feats if pd.api.types.is_numeric_dtype(m[f])]   # drop string cols (e.g. env_source_file)
    if modality != "all":
        want = {"audio": ("aud_",), "video": ("vid_",), "env": ("env_",),
                "av": ("aud_", "vid_")}[modality]
        feats = [f for f in feats if f.startswith(want)]
    fus = (m[m["coverage_state"] == "both_lit"].sort_values("time")
           .dropna(subset=feats).reset_index(drop=True))
    X = fus[feats].to_numpy(float)
    if representation == "DETREND":
        X = detrend_dayindex(X, fus["env_day_index"].to_numpy(float))
    X = robust_standardize(X)
    dims = {k: sum(f.startswith(p) for f in feats)
            for k, p in [("aud", "aud_"), ("vid", "vid_"), ("env", "env_")]}
    dims = {k: v for k, v in dims.items() if v > 0}
    # reorder columns to modality-grouped order (needed for the late-fusion split)
    order = ([f for f in feats if f.startswith("aud_")] +
             [f for f in feats if f.startswith("vid_")] +
             [f for f in feats if f.startswith("env_")])
    idx = [feats.index(f) for f in order]
    return fus, order, X[:, idx], dims


def make_detector(kind, dims, epochs, quick):
    if kind == "classical":
        return MahalanobisDetector()
    from src.models.dl_detector import AEDetector
    return AEDetector(dims, kind=("early" if kind == "dl_early" else "late"),
                      epochs=(40 if quick else epochs), verbose=False)


# ======================================================================
# Synthetic-injection evaluation (shared protocol)
# ======================================================================
def fit_detector(det, Xtr):
    try:
        det.fit(Xtr)          # classical
    except TypeError:
        det.fit(Xtr, None)
    return det


def gate_closed(scores, base_scores):
    """Run the alpha_t gate on a score stream, calibrated on clean scores."""
    resid = np.sqrt(scores) - np.median(np.sqrt(base_scores))
    scale = robust_scale(np.sqrt(base_scores) - np.median(np.sqrt(base_scores)))
    return AlphaGate(scale=scale, **GATE_PARAMS).run(resid)["closed"]


def event_trials(det, Xtr, Xpool, trials, seed=0, timeline=240,
                 n_sustained=3, sustained_len=(12, 30), sustained_mag=2.5,
                 n_spikes=5, spike_mag=10.0):
    """Per-trial event-level counts for DL-alone (threshold) and DL+gate."""
    rng = np.random.default_rng(seed)
    # Calibrate threshold + gate on CLEAN hours from the evaluation distribution,
    # not the training set — the autoencoder underestimates its own training
    # error, which otherwise saturates the gate closed on every test hour.
    base = det.score(Xpool)
    thr = np.quantile(base, 0.99)
    n = len(Xpool); d = Xpool.shape[1]
    rec = {k: [] for k in ("tp_a", "fp_a", "fn_a", "tp_g", "fp_g", "fn_g", "lat")}
    for _ in range(trials):
        T = Xpool[rng.integers(0, n, size=timeline)].copy()
        slots = rng.permutation(np.arange(20, timeline - 40, 8))
        spans, spikes = [], []
        for s in slots[:n_sustained]:
            L = int(rng.integers(*sustained_len))
            T[s:s + L] += sustained_mag * rng.choice([-1., 1.], size=(1, d))
            spans.append((s, s + L))
        for s in slots[n_sustained:n_sustained + n_spikes]:
            T[s] += spike_mag * rng.choice([-1., 1.], size=d); spikes.append(s)
        sc = det.score(T)
        alone = sc > thr
        closed = gate_closed(sc, base)
        ta = fa = na = tg = fg = ng = 0
        for a, b in spans:
            if alone[a:b].any(): ta += 1
            else: na += 1
            if closed[a:b].any():
                tg += 1
                rec["lat"].append(int(np.argmax(closed[a:b])))
            else: ng += 1
        for p in spikes:
            if alone[p:p + 2].any(): fa += 1
            if closed[p:p + 2].any(): fg += 1
        for k, v in [("tp_a", ta), ("fp_a", fa), ("fn_a", na),
                     ("tp_g", tg), ("fp_g", fg), ("fn_g", ng)]:
            rec[k].append(v)
    return {k: np.array(v) for k, v in rec.items()}


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def bootstrap_prf(tp, fp, fn, B=2000, seed=0):
    """95% CI on P/R/F1 by resampling trials."""
    rng = np.random.default_rng(seed)
    n = len(tp)
    fs, ps, rs = [], [], []
    for _ in range(B):
        i = rng.integers(0, n, size=n)
        p, r, f = prf(tp[i].sum(), fp[i].sum(), fn[i].sum())
        ps.append(p); rs.append(r); fs.append(f)
    def ci(a): return (round(np.percentile(a, 2.5), 3), round(np.percentile(a, 97.5), 3))
    P, R, F = prf(tp.sum(), fp.sum(), fn.sum())
    return dict(precision=round(P, 3), recall=round(R, 3), f1=round(F, 3),
                f1_lo=ci(fs)[0], f1_hi=ci(fs)[1],
                prec_lo=ci(ps)[0], prec_hi=ci(ps)[1])


def injection_auc(det, Xtr, Xte, mags, seed=0):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    rows = []
    for m in mags:
        Xp = Xte + m * rng.choice([-1., 1.], size=Xte.shape)
        y = np.r_[np.zeros(len(Xte)), np.ones(len(Xp))]
        s = np.r_[det.score(Xte), det.score(Xp)]
        rows.append((m, roc_auc_score(y, s)))
    return rows


# ======================================================================
# Runner
# ======================================================================
def temporal_split(X, frac=0.7):
    cut = int(len(X) * frac)
    return X[:cut], X[cut:]


def run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    methods = {"classical": "both", "dl": "both", "both": "both"}[args.methods]
    do_classical = args.methods in ("classical", "both")
    do_dl = args.methods in ("dl", "both")
    eval_spine = args.eval_spine or args.spine
    cross = args.eval_spine is not None
    mags = [1, 2, 3, 4, 6]

    main_rows, mag_rows, oper_rows = [], [], []
    fitted = {}   # keep detectors + data for later figures

    def do_method(label, kind, which):
        _, order, Xtr_full, dims = load(args.spine, args.features, which, "DETREND")
        _, _, Xev_full, _ = load(eval_spine, args.features, which, "DETREND")
        Xtr, Xte_self = temporal_split(Xtr_full)
        Xpool = Xev_full if cross else Xte_self
        det = fit_detector(make_detector(kind, dims, args.epochs, args.quick), Xtr)
        rec = event_trials(det, Xtr, Xpool, args.trials)
        for tag, (tp, fp, fn) in {
            f"{label}": (rec["tp_a"], rec["fp_a"], rec["fn_a"]),
            f"{label}+Gate": (rec["tp_g"], rec["fp_g"], rec["fn_g"]),
        }.items():
            main_rows.append({"method": tag, **bootstrap_prf(tp, fp, fn)})
        for m, a in injection_auc(det, Xtr, Xte_self, mags):
            mag_rows.append({"method": label, "magnitude": m, "auc": round(a, 3)})
        oper_rows.append({"method": label,
                          "median_latency_h": float(np.median(rec["lat"])) if len(rec["lat"]) else np.nan,
                          "spike_FP_per_trial_alone": round(rec["fp_a"].mean(), 3),
                          "spike_FP_per_trial_gate": round(rec["fp_g"].mean(), 3)})
        fitted[label] = dict(det=det, Xtr=Xtr, Xpool=Xpool, order=order, dims=dims)
        print(f"[done] {label}")

    if do_classical:
        do_method("Classical", "classical", "lean")
    if do_dl:
        do_method("DL", "dl_late", "rich")

    pd.DataFrame(main_rows).to_csv(OUT / "results_main.csv", index=False)
    pd.DataFrame(mag_rows).to_csv(OUT / "results_magnitude.csv", index=False)
    pd.DataFrame(oper_rows).to_csv(OUT / "results_operating.csv", index=False)

    # ---- modality ablation (F1 with gate) ----
    mod_rows = []
    for label, kind, which in ([("Classical", "classical", "lean")] if do_classical else []) + \
                              ([("DL", "dl_late", "rich")] if do_dl else []):
        for mod in ["audio", "video", "env", "av", "all"]:
            _, order, X, dims = load(args.spine, args.features, which, "DETREND", modality=mod)
            if X.shape[1] == 0:
                continue
            Xtr, Xte = temporal_split(X)
            det = fit_detector(make_detector(kind, dims, args.epochs, args.quick), Xtr)
            rec = event_trials(det, Xtr, Xte, max(100, args.trials // 2))
            _, _, f = prf(rec["tp_g"].sum(), rec["fp_g"].sum(), rec["fn_g"].sum())
            mod_rows.append({"method": label, "modality": mod, "f1_gate": round(f, 3),
                             "n_features": X.shape[1]})
        print(f"[done] modality ablation {label}")
    pd.DataFrame(mod_rows).to_csv(OUT / "results_modality.csv", index=False)

    # ---- representation ablation (RAW/DETREND x lean/rich) ----
    rep_rows = []
    combos = ([("classical", "lean", "Classical")] if do_classical else []) + \
             ([("dl_late", "rich", "DL")] if do_dl else []) + \
             ([("classical", "rich", "Classical")] if do_classical else []) + \
             ([("dl_late", "lean", "DL")] if do_dl else [])
    for kind, which, label in combos:
        for rep in ["RAW", "DETREND"]:
            _, order, X, dims = load(args.spine, args.features, which, rep)
            Xtr, Xte = temporal_split(X)
            det = fit_detector(make_detector(kind, dims, args.epochs, args.quick), Xtr)
            rec = event_trials(det, Xtr, Xte, max(100, args.trials // 2))
            _, _, f = prf(rec["tp_g"].sum(), rec["fp_g"].sum(), rec["fn_g"].sum())
            rep_rows.append({"method": label, "features": which, "representation": rep,
                             "f1_gate": round(f, 3)})
        print(f"[done] representation ablation {label}-{which}")
    pd.DataFrame(rep_rows).to_csv(OUT / "results_representation.csv", index=False)

    make_figures(main_rows, mag_rows, mod_rows, rep_rows, fitted, cross)
    print(f"\nAll outputs in {OUT}")


# ======================================================================
# Figures (publication style, 600 dpi)
# ======================================================================
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25, "figure.dpi": 120})
    return plt


def make_figures(main_rows, mag_rows, mod_rows, rep_rows, fitted, cross):
    plt = _style()
    C = {"DL": "#2e7d32", "Classical": "#1565c0", "gate": "#e07b39"}
    tag = "cross-room" if cross else "temporal holdout"

    # ---- fig_main: 4-way P/R/F1 with CI ----
    dfm = pd.DataFrame(main_rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    order = [m for m in ["Classical", "Classical+Gate", "DL", "DL+Gate"] if m in set(dfm.method)]
    x = np.arange(len(order)); w = 0.26
    for i, (met, key) in enumerate([("precision", "prec"), ("recall", None), ("f1", "f1")]):
        vals = [dfm[dfm.method == m][met].values[0] for m in order]
        err = None
        if key and f"{key}_lo" in dfm:
            lo = [dfm[dfm.method == m][f"{key}_lo"].values[0] for m in order]
            hi = [dfm[dfm.method == m][f"{key}_hi"].values[0] for m in order]
            err = [np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)]
        ax.bar(x + (i - 1) * w, vals, w, yerr=err, capsize=3,
               label=met.upper(), color=["#9e9e9e", "#607d8b", "#2e7d32"][i])
    ax.set_xticks(x); ax.set_xticklabels(order); ax.set_ylim(0, 1.05)
    ax.set_ylabel("score"); ax.legend(ncol=3, loc="upper left")
    ax.set_title(f"Detector comparison with 95% CI ({tag})", fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "fig_main.png", dpi=600); plt.close(fig)

    # ---- fig_magnitude ----
    dfa = pd.DataFrame(mag_rows)
    if len(dfa):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for met in dfa.method.unique():
            d = dfa[dfa.method == met]
            ax.plot(d.magnitude, d.auc, "o-", label=met, color=C.get(met, "#333"))
        ax.axhline(0.5, color="gray", ls="--", lw=.7); ax.set_ylim(.45, 1.02)
        ax.set_xlabel("injection magnitude (σ)"); ax.set_ylabel("per-hour ROC-AUC")
        ax.legend(); ax.set_title("Detection vs anomaly magnitude", fontweight="bold")
        fig.tight_layout(); fig.savefig(OUT / "fig_magnitude.png", dpi=600); plt.close(fig)

    # ---- fig_modality ----
    dmo = pd.DataFrame(mod_rows)
    if len(dmo):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        mods = ["audio", "video", "env", "av", "all"]
        methods = list(dmo.method.unique()); w = 0.8 / max(1, len(methods))
        for j, met in enumerate(methods):
            d = dmo[dmo.method == met].set_index("modality").reindex(mods)
            ax.bar(np.arange(len(mods)) + j * w, d["f1_gate"].values, w,
                   label=met, color=C.get(met, "#333"))
        ax.set_xticks(np.arange(len(mods)) + w * (len(methods) - 1) / 2)
        ax.set_xticklabels(["audio", "video", "env", "audio+video", "all"])
        ax.set_ylabel("F1 (with gate)"); ax.set_ylim(0, 1.05); ax.legend()
        ax.set_title("Modality ablation — does fusion help?", fontweight="bold")
        fig.tight_layout(); fig.savefig(OUT / "fig_modality.png", dpi=600); plt.close(fig)

    # ---- fig_representation ----
    dre = pd.DataFrame(rep_rows)
    if len(dre):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        dre["cfg"] = dre["method"] + "-" + dre["features"]
        cfgs = dre.cfg.unique(); w = 0.35
        for i, rep in enumerate(["RAW", "DETREND"]):
            d = dre[dre.representation == rep].set_index("cfg").reindex(cfgs)
            ax.bar(np.arange(len(cfgs)) + i * w, d["f1_gate"].values, w,
                   label=rep, color=["#9e9e9e", "#2e7d32"][i])
        ax.set_xticks(np.arange(len(cfgs)) + w / 2); ax.set_xticklabels(cfgs, rotation=15)
        ax.set_ylabel("F1 (with gate)"); ax.set_ylim(0, 1.05); ax.legend()
        ax.set_title("Representation ablation — RAW vs DETREND, lean vs rich", fontweight="bold")
        fig.tight_layout(); fig.savefig(OUT / "fig_representation.png", dpi=600); plt.close(fig)

    # ---- fig_trace + fig_operating (from one detector) ----
    key = "DL" if "DL" in fitted else ("Classical" if "Classical" in fitted else None)
    if key:
        trace_and_operating(plt, fitted[key], key)


def trace_and_operating(plt, F, label):
    det, Xtr, Xpool = F["det"], F["Xtr"], F["Xpool"]
    rng = np.random.default_rng(7)
    d = Xpool.shape[1]; timeline = 240
    T = Xpool[rng.integers(0, len(Xpool), size=timeline)].copy()
    base = det.score(Xpool); thr = np.quantile(base, 0.99)   # calibrate on clean eval hours
    spans = [(60, 90)]; spikes = [140, 180]
    for a, b in spans:
        T[a:b] += 2.5 * rng.choice([-1., 1.], size=(1, d))
    for p in spikes:
        T[p] += 10.0 * rng.choice([-1., 1.], size=d)
    sc = det.score(T)
    closed = gate_closed(sc, base)

    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1]})
    ax[0].plot(sc, color="#37474f", lw=1.1, label="anomaly score")
    ax[0].axhline(thr, color="#c62828", ls="--", lw=.8, label="DL-alone threshold")
    for a, b in spans:
        ax[0].axvspan(a, b, color="#2e7d32", alpha=.15, label="sustained (true)")
    for p in spikes:
        ax[0].axvline(p, color="#e07b39", lw=1.2, ls=":", label="spike (benign)")
    h, l = ax[0].get_legend_handles_labels()
    ax[0].legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(), fontsize=8, ncol=2)
    ax[0].set_ylabel("score"); ax[0].set_title(f"{label}: score, threshold, gate", fontweight="bold")
    ax[1].fill_between(range(timeline), 0, closed.astype(float), color="#c62828", alpha=.3)
    ax[1].set_ylabel("gate\nclosed"); ax[1].set_yticks([0, 1]); ax[1].set_xlabel("hour")
    fig.tight_layout(); fig.savefig(OUT / "fig_trace.png", dpi=600); plt.close(fig)

    # operating: latency distribution + spike FP alone vs gate over many trials
    rec = event_trials(det, Xtr, Xpool, 300)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    if len(rec["lat"]):
        ax[0].hist(rec["lat"], bins=range(0, 20), color="#2e7d32", alpha=.8)
    ax[0].set_xlabel("hours to gate closure"); ax[0].set_ylabel("count")
    ax[0].set_title("Detection latency (sustained events)", fontweight="bold")
    ax[1].bar(["DL alone", "DL + gate"],
              [rec["fp_a"].mean(), rec["fp_g"].mean()], color=["#9e9e9e", "#2e7d32"])
    ax[1].set_ylabel("benign-spike false alarms / trial")
    ax[1].set_title("False alarms on benign spikes", fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "fig_operating.png", dpi=600); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spine", default=str(RESULTS_DIR / "spine_room2_rich.csv"))
    ap.add_argument("--eval-spine", default=None, help="Cross-room: evaluate on this spine (fit on --spine).")
    ap.add_argument("--features", default=str(RESULTS_DIR / "recommended_features.csv"))
    ap.add_argument("--methods", default="both", choices=["classical", "dl", "both"])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--quick", action="store_true", help="Fast DL (40 epochs) for smoke tests.")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
