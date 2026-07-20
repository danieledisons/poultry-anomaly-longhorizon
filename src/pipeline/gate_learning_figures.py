#!/usr/bin/env python3
"""
gate_learning_figures.py — visualize, as trend lines, HOW an anomaly corrupts an
online detector's notion of "normal", and how the alpha_t gate prevents it.

Two figures (RESULTS_DIR/gate_learning/):

  fig_baseline_drift.png
      The model's learned "normal" (its baseline, projected onto the anomaly
      axis) plotted over time. 0 = the true clean normal, 1 = the model has fully
      adopted the anomaly as normal. Without the gate the baseline CLIMBS toward
      1 (learns the wrong thing); with the gate it stays near 0 (protected).

  fig_spike_frequency.png
      Repeated BRIEF spikes (the gate is meant to ignore these as alarms). As the
      spikes get more frequent, an ungated detector still slowly LEARNS them into
      its baseline; the gate engages once they persist and keeps the baseline clean.
      Trend: final baseline corruption vs spike frequency, ungated vs gated.

Both reuse the online detector from contamination_experiment.py.

Usage
-----
    python src/pipeline/gate_learning_figures.py --trials 300
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import RESULTS_DIR
from src.pipeline.contamination_experiment import load_features, run_online

OUT = RESULTS_DIR / "gate_learning"
GREY, GREEN, PURPLE, RED = "#616161", "#2e7d32", "#8e24aa", "#c62828"


def _calib(X, rng):
    mu0 = X[rng.integers(0, len(X), 200)].mean(0)
    dists = np.array([np.linalg.norm(X[i] - mu0) for i in rng.integers(0, len(X), 500)])
    return mu0, dists.mean(), dists.std() + 1e-9


# ======================================================================
# 1. Baseline-drift trend (single sustained anomaly)
# ======================================================================
def baseline_drift(X, trials, timeline=320, warm=60, anomaly=(80, 300), mag=7.0, lam=0.06):
    a0, a1 = anomaly
    curves = {False: [], True: []}
    rng = np.random.default_rng(0)
    for _ in range(trials):
        d = X.shape[1]
        T = X[rng.integers(0, len(X), timeline)].copy()
        direction = rng.choice([-1., 1.], d); direction /= np.linalg.norm(direction)
        mu0, center, scale = _calib(X, rng)
        base0 = mu0 @ direction
        T[a0:a1] += mag * direction
        for gated in (False, True):
            _, mn, _ = run_online(T, mu0, center, scale, lam, gated, direction, warm)
            curves[gated].append((mn - base0) / mag)     # fraction of anomaly absorbed
    return {g: np.array(v) for g, v in curves.items()}, (a0, a1)


def fig_baseline_drift(curves, span, plt):
    a0, a1 = span
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.axhspan(0.9, 1.05, color=RED, alpha=.06)
    ax.axhline(1.0, color=RED, ls=":", lw=1.2)
    ax.axhline(0.0, color="#1565c0", ls=":", lw=1.2)
    ax.axvspan(a0, a1, color=PURPLE, alpha=.10)
    x = np.arange(curves[False].shape[1])
    for gated, color, name in [(False, GREY, "Ungated (adapts online)"),
                               (True, GREEN, "Gated (alpha_t freezes learning)")]:
        m = curves[gated].mean(0); lo = np.percentile(curves[gated], 2.5, 0); hi = np.percentile(curves[gated], 97.5, 0)
        ax.plot(x, m, color=color, lw=2.2, label=name)
        ax.fill_between(x, lo, hi, color=color, alpha=.15)
    ax.text(a1 - 2, 1.02, "model fully believes the anomaly is normal", color=RED, ha="right", fontsize=9)
    ax.text(5, 0.03, "true normal", color="#1565c0", fontsize=9)
    ax.text((a0 + a1) / 2, -0.14, "persistent anomaly present", color=PURPLE, ha="center", fontsize=9)
    ax.set_xlabel("hour"); ax.set_ylabel("fraction of anomaly absorbed into 'normal'")
    ax.set_ylim(-0.2, 1.12); ax.legend(loc="center right")
    ax.set_title("What the detector believes is 'normal' drifts toward the anomaly — unless the gate freezes learning",
                 fontsize=12, fontweight="normal")
    fig.tight_layout(); fig.savefig(OUT / "fig_baseline_drift.png", dpi=600); plt.close(fig)


# ======================================================================
# 2. Frequent brief spikes -> slow corruption of the baseline
# ======================================================================
def spike_frequency(X, trials, intervals=(48, 24, 16, 12, 8, 6, 4, 3),
                    timeline=480, warm=60, spike_mag=8.0, spike_len=1, lam=0.06):
    res = {False: {"m": [], "lo": [], "hi": [], "f": []},
           True: {"m": [], "lo": [], "hi": [], "f": []}}
    rng = np.random.default_rng(1)
    for I in intervals:
        freq = 24.0 / I                                   # spikes per day
        vals = {False: [], True: []}
        for _ in range(trials):
            d = X.shape[1]
            T = X[rng.integers(0, len(X), timeline)].copy()
            direction = rng.choice([-1., 1.], d); direction /= np.linalg.norm(direction)
            mu0, center, scale = _calib(X, rng)
            base0 = mu0 @ direction
            for s in range(warm + 20, timeline - spike_len, I):
                T[s:s + spike_len] += spike_mag * direction
            for gated in (False, True):
                _, mn, _ = run_online(T, mu0, center, scale, lam, gated, direction, warm)
                vals[gated].append(float(np.mean((mn[timeline // 2:] - base0)) / spike_mag))
        for gated in (False, True):
            a = np.array(vals[gated])
            res[gated]["m"].append(a.mean()); res[gated]["f"].append(freq)
            res[gated]["lo"].append(np.percentile(a, 2.5)); res[gated]["hi"].append(np.percentile(a, 97.5))
    return res


def fig_spike_frequency(res, plt):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for gated, color, name in [(False, GREY, "Ungated (learns the spikes)"),
                               (True, GREEN, "Gated (baseline protected)")]:
        r = res[gated]
        f = np.array(r["f"]); m = np.array(r["m"])
        ax.plot(f, m, "o-", color=color, lw=2.2, label=name)
        ax.fill_between(f, r["lo"], r["hi"], color=color, alpha=.15)
    ax.set_xlabel("brief-spike frequency (spikes per day)")
    ax.set_ylabel("baseline corruption (fraction of spike absorbed)")
    ax.set_ylim(bottom=-0.02); ax.legend()
    ax.set_title("Frequent brief spikes slowly poison an online baseline; the gate resists once they persist",
                 fontsize=12, fontweight="normal")
    fig.tight_layout(); fig.savefig(OUT / "fig_spike_frequency.png", dpi=600); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spine", default=str(RESULTS_DIR / "spine_room2_rich.csv"))
    ap.add_argument("--features", default=str(RESULTS_DIR / "recommended_features.csv"))
    ap.add_argument("--trials", type=int, default=300)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.3, "legend.frameon": False})
    X = load_features(a.spine, a.features)

    curves, span = baseline_drift(X, a.trials)
    fig_baseline_drift(curves, span, plt)
    print(f"[baseline drift] ungated final={curves[False][:, -60:].mean():.2f}  "
          f"gated final={curves[True][:, -60:].mean():.2f}  (1.0 = fully learned anomaly)")

    res = spike_frequency(X, max(150, a.trials // 2))
    fig_spike_frequency(res, plt)
    print(f"[spike freq] ungated corruption @max freq={res[False]['m'][-1]:.2f}  "
          f"gated={res[True]['m'][-1]:.2f}")
    print(f"\nWrote fig_baseline_drift.png, fig_spike_frequency.png in {OUT}")


if __name__ == "__main__":
    main()
