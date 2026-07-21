#!/usr/bin/env python3
"""
The Approach-C ablation summarised as a trilemma tradeoff plot.

Reads the per-room config summaries and plots detection (injection retention) vs
false-alarm rate for every mechanism. The 'good' corner is top-left (high
detection, low false alarm). The point of the figure: NO simple gating mechanism
reaches it -- adaptation kills detection, freezing kills the stationary null.
This defines the remaining contribution (a properly designed directional /
learned-dynamics slow state), rather than claiming a premature win.

Usage: python validation_new/approach_c/tradeoff_plot.py
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
LABEL = {"fixed": "fixed cov (#1)", "online_ungated": "adapt both (ungated)",
         "online_gated": "freeze both (naive gate)", "robust": "soft robust",
         "gated_floor": "freeze mean / floor cov", "meangate_covfree": "freeze mean / free cov",
         "directional": "directional (naive)", "predictive": "predictive velocity",
         "hybrid": "predmean + robust cov", "stable_directional": "STABLE directional (C)"}
COL = {"fixed": "tab:red", "online_ungated": "tab:blue", "online_gated": "tab:green",
       "robust": "tab:purple", "gated_floor": "tab:orange", "meangate_covfree": "tab:brown",
       "directional": "tab:pink", "predictive": "tab:gray", "hybrid": "tab:olive",
       "stable_directional": "black"}
import sys
ITYPE = sys.argv[1] if len(sys.argv) > 1 else "step"

fig, ax = plt.subplots(figsize=(8.5, 6.5))
for R, mk in [("2", "o"), ("6", "s")]:
    f = HERE / "csv" / f"approachc_room{R}_{ITYPE}_summary.csv"
    if not f.exists():
        continue
    df = pd.read_csv(f)
    for _, row in df.iterrows():
        big = row.config == "stable_directional"
        ax.scatter(min(row.late_false_alarm, 1.0), row.inj_retention, marker=mk,
                   s=260 if big else 120, color=COL.get(row.config, "k"),
                   edgecolor="k", lw=1.5 if big else 0.8, zorder=4 if big else 3,
                   label=f"{LABEL.get(row.config, row.config)}" if R == "2" else None)
        ax.annotate(f"R{R}", (min(row.late_false_alarm, 1.0), row.inj_retention),
                    fontsize=6, xytext=(4, 4), textcoords="offset points")

ax.axhspan(0.8, 1.05, xmin=0, xmax=0.15, color="green", alpha=0.06)
ax.annotate("desired:\nhigh detection,\nlow false alarm", (0.02, 0.86), fontsize=9, color="green")
ax.set_xlabel("late-cycle false-alarm rate  (lower is better)")
ax.set_ylabel("injection detection retention  (higher is better)")
ax.set_title(f"Approach-C trilemma ({ITYPE} anomaly): detection vs false alarm\n"
             "STABLE directional (black) is the first to move toward the good corner")
ax.set_xlim(-0.03, 1.0); ax.set_ylim(-0.05, 1.08)
ax.legend(fontsize=7, loc="center right", title="mechanism (circle=R2, square=R6)")
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(HERE / "figs" / f"fig_approachc_tradeoff_{ITYPE}.png", dpi=140)
print("wrote", HERE / "figs" / f"fig_approachc_tradeoff_{ITYPE}.png")
