#!/usr/bin/env python3
"""
feature_selection.py — turn the 194 rich features into a rich-but-LEAN set.

Three filters, in order (see the thesis rationale: with only ~440 fusion hours,
every feature must earn its place or the DL model overfits):

  1. Variance filter  — drop near-constant / dead features.
  2. Within-modality redundancy — greedily drop any feature that is >|corr|
     threshold with an already-kept feature of the same modality (adjacent mel
     bands and neighbouring grid cells are highly collinear).
  3. Cross-modal ranking — on the both_lit fusion hours, score each survivor by
     its strongest correlation with the OTHER modality (and env). Features where
     the modalities inform each other are the ones fusion should keep.

Outputs: selected_features.csv (the lean set + why kept), a cross-modal top-pairs
table, and a correlation heatmap of the final set.

Usage
-----
    python src/pipeline/feature_selection.py \
        --spine results/spine_room2_rich.csv --corr-thresh 0.90
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import RESULTS_DIR


def modality_cols(df, prefix):
    cols = [c for c in df.columns if c.startswith(prefix)]
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


def greedy_dedup(df_rows, cols, thresh):
    """Keep a feature only if it's not >|thresh| correlated with an already-kept
    one. Order by descending variance so the most informative survive."""
    sub = df_rows[cols].dropna(how="all")
    var = sub.var().sort_values(ascending=False)
    ordered = [c for c in var.index if var[c] > 0]      # variance filter (drop dead)
    corr = sub[ordered].corr().abs()
    kept = []
    for c in ordered:
        if all(corr.loc[c, k] <= thresh for k in kept):
            kept.append(c)
    dropped_dead = [c for c in cols if c not in ordered]
    return kept, dropped_dead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spine", default=str(RESULTS_DIR / "spine_room2_rich.csv"))
    ap.add_argument("--corr-thresh", type=float, default=0.80)
    ap.add_argument("--cap-audio", type=int, default=10,
                    help="Max audio features in the final recommended set.")
    ap.add_argument("--cap-video", type=int, default=8)
    ap.add_argument("--cap-env", type=int, default=6)
    ap.add_argument("--out", default=str(RESULTS_DIR / "selected_features.csv"))
    args = ap.parse_args()

    m = pd.read_csv(args.spine, parse_dates=["time"])
    aud = modality_cols(m, "aud_")
    vid = modality_cols(m, "vid_")
    env = modality_cols(m, "env_")

    # rows where each modality is real
    aud_rows = m[m["has_audio"]]
    vid_rows = m[m["has_video_lit"]]
    fus = m[m["coverage_state"] == "both_lit"]   # both modalities live

    # --- filters 1+2 per modality ---
    aud_keep, aud_dead = greedy_dedup(aud_rows, aud, args.corr_thresh)
    vid_keep, vid_dead = greedy_dedup(vid_rows, vid, args.corr_thresh)
    env_keep, env_dead = greedy_dedup(m, env, args.corr_thresh)

    print(f"AUDIO : {len(aud)} -> {len(aud_keep)} kept "
          f"({len(aud_dead)} dead, {len(aud)-len(aud_keep)-len(aud_dead)} redundant)")
    print(f"VIDEO : {len(vid)} -> {len(vid_keep)} kept "
          f"({len(vid_dead)} dead, {len(vid)-len(vid_keep)-len(vid_dead)} redundant)")
    print(f"ENV   : {len(env)} -> {len(env_keep)} kept")

    # --- filter 3: cross-modal correlation on fusion hours ---
    xcorr = fus[aud_keep + vid_keep + env_keep].corr().abs()

    def top_cross(feat, other_cols):
        s = xcorr.loc[feat, other_cols].drop(labels=[feat], errors="ignore")
        return (s.idxmax(), round(float(s.max()), 3)) if len(s) else ("", 0.0)

    rows = []
    for f in aud_keep:
        partner, r = top_cross(f, vid_keep + env_keep)
        rows.append(("audio", f, partner, r))
    for f in vid_keep:
        partner, r = top_cross(f, aud_keep + env_keep)
        rows.append(("video", f, partner, r))
    for f in env_keep:
        partner, r = top_cross(f, aud_keep + vid_keep)
        rows.append(("env", f, partner, r))

    sel = pd.DataFrame(rows, columns=["modality", "feature",
                                      "top_cross_partner", "top_cross_corr"])
    sel = sel.sort_values(["modality", "top_cross_corr"], ascending=[True, False])
    sel.to_csv(args.out, index=False)

    # --- final recommended set: cap each modality by cross-modal relevance ---
    caps = {"audio": args.cap_audio, "video": args.cap_video, "env": args.cap_env}
    rec = pd.concat([sel[sel["modality"] == mod].head(k) for mod, k in caps.items()],
                    ignore_index=True)
    rec_path = str(Path(args.out).with_name("recommended_features.csv"))
    rec.to_csv(rec_path, index=False)

    print(f"\nDEDUP SET: {len(sel)} features "
          f"(audio={len(aud_keep)}, video={len(vid_keep)}, env={len(env_keep)})")
    print(f"RECOMMENDED LEAN SET: {len(rec)} features "
          f"(audio<={args.cap_audio}, video<={args.cap_video}, env<={args.cap_env})")
    print(rec.to_string(index=False))
    print("\nStrongest cross-modal audio<->video pairs (fusion hours):")
    av = xcorr.loc[aud_keep, vid_keep].stack().sort_values(ascending=False)
    seen = set(); shown = 0
    for (fa, fv), r in av.items():
        if shown >= 10:
            break
        print(f"  {fa:22} <-> {fv:18}  |r|={r:.3f}")
        shown += 1
    print(f"\nWrote lean feature set -> {args.out}")

    # --- heatmap of the lean set ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        feats = rec["feature"].tolist()   # compact recommended set only
        C = fus[feats].corr()
        side = min(max(6, len(feats) * 0.35), 12)   # bound canvas so 600dpi is safe
        fig, ax = plt.subplots(figsize=(side, side))
        im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(feats))); ax.set_xticklabels(feats, rotation=90, fontsize=6)
        ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats, fontsize=6)
        ax.set_title("Lean rich-feature correlation (fusion hours)")
        fig.colorbar(im, fraction=0.046, pad=0.04)
        fig.tight_layout()
        png = str(Path(args.out).with_name("selected_features_corr.png"))
        fig.savefig(png, dpi=600); plt.close(fig)
        print(f"Wrote heatmap          -> {png}")
    except ImportError:
        print("(matplotlib not available — skipped heatmap)")


if __name__ == "__main__":
    main()
