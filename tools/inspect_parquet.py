#!/usr/bin/env python3
"""Peek at a parquet file's schema and first rows.

Run: python tools/inspect_parquet.py
"""
import sys
import pandas as pd
import numpy as np

path = sys.argv[1]
df = pd.read_parquet(path)

print("="*70)
print(f"FILE: {path}")
print(f"shape: {df.shape[0]} rows (days) x {df.shape[1]} cols")
print("="*70)

# --- metadata / non-embedding columns ---
meta_cols = [c for c in df.columns if not c.startswith(("emb_mean_", "emb_std_"))]
print("\nMETADATA COLUMNS:", meta_cols)
print("\nPER-DAY SUMMARY:")
show = [c for c in ["date", "room", "month", "n_files", "n_windows"] if c in df.columns]
print(df[show].to_string(index=False))

# --- coverage ---
print("\nCOVERAGE:")
print(f"  date range : {df['date'].min()} -> {df['date'].max()}")
print(f"  n days     : {len(df)}")
print(f"  total files: {df['n_files'].sum() if 'n_files' in df else 'n/a'}")
print(f"  total windows: {df['n_windows'].sum() if 'n_windows' in df else 'n/a'}")

# --- embedding sanity ---
emb_mean_cols = [c for c in df.columns if c.startswith("emb_mean_")]
print(f"\nEMBEDDING CHECK ({len(emb_mean_cols)} mean dims):")
E = df[emb_mean_cols].values
print(f"  value range : {E.min():.3f} to {E.max():.3f}")
print(f"  any NaN?    : {np.isnan(E).any()}")
print(f"  any all-zero rows? : {(np.abs(E).sum(axis=1) == 0).any()}")
# day-to-day variation: are embeddings actually different across days?
if len(df) > 1:
    day_std = E.std(axis=0).mean()
    print(f"  mean across-day std per dim: {day_std:.4f}  (near-0 = suspiciously identical days)")

# --- quality csv if present ---
qpath = path.replace(".parquet", "_quality.csv")
try:
    q = pd.read_csv(qpath)
    print(f"\nQUALITY LOG ({qpath}): {len(q)} file rows")
    if "clipped" in q:
        print(f"  clipped files: {q['clipped'].sum()} / {len(q)}")
    if "issue" in q:
        issues = q["issue"].dropna()
        if len(issues):
            print(f"  issues logged:\n{issues.value_counts().to_string()}")
except FileNotFoundError:
    print(f"\n(no quality csv found at {qpath})")