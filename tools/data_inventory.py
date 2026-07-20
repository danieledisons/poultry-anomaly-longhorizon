#!/usr/bin/env python3
"""data_inventory.py - completeness / missingness inventory for the Room 2 features.
"""
import argparse, pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def span_stats(t):
    t = pd.to_datetime(t).dropna().sort_values()
    if len(t)==0: return None
    start, end = t.min(), t.max()
    expected = int((end - start)/pd.Timedelta(hours=1)) + 1
    present = t.dt.floor('h').nunique()
    return dict(start=start, end=end, days=(end-start).days,
                present_hours=present, expected_hours=expected,
                missing_pct=round(100*(1-present/expected),1))


def make_figure(m, inv, out_png, dpi=600):
    """Coverage timeline (where each modality is present) + missingness bar."""
    spine = pd.date_range(m['time'].min().floor('h'), m['time'].max().ceil('h'), freq='h')
    vp = m['vid_row_present'].fillna(False) if 'vid_row_present' in m else m['vid_flow_mean_avg'].notna()
    present = {
        "Video (lit)":   m.loc[m['vid_flow_mean_avg'].notna(),'time'].dt.floor('h'),
        "Video (dark)":  m.loc[vp & m['vid_flow_mean_avg'].isna(),'time'].dt.floor('h'),
        "Audio":         m.loc[m['aud_rms_db_mean'].notna(),'time'].dt.floor('h'),
        "Environment":   m.loc[m['env_temp_day_mean_c'].notna(),'time'].dt.floor('h'),
    }
    fused = m[m['vid_flow_mean_avg'].notna() & m['aud_rms_db_mean'].notna() & m['env_temp_day_mean_c'].notna()]['time'].dt.floor('h')
    present["Fused (all 3)"] = fused
    colors = {"Video (lit)":"#1f77b4","Video (dark)":"#9e9e9e","Audio":"#ff7f0e","Environment":"#2ca02c","Fused (all 3)":"#c62828"}
    order = ["Video (lit)","Video (dark)","Audio","Environment","Fused (all 3)"]

    fig, ax = plt.subplots(2,1, figsize=(9,4.2), gridspec_kw={'height_ratios':[3,1.4]})
    # coverage strips
    for i,name in enumerate(order):
        hrs = set(present[name]); flags = np.array([t in hrs for t in spine])
        # build contiguous runs
        runs=[]; start=None
        for j,fl in enumerate(flags):
            if fl and start is None: start=j
            if (not fl) and start is not None: runs.append((spine[start], spine[j-1])); start=None
        if start is not None: runs.append((spine[start], spine[-1]))
        for a,b in runs:
            ax[0].barh(i, (b-a)/np.timedelta64(1,'D')+1/24, left=a, height=0.62, color=colors[name])
    ax[0].set_yticks(range(len(order))); ax[0].set_yticklabels(order, fontsize=8)
    ax[0].invert_yaxis(); ax[0].set_title("Data coverage timeline (Room 2)", fontsize=10, fontweight='bold')
    ax[0].tick_params(axis='x', labelsize=7); ax[0].grid(axis='x', alpha=.25)

    # missingness bar
    b = inv[~inv['modality'].isin(["Video (all recorded)"])].copy()
    ax[1].bar(range(len(b)), b['missingness_pct'], color=[ '#888' for _ in range(len(b))])
    ax[1].set_xticks(range(len(b))); ax[1].set_xticklabels([x.replace(' ','\n') for x in b['modality']], fontsize=7)
    ax[1].set_ylabel("missing %", fontsize=8); ax[1].set_ylim(0,100)
    for j,v in enumerate(b['missingness_pct']): ax[1].text(j, v+2, f"{v:.0f}", ha='center', fontsize=7)
    ax[1].set_title("Missingness by modality", fontsize=9)
    fig.tight_layout(); fig.savefig(out_png, dpi=dpi, bbox_inches='tight')
    print(f"[write] {out_png}  ({dpi} dpi)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", default="./room2_merged_hourly.csv")
    ap.add_argument("--out", default="./data_inventory.csv")
    ap.add_argument("--fig", default="./data_inventory_coverage.png")
    ap.add_argument("--dpi", type=int, default=600)
    a = ap.parse_args()
    m = pd.read_csv(a.merged); m['time']=pd.to_datetime(m['time'])
    fused = m[m['vid_flow_mean_avg'].notna() & m['aud_rms_db_mean'].notna() & m['env_temp_day_mean_c'].notna()]
    n_fused = len(fused)

    rows=[]
    vid_present = m['vid_row_present'].fillna(False) if 'vid_row_present' in m else m['vid_flow_mean_avg'].notna()
    vid_lit  = m['vid_flow_mean_avg'].notna()
    vid_dark = vid_present & m['vid_flow_mean_avg'].isna()
    defs = {
      "Video (all recorded)": m.loc[vid_present,'time'],
      "Video (lit / usable)": m.loc[vid_lit,'time'],
      "Video (dark / unlit)": m.loc[vid_dark,'time'],
      "Audio":                m.loc[m['aud_rms_db_mean'].notna(),'time'],
      "Environment":          m.loc[m['env_temp_day_mean_c'].notna(),'time'],
      "Fused (all 3)":        fused['time'],
    }
    for name, t in defs.items():
        st = span_stats(t)
        if st is None: continue
        rows.append({"modality":name,
                     "date_range":f"{st['start'].date()} to {st['end'].date()}",
                     "duration_days":st['days'],
                     "hours_present":st['present_hours'],
                     "hours_expected":st['expected_hours'],
                     "missingness_pct":st['missing_pct'],
                     "share_of_fused_pct":round(100*st['present_hours']/max(n_fused,1),1) if name!="Fused (all 3)" else 100.0})
    inv=pd.DataFrame(rows)
    pd.set_option('display.width',160,'display.max_columns',20)
    print("\n=== ROOM 2 DATA INVENTORY (aligned hourly table) ===")
    print(inv.to_string(index=False))
    print(f"\nFused window (all three modalities): {n_fused} hours")
    inv.to_csv(a.out,index=False); print(f"[write] {a.out}")
    make_figure(m, inv, a.fig, dpi=a.dpi)

if __name__=="__main__":
    main()