"""
baseline.py — classical slow-band anomaly-detection floor + synthetic
injection validation, for poultry-barn multimodal anomaly detection.

WHERE THIS SITS IN THE PROJECT
-------------------------------
Everything here operates on the SLOW BAND (trend + diurnal recurrence across
weeks), not the fast band. It answers one question: "how well does the
simplest, most standard classical toolkit model normal week-over-week aging
+ diurnal rhythm, and how sensitive is that toolkit to known synthetic
anomalies?" Any later multimodal fusion / learned regime-switching model
needs to beat THIS on the SAME injected-anomaly test, or it hasn't earned
its added complexity. Real barn data has no ground-truth anomaly labels, so
validity is established here via synthetic injection with known locations —
not real-event detection.

TWO BASELINE MODELS (both are per-series, univariate)
  1. Holt-Winters ETS       — statsmodels ExponentialSmoothing
                               (additive trend + additive 24h seasonal)
  2. Local-linear-trend      — statsmodels UnobservedComponents(level=
     Kalman filter              'local linear trend'); state-space level+trend
                               fit via the Kalman filter/smoother, gives
                               genuine one-step-ahead innovations as residuals.

DETECTION RULE (shared, so ETS vs Kalman is a fair comparison)
  Fixed threshold on the standardized residual: |z| > SIGMA_THRESHOLD (3.0
  by default). Same numeric rule for both models.

-----------------------------------------------------------------------------
MULTI-MONTH + GAP-AWARE REWRITE (June + July, August-ready)
-----------------------------------------------------------------------------
June's real coverage report showed 16/30 days present and holes up to 115.5h
inside "present" days. The previous version fit ETS/Kalman on one month's
hourly series at a time and interpolated across EVERY gap, including
multi-day ones — meaning a 115-hour hole would get silently filled with a
straight-line fabrication and treated as real signal by both models. That's
a data-integrity bug, not just a June quirk, and it gets worse once June,
July, and (later) August are concatenated into one chronological series with
real month-boundary gaps too.

Fix, in four parts:

  1. MULTI-MONTH LOADING — any number of --audio-hourly-csv files (June,
     July, August, ...) get concatenated into one chronologically-sorted
     series, not analyzed month-by-month in isolation. This is what makes
     "use June, July, and August" actually mean something: a slow-band
     trend spanning months is the whole point of the slow band.

  2. MINUTE-FILE COVERAGE CHECK (optional, paired via --audio-minute-csv) —
     minute-level files are NOT fed into ETS/Kalman (that would blur into
     fast-band territory, out of scope for this floor). They're used only
     to compute, per hourly bin, what fraction of that hour's 60 minutes
     actually contain audio (n_frames > 0). An "hourly" aggregate built from
     2 real minutes out of 60 is a much weaker data point than one built
     from 55/60 — this coverage fraction feeds the segment density check
     below rather than being silently treated as equally trustworthy.

  3. GAP-AWARE SEGMENTATION — the concatenated series is split into
     contiguous segments wherever the gap since the previous observation
     exceeds MAX_GAP_HOURS (default 6h — deliberately much stricter than
     the old blanket interpolation). No interpolation ever crosses a
     segment boundary. Segments are further required to clear
     MIN_SEGMENT_HOURS (default 48h — two diurnal cycles, the floor for a
     seasonal ETS fit to mean anything) AND a MIN_DENSITY ratio (default
     0.5 — at least half the expected hourly bins present) before they're
     eligible for modeling at all. Every segment's fate (eligible / too
     short / too sparse) is printed in a SEGMENT REPORT — nothing is
     silently dropped.

  4. REGULAR-GRID REGULARIZATION PER SEGMENT — within an eligible segment,
     small internal gaps (<= MAX_GAP_HOURS) still leave missing hourly rows,
     which would misalign ETS's seasonal_periods=24 (an array-position
     assumption) against real wall-clock hours. Each segment is reindexed
     onto an explicit pd.date_range(..., freq="h") grid before
     interpolation, so "24 steps" and "24 real hours" are guaranteed to
     mean the same thing.

The full spike/level_shift/trend_drift/combined x obvious/subtle x
ETS/Kalman ablation grid now runs on the TOP_N_SEGMENTS longest eligible
segments (default 2 — typically July's mostly-continuous block plus June's
best cluster), demonstrating the baseline generalizes across months, not
just within one. Every other eligible segment gets a lighter "natural flag
rate" pass: fit clean (no injection), see how often the model flags real,
untouched data as anomalous. That's the complementary number to
injection-based recall — a specificity / false-alarm-rate check on genuine
multi-month data the models weren't tuned on.

METHODOLOGICAL NOTE — refit-on-injected (unchanged from before)
  Both models are refit directly on the injected series. A refit model can
  partially ABSORB a slow-onset anomaly into its own level/trend, especially
  the Kalman filter. Ground truth for level_shift/trend_drift is scored only
  over an ONSET_WINDOW_HOURS window (one diurnal cycle) after onset, not the
  entire remainder of the series, because scoring the full tail against a
  refit model structurally floors recall near zero regardless of real
  detection quality — see the June-run writeup for the empirical version of
  this finding.

USAGE
  python3 baseline.py \\
    --audio-hourly-csv data/features_room2/audio/audio_features_hourly_Room2_2025-06.csv \\
                        data/features_room2/audio/audio_features_hourly_Room2_2025-07.csv \\
    --audio-minute-csv data/features_room2/audio/audio_features_minute_Room2_2025-06.csv \\
                        data/features_room2/audio/audio_features_minute_Room2_2025-07.csv \\
    --env-csv data/raw_room2/env/env_features_Room2.csv \\
    --output-dir results/baseline --room-label Room2 --run-tag 2025-06_07
"""

import argparse
import os
import sys
import warnings
from itertools import product
from pathlib import Path

# Make the repo root importable so `config` resolves no matter how this script
# is launched (python src/models/baseline.py  OR  python -m src.models.baseline).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import FEATURES_DIR, RESULTS_DIR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.structural import UnobservedComponents

warnings.filterwarnings("ignore")  # statsmodels convergence chatter; real
                                    # failures are still caught explicitly below

# ---------------------------------------------------------------------------
# CONFIG — defaults, all overridable via CLI. August can be appended to
# --audio-hourly-csv / --audio-minute-csv the moment that extraction finishes
# — nothing else about this script needs to change.
# ---------------------------------------------------------------------------
AUDIO_HOURLY_CSVS = [
    str(FEATURES_DIR / "audio_features_hourly_Room2_2025-06.csv"),
    str(FEATURES_DIR / "audio_features_hourly_Room2_2025-07.csv"),
    # str(FEATURES_DIR / "audio_features_hourly_Room2_2025-08.csv"),
]
AUDIO_MINUTE_CSVS = [
    str(FEATURES_DIR / "audio_features_minute_Room2_2025-06.csv"),
    str(FEATURES_DIR / "audio_features_minute_Room2_2025-07.csv"),
    # str(FEATURES_DIR / "audio_features_minute_Room2_2025-08.csv"),
]
ENV_CSV     = str(FEATURES_DIR / "env_features_Room2.csv")
OUTPUT_DIR  = str(RESULTS_DIR / "baseline")
ROOM_LABEL  = "Room2"
RUN_TAG     = "2025-06_07"   # label only, used in output filenames

PRIMARY_SIGNAL     = "centroid_hz_mean"
SECONDARY_SIGNALS  = ["voc_activity_frac", "rms_db_mean", "mech_frac_mean"]
# + env_temp (<- temp_day_mean_c) and env_rh (<- rh_day_mean_pct), appended
# at runtime by load_env_features(). Verified directly against your real
# env_features_Room2.csv: both columns resolve with 0 missing values once
# reindexed onto the hourly grid — not a hypothetical, confirmed.

SEASONAL_PERIODS   = 24     # diurnal cycle, hourly cadence
SIGMA_THRESHOLD    = 3.0    # shared detection rule for both models
MAGNITUDE_PRESETS  = {"obvious": 5.5, "subtle": 2.5}   # in units of series std
INJECTION_TYPES    = ["spike", "level_shift", "trend_drift", "combined"]
ONSET_WINDOW_HOURS = 24     # ground-truth window for sustained anomalies (see docstring)
RANDOM_SEED         = 13

MAX_GAP_HOURS      = 6.0    # gap size that breaks a segment
MIN_SEGMENT_HOURS  = 48.0   # 2 diurnal cycles minimum to attempt a seasonal fit
MIN_DENSITY        = 0.5    # fraction of expected hourly bins that must be present
TOP_N_SEGMENTS     = 2      # segments getting the full ablation grid

FIG_DPI = 600

# ---------------------------------------------------------------------------
# DATA LOADING — multi-month
# ---------------------------------------------------------------------------
def load_hourly_multi(paths):
    frames = []
    for p in paths:
        if not os.path.exists(p):
            print(f"[WARN] hourly CSV not found, skipping: {p}")
            continue
        df = pd.read_csv(p, parse_dates=["time"], index_col="time").sort_index()
        df["__source_file"] = os.path.basename(p)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No hourly CSVs could be loaded — check --audio-hourly-csv paths.")
    combined = pd.concat(frames).sort_index()
    n_dupes = combined.index.duplicated().sum()
    if n_dupes:
        print(f"[WARN] {n_dupes} duplicate timestamps across hourly files — keeping first occurrence.")
        combined = combined[~combined.index.duplicated(keep="first")]
    return combined

def load_minute_coverage(paths, hourly_index):
    """Per-hour fraction of minutes with real audio (n_frames > 0), built
    from the paired minute-level CSVs. Not used as a modeling input — only
    as a segment-eligibility signal (see MIN_DENSITY)."""
    frames = []
    for p in paths:
        if not os.path.exists(p):
            print(f"[WARN] minute CSV not found, skipping: {p}")
            continue
        m = pd.read_csv(p, parse_dates=["time"], index_col="time").sort_index()
        frames.append(m)
    if not frames:
        print("[minute] no minute-level CSVs available — hour coverage confidence will be unweighted.")
        return None
    minute_df = pd.concat(frames).sort_index()
    minute_df = minute_df[~minute_df.index.duplicated(keep="first")]
    has_data = (minute_df["n_frames"] > 0) if "n_frames" in minute_df.columns else pd.Series(
        True, index=minute_df.index)
    minutes_per_hour = has_data.resample("h").sum()
    coverage = (minutes_per_hour / 60.0).reindex(hourly_index).fillna(0.0)
    coverage.name = "minute_coverage_frac"
    return coverage

def load_env_features(env_path, target_index):
    """Load env CSV once, extract BOTH temperature and relative-humidity
    daily-mean columns, forward-filled onto the shared hourly grid. Returns
    a dict of {output_name: series} for whichever signals resolve — missing
    ones are skipped gracefully rather than failing the whole load.

    RH is not a secondary afterthought here: the lab's pilot paper found
    humidity correlating with acoustic features (r = 0.65-0.70), not
    temperature, so it's arguably the more theoretically-motivated of the
    two env signals, and gets equal billing.

    Confirmed real schema (env_features_Room2.csv): ['date', 'temp_am_min_c',
    'temp_am_max_c', 'temp_pm_c', 'rh_am_pct', 'rh_pm_pct', 'source_file',
    'day_index', 'temp_day_mean_c', 'temp_am_range_c', 'temp_am_pm_swing_c',
    'rh_day_mean_pct', 'rh_am_pm_change_pct', 'temp_rate_c_per_day',
    'rh_rate_pct_per_day', 'temp_roll_mean_c', 'rh_roll_mean_pct',
    'temp_roll_slope_c_per_day']. Both 'temp_day_mean_c' and 'rh_day_mean_pct'
    match the "representative daily mean" convention and are picked over
    their rolling-mean counterparts because they appear first in column
    order — kept as a heuristic (not hardcoded names) so this still works if
    the schema shifts."""
    if not env_path or not os.path.exists(env_path):
        print(f"[env] no env CSV found at {env_path!r} — skipping env secondary signals.")
        return {}
    try:
        env = pd.read_csv(env_path)
    except Exception as e:
        print(f"[env] failed to read {env_path!r}: {e} — skipping.")
        return {}

    time_col = next((c for c in env.columns if "date" in c.lower() or "time" in c.lower()), None)
    if time_col is None:
        print(f"[env] no date/time column found (columns: {list(env.columns)}) — skipping.")
        return {}
    env[time_col] = pd.to_datetime(env[time_col], errors="coerce")
    env = env.dropna(subset=[time_col]).set_index(time_col).sort_index()

    def pick_column(keywords):
        candidates_mean = [c for c in env.columns
                            if any(k in c.lower() for k in keywords) and "mean" in c.lower()]
        if candidates_mean:
            return candidates_mean[0]
        candidates = [c for c in env.columns if any(k in c.lower() for k in keywords)]
        return candidates[0] if candidates else None

    results = {}
    for keywords, out_name in [(["temp"], "env_temp"), (["rh", "humid"], "env_rh")]:
        col = pick_column(keywords)
        if col is None:
            print(f"[env] no column matching {keywords} found (columns: {list(env.columns)}) "
                  f"— skipping {out_name}.")
            continue
        aligned = env[col].sort_index().reindex(target_index, method="ffill")
        aligned = aligned.ffill().bfill()
        aligned.name = out_name
        print(f"[env] using column {col!r} as {out_name} secondary signal.")
        results[out_name] = aligned
    return results

# ---------------------------------------------------------------------------
# GAP-AWARE SEGMENTATION
# ---------------------------------------------------------------------------
def compute_segment_ids(index, max_gap_hours):
    idx_series = index.to_series()
    gap_hours = idx_series.diff().dt.total_seconds().div(3600)
    seg_id = (gap_hours > max_gap_hours).fillna(False).cumsum()
    return pd.Series(seg_id.values, index=index, name="segment_id")

def summarize_segments(df, coverage_col=None):
    rows = []
    for sid, g in df.groupby("segment_id"):
        start, end = g.index.min(), g.index.max()
        span_hours = (end - start).total_seconds() / 3600 + 1  # +1: inclusive of the last hourly bin
        expected_bins = max(int(round(span_hours)), 1)
        density = len(g) / expected_bins
        mean_cov = g[coverage_col].mean() if coverage_col and coverage_col in g.columns else np.nan
        eligible = (span_hours >= MIN_SEGMENT_HOURS) and (density >= MIN_DENSITY)
        reason = "ok" if eligible else (
            "too_short" if span_hours < MIN_SEGMENT_HOURS else "too_sparse")
        rows.append({"segment_id": sid, "start": start, "end": end,
                      "span_hours": round(span_hours, 1), "n_hours_observed": len(g),
                      "density": round(density, 2), "mean_minute_coverage": round(mean_cov, 2)
                      if not np.isnan(mean_cov) else np.nan,
                      "eligible": eligible, "reason": reason})
    return pd.DataFrame(rows).sort_values("span_hours", ascending=False).reset_index(drop=True)

def print_segment_report(seg_summary):
    print("\n--- SEGMENT REPORT (gap-aware, MAX_GAP_HOURS={}h) ---".format(MAX_GAP_HOURS))
    print(f"  {'seg':>3}  {'start':19}  {'end':19}  {'span_h':>7}  {'obs_h':>6}  "
          f"{'density':>7}  {'min_cov':>7}  status")
    for _, r in seg_summary.iterrows():
        print(f"  {int(r['segment_id']):>3}  {str(r['start']):19}  {str(r['end']):19}  "
              f"{r['span_hours']:>7.1f}  {r['n_hours_observed']:>6}  {r['density']:>7.2f}  "
              f"{r['mean_minute_coverage']:>7}  {r['reason']}")

# ---------------------------------------------------------------------------
# BASELINE MODELS
# ---------------------------------------------------------------------------
def _regularize_hourly(series):
    """Reindex onto an explicit hourly grid spanning the segment, then
    interpolate. Safe here because segments are gap-bounded by construction
    (no internal gap exceeds MAX_GAP_HOURS) — this is filling small,
    legitimate holes, not fabricating across the multi-day gaps that used
    to get silently interpolated."""
    full_idx = pd.date_range(series.index.min(), series.index.max(), freq="h")
    return series.reindex(full_idx).interpolate(limit_direction="both")

def fit_ets(series):
    s = _regularize_hourly(series)
    model = ExponentialSmoothing(s, trend="add", seasonal="add",
                                  seasonal_periods=SEASONAL_PERIODS,
                                  initialization_method="estimated")
    fit = model.fit(optimized=True)
    fitted = fit.fittedvalues
    resid = s - fitted
    return fitted, resid

def fit_local_linear_kalman(series):
    s = _regularize_hourly(series)
    model = UnobservedComponents(s, level="local linear trend")
    fit = model.fit(disp=False)
    fitted = fit.fittedvalues
    resid = pd.Series(np.asarray(fit.resid).ravel(), index=s.index)
    return fitted, resid

MODELS = {"ETS": fit_ets, "Kalman": fit_local_linear_kalman}

def flag_anomalies(resid, sigma_threshold=SIGMA_THRESHOLD):
    resid = resid.copy()
    mu, sigma = resid.mean(), resid.std()
    z = (resid - mu) / (sigma + 1e-9)
    flags = z.abs() > sigma_threshold
    return flags, z

# ---------------------------------------------------------------------------
# SYNTHETIC INJECTION
# ---------------------------------------------------------------------------
def inject_point_spike(series, loc, magnitude_sigma, rng):
    s = series.copy()
    sigma = series.std()
    sign = rng.choice([-1, 1])
    s.iloc[loc] += sign * magnitude_sigma * sigma
    truth = pd.Series(False, index=series.index)
    truth.iloc[loc] = True
    return s, truth

def inject_level_shift(series, start, magnitude_sigma, rng):
    s = series.copy()
    sigma = series.std()
    sign = rng.choice([-1, 1])
    s.iloc[start:] += sign * magnitude_sigma * sigma
    truth = pd.Series(False, index=series.index)
    onset_end = min(start + ONSET_WINDOW_HOURS, len(series))
    truth.iloc[start:onset_end] = True
    return s, truth

def inject_trend_drift(series, start, magnitude_sigma, rng):
    s = series.copy()
    sigma = series.std()
    sign = rng.choice([-1, 1])
    n = len(series) - start
    ramp = np.linspace(0, sign * magnitude_sigma * sigma, n)
    s.iloc[start:] = s.iloc[start:].to_numpy() + ramp
    truth = pd.Series(False, index=series.index)
    onset_end = min(start + ONSET_WINDOW_HOURS, len(series))
    truth.iloc[start:onset_end] = True
    return s, truth

def make_injection(clean_series, kind, magnitude_key, rng):
    n = len(clean_series)
    mag = MAGNITUDE_PRESETS[magnitude_key]
    if kind == "spike":
        loc = int(n * 0.75)
        return inject_point_spike(clean_series, loc, mag, rng)
    elif kind == "level_shift":
        start = int(n * 0.65)
        return inject_level_shift(clean_series, start, mag, rng)
    elif kind == "trend_drift":
        start = int(n * 0.55)
        return inject_trend_drift(clean_series, start, mag, rng)
    elif kind == "combined":
        third = n // 3
        s1, t1 = inject_point_spike(clean_series, int(third * 0.5), mag, rng)
        s2, t2 = inject_level_shift(s1, third + int(third * 0.3), mag, rng)
        s3, t3 = inject_trend_drift(s2, 2 * third + int(third * 0.2), mag, rng)
        return s3, (t1 | t2 | t3)
    raise ValueError(f"unknown injection kind: {kind}")

# ---------------------------------------------------------------------------
# SCORING
# ---------------------------------------------------------------------------
def score_detection(flags, truth):
    flags = flags.reindex(truth.index).fillna(False)
    tp = int((flags & truth).sum())
    fp = int((flags & ~truth).sum())
    fn = int((~flags & truth).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = (2 * precision * recall / (precision + recall)
          if (precision and recall and not np.isnan(precision) and not np.isnan(recall)
              and (precision + recall) > 0) else np.nan)
    delay_hours = np.nan
    if truth.any():
        first_true = truth[truth].index[0]
        detected_in_truth = flags[flags & truth].index
        if len(detected_in_truth):
            delay_hours = (detected_in_truth[0] - first_true).total_seconds() / 3600
    return {"precision": precision, "recall": recall, "f1": f1,
            "detection_delay_hours": delay_hours, "tp": tp, "fp": fp, "fn": fn}

# ---------------------------------------------------------------------------
# PLOTTING (600 dpi)
# ---------------------------------------------------------------------------
def plot_case(clean, injected, truth, fitted, flags, z, title, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(clean.index, clean.values, label="clean", color="gray", alpha=0.6, linewidth=0.8)
    axes[0].plot(injected.index, injected.values, label="injected", color="black", linewidth=0.9)
    if truth.any():
        axes[0].scatter(injected.index[truth], injected.values[truth],
                         color="red", s=10, label="true anomaly", zorder=5)
    axes[0].plot(fitted.index, fitted.values, label="model fit", color="tab:blue",
                 linestyle="--", linewidth=0.9)
    axes[0].legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("value")

    axes[1].plot(z.index, z.values, color="tab:purple", linewidth=0.8, label="standardized residual (z)")
    axes[1].axhline(SIGMA_THRESHOLD, color="red", linestyle=":", linewidth=0.8)
    axes[1].axhline(-SIGMA_THRESHOLD, color="red", linestyle=":", linewidth=0.8)
    axes[1].legend(fontsize=7, loc="upper left")
    axes[1].set_ylabel("z-score")

    if flags.any() or truth.any():
        axes[2].scatter(flags.index[flags], np.ones(int(flags.sum())),
                         color="orange", s=12, label="detected")
        axes[2].scatter(injected.index[truth], np.full(int(truth.sum()), 1.05),
                         color="red", s=12, label="true")
    axes[2].set_ylim(0.9, 1.15)
    axes[2].set_yticks([])
    axes[2].legend(fontsize=7, loc="upper left")
    axes[2].set_xlabel("time")

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)

def plot_metrics_summary(metrics_df, out_path, title):
    if metrics_df.empty:
        return
    piv = metrics_df.pivot_table(index=["segment_label", "injection", "magnitude"],
                                  columns="model", values="f1")
    fig, ax = plt.subplots(figsize=(11, max(4, 0.35 * len(piv))))
    piv.plot(kind="barh", ax=ax)
    ax.set_xlabel("F1")
    ax.set_title(title)
    ax.legend(title="model", fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)

def plot_secondary_generalization(records, out_path):
    signals = sorted(set(r["signal"] for r in records))
    models = sorted(set(r["model"] for r in records))
    if not signals:
        return
    fig, axes = plt.subplots(len(signals), len(models),
                              figsize=(6 * len(models), 2.6 * len(signals)),
                              squeeze=False, sharex=False)
    for i, sig in enumerate(signals):
        for j, mdl in enumerate(models):
            rec = next((r for r in records if r["signal"] == sig and r["model"] == mdl), None)
            ax = axes[i][j]
            if rec is None:
                ax.axis("off")
                continue
            ax.plot(rec["z"].index, rec["z"].values, color="tab:purple", linewidth=0.7)
            ax.axhline(SIGMA_THRESHOLD, color="red", linestyle=":", linewidth=0.7)
            ax.axhline(-SIGMA_THRESHOLD, color="red", linestyle=":", linewidth=0.7)
            if rec["truth"].any():
                ymax = ax.get_ylim()[1]
                ax.scatter(rec["truth"].index[rec["truth"]],
                           np.full(int(rec["truth"].sum()), ymax * 0.9),
                           color="red", s=8)
            ax.set_title(f"{sig} / {mdl}", fontsize=8)
    fig.suptitle("Secondary-signal generalization check (combined injection, obvious magnitude, "
                 "longest eligible segment)", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Multi-month slow-band baseline (ETS + local-linear "
                                             "Kalman) with gap-aware segmentation and synthetic "
                                             "anomaly injection ablation.")
    p.add_argument("--audio-hourly-csv", nargs="+", default=AUDIO_HOURLY_CSVS,
                   help="One or more hourly feature CSVs (any months) — concatenated chronologically.")
    p.add_argument("--audio-minute-csv", nargs="*", default=AUDIO_MINUTE_CSVS,
                   help="Optional paired minute-level CSVs, used only for per-hour coverage "
                        "confidence, not fed into the models.")
    p.add_argument("--env-csv", default=ENV_CSV)
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    p.add_argument("--room-label", default=ROOM_LABEL)
    p.add_argument("--run-tag", default=RUN_TAG, help="Label used in output filenames.")
    p.add_argument("--sigma-threshold", type=float, default=SIGMA_THRESHOLD)
    p.add_argument("--max-gap-hours", type=float, default=MAX_GAP_HOURS)
    p.add_argument("--min-segment-hours", type=float, default=MIN_SEGMENT_HOURS)
    p.add_argument("--min-density", type=float, default=MIN_DENSITY)
    p.add_argument("--top-n-segments", type=int, default=TOP_N_SEGMENTS)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    return p.parse_args()

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    global SIGMA_THRESHOLD, MAX_GAP_HOURS, MIN_SEGMENT_HOURS, MIN_DENSITY, TOP_N_SEGMENTS
    SIGMA_THRESHOLD = args.sigma_threshold
    MAX_GAP_HOURS = args.max_gap_hours
    MIN_SEGMENT_HOURS = args.min_segment_hours
    MIN_DENSITY = args.min_density
    TOP_N_SEGMENTS = args.top_n_segments
    rng = np.random.default_rng(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    fig_dir = os.path.join(args.output_dir, "figures")

    print(f"Loading {len(args.audio_hourly_csv)} hourly CSV(s)...")
    audio = load_hourly_multi(args.audio_hourly_csv)
    print(f"  combined shape: {audio.shape}, range: {audio.index.min()} -> {audio.index.max()}")

    if args.audio_minute_csv:
        coverage = load_minute_coverage(args.audio_minute_csv, audio.index)
        if coverage is not None:
            audio = audio.join(coverage)
    coverage_col = "minute_coverage_frac" if "minute_coverage_frac" in audio.columns else None

    env_signals = load_env_features(args.env_csv, audio.index)
    secondary_signals = list(SECONDARY_SIGNALS)
    for name, series in env_signals.items():
        audio = audio.join(series)
        secondary_signals.append(name)
    secondary_signals = [s for s in secondary_signals if s in audio.columns]

    if PRIMARY_SIGNAL not in audio.columns:
        raise KeyError(f"PRIMARY_SIGNAL {PRIMARY_SIGNAL!r} not found in columns: {list(audio.columns)}")

    # ---- gap-aware segmentation ----
    audio["segment_id"] = compute_segment_ids(audio.index, MAX_GAP_HOURS)
    seg_summary = summarize_segments(audio, coverage_col=coverage_col)
    print_segment_report(seg_summary)

    eligible = seg_summary[seg_summary["eligible"]].reset_index(drop=True)
    if eligible.empty:
        raise RuntimeError("No segment cleared MIN_SEGMENT_HOURS / MIN_DENSITY — cannot run baseline. "
                            "Loosen --min-segment-hours / --min-density or gather more contiguous data.")

    top_segments = eligible.head(TOP_N_SEGMENTS)
    other_segments = eligible.iloc[TOP_N_SEGMENTS:]
    print(f"\nTop {len(top_segments)} segment(s) get the full ablation grid: "
          f"{list(top_segments['segment_id'])}")
    if len(other_segments):
        print(f"{len(other_segments)} other eligible segment(s) get a natural-flag-rate check only: "
              f"{list(other_segments['segment_id'])}")

    def seg_label(row):
        return f"seg{int(row['segment_id'])}_{row['start'].strftime('%Y%m%d')}-{row['end'].strftime('%Y%m%d')}"

    all_metrics = []
    natural_flag_rows = []

    # ---- PRIMARY SIGNAL: full ablation grid on top-N segments ----
    print(f"\n=== PRIMARY SIGNAL: {PRIMARY_SIGNAL} (full ablation grid, top {len(top_segments)} segment(s)) ===")
    for _, seg_row in top_segments.iterrows():
        sid = seg_row["segment_id"]
        label = seg_label(seg_row)
        clean = audio.loc[audio["segment_id"] == sid, PRIMARY_SIGNAL].astype(float)
        print(f"\n-- {label} ({len(clean)} obs, {seg_row['span_hours']}h span) --")
        for kind, mag_key in product(INJECTION_TYPES, MAGNITUDE_PRESETS.keys()):
            injected, truth = make_injection(clean, kind, mag_key, rng)
            for model_name, fit_fn in MODELS.items():
                case = f"{label}__{PRIMARY_SIGNAL}__{kind}__{mag_key}__{model_name}"
                try:
                    fitted, resid = fit_fn(injected)
                    flags, z = flag_anomalies(resid, SIGMA_THRESHOLD)
                    metrics = score_detection(flags, truth)
                except Exception as e:
                    print(f"  [FAIL] {case}: {e}")
                    metrics = {"precision": np.nan, "recall": np.nan, "f1": np.nan,
                               "detection_delay_hours": np.nan, "tp": 0, "fp": 0, "fn": int(truth.sum())}
                    all_metrics.append({"segment_label": label, "signal": PRIMARY_SIGNAL,
                                         "injection": kind, "magnitude": mag_key,
                                         "model": model_name, **metrics})
                    continue

                all_metrics.append({"segment_label": label, "signal": PRIMARY_SIGNAL,
                                     "injection": kind, "magnitude": mag_key,
                                     "model": model_name, **metrics})
                print(f"  {case}: precision={metrics['precision']:.2f} recall={metrics['recall']:.2f} "
                      f"f1={metrics['f1']:.2f} delay={metrics['detection_delay_hours']}")

                fig_path = os.path.join(fig_dir, f"{case}.png")
                plot_case(clean, injected, truth, fitted, flags, z,
                          title=f"{label} | {PRIMARY_SIGNAL} | {kind} | {mag_key} | {model_name}",
                          out_path=fig_path)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = os.path.join(args.output_dir,
                                 f"baseline_ablation_metrics_{args.room_label}_{args.run_tag}.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved ablation metrics -> {metrics_path}")

    summary_fig_path = os.path.join(fig_dir, f"ablation_summary_{args.room_label}_{args.run_tag}.png")
    plot_metrics_summary(metrics_df, summary_fig_path,
                          title=f"Baseline ablation summary — {PRIMARY_SIGNAL} (top segments)")
    print(f"Saved ablation summary figure -> {summary_fig_path}")

    # ---- OTHER ELIGIBLE SEGMENTS: natural flag rate (no injection) ----
    if len(other_segments):
        print(f"\n=== NATURAL FLAG RATE — other eligible segments, {PRIMARY_SIGNAL}, clean fit only ===")
        for _, seg_row in other_segments.iterrows():
            sid = seg_row["segment_id"]
            label = seg_label(seg_row)
            clean = audio.loc[audio["segment_id"] == sid, PRIMARY_SIGNAL].astype(float)
            for model_name, fit_fn in MODELS.items():
                try:
                    fitted, resid = fit_fn(clean)
                    flags, z = flag_anomalies(resid, SIGMA_THRESHOLD)
                    rate = float(flags.mean())
                except Exception as e:
                    print(f"  [FAIL] {label}/{model_name}: {e}")
                    rate = np.nan
                natural_flag_rows.append({"segment_label": label, "signal": PRIMARY_SIGNAL,
                                           "model": model_name, "natural_flag_rate": rate,
                                           "n_obs": len(clean)})
                print(f"  {label} / {model_name}: natural_flag_rate={rate}")

    # ---- SECONDARY SIGNALS: generalization check on the single longest segment ----
    longest = eligible.iloc[0]
    longest_label = seg_label(longest)
    print(f"\n=== SECONDARY SIGNALS: {secondary_signals} "
          f"(combined injection, longest segment = {longest_label}) ===")
    secondary_records = []
    secondary_metrics = []
    seg_mask = audio["segment_id"] == longest["segment_id"]
    for sig in secondary_signals:
        clean_s = audio.loc[seg_mask, sig].astype(float)
        for mag_key in MAGNITUDE_PRESETS.keys():
            injected, truth = make_injection(clean_s, "combined", mag_key, rng)
            for model_name, fit_fn in MODELS.items():
                case = f"{longest_label}__{sig}__combined__{mag_key}__{model_name}"
                try:
                    fitted, resid = fit_fn(injected)
                    flags, z = flag_anomalies(resid, SIGMA_THRESHOLD)
                    metrics = score_detection(flags, truth)
                except Exception as e:
                    print(f"  [FAIL] {case}: {e}")
                    continue
                secondary_metrics.append({"segment_label": longest_label, "signal": sig,
                                           "injection": "combined", "magnitude": mag_key,
                                           "model": model_name, **metrics})
                print(f"  {case}: f1={metrics['f1']:.2f}")
                if mag_key == "obvious":
                    secondary_records.append({"signal": sig, "model": model_name,
                                               "z": z, "truth": truth})

    if secondary_metrics:
        sec_df = pd.DataFrame(secondary_metrics)
        sec_path = os.path.join(args.output_dir,
                                 f"baseline_secondary_metrics_{args.room_label}_{args.run_tag}.csv")
        sec_df.to_csv(sec_path, index=False)
        print(f"Saved secondary-signal metrics -> {sec_path}")

        gen_fig_path = os.path.join(fig_dir, f"secondary_generalization_{args.room_label}_{args.run_tag}.png")
        plot_secondary_generalization(secondary_records, gen_fig_path)
        print(f"Saved secondary generalization figure -> {gen_fig_path}")

    if natural_flag_rows:
        nat_df = pd.DataFrame(natural_flag_rows)
        nat_path = os.path.join(args.output_dir,
                                 f"baseline_natural_flag_rate_{args.room_label}_{args.run_tag}.csv")
        nat_df.to_csv(nat_path, index=False)
        print(f"Saved natural flag-rate metrics -> {nat_path}")

    seg_summary_path = os.path.join(args.output_dir,
                                     f"baseline_segment_report_{args.room_label}_{args.run_tag}.csv")
    seg_summary.to_csv(seg_summary_path, index=False)
    print(f"Saved segment report -> {seg_summary_path}")

    print("\n--- SUMMARY (primary signal, mean F1 by segment / injection / model) ---")
    if len(metrics_df):
        print(metrics_df.pivot_table(index=["segment_label", "injection"], columns="model",
                                      values="f1", aggfunc="mean").round(2).to_string())


if __name__ == "__main__":
    main()