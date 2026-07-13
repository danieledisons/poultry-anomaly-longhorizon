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
  by default). Same numeric rule for both models — if the rule itself
  differed per model, any performance gap could just be an artifact of a
  differently-calibrated interval rather than a real quality difference.

SIGNAL SCOPE (curated multi-signal set)
  PRIMARY_SIGNAL   = "centroid_hz_mean" — your own extract_audio_features.py
                      docstring calls this "the audio GROWTH signal" (chicks
                      high-pitched -> mature birds lower-pitched). This one
                      gets the FULL ablation grid + one figure per case.
  SECONDARY_SIGNALS = voc_activity_frac, rms_db_mean, mech_frac_mean, and
                      env temperature (if the env CSV's schema resolves).
                      These get the same detection test but only the
                      "combined" injection case at both magnitudes, logged
                      into the same metrics table + ONE shared generalization
                      figure, not a full 16-figure set each. Rationale: 5
                      signals x 4 injection types x 2 magnitudes x 2 models
                      = 80 combinations if fully expanded — good for
                      completeness, bad for a progress report anyone can
                      actually read. Full rigor lives in the metrics CSV;
                      figures are curated to the story.

INJECTION TYPES x MAGNITUDES (the ablation grid, "mixed" per design choice)
  spike        — single-frame point anomaly (sudden vocalization/transient
                 burst analog)
  level_shift  — step change sustained to the end of the series (HVAC/fan
                 failure analog)
  trend_drift  — linear ramp to the end of the series (undetected
                 growth-rate deviation analog)
  combined     — all three injected into non-overlapping thirds of the same
                 series
  magnitudes: "obvious" (~5.5 sigma of the series) and "subtle" (~2.5 sigma)

METHODOLOGICAL NOTE — refit-on-injected (read before trusting the numbers)
  Both models are refit directly on the INJECTED series, not fit-once-on-
  clean-then-scored-out-of-sample. This is a deliberate scope simplification,
  not an oversight: it's simpler to implement correctly and it produces an
  honest, if slightly conservative, test. The real risk it introduces is
  that a model can partially ABSORB a slow-onset anomaly (level_shift,
  trend_drift) into its own fitted level/trend during refitting, especially
  the Kalman filter, which adapts online. If you see recall drop off for
  level_shift/trend_drift specifically, that is not a bug — it is itself an
  informative finding: a baseline that "learns away" the anomaly quickly is
  a weaker detector, and that gap is exactly what a smarter regime-aware
  model should close. Report it as a finding, not hide it.

USAGE
  python3 baseline.py \\
    --audio-csv /path/to/audio_features_hourly_Room2_2025-07.csv \\
    --env-csv /path/to/env_features_Room2.csv \\
    --output-dir /path/to/results/baseline \\
    --room-label Room2 --month-tag 2025-07
"""

import argparse
import os
import warnings
from itertools import product

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
# CONFIG — defaults, all overridable via CLI so this can point at any
# room/month once June/August feature extraction finishes (same pattern as
# extract_audio_features.py).
# ---------------------------------------------------------------------------
AUDIO_CSV   = "/Users/daniel/Documents/LongHorizon/data/features_room2/audio/audio_features_hourly_Room2_2025-07.csv"
ENV_CSV     = "/Users/daniel/Documents/LongHorizon/data/raw_room2/env/env_features_Room2.csv"
OUTPUT_DIR  = "/Users/daniel/Documents/LongHorizon/results/baseline"
ROOM_LABEL  = "Room2"
MONTH_TAG   = "2025-07"

PRIMARY_SIGNAL     = "centroid_hz_mean"
SECONDARY_SIGNALS  = ["voc_activity_frac", "rms_db_mean", "mech_frac_mean"]  # + env_temp if resolved

SEASONAL_PERIODS   = 24     # diurnal cycle, hourly cadence
SIGMA_THRESHOLD    = 3.0    # shared detection rule for both models
MAGNITUDE_PRESETS  = {"obvious": 5.5, "subtle": 2.5}   # in units of series std
INJECTION_TYPES    = ["spike", "level_shift", "trend_drift", "combined"]
RANDOM_SEED         = 13

FIG_DPI = 600

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
def load_audio_hourly(path):
    df = pd.read_csv(path, parse_dates=["time"], index_col="time")
    return df.sort_index()

def load_env_temp_aligned(env_path, target_index):
    """Best-effort load of a daily env temperature column, forward-filled
    onto the shared hourly grid (env has no sub-daily detail of its own —
    per project convention it's aligned to whatever higher-rate modality
    sets the cadence). Schema of env_features_Room2.csv wasn't confirmed
    ahead of time, so this searches by column-name heuristics and fails
    gracefully (prints available columns) rather than guessing wrong."""
    if not env_path or not os.path.exists(env_path):
        print(f"[env] no env CSV found at {env_path!r} — skipping env secondary signal.")
        return None
    try:
        env = pd.read_csv(env_path)
    except Exception as e:
        print(f"[env] failed to read {env_path!r}: {e} — skipping.")
        return None

    time_col = next((c for c in env.columns if "date" in c.lower() or "time" in c.lower()), None)
    if time_col is None:
        print(f"[env] no date/time column found (columns: {list(env.columns)}) — skipping.")
        return None
    env[time_col] = pd.to_datetime(env[time_col], errors="coerce")
    env = env.dropna(subset=[time_col]).set_index(time_col).sort_index()

    temp_col = next((c for c in env.columns if "temp" in c.lower() and "mean" in c.lower()), None)
    if temp_col is None:
        temp_col = next((c for c in env.columns if "temp" in c.lower()), None)
    if temp_col is None:
        print(f"[env] no temperature-like column found (columns: {list(env.columns)}) — skipping.")
        return None

    aligned = env[temp_col].sort_index().reindex(target_index, method="ffill")
    aligned = aligned.ffill().bfill()
    aligned.name = "env_temp"
    print(f"[env] using column {temp_col!r} as env_temp secondary signal.")
    return aligned

# ---------------------------------------------------------------------------
# BASELINE MODELS
# ---------------------------------------------------------------------------
def fit_ets(series):
    s = series.interpolate(limit_direction="both")
    model = ExponentialSmoothing(s, trend="add", seasonal="add",
                                  seasonal_periods=SEASONAL_PERIODS,
                                  initialization_method="estimated")
    fit = model.fit(optimized=True)
    fitted = fit.fittedvalues
    resid = s - fitted
    return fitted, resid

def fit_local_linear_kalman(series):
    s = series.interpolate(limit_direction="both")
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
#
# ONSET_WINDOW_HOURS bounds the ground-truth "should-be-flagged" region for
# sustained anomalies (level_shift, trend_drift). This is NOT the duration of
# the injected value change itself (a level shift stays shifted permanently,
# same as a real HVAC/fan failure would) — it's the duration over which
# DETECTION is plausible under the refit-on-injected methodology documented
# at the top of this file. A refit model absorbs a sustained shift into its
# own level within roughly one diurnal cycle, after which the "anomaly"
# becomes indistinguishable from the model's new normal — so scoring recall
# against the full post-shift tail structurally floors every model near
# zero, regardless of real detection quality. Scoring against the onset
# window instead answers the actually meaningful question: did the model
# notice the transition while it was still novel?
# ---------------------------------------------------------------------------
ONSET_WINDOW_HOURS = 24   # one diurnal cycle

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

def plot_metrics_summary(metrics_df, out_path):
    piv = metrics_df.pivot_table(index=["injection", "magnitude"], columns="model", values="f1")
    fig, ax = plt.subplots(figsize=(9, 5))
    piv.plot(kind="bar", ax=ax)
    ax.set_ylabel("F1 (primary signal)")
    ax.set_title(f"Baseline ablation summary — {PRIMARY_SIGNAL}")
    ax.legend(title="model", fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)

def plot_secondary_generalization(records, out_path):
    """One shared figure: z-score traces for every secondary signal under
    the 'combined' injection at the 'obvious' magnitude, one row per signal,
    one column per model. Generalization check, not full ablation detail."""
    signals = sorted(set(r["signal"] for r in records))
    models = sorted(set(r["model"] for r in records))
    if not signals:
        return
    fig, axes = plt.subplots(len(signals), len(models),
                              figsize=(6 * len(models), 2.6 * len(signals)),
                              squeeze=False, sharex=True)
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
    fig.suptitle("Secondary-signal generalization check (combined injection, obvious magnitude)",
                 fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Slow-band baseline (ETS + local-linear Kalman) "
                                             "with synthetic anomaly injection ablation.")
    p.add_argument("--audio-csv", default=AUDIO_CSV)
    p.add_argument("--env-csv", default=ENV_CSV)
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    p.add_argument("--room-label", default=ROOM_LABEL)
    p.add_argument("--month-tag", default=MONTH_TAG)
    p.add_argument("--sigma-threshold", type=float, default=SIGMA_THRESHOLD)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    return p.parse_args()

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    global SIGMA_THRESHOLD
    SIGMA_THRESHOLD = args.sigma_threshold
    rng = np.random.default_rng(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    fig_dir = os.path.join(args.output_dir, "figures")

    print(f"Loading audio hourly features: {args.audio_csv}")
    audio = load_audio_hourly(args.audio_csv)
    print(f"  shape: {audio.shape}, range: {audio.index.min()} -> {audio.index.max()}")

    env_temp = load_env_temp_aligned(args.env_csv, audio.index)
    secondary_signals = list(SECONDARY_SIGNALS)
    if env_temp is not None:
        audio = audio.join(env_temp)
        secondary_signals.append("env_temp")
    secondary_signals = [s for s in secondary_signals if s in audio.columns]

    missing = [s for s in [PRIMARY_SIGNAL] + secondary_signals if s not in audio.columns]
    if PRIMARY_SIGNAL not in audio.columns:
        raise KeyError(f"PRIMARY_SIGNAL {PRIMARY_SIGNAL!r} not found in audio CSV columns: "
                        f"{list(audio.columns)}")
    if missing:
        print(f"[WARN] secondary signals not found and will be skipped: {missing}")
        secondary_signals = [s for s in secondary_signals if s not in missing]

    all_metrics = []

    # ---- PRIMARY SIGNAL: full ablation grid ----
    print(f"\n=== PRIMARY SIGNAL: {PRIMARY_SIGNAL} (full ablation grid) ===")
    clean = audio[PRIMARY_SIGNAL].astype(float)
    for kind, mag_key in product(INJECTION_TYPES, MAGNITUDE_PRESETS.keys()):
        injected, truth = make_injection(clean, kind, mag_key, rng)
        for model_name, fit_fn in MODELS.items():
            case = f"{PRIMARY_SIGNAL}__{kind}__{mag_key}__{model_name}"
            try:
                fitted, resid = fit_fn(injected)
                flags, z = flag_anomalies(resid, SIGMA_THRESHOLD)
                metrics = score_detection(flags, truth)
            except Exception as e:
                print(f"  [FAIL] {case}: {e}")
                metrics = {"precision": np.nan, "recall": np.nan, "f1": np.nan,
                           "detection_delay_hours": np.nan, "tp": 0, "fp": 0, "fn": int(truth.sum())}
                all_metrics.append({"signal": PRIMARY_SIGNAL, "injection": kind,
                                     "magnitude": mag_key, "model": model_name, **metrics})
                continue

            all_metrics.append({"signal": PRIMARY_SIGNAL, "injection": kind,
                                 "magnitude": mag_key, "model": model_name, **metrics})
            print(f"  {case}: precision={metrics['precision']:.2f} recall={metrics['recall']:.2f} "
                  f"f1={metrics['f1']:.2f} delay={metrics['detection_delay_hours']}")

            fig_path = os.path.join(fig_dir, f"{case}.png")
            plot_case(clean, injected, truth, fitted, flags, z,
                      title=f"{PRIMARY_SIGNAL} | {kind} | {mag_key} | {model_name}",
                      out_path=fig_path)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = os.path.join(args.output_dir,
                                 f"baseline_ablation_metrics_{args.room_label}_{args.month_tag}.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved ablation metrics -> {metrics_path}")

    summary_fig_path = os.path.join(fig_dir, f"ablation_summary_{args.room_label}_{args.month_tag}.png")
    plot_metrics_summary(metrics_df[metrics_df["signal"] == PRIMARY_SIGNAL], summary_fig_path)
    print(f"Saved ablation summary figure -> {summary_fig_path}")

    # ---- SECONDARY SIGNALS: generalization check (combined injection only) ----
    print(f"\n=== SECONDARY SIGNALS: {secondary_signals} (combined injection, both magnitudes) ===")
    secondary_records = []
    secondary_metrics = []
    for sig in secondary_signals:
        clean_s = audio[sig].astype(float)
        for mag_key in MAGNITUDE_PRESETS.keys():
            injected, truth = make_injection(clean_s, "combined", mag_key, rng)
            for model_name, fit_fn in MODELS.items():
                case = f"{sig}__combined__{mag_key}__{model_name}"
                try:
                    fitted, resid = fit_fn(injected)
                    flags, z = flag_anomalies(resid, SIGMA_THRESHOLD)
                    metrics = score_detection(flags, truth)
                except Exception as e:
                    print(f"  [FAIL] {case}: {e}")
                    continue
                secondary_metrics.append({"signal": sig, "injection": "combined",
                                           "magnitude": mag_key, "model": model_name, **metrics})
                print(f"  {case}: f1={metrics['f1']:.2f}")
                if mag_key == "obvious":
                    secondary_records.append({"signal": sig, "model": model_name,
                                               "z": z, "truth": truth})

    if secondary_metrics:
        sec_df = pd.DataFrame(secondary_metrics)
        sec_path = os.path.join(args.output_dir,
                                 f"baseline_secondary_metrics_{args.room_label}_{args.month_tag}.csv")
        sec_df.to_csv(sec_path, index=False)
        print(f"Saved secondary-signal metrics -> {sec_path}")

        gen_fig_path = os.path.join(fig_dir, f"secondary_generalization_{args.room_label}_{args.month_tag}.png")
        plot_secondary_generalization(secondary_records, gen_fig_path)
        print(f"Saved secondary generalization figure -> {gen_fig_path}")

    print("\n--- SUMMARY (primary signal, mean F1 by injection type / model) ---")
    if len(metrics_df):
        print(metrics_df.pivot_table(index="injection", columns="model", values="f1", aggfunc="mean")
              .round(2).to_string())


if __name__ == "__main__":
    main()