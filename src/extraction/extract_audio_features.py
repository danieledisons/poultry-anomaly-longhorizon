"""
Audio feature extraction for poultry-barn multimodal anomaly detection.

Reads a folder of audio files (WAV / FLAC / MP4-audio), computes interpretable
frame-level descriptors from a log-mel base, tags every frame with a wall-clock
timestamp, and aggregates to MINUTE-level and HOUR-level tables.

Output tables:
  - audio_features_minute_Room2_<MONTH_TAG>.csv   (base granularity)
  - audio_features_hourly_Room2_<MONTH_TAG>.csv   (slow-band cadence)
  - audio_features_<MONTH_TAG>_time_coverage.csv  (which days/hours actually
    have data — READ THIS before feeding output into baseline.py)

Band assignment (for later modelling):
  FAST band  -> per-frame / per-minute descriptors, vocalization-activity index,
                transient rate (also feeds the human-presence detector)
  SLOW band  -> hourly aggregates; watch spectral_centroid + voc_band_frac trend
                across weeks as the audio GROWTH signal (chicks high-pitched ->
                mature birds lower-pitched)

IMPORTANT: this is a plumbing/validation script. On a few days of data you are
checking (a) timestamps parse, (b) values are sane, (c) diurnal rhythm shows.
You are NOT reading any slow trajectory yet.

-----------------------------------------------------------------------------
EDITS (June batch):
  1. Source data is NOT a flat folder of wav files. It's nested by recording
     cluster, e.g.:
         Room 2 (June)/
           Room 2 (6, 7, 8 June)/
           Room 2 (9, 10, 11 June)/
           Room 2 (13, 15, 16 June)/
           Room 2 (18, 19, 21 June)/
           Room (23, 25, 30 June)/          <- inconsistent naming, no "2"
     `find_audio_files()` now recurses under INPUT_ROOT and does NOT pattern-
     match subfolder names, so naming inconsistencies (missing "2", extra
     spaces, parens) can't break discovery.
  2. MONTH_TAG was hardcoded "2025-07" in the source script — that's a July
     leftover and would have silently mislabeled June's output. Fixed to
     "2025-06".
  3. Paths switched from the workstation (/home/daniel/workspace/...) to the
     laptop (/Users/daniel/Documents/LongHorizon/...) per the local-run plan.
  4. Gaps are no longer invisible. The folder names above already tell you
     June has no recordings on 1-5, 12, 14, 17, 20, 22, 24, 26-29. This script
     now (a) reports missing calendar days and any >2h holes within
     "present" days, (b) writes a coverage CSV, and (c) adds a
     `gap_hours_since_prev` column to both output tables so baseline.py can
     see real elapsed time between observations instead of assuming a clean
     regular index. Do NOT feed this into Holt-Winters/Kalman as if it were
     continuous — fit per contiguous block, or gap-mask explicitly.
  5. FOLDER SHAPE — find_audio_files() already handles both "a folder of
     folders" (June: date-cluster subfolders, inconsistent naming) and "a
     folder of plain wav files" (July/August, if that's how they land)
     identically, because Path.rglob("*") recurses at every depth INCLUDING
     the root's own immediate files — a flat folder is just a nesting depth
     of zero to it. No structural change was needed here; re-verified while
     adding STE below, still correct, left as-is.
  6. STE (Short-Time Energy) ADDED — the lab's own prior pilot paper flags
     RMS *and* STE as major developmental features; this script previously
     only saved RMS. The needed quantity was already being computed every
     frame as `total_e` (per-frame power summed across all frequency bins,
     used internally to normalize the band-fraction features) and then
     discarded. It's now captured as `ste_db` (10*log10 of that per-frame
     energy — power-domain dB, not the 20*log10 used for the amplitude-like
     RMS) and aggregated into the hourly/minute tables like every other
     descriptor. Re-running June and July with this script backfills STE for
     both; August should be run with this version from the start so all
     three months carry a consistent feature set.
  7. FAN-TONE NOTCH/REDUCE FILTER ADDED — baseline.py's correlation
     diagnostic showed centroid_hz_mean strongly fan-confounded (r=-0.62 to
     -0.84 vs mech_frac_mean across all 5 real segments) and ste_db_mean
     moderately confounded (r=0.32-0.69). By-ear EQ testing (two passes)
     found 21 Hz and 64 Hz should be fully notched and 125 Hz partially
     reduced (to 30% power) for the clearest result. Implemented as
     build_fan_gain() + a per-bin power mask applied to a COPY of the
     spectrogram, producing new centroid_hz_clean / ste_db_clean columns
     alongside (not replacing) centroid_hz / ste_db, so baseline.py's
     correlation check can compare before vs after directly. Defaults are
     CLI-overridable (--notch-freqs-hz, --notch-width-hz, --reduce-freqs-hz,
     --reduce-width-hz, --reduce-gain) since these were tuned by ear for
     Room 2's fan and another room may need different values. Re-running
     June/July/August is needed to backfill these two columns, same as
     item 6 above.
-----------------------------------------------------------------------------
"""

import argparse
import calendar
import concurrent.futures as cf
import os
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import RAW_DIR, FEATURES_DIR

import librosa
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG  — these are now DEFAULTS only. Override via CLI flags so you can run
# two months concurrently in two terminals without editing this file (and
# without one run's config clobbering the other's while both are executing).
#
#   Terminal 1 (already running, June):
#     python3 extract_audio_features.py
#       (no flags needed — defaults below are June's settings)
#
#   Terminal 2 (August, run at the same time):
#     python3 extract_audio_features.py \
#       --input-root "/Users/daniel/Documents/LongHorizon/data/raw_room2/Room 2 (August)" \
#       --month-tag 2025-08
#
# Each run is a separate OS process with its own copy of these values in
# memory, so there's no shared-state risk between them — only real cost is
# CPU contention, which the M5's multiple cores absorb fine for two audio
# extraction jobs.
# ---------------------------------------------------------------------------
INPUT_ROOT   = str(RAW_DIR / "Room 2 (June)")
OUTPUT_DIR   = str(FEATURES_DIR / "audio")
ROOM_LABEL   = "Room2"
MONTH_TAG    = "2025-06"   # <-- change per batch (e.g. 2025-08). Keeps each month
                           #     in its own file so runs never overwrite each other.

AUDIO_EXTENSIONS = (".wav", ".WAV", ".flac", ".FLAC")
GAP_REPORT_THRESHOLD_HOURS = 2.0   # holes bigger than this get flagged individually

SR           = 16000     # resample target; 16 kHz covers flock vocal range
N_FFT        = 1024
HOP          = 512       # ~32 ms hop at 16 kHz -> short-frame cadence
N_MELS       = 64

# --- Denoising (stationary spectral gating; matches the wav2vec2 pipeline) -----
DENOISE            = True
DENOISE_PROP       = 0.80
DENOISE_STATIONARY = True

VOC_BAND_HZ      = (2000, 6000)
FAN_BAND_HZ      = (0, 500)

# Empirically found by ear (refined over two listening passes): fully
# removing 21 Hz and 64 Hz, and partially reducing (not fully zeroing) 125 Hz,
# gave the cleanest result. That's a much more surgical fix than the broad
# FAN_BAND_HZ range mech_frac uses — this touches only those three tones
# rather than discarding everything under 500 Hz, so genuine low-frequency
# acoustic content that ISN'T the fan doesn't get thrown away too.
# Room-specific — if another room has a different fan/motor, override via
# --notch-freqs-hz / --reduce-freqs-hz.
NOTCH_FREQS_HZ   = [21.0, 64.0]   # fully zeroed
NOTCH_WIDTH_HZ   = 8.0             # +/- around each notch center; at
                                    # SR=16000/N_FFT=1024 (15.625 Hz/bin) this
                                    # reliably captures exactly the nearest
                                    # bin without bleeding into neighbors.
REDUCE_FREQS_HZ  = [125.0]         # partially attenuated, not fully removed
REDUCE_WIDTH_HZ  = 8.0
REDUCE_GAIN      = 0.3             # power multiplier at these bins (1.0 =
                                    # untouched, 0.0 = same as a full notch);
                                    # tune by ear if 125 Hz still bleeds through

VOC_FRAC_THR      = 0.15
VOC_REL_DB_THR    = 2.0
VOC_FLATNESS_THR  = 0.20
VOC_BASELINE_WIN  = 200
TRANSIENT_DB_JUMP = 12.0

# ---------------------------------------------------------------------------
# FILE DISCOVERY — recurses through nested/inconsistently-named date folders
# ---------------------------------------------------------------------------
def find_audio_files(root_dir):
    """Recursively collect audio files under root_dir, regardless of how many
    levels of subfolders exist or how they're named. Fixes the June layout
    where clips live under per-cluster folders like 'Room 2 (6, 7, 8 June)'
    and one folder is misnamed 'Room (23, 25, 30 June)' (no '2').

    Also skips macOS AppleDouble shadow files (e.g. '._S4A27290_....wav').
    These get silently created for every real file when copying to/from
    exFAT/FAT32-formatted external drives, share the same extension as the
    real clip, and always fail to load (they contain metadata, not audio) —
    so they were previously counted in the file total and wasted a load
    attempt (+ slow audioread fallback) on every batch."""
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"INPUT_ROOT does not exist: {root_dir!r}")
    files = [p for p in root.rglob("*")
             if p.suffix in AUDIO_EXTENSIONS and not p.name.startswith("._")]
    return sorted(str(p) for p in files)

# ---------------------------------------------------------------------------
# TIMESTAMP PARSING  — the single most important thing to get right
# ---------------------------------------------------------------------------
FILENAME_TS_REGEX = r"(\d{8}_\d{6})"       # captures 20250701_083000
FILENAME_TS_FMT   = "%Y%m%d_%H%M%S"

def parse_start_time(path):
    """Return the wall-clock start time of a clip. Falls back to file mtime."""
    name = os.path.basename(path)
    m = re.search(FILENAME_TS_REGEX, name)
    if m:
        try:
            return datetime.strptime(m.group(1), FILENAME_TS_FMT), "filename"
        except ValueError:
            pass
    return datetime.fromtimestamp(os.path.getmtime(path)), "mtime"

# ---------------------------------------------------------------------------
# FAN-TONE GAIN MASK — notch (zero) + partial-reduce, built once per call
# ---------------------------------------------------------------------------
def build_fan_gain(freqs):
    """Per-frequency-bin POWER multiplier: 0.0 at notched tones, REDUCE_GAIN
    at partially-attenuated tones, 1.0 everywhere else. Applied to the power
    spectrogram (S) before recomputing centroid/STE so those two descriptors
    stop being dominated by the fan's specific tonal frequencies, without
    discarding the whole low-frequency range the way a blanket high-pass
    (or FAN_BAND_HZ) would.

    Reads NOTCH_FREQS_HZ / REDUCE_FREQS_HZ etc. as module globals AT CALL
    TIME (not as default-argument values) specifically so --notch-freqs-hz /
    --reduce-freqs-hz CLI overrides in main() actually take effect — Python
    binds default-argument values once at function-definition time, which
    would silently ignore any later `global` reassignment."""
    gain = np.ones_like(freqs)
    for center in NOTCH_FREQS_HZ:
        gain[(freqs >= center - NOTCH_WIDTH_HZ) & (freqs <= center + NOTCH_WIDTH_HZ)] = 0.0
    for center in REDUCE_FREQS_HZ:
        mask = (freqs >= center - REDUCE_WIDTH_HZ) & (freqs <= center + REDUCE_WIDTH_HZ)
        gain[mask] = np.minimum(gain[mask], REDUCE_GAIN)  # don't un-notch an overlapping notch
    return gain

# ---------------------------------------------------------------------------
# PER-FILE FEATURE EXTRACTION
# ---------------------------------------------------------------------------
def extract_file(path):
    y_raw, _ = librosa.load(path, sr=SR, mono=True)
    if y_raw.size == 0:
        return None
    start_time, ts_source = parse_start_time(path)

    S_raw = np.abs(librosa.stft(y_raw, n_fft=N_FFT, hop_length=HOP)) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    total_raw = S_raw.sum(axis=0) + 1e-10
    fan_idx = (freqs >= FAN_BAND_HZ[0]) & (freqs < FAN_BAND_HZ[1])
    mech_frac = S_raw[fan_idx, :].sum(axis=0) / total_raw

    if DENOISE:
        import noisereduce as nr
        y = nr.reduce_noise(y=y_raw, sr=SR,
                            stationary=DENOISE_STATIONARY,
                            prop_decrease=DENOISE_PROP).astype(np.float32)
    else:
        y = y_raw

    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)) ** 2
    mel = librosa.feature.melspectrogram(S=S, sr=SR, n_mels=N_MELS)
    logmel = librosa.power_to_db(mel + 1e-10)

    centroid  = librosa.feature.spectral_centroid(S=np.sqrt(S), sr=SR)[0]
    rolloff   = librosa.feature.spectral_rolloff(S=np.sqrt(S), sr=SR, roll_percent=0.85)[0]
    flatness  = librosa.feature.spectral_flatness(S=np.sqrt(S))[0]
    zcr       = librosa.feature.zero_crossing_rate(y, frame_length=2 * HOP, hop_length=HOP)[0]
    rms       = librosa.feature.rms(S=np.sqrt(S), frame_length=N_FFT)[0]
    rms_db    = 20.0 * np.log10(rms + 1e-10)
    flux = np.sqrt(np.sum(np.diff(np.sqrt(S), axis=1).clip(min=0) ** 2, axis=0))
    flux = np.concatenate([[0.0], flux])

    total_e = S.sum(axis=0) + 1e-10
    # Short-Time Energy: per-frame power summed across all frequency bins
    # (denoised signal, same processing chain as rms_db/centroid/etc — unlike
    # mech_frac, which deliberately stays on raw audio). Power-domain dB uses
    # 10*log10, not the 20*log10 used for the amplitude-like RMS.
    ste_db = 10.0 * np.log10(total_e)

    # --- Fan-tone-notched versions of centroid/STE (see build_fan_gain) ---
    # centroid_hz/ste_db above are KEPT as-is (nothing removed) so before/
    # after can still be compared via baseline.py's correlation diagnostic.
    # These "_clean" versions are the ones that should actually decouple from
    # mech_frac_mean once the fan tones are gone.
    fan_gain = build_fan_gain(freqs)
    S_clean = S * fan_gain[:, None]
    mag_clean = np.sqrt(S_clean)
    total_e_clean = S_clean.sum(axis=0) + 1e-10
    centroid_clean = (freqs[:, None] * mag_clean).sum(axis=0) / (mag_clean.sum(axis=0) + 1e-10)
    ste_db_clean = 10.0 * np.log10(total_e_clean)

    def band_frac(lo, hi):
        idx = (freqs >= lo) & (freqs < hi)
        return S[idx, :].sum(axis=0) / total_e
    mid_frac  = band_frac(500, 2000)
    voc_frac  = band_frac(*VOC_BAND_HZ)
    high_frac = band_frac(6000, SR / 2)

    rms_baseline = (pd.Series(rms_db)
                    .rolling(VOC_BASELINE_WIN, min_periods=10, center=True)
                    .median().to_numpy())
    rel_db = rms_db - rms_baseline
    voc_active = (voc_frac > VOC_FRAC_THR) & \
                 (rel_db > VOC_REL_DB_THR) & \
                 (flatness < VOC_FLATNESS_THR)

    base = pd.Series(rms_db).rolling(50, min_periods=5, center=True).median().to_numpy()
    transient = (rms_db - base) > TRANSIENT_DB_JUMP

    n = min(centroid.shape[0], zcr.shape[0], rms_db.shape[0], flux.shape[0], ste_db.shape[0],
            centroid_clean.shape[0], ste_db_clean.shape[0])
    centroid, rolloff, flatness = centroid[:n], rolloff[:n], flatness[:n]
    zcr, rms_db, flux = zcr[:n], rms_db[:n], flux[:n]
    mech_frac, mid_frac = mech_frac[:n], mid_frac[:n]
    voc_frac, high_frac = voc_frac[:n], high_frac[:n]
    voc_active, transient = voc_active[:n], transient[:n]
    ste_db = ste_db[:n]
    centroid_clean, ste_db_clean = centroid_clean[:n], ste_db_clean[:n]

    frame_times = np.arange(n) * HOP / SR
    wall = [start_time + timedelta(seconds=float(t)) for t in frame_times]

    df = pd.DataFrame({
        "time": wall,
        "rms_db": rms_db, "centroid_hz": centroid, "rolloff_hz": rolloff,
        "flatness": flatness, "zcr": zcr, "flux": flux,
        "mech_frac": mech_frac, "mid_frac": mid_frac,
        "voc_frac": voc_frac, "high_frac": high_frac,
        "ste_db": ste_db,
        "centroid_hz_clean": centroid_clean, "ste_db_clean": ste_db_clean,
        "voc_active": voc_active.astype(float),
        "transient": transient.astype(float),
    })
    df["source_file"] = os.path.basename(path)
    df["ts_source"] = ts_source
    return df

# ---------------------------------------------------------------------------
# PARALLEL EXECUTION HELPERS — extract_file() is pure CPU (librosa/numpy/scipy,
# no GPU calls anywhere in this pipeline), so the only way to go faster is
# more cores working at once, not a bigger GPU. ProcessPoolExecutor uses
# 'spawn' on macOS, which re-imports this module fresh in each worker — so
# CLI-overridden NOTCH_FREQS_HZ etc. (set via `global` in main(), see below)
# would NOT reach the workers unless explicitly re-applied there. _init_worker
# does that once per worker process at pool startup.
# ---------------------------------------------------------------------------
def _init_worker(notch_freqs, notch_width, reduce_freqs, reduce_width, reduce_gain):
    global NOTCH_FREQS_HZ, NOTCH_WIDTH_HZ, REDUCE_FREQS_HZ, REDUCE_WIDTH_HZ, REDUCE_GAIN
    NOTCH_FREQS_HZ, NOTCH_WIDTH_HZ = notch_freqs, notch_width
    REDUCE_FREQS_HZ, REDUCE_WIDTH_HZ, REDUCE_GAIN = reduce_freqs, reduce_width, reduce_gain

def _extract_one(path):
    """Top-level (picklable) wrapper so per-file errors don't kill the pool —
    returns (path, df_or_None, error_message_or_None) instead of raising."""
    try:
        return path, extract_file(path), None
    except Exception as e:
        return path, None, (str(e) or type(e).__name__)

# ---------------------------------------------------------------------------
# AGGREGATION
# ---------------------------------------------------------------------------
DESCRIPTORS = ["rms_db", "centroid_hz", "rolloff_hz", "flatness", "zcr", "flux",
               "mech_frac", "mid_frac", "voc_frac", "high_frac", "ste_db",
               "centroid_hz_clean", "ste_db_clean"]

def aggregate(frame_df, freq):
    """Aggregate frame-level features to a time bin ('min' or 'h'). Only bins
    with actual frames are emitted — missing periods are NOT zero-filled or
    interpolated here, they simply don't appear. See add_gap_column() below
    for how that absence gets surfaced explicitly."""
    g = frame_df.set_index("time").groupby(pd.Grouper(freq=freq))
    out = {}
    for col in DESCRIPTORS:
        out[f"{col}_mean"] = g[col].mean()
        out[f"{col}_std"]  = g[col].std()
        out[f"{col}_p10"]  = g[col].quantile(0.10)
        out[f"{col}_p90"]  = g[col].quantile(0.90)
    out["voc_activity_frac"] = g["voc_active"].mean()
    out["transient_rate"]    = g["transient"].mean()
    out["n_frames"]          = g["rms_db"].count()
    res = pd.DataFrame(out)
    res = res[res["n_frames"] > 0]
    return add_gap_column(res)

def add_gap_column(agg_df):
    """Elapsed time (hours) since the previous non-empty bin. Makes silent
    multi-day jumps visible in the CSV itself, not just in a console log."""
    idx = agg_df.index.to_series()
    agg_df = agg_df.copy()
    agg_df["gap_hours_since_prev"] = idx.diff().dt.total_seconds().div(3600)
    return agg_df

# ---------------------------------------------------------------------------
# TIME COVERAGE REPORT — makes the June recording gaps explicit
# ---------------------------------------------------------------------------
def analyze_time_coverage(frame_df, month_tag, study_start_date=None):
    """Report which calendar days actually have recordings and flag any
    within-data holes larger than GAP_REPORT_THRESHOLD_HOURS. Poultry
    recording here is NOT continuous (validation days were sampled in
    clusters), so downstream models (baseline.py: Holt-Winters / local-linear
    Kalman) must not assume an unbroken hourly index — fit per contiguous
    block, or pass gap info through explicitly.

    study_start_date (optional date): the actual first day monitoring began
    (e.g. 2025-06-05, the flock's first recorded day). Without this, "missing
    days" gets scored against the FULL calendar month, which overstates
    missingness for any month that didn't start on day 1 — June's real
    coverage is 16/26 study days present, not 16/30 calendar days, once the
    4 pre-study days are correctly excluded rather than counted as gaps."""
    days_present = sorted(frame_df["time"].dt.date.unique())
    year, month = map(int, month_tag.split("-"))
    n_days_in_month = calendar.monthrange(year, month)[1]
    all_days = {date(year, month, d) for d in range(1, n_days_in_month + 1)}

    if study_start_date is not None:
        all_days = {d for d in all_days if d >= study_start_date}

    missing_days = sorted(all_days - set(days_present))
    expected_days = len(all_days)

    sorted_times = frame_df["time"].sort_values().reset_index(drop=True)
    gaps = sorted_times.diff().dropna()
    big_gaps = gaps[gaps > pd.Timedelta(hours=GAP_REPORT_THRESHOLD_HOURS)]
    gap_rows = []
    if len(big_gaps):
        gap_starts = sorted_times.shift(1)[big_gaps.index]
        for start, size in zip(gap_starts, big_gaps):
            gap_rows.append({"gap_start": start, "gap_hours": size.total_seconds() / 3600})

    print("\n--- TIME COVERAGE REPORT ---")
    if study_start_date is not None:
        print(f"  Study start date      : {study_start_date.isoformat()} "
              f"(days before this excluded from the denominator, not counted as missing)")
    print(f"  Days with recordings : {len(days_present)} / {expected_days}")
    print(f"  Missing days (within study window): {[d.isoformat() for d in missing_days]}")
    print(f"  Holes > {GAP_REPORT_THRESHOLD_HOURS}h within recorded data: {len(gap_rows)}")
    for row in gap_rows:
        print(f"    {row['gap_start']}  -> +{row['gap_hours']:.1f}h")

    coverage_df = pd.DataFrame({
        "missing_days": pd.Series([d.isoformat() for d in missing_days]),
    })
    coverage_df.attrs["expected_days"] = expected_days
    coverage_df.attrs["days_present"] = len(days_present)
    gaps_df = pd.DataFrame(gap_rows)
    return coverage_df, gaps_df

# ---------------------------------------------------------------------------
# CLI ARGS — lets two months run concurrently in separate terminals without
# editing this file or racing on the same in-memory config.
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Audio feature extraction (per-month batch).")
    p.add_argument("--input-root", default=INPUT_ROOT,
                   help="Root folder to recursively search for audio files (default: June).")
    p.add_argument("--output-dir", default=OUTPUT_DIR,
                   help="Where to write the output CSVs.")
    p.add_argument("--room-label", default=ROOM_LABEL,
                   help="Room tag used in output filenames.")
    p.add_argument("--month-tag", default=MONTH_TAG,
                   help="Month tag (e.g. 2025-08) used in output filenames and the "
                        "calendar-day coverage check.")
    p.add_argument("--study-start-date", default=None,
                   help="Actual first day monitoring began, e.g. 2025-06-05. When set, "
                        "days before this are excluded from the coverage denominator "
                        "instead of being counted as missing. Omit for months that "
                        "started on calendar day 1 (unchanged behavior).")
    p.add_argument("--notch-freqs-hz", type=float, nargs="*", default=NOTCH_FREQS_HZ,
                   help=f"Frequencies (Hz) to fully zero out before recomputing "
                        f"centroid_hz_clean/ste_db_clean (default: {NOTCH_FREQS_HZ}, "
                        f"tuned by ear for Room 2's fan).")
    p.add_argument("--notch-width-hz", type=float, default=NOTCH_WIDTH_HZ,
                   help=f"+/- band around each notch center (default: {NOTCH_WIDTH_HZ}).")
    p.add_argument("--reduce-freqs-hz", type=float, nargs="*", default=REDUCE_FREQS_HZ,
                   help=f"Frequencies (Hz) to partially attenuate rather than fully "
                        f"notch (default: {REDUCE_FREQS_HZ}).")
    p.add_argument("--reduce-width-hz", type=float, default=REDUCE_WIDTH_HZ,
                   help=f"+/- band around each reduce center (default: {REDUCE_WIDTH_HZ}).")
    p.add_argument("--reduce-gain", type=float, default=REDUCE_GAIN,
                   help=f"Power multiplier at reduce-band bins, 0-1 (default: {REDUCE_GAIN}).")
    p.add_argument("--workers", type=int, default=None,
                   help="Parallel worker processes for feature extraction (default: all "
                        "CPU cores). This pipeline is pure CPU (librosa/numpy/scipy) — "
                        "it does not use a GPU, so this is the actual speed lever, not "
                        "which machine/GPU you run it on. Set to 1 to force serial "
                        "execution (e.g. for debugging).")
    return p.parse_args()

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    global NOTCH_FREQS_HZ, NOTCH_WIDTH_HZ, REDUCE_FREQS_HZ, REDUCE_WIDTH_HZ, REDUCE_GAIN
    args = parse_args()
    input_root, output_dir = args.input_root, args.output_dir
    room_label, month_tag = args.room_label, args.month_tag
    study_start_date = (datetime.strptime(args.study_start_date, "%Y-%m-%d").date()
                         if args.study_start_date else None)

    # Reassigned here (not read as function-default args) so build_fan_gain()
    # picks these up at call time — see the note in build_fan_gain()'s docstring.
    NOTCH_FREQS_HZ  = args.notch_freqs_hz
    NOTCH_WIDTH_HZ  = args.notch_width_hz
    REDUCE_FREQS_HZ = args.reduce_freqs_hz
    REDUCE_WIDTH_HZ = args.reduce_width_hz
    REDUCE_GAIN     = args.reduce_gain
    if (NOTCH_FREQS_HZ != [21.0, 64.0] or REDUCE_FREQS_HZ != [125.0]
            or REDUCE_GAIN != 0.3):
        print(f"[fan-filter] using non-default settings: notch={NOTCH_FREQS_HZ}"
              f"(+/-{NOTCH_WIDTH_HZ}Hz)  reduce={REDUCE_FREQS_HZ}"
              f"(+/-{REDUCE_WIDTH_HZ}Hz, gain={REDUCE_GAIN})")

    files = find_audio_files(input_root)
    if not files:
        raise FileNotFoundError(f"No audio files found under {input_root!r}")
    print(f"Found {len(files)} audio files under {input_root}")

    n_workers = args.workers or os.cpu_count() or 1
    print(f"Extracting with {n_workers} worker process(es) "
          f"(CPU-bound pipeline — more cores is the lever, not a GPU)")

    frames, sources = [], {}
    failed_files = []
    init_args = (NOTCH_FREQS_HZ, NOTCH_WIDTH_HZ, REDUCE_FREQS_HZ, REDUCE_WIDTH_HZ, REDUCE_GAIN)
    with cf.ProcessPoolExecutor(max_workers=n_workers,
                                 initializer=_init_worker, initargs=init_args) as ex:
        futures = {ex.submit(_extract_one, f): f for f in files}
        for i, fut in enumerate(cf.as_completed(futures), 1):
            f = futures[fut]
            _, df, reason = fut.result()
            base = os.path.basename(f)
            if reason is not None:
                print(f"  [WARN] failed on {base}: {reason}")
                failed_files.append({"file": base, "reason": reason})
                continue
            if df is None:
                print(f"  [WARN] empty audio: {base}")
                failed_files.append({"file": base, "reason": "empty audio"})
                continue
            frames.append(df)
            src = df["ts_source"].iloc[0]
            sources[src] = sources.get(src, 0) + 1
            print(f"  [{i}/{len(files)}] {base}  "
                  f"{df['time'].iloc[0]} -> {df['time'].iloc[-1]}  ({len(df)} frames)")

    if failed_files:
        print(f"\n--- FAILED FILES: {len(failed_files)} / {len(files)} "
              f"({len(failed_files)/len(files)*100:.1f}%) could not be loaded ---")
        for row in failed_files:
            print(f"    {row['file']}: {row['reason']}")
        if len(failed_files) / len(files) > 0.05:
            print("  [!] Over 5% of files failed — worth checking for a systemic issue "
                  "(bad SD card, wrong codec) rather than assuming isolated corruption.")

    all_frames = pd.concat(frames, ignore_index=True).sort_values("time")

    print("\n--- TIMESTAMP SOURCE ---")
    for k, v in sources.items():
        print(f"  {k}: {v} files")
    if "mtime" in sources:
        print("  [!] Some files used mtime fallback — verify these are correct.")

    coverage_df, gaps_df = analyze_time_coverage(all_frames, month_tag, study_start_date)

    minute = aggregate(all_frames, "min")
    hourly = aggregate(all_frames, "h")

    os.makedirs(output_dir, exist_ok=True)
    minute_path = os.path.join(output_dir, f"audio_features_minute_{room_label}_{month_tag}.csv")
    hourly_path = os.path.join(output_dir, f"audio_features_hourly_{room_label}_{month_tag}.csv")
    coverage_path = os.path.join(output_dir, f"audio_features_{room_label}_{month_tag}_time_coverage.csv")
    gaps_path = os.path.join(output_dir, f"audio_features_{room_label}_{month_tag}_gaps.csv")
    failed_path = os.path.join(output_dir, f"audio_features_{room_label}_{month_tag}_failed_files.csv")

    minute.to_csv(minute_path)
    hourly.to_csv(hourly_path)
    coverage_df.to_csv(coverage_path, index=False)
    gaps_df.to_csv(gaps_path, index=False)
    pd.DataFrame(failed_files).to_csv(failed_path, index=False)

    print(f"\nSaved minute-level: {minute.shape}  hourly: {hourly.shape}")
    print(f"  -> {minute_path}")
    print(f"  -> {hourly_path}")
    print(f"  -> {coverage_path}  (missing days within study window)")
    print(f"  -> {gaps_path}  (holes > {GAP_REPORT_THRESHOLD_HOURS}h within recorded data)")
    print(f"  -> {failed_path}  ({len(failed_files)} file(s) that could not be loaded)")

    print("\n--- SANITY CHECK (do these look plausible?) ---")
    print(f"  centroid_hz mean : {all_frames['centroid_hz'].mean():.0f} Hz "
          f"(chicks high, mature birds lower)")
    print(f"  voc_activity     : {all_frames['voc_active'].mean():.3f} "
          f"(fraction vocal-active — tune thresholds if ~0 or ~1)")
    print(f"  transient_rate   : {all_frames['transient'].mean():.4f} "
          f"(spikes = doors/humans/impacts)")
    print(f"  ste_db mean      : {all_frames['ste_db'].mean():.1f} dB "
          f"(short-time energy — should track rms_db closely but is not identical "
          f"after hourly aggregation; compare trends, don't expect exact overlap)")

    print("\n--- FAN-NOTCH BEFORE/AFTER (centroid/STE with fan tones removed) ---")
    c_raw, c_clean = all_frames["centroid_hz"], all_frames["centroid_hz_clean"]
    s_raw, s_clean = all_frames["ste_db"], all_frames["ste_db_clean"]
    print(f"  centroid_hz      : raw={c_raw.mean():.0f} Hz  ->  clean={c_clean.mean():.0f} Hz "
          f"(delta={c_clean.mean() - c_raw.mean():+.0f} Hz)")
    print(f"  ste_db           : raw={s_raw.mean():.1f} dB  ->  clean={s_clean.mean():.1f} dB "
          f"(delta={s_clean.mean() - s_raw.mean():+.1f} dB)")
    print("  Expect centroid_clean to shift UP (fan energy pulled it down) and "
          "ste_db_clean to shift DOWN (less total energy once fan tones are removed). "
          "Zero/near-zero delta on either usually means the notch bins fell between "
          "spectral bins for this SR/N_FFT, or DENOISE already suppressed the tones "
          "upstream — worth a spot check, not just trusting the number.")
    print("  This is a mean over the whole month, not proof of decoupling — re-run "
          "baseline.py's correlation diagnostic on centroid_hz_clean/ste_db_clean vs "
          "mech_frac_mean per segment to confirm the fan confound actually dropped.")

    print("\n--- CALIBRATION DIAGNOSTICS (set thresholds from these) ---")
    vf = all_frames["voc_frac"]
    fl = all_frames["flatness"]
    mf = all_frames["mech_frac"]
    print(f"  voc_frac   (2-6 kHz energy frac)  "
          f"p50={vf.median():.3f}  p75={vf.quantile(.75):.3f}  p90={vf.quantile(.90):.3f}"
          f"   -> VOC_FRAC_THR near p75-p90")
    print(f"  flatness   (tonal<..<noise)       "
          f"p25={fl.quantile(.25):.3f}  p50={fl.median():.3f}  p75={fl.quantile(.75):.3f}"
          f"   -> VOC_FLATNESS_THR near p25-p50")
    print(f"  mech_frac  (0-500 Hz fan band)    "
          f"p50={mf.median():.3f}  (kept as a ventilation/growth proxy, not removed)")
    print("  NOTE: after adjusting, open one HIGH-voc_active minute and one LOW one")
    print("        and LISTEN — confirm the flag matches audible bird calls.")


if __name__ == "__main__":
    main()