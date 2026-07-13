"""
Audio feature extraction for poultry-barn multimodal anomaly detection.

Reads a folder of audio files (WAV / FLAC / MP4-audio), computes interpretable
frame-level descriptors from a log-mel base, tags every frame with a wall-clock
timestamp, and aggregates to MINUTE-level and HOUR-level tables.

Output tables:
  - audio_features_minute_Room2.csv   (base granularity)
  - audio_features_hourly_Room2.csv    (slow-band cadence)

Band assignment (for later modelling):
  FAST band  -> per-frame / per-minute descriptors, vocalization-activity index,
                transient rate (also feeds the human-presence detector)
  SLOW band  -> hourly aggregates; watch spectral_centroid + voc_band_frac trend
                across weeks as the audio GROWTH signal (chicks high-pitched ->
                mature birds lower-pitched)

IMPORTANT: this is a plumbing/validation script. On a few days of data you are
checking (a) timestamps parse, (b) values are sane, (c) diurnal rhythm shows.
You are NOT reading any slow trajectory yet.
"""

import glob
import os
import re
from datetime import datetime, timedelta

import librosa
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG  — edit these
# ---------------------------------------------------------------------------
INPUT_GLOB   = "/home/daniel/workspace/projects/longhorizon/data/raw_room2/audio/July/*.wav"   # <-- point at your clips
OUTPUT_DIR   = "/home/daniel/workspace/projects/longhorizon/data/features_room2/audio"
ROOM_LABEL   = "Room2"
MONTH_TAG    = "2025-07"   # <-- change per batch (e.g. 2025-08). Keeps each month
                           #     in its own file so runs never overwrite each other.

SR           = 16000     # resample target; 16 kHz covers flock vocal range
N_FFT        = 1024
HOP          = 512       # ~32 ms hop at 16 kHz -> short-frame cadence
N_MELS       = 64

# --- Denoising (stationary spectral gating; matches the wav2vec2 pipeline) -----
# The barn is fan-dominated (~99% of energy in 0-500 Hz), so light stationary
# denoising helps recover the faint 2-6 kHz vocalization band. Conservative on
# purpose. IMPORTANT: fan-band features (mech_frac) are measured on the RAW audio
# BEFORE denoising, so we don't erase the ventilation/growth signal we track.
DENOISE            = True    # toggle: run with True vs False to A/B whether it helps
DENOISE_PROP       = 0.80    # <1.0 under-cleans: keeps ~20% ambient bed (protects calls)
DENOISE_STATIONARY = True    # stationary mode targets the steady hum, spares transients

# --- Vocalization band (Hz): where hen/chick calls concentrate. TUNE by listening.
VOC_BAND_HZ      = (2000, 6000)
FAN_BAND_HZ      = (0, 500)      # steady mechanical/fan hum lives here (now a KEPT feature)
#
# Vocalization gate is now RELATIVE + BAND-RATIO based, so a constant fan floor
# no longer saturates it, and it transfers across weeks as flock loudness drifts.
# A frame is "vocalization-active" if BOTH:
#   (1) its voc-band energy fraction is high enough (call energy, not hum), AND
#   (2) it is loud RELATIVE to a rolling baseline of recent frames (a real call,
#       not the ambient bed).
# Tonality (low flatness) is used as an additional gate to reject broadband noise.
VOC_FRAC_THR      = 0.15    # min fraction of energy in 2-6 kHz to count as a call
VOC_REL_DB_THR    = 2.0     # frame RMS must exceed rolling baseline by this many dB
                            #   (GENTLE by default: band-ratio + tonality do the main
                            #    work; raise this only if hum-modulation leaks through)
VOC_FLATNESS_THR  = 0.20    # tonal gate: spectral flatness BELOW this = tonal
                            #   (recorder-dependent; set from printed p25-p50 diagnostic)
VOC_BASELINE_WIN  = 200     # frames (~6 s) for the rolling loudness baseline
# --- Transient (door/human/impulse) detection for the human-presence covariate:
TRANSIENT_DB_JUMP = 12.0   # frame RMS rising >12 dB above local baseline

# ---------------------------------------------------------------------------
# TIMESTAMP PARSING  — the single most important thing to get right
# ---------------------------------------------------------------------------
# Try to read a start time from the filename. Adjust FILENAME_TS_REGEX /
# FILENAME_TS_FMT to match your files, e.g. "ROOM2_20250701_083000.WAV".
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
    # Fallback: file modification time (LESS reliable — verify!)
    return datetime.fromtimestamp(os.path.getmtime(path)), "mtime"

# ---------------------------------------------------------------------------
# PER-FILE FEATURE EXTRACTION
# ---------------------------------------------------------------------------
def extract_file(path):
    y_raw, _ = librosa.load(path, sr=SR, mono=True)
    if y_raw.size == 0:
        return None
    start_time, ts_source = parse_start_time(path)

    # --- RAW spectrogram: used for the fan/ventilation band, which we do NOT denoise
    S_raw = np.abs(librosa.stft(y_raw, n_fft=N_FFT, hop_length=HOP)) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    total_raw = S_raw.sum(axis=0) + 1e-10
    fan_idx = (freqs >= FAN_BAND_HZ[0]) & (freqs < FAN_BAND_HZ[1])
    mech_frac = S_raw[fan_idx, :].sum(axis=0) / total_raw   # ventilation proxy (raw)

    # --- Optional stationary denoise to recover the faint 2-6 kHz vocal band ---
    if DENOISE:
        import noisereduce as nr
        y = nr.reduce_noise(y=y_raw, sr=SR,
                            stationary=DENOISE_STATIONARY,
                            prop_decrease=DENOISE_PROP).astype(np.float32)
    else:
        y = y_raw

    # Short-frame spectral features (on denoised audio if enabled)
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)) ** 2  # power
    mel = librosa.feature.melspectrogram(S=S, sr=SR, n_mels=N_MELS)
    logmel = librosa.power_to_db(mel + 1e-10)

    centroid  = librosa.feature.spectral_centroid(S=np.sqrt(S), sr=SR)[0]
    rolloff   = librosa.feature.spectral_rolloff(S=np.sqrt(S), sr=SR, roll_percent=0.85)[0]
    flatness  = librosa.feature.spectral_flatness(S=np.sqrt(S))[0]
    zcr       = librosa.feature.zero_crossing_rate(y, frame_length=2 * HOP, hop_length=HOP)[0]
    rms       = librosa.feature.rms(S=np.sqrt(S), frame_length=N_FFT)[0]
    rms_db    = 20.0 * np.log10(rms + 1e-10)
    # Spectral flux (positive changes between consecutive frames)
    flux = np.sqrt(np.sum(np.diff(np.sqrt(S), axis=1).clip(min=0) ** 2, axis=0))
    flux = np.concatenate([[0.0], flux])

    # Band-energy fractions on the (possibly denoised) signal
    total_e = S.sum(axis=0) + 1e-10
    def band_frac(lo, hi):
        idx = (freqs >= lo) & (freqs < hi)
        return S[idx, :].sum(axis=0) / total_e
    mid_frac  = band_frac(500, 2000)
    voc_frac  = band_frac(*VOC_BAND_HZ)
    high_frac = band_frac(6000, SR / 2)

    # Vocalization-active frames: relative loudness AND call-band energy AND tonal.
    # Rolling baseline = local ambient level, so a steady fan floor cancels out.
    rms_baseline = (pd.Series(rms_db)
                    .rolling(VOC_BASELINE_WIN, min_periods=10, center=True)
                    .median().to_numpy())
    rel_db = rms_db - rms_baseline
    voc_active = (voc_frac > VOC_FRAC_THR) & \
                 (rel_db > VOC_REL_DB_THR) & \
                 (flatness < VOC_FLATNESS_THR)

    # Transient frames: RMS jump above a local rolling baseline
    base = pd.Series(rms_db).rolling(50, min_periods=5, center=True).median().to_numpy()
    transient = (rms_db - base) > TRANSIENT_DB_JUMP

    # STFT-derived features and ZCR can differ by one frame — trim to common length
    n = min(centroid.shape[0], zcr.shape[0], rms_db.shape[0], flux.shape[0])
    centroid, rolloff, flatness = centroid[:n], rolloff[:n], flatness[:n]
    zcr, rms_db, flux = zcr[:n], rms_db[:n], flux[:n]
    mech_frac, mid_frac = mech_frac[:n], mid_frac[:n]
    voc_frac, high_frac = voc_frac[:n], high_frac[:n]
    voc_active, transient = voc_active[:n], transient[:n]

    frame_times = np.arange(n) * HOP / SR
    wall = [start_time + timedelta(seconds=float(t)) for t in frame_times]

    df = pd.DataFrame({
        "time": wall,
        "rms_db": rms_db, "centroid_hz": centroid, "rolloff_hz": rolloff,
        "flatness": flatness, "zcr": zcr, "flux": flux,
        "mech_frac": mech_frac, "mid_frac": mid_frac,
        "voc_frac": voc_frac, "high_frac": high_frac,
        "voc_active": voc_active.astype(float),
        "transient": transient.astype(float),
    })
    df["source_file"] = os.path.basename(path)
    df["ts_source"] = ts_source
    return df

# ---------------------------------------------------------------------------
# AGGREGATION
# ---------------------------------------------------------------------------
DESCRIPTORS = ["rms_db", "centroid_hz", "rolloff_hz", "flatness", "zcr", "flux",
               "mech_frac", "mid_frac", "voc_frac", "high_frac"]

def aggregate(frame_df, freq):
    """Aggregate frame-level features to a time bin ('min' or 'h')."""
    g = frame_df.set_index("time").groupby(pd.Grouper(freq=freq))
    out = {}
    for col in DESCRIPTORS:
        out[f"{col}_mean"] = g[col].mean()
        out[f"{col}_std"]  = g[col].std()
        out[f"{col}_p10"]  = g[col].quantile(0.10)
        out[f"{col}_p90"]  = g[col].quantile(0.90)
    out["voc_activity_frac"] = g["voc_active"].mean()   # fraction of frames vocal-active
    out["transient_rate"]    = g["transient"].mean()    # fraction of frames transient
    out["n_frames"]          = g["rms_db"].count()
    res = pd.DataFrame(out)
    return res[res["n_frames"] > 0]

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    files = sorted(glob.glob(INPUT_GLOB))
    if not files:
        raise FileNotFoundError(f"No files matched {INPUT_GLOB!r}")
    print(f"Found {len(files)} audio files")

    frames, sources = [], {}
    for i, f in enumerate(files, 1):
        try:
            df = extract_file(f)
        except Exception as e:
            print(f"  [WARN] failed on {os.path.basename(f)}: {e}")
            continue
        if df is None:
            print(f"  [WARN] empty audio: {os.path.basename(f)}")
            continue
        frames.append(df)
        sources[df["ts_source"].iloc[0]] = sources.get(df["ts_source"].iloc[0], 0) + 1
        print(f"  [{i}/{len(files)}] {os.path.basename(f)}  "
              f"{df['time'].iloc[0]} -> {df['time'].iloc[-1]}  ({len(df)} frames)")

    all_frames = pd.concat(frames, ignore_index=True).sort_values("time")

    # ---- timestamp sanity report (READ THIS) ----
    print("\n--- TIMESTAMP SOURCE ---")
    for k, v in sources.items():
        print(f"  {k}: {v} files")
    if "mtime" in sources:
        print("  [!] Some files used mtime fallback — verify these are correct.")

    minute = aggregate(all_frames, "min")
    hourly = aggregate(all_frames, "h")

    minute_path = os.path.join(OUTPUT_DIR, f"audio_features_minute_{ROOM_LABEL}_{MONTH_TAG}.csv")
    hourly_path = os.path.join(OUTPUT_DIR, f"audio_features_hourly_{ROOM_LABEL}_{MONTH_TAG}.csv")
    minute.to_csv(minute_path)
    hourly.to_csv(hourly_path)
    print(f"\nSaved minute-level: {minute.shape}  hourly: {hourly.shape}")
    print(f"  -> {minute_path}")
    print(f"  -> {hourly_path}")

    # ---- quick sanity numbers ----
    print("\n--- SANITY CHECK (do these look plausible?) ---")
    print(f"  centroid_hz mean : {all_frames['centroid_hz'].mean():.0f} Hz "
          f"(chicks high, mature birds lower)")
    print(f"  voc_activity     : {all_frames['voc_active'].mean():.3f} "
          f"(fraction vocal-active — tune thresholds if ~0 or ~1)")
    print(f"  transient_rate   : {all_frames['transient'].mean():.4f} "
          f"(spikes = doors/humans/impacts)")

    # ---- calibration diagnostics: use these to set the gate, don't guess ----
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