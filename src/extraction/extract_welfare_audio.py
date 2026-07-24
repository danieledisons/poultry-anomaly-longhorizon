#!/usr/bin/env python3
"""
WELFARE-RELEVANT AUDIO FEATURE EXTRACTION  (poultry vocalisation biomarkers).

Replaces the broadband hourly log-mel MEANS (dominated by ventilation noise) with a
welfare-grounded feature set drawn from the poultry-acoustics literature (Lee15,
Du20, Tao22, Mah21, Man25b). Produces per-hour aggregates.

Runs on the SERVER (raw audio) with a GPU optional. Writes to a NEW filename;
never overwrites existing features.

Feature groups (per analysis frame, then aggregated to the hour):
  energy/amplitude : RMS, short-time energy, amplitude envelope stats
  spectral shape   : centroid, bandwidth/spread, rolloff, flatness, contrast,
                     spectral entropy, high-frequency-energy fraction
  cepstral         : MFCC 1-13 + delta + delta-delta
  band energies    : fan/mechanical band, vocalisation band, high band, and the
                     fan-normalised voc/mech ratio (gain- and ventilation-robust)
  voice/pitch      : F0 mean/max/min/range/IQR/std, voiced fraction (Praat)
  perturbation     : jitter, shimmer, harmonic-to-noise ratio (Praat)
  formants         : F1, F2, F3 (Praat)
  disorder         : wavelet entropy (respiratory-disease sensitive, Mah21)
  zero crossings   : ZCR
Event / call statistics (hourly, from a harmonicity-gated call detector):
  call_rate (calls/min), inter-call interval median & IQR, call duration median &
  IQR, call energy p50/p90, peak-frequency p50/p90, burstiness (Fano factor),
  chorus_index (fraction of frames with overlapping calls)

Aggregation: for each continuous descriptor we store hourly mean, std, p10, p50,
p90; for events we store hourly counts/rates. Output is one row per hour.

Improved fan-noise handling (beyond the existing stationary denoise):
  1. stationary spectral gating (as before, keeps 20% ambient bed)
  2. adaptive spectral-subtraction of the persistent fan spectrum estimated from the
     quietest frames of each file
  3. harmonic notch at the estimated fan fundamental + harmonics
This is applied for FEATURE extraction; the raw file is never modified.

Dependencies: numpy pandas soundfile librosa scipy noisereduce pywt praat-parselmouth
  pip install soundfile librosa noisereduce pywt praat-parselmouth --upgrade

Usage:
  python src/extraction/extract_welfare_audio.py \
      --room "Room 2" --in "/mnt/.../Audio data/Room2" \
      --out features/audio_welfare_biomarkers_Room2.csv
  (point --in at the room folder; it recurses into session subfolders)
"""
from __future__ import annotations
import argparse, logging, re, sys, signal
from pathlib import Path
from datetime import datetime, timedelta


class _FileTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _FileTimeout()

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from scipy.signal import butter, sosfilt
import noisereduce as nr
import pywt

try:
    import parselmouth
    from parselmouth.praat import call as praat_call
    HAVE_PRAAT = True
except Exception:
    HAVE_PRAAT = False

# ---------------------------------------------------------------------------
# CONFIG — single source of truth, identical for every room (never per-room).
# ---------------------------------------------------------------------------
TARGET_SR   = 16_000
HPF_HZ      = 120       # high-pass: remove subsonic rumble AND most of the fan fundamental
FRAME_S     = 1.0        # analysis frame length (s) for spectral/cepstral features
HOP_S       = 0.5        # frame hop (s)
# --- DENOISE (heavier, two-stage; override on the CLI) ---
DENOISE          = True
DENOISE_TWO_PASS = True   # pass 1 non-stationary (tracks drifting fan), pass 2 stationary
PROP_DECREASE_NS = 0.90   # non-stationary strength (time-varying noise)
PROP_DECREASE_ST = 0.90   # stationary strength (steady fan bed)
FAN_SUBTRACT     = True
PEAK_TARGET = 0.97
CLIP_FRAC_THR = 0.005

# Frequency bands (Hz) — tune to your flock/room if the fan differs.
FAN_BAND  = (80, 500)       # ventilation / mechanical
VOC_BAND  = (1500, 5000)    # chicken vocalisation range
HIGH_BAND = (5000, 8000)

# Call detector
CALL_MIN_DUR_S = 0.05
CALL_MAX_DUR_S = 2.0
CALL_SNR_DB    = 6.0        # voc-band energy above adaptive noise floor to be a call
CALL_MIN_HNR   = 3.0        # harmonicity gate: reject broadband machinery transients
N_MFCC = 13
PRAAT_WIN_S    = 20         # run Praat voice-quality on the loudest 20 s per file only

PERCENTILES = [10, 50, 90]


def _loudest_center(y, w):
    """Index of the center of the loudest w-sample window, in O(n) via cumsum.
    (np.convolve here is direct O(n*w) and hangs for hours on 1-hour files.)"""
    a = np.abs(y)
    if len(a) <= w:
        return len(a) // 2
    c = np.cumsum(a)
    wsum = c[w:] - c[:-w]                 # sliding-window sums, length n-w
    return int(np.argmax(wsum)) + w // 2


def setup_logging(logfile):
    Path(logfile).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(logfile), logging.StreamHandler(sys.stdout)])


# --- timestamping: return a full datetime (with time when the filename has it) ----
FNAME_DT_PATTERNS = [
    (re.compile(r"(\d{8})_(\d{6})"), "%Y%m%d%H%M%S"),   # AudioMoth 20250831_083412
    (re.compile(r"(\d{6})_(\d{6})"), "%y%m%d%H%M%S"),   # YYMMDD_HHMMSS
]
FNAME_DATE_PATTERNS = [
    (re.compile(r"(\d{8})"), "%Y%m%d"),
    (re.compile(r"(\d{6})"), "%y%m%d"),
]

def file_start_datetime(path: Path):
    stem = path.stem
    for pat, fmt in FNAME_DT_PATTERNS:
        m = pat.search(stem)
        if m:
            try:
                return datetime.strptime(m.group(1) + m.group(2), fmt)
            except ValueError:
                pass
    for pat, fmt in FNAME_DATE_PATTERNS:
        m = pat.search(stem)
        if m:
            try:                       # date only -> assume midnight start of file
                return datetime.strptime(m.group(1), fmt)
            except ValueError:
                pass
    try:                               # last resort: file mtime
        return datetime.fromtimestamp(path.stat().st_mtime)
    except Exception:
        return None


# --- preprocessing + improved fan removal ----------------------------------
def _highpass(y, hz):
    if hz <= 0:
        return y
    sos = butter(4, hz, btype="highpass", fs=TARGET_SR, output="sos")
    return sosfilt(sos, y).astype(np.float32)

def _fan_spectral_subtract(y):
    """Estimate the persistent fan spectrum from the quietest 10% of frames and
    subtract it (soft), then notch the fan fundamental + harmonics."""
    S = librosa.stft(y, n_fft=1024, hop_length=512)
    mag, phase = np.abs(S), np.angle(S)
    frame_energy = mag.sum(0)
    quiet = mag[:, frame_energy <= np.percentile(frame_energy, 10)]
    if quiet.shape[1] >= 3:
        noise = np.median(quiet, axis=1, keepdims=True)
        mag = np.maximum(mag - 0.9 * noise, 0.05 * mag)   # soft spectral subtraction
    # notch fan fundamental (dominant peak in FAN_BAND) + first 3 harmonics
    freqs = librosa.fft_frequencies(sr=TARGET_SR, n_fft=1024)
    fanmask = (freqs >= FAN_BAND[0]) & (freqs <= FAN_BAND[1])
    if fanmask.any():
        f0_fan = freqs[fanmask][np.argmax(np.median(mag[fanmask], axis=1))]
        for h in range(1, 4):
            k = np.argmin(np.abs(freqs - h * f0_fan))
            mag[max(k - 1, 0):k + 2] *= 0.3
    return librosa.istft(mag * np.exp(1j * phase), hop_length=512).astype(np.float32)

def preprocess(y, sr, return_stages=False):
    """Explicit, staged preprocessing. NEVER extract features from raw audio.

    Stage 0: mono mix + clip check (quality flag)
    Stage 1: resample to 16 kHz
    Stage 2: high-pass filter (remove subsonic rumble + fan fundamental)
    Stage 3: heavy denoise
             pass 1 = non-stationary spectral gating (tracks the time-varying fan)
             pass 2 = stationary spectral gating (removes the steady fan bed)
    Stage 4: fan spectral subtraction + harmonic notch (residual ventilation lines)
    Stage 5: peak normalisation (cross-file loudness consistency)

    Returns (clean, flags); if return_stages, also returns the post-resample RAW
    signal and the post-denoise signal for before/after inspection.
    """
    flags = {}
    # Stage 0
    if y.ndim > 1:
        y = y.mean(axis=1)
    flags["clip_frac"] = float(np.mean(np.abs(y) > 0.999))
    flags["clipped"] = flags["clip_frac"] > CLIP_FRAC_THR
    # Stage 1
    if sr != TARGET_SR:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
    raw16k = y.copy()                                   # for before/after export
    # Stage 2
    y = _highpass(y, HPF_HZ)
    # Stage 3 (heavy, two-stage)
    if DENOISE:
        y = nr.reduce_noise(y=y, sr=TARGET_SR, stationary=False,
                            prop_decrease=PROP_DECREASE_NS).astype(np.float32)
        if DENOISE_TWO_PASS:
            y = nr.reduce_noise(y=y, sr=TARGET_SR, stationary=True,
                                prop_decrease=PROP_DECREASE_ST).astype(np.float32)
    denoised = y.copy()
    # Stage 4
    if FAN_SUBTRACT:
        y = _fan_spectral_subtract(y)
    # Stage 5
    peak = float(np.max(np.abs(y))) + 1e-9
    y = (y * (PEAK_TARGET / peak)).astype(np.float32)
    if return_stages:
        return y, flags, raw16k, denoised
    return y, flags


# --- band energy helper -----------------------------------------------------
def _band_energy(S_pow, freqs, band):
    m = (freqs >= band[0]) & (freqs < band[1])
    return S_pow[m].sum(0)


def spectral_entropy(S_pow):
    p = S_pow / (S_pow.sum(0, keepdims=True) + 1e-12)
    return -(p * np.log(p + 1e-12)).sum(0) / np.log(p.shape[0])


def wavelet_entropy(y):
    try:
        coeffs = pywt.wavedec(y, "db4", level=5)
        energies = np.array([np.sum(c ** 2) for c in coeffs])
        p = energies / (energies.sum() + 1e-12)
        return float(-(p * np.log(p + 1e-12)).sum())
    except Exception:
        return np.nan


# --- Praat voice-quality (per frame is expensive; done per-file segment) ----
def praat_voice_features(y):
    """F0 stats, jitter, shimmer, HNR, formants, voiced fraction for a signal."""
    out = dict(f0_mean=np.nan, f0_max=np.nan, f0_min=np.nan, f0_range=np.nan,
               f0_iqr=np.nan, f0_std=np.nan, voiced_frac=np.nan,
               jitter=np.nan, shimmer=np.nan, hnr=np.nan,
               f1=np.nan, f2=np.nan, f3=np.nan)
    if not HAVE_PRAAT or len(y) < TARGET_SR // 2:
        return out
    # Praat is the slowest stage; running it on a whole (potentially long) file is
    # impractical. Analyse the LOUDEST PRAAT_WIN_S seconds instead — that is where
    # vocalisations concentrate, and it caps cost per file to a constant.
    if len(y) > PRAAT_WIN_S * TARGET_SR:
        win = int(PRAAT_WIN_S * TARGET_SR)
        c = _loudest_center(y, TARGET_SR)
        lo = max(0, c - win // 2); y = y[lo:lo + win]
    try:
        snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=TARGET_SR)
        pitch = snd.to_pitch(pitch_floor=200, pitch_ceiling=6000)  # chicks are high
        f0 = pitch.selected_array["frequency"]; f0 = f0[f0 > 0]
        if f0.size:
            out.update(f0_mean=float(np.mean(f0)), f0_max=float(np.max(f0)),
                       f0_min=float(np.min(f0)), f0_range=float(np.ptp(f0)),
                       f0_iqr=float(np.subtract(*np.percentile(f0, [75, 25]))),
                       f0_std=float(np.std(f0)),
                       voiced_frac=float(f0.size / max(len(pitch.selected_array), 1)))
        pp = praat_call(snd, "To PointProcess (periodic, cc)", 200, 6000)
        out["jitter"] = float(praat_call(pp, "Get jitter (local)", 0, 0, 1e-4, 0.02, 1.3))
        out["shimmer"] = float(praat_call([snd, pp], "Get shimmer (local)", 0, 0, 1e-4, 0.02, 1.3, 1.6))
        harm = praat_call(snd, "To Harmonicity (cc)", 0.01, 200, 0.1, 1.0)
        out["hnr"] = float(praat_call(harm, "Get mean", 0, 0))
        fm = praat_call(snd, "To Formant (burg)", 0.0, 5, 8000, 0.025, 50)
        for i, key in enumerate(["f1", "f2", "f3"], start=1):
            out[key] = float(praat_call(fm, "Get mean", i, 0, 0, "hertz"))
    except Exception:
        pass
    return out


def detect_calls(voc_e, hnr_frame, hop_s):
    """Return list of (start_idx, dur_frames, peak_energy) from a voc-band energy
    envelope gated by harmonicity. Adaptive noise floor = trailing median."""
    e_db = 10 * np.log10(voc_e + 1e-12)
    floor = pd.Series(e_db).rolling(int(30 / hop_s), min_periods=5).median().to_numpy()
    active = (e_db > floor + CALL_SNR_DB) & (hnr_frame > CALL_MIN_HNR)
    calls = []; i = 0; n = len(active)
    while i < n:
        if active[i]:
            j = i
            while j + 1 < n and active[j + 1]:
                j += 1
            dur = (j - i + 1) * hop_s
            if CALL_MIN_DUR_S <= dur <= CALL_MAX_DUR_S:
                calls.append((i, j - i + 1, float(voc_e[i:j + 1].max())))
            i = j + 1
        else:
            i += 1
    return calls


def export_denoise_sample(path, outdir, seconds=60):
    """Write a before/after clip so the denoiser can be inspected by ear.
    Exports the LOUDEST `seconds` window (where vocalisations concentrate):
      <stem>_00_raw16k.wav      resampled mono, NO denoise
      <stem>_01_denoised.wav    after Stage 3 (heavy denoise), before normalisation
      <stem>_02_final.wav       full pipeline output actually used for features
    """
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    y, sr = sf.read(path)
    final, flags, raw16k, denoised = preprocess(y, sr, return_stages=True)
    win = int(seconds * TARGET_SR)
    if len(raw16k) > win:                               # locate the loudest window (O(n))
        c = _loudest_center(raw16k, TARGET_SR); lo = max(0, c - win // 2)
    else:
        lo = 0; win = len(raw16k)
    sl = slice(lo, lo + win)
    stem = Path(path).stem
    sf.write(outdir / f"{stem}_00_raw16k.wav", raw16k[sl], TARGET_SR)
    sf.write(outdir / f"{stem}_01_denoised.wav", denoised[sl], TARGET_SR)
    sf.write(outdir / f"{stem}_02_final.wav", final[sl], TARGET_SR)
    logging.info(f"DENOISE SAMPLE written to {outdir}/  (raw vs denoised vs final, "
                 f"{seconds}s of {stem})  clip_frac={flags['clip_frac']:.4f}")


def process_file(path, start_dt):
    """Return a DataFrame of FRAME rows (timestamped) + a list of call events."""
    y, sr = sf.read(path)
    y, flags = preprocess(y, sr)
    hop = int(HOP_S * TARGET_SR); win = int(FRAME_S * TARGET_SR)
    S = np.abs(librosa.stft(y, n_fft=win, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=TARGET_SR, n_fft=win)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=TARGET_SR, hop_length=hop)

    rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop)[0][:S.shape[1]]
    centroid = librosa.feature.spectral_centroid(S=S, sr=TARGET_SR)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=TARGET_SR)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=TARGET_SR)[0]
    flatness = librosa.feature.spectral_flatness(S=np.sqrt(S))[0]
    contrast = librosa.feature.spectral_contrast(S=S, sr=TARGET_SR).mean(0)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=win, hop_length=hop)[0][:S.shape[1]]
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=N_MFCC)
    dmfcc = librosa.feature.delta(mfcc); ddmfcc = librosa.feature.delta(mfcc, order=2)
    sent = spectral_entropy(S)
    fan_e = _band_energy(S, freqs, FAN_BAND)
    voc_e = _band_energy(S, freqs, VOC_BAND)
    high_e = _band_energy(S, freqs, HIGH_BAND)
    tot_e = S.sum(0) + 1e-12

    cols = dict(rms=rms, centroid=centroid, bandwidth=bandwidth, rolloff=rolloff,
                flatness=flatness, contrast=contrast, zcr=zcr, spec_entropy=sent,
                fan_frac=fan_e / tot_e, voc_frac=voc_e / tot_e, high_frac=high_e / tot_e,
                voc_mech_ratio=voc_e / (fan_e + 1e-9))
    for k in range(N_MFCC):
        cols[f"mfcc{k:02d}"] = mfcc[k]; cols[f"dmfcc{k:02d}"] = dmfcc[k]
        cols[f"ddmfcc{k:02d}"] = ddmfcc[k]
    frame_df = pd.DataFrame(cols)
    frame_df["time"] = [start_dt + timedelta(seconds=float(t)) for t in times]

    # per-frame HNR proxy for the call gate (fast): tonality = 1 - flatness scaled
    hnr_frame = np.clip((1 - flatness) * 20, 0, 40)
    calls = detect_calls(voc_e, hnr_frame, HOP_S)
    peak_freq = freqs[np.argmax(S, axis=0)]
    call_events = [dict(time=start_dt + timedelta(seconds=float(times[i])),
                        dur_s=d * HOP_S, energy=e, peak_hz=float(peak_freq[i]))
                   for (i, d, e) in calls]
    # attach a coarse file-level Praat block (voice quality on the whole file)
    frame_df.attrs["praat"] = praat_voice_features(y)
    frame_df.attrs["flags"] = flags
    return frame_df, call_events


def aggregate_hourly(frame_df, call_df, praat_rows):
    frame_df["hour"] = frame_df["time"].dt.floor("h")
    cont_cols = [c for c in frame_df.columns if c not in ("time", "hour")]
    agg = {}
    for c in cont_cols:
        g = frame_df.groupby("hour")[c]
        agg[f"{c}_mean"] = g.mean(); agg[f"{c}_std"] = g.std()
        for p in PERCENTILES:
            agg[f"{c}_p{p}"] = g.quantile(p / 100)
    out = pd.DataFrame(agg)

    if len(call_df):
        call_df["hour"] = call_df["time"].dt.floor("h")
        cg = call_df.groupby("hour")
        out["call_rate_per_min"] = cg.size() / 60.0
        out["call_dur_med"] = cg["dur_s"].median()
        out["call_dur_iqr"] = cg["dur_s"].quantile(.75) - cg["dur_s"].quantile(.25)
        out["call_energy_p50"] = cg["energy"].median()
        out["call_energy_p90"] = cg["energy"].quantile(.9)
        out["call_peakhz_p50"] = cg["peak_hz"].median()
        out["call_peakhz_p90"] = cg["peak_hz"].quantile(.9)
        # inter-call interval + burstiness (Fano factor of counts per minute)
        def iei_stats(s):
            t = np.sort(s.values.astype("datetime64[s]").astype(float))
            d = np.diff(t)
            return pd.Series({"iei_med": np.median(d) if len(d) else np.nan,
                              "iei_iqr": (np.percentile(d, 75) - np.percentile(d, 25)) if len(d) > 3 else np.nan})
        out = out.join(cg["time"].apply(iei_stats).unstack())
        # burstiness: variance/mean of per-minute counts within the hour
        call_df["minute"] = call_df["time"].dt.floor("min")
        pm = call_df.groupby(["hour", "minute"]).size().groupby("hour")
        out["burstiness_fano"] = pm.var() / (pm.mean() + 1e-9)

    if praat_rows:
        pr = pd.DataFrame(praat_rows).set_index("hour")
        out = out.join(pr, how="left")
    out = out.reset_index().rename(columns={"hour": "time"})
    return out


def main():
    global PROP_DECREASE_NS, PROP_DECREASE_ST, DENOISE_TWO_PASS
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", required=True)
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exts", default="wav,WAV,flac")
    ap.add_argument("--log", default="src/welfare_audio.log")
    ap.add_argument("--denoise_ns", type=float, default=PROP_DECREASE_NS,
                    help="non-stationary denoise strength 0-1 (higher = heavier)")
    ap.add_argument("--denoise_st", type=float, default=PROP_DECREASE_ST,
                    help="stationary denoise strength 0-1 (higher = heavier)")
    ap.add_argument("--sample_out", default=None,
                    help="dir to write a before/after denoise clip from the first file, then continue")
    ap.add_argument("--sample_only", action="store_true",
                    help="write the denoise sample and EXIT (no feature extraction)")
    ap.add_argument("--sample_seconds", type=int, default=60)
    ap.add_argument("--file_timeout", type=int, default=180,
                    help="max seconds per file before it is skipped (prevents one bad file hanging the run)")
    ap.add_argument("--single_pass", action="store_true",
                    help="use one denoise pass instead of two (roughly halves runtime)")
    args = ap.parse_args()
    if args.single_pass:
        DENOISE_TWO_PASS = False
    setup_logging(args.log)
    PROP_DECREASE_NS = args.denoise_ns; PROP_DECREASE_ST = args.denoise_st
    logging.info(f"denoise: non-stationary={PROP_DECREASE_NS} stationary={PROP_DECREASE_ST} "
                 f"two_pass={DENOISE_TWO_PASS} hpf={HPF_HZ}Hz fan_subtract={FAN_SUBTRACT}")
    if not HAVE_PRAAT:
        logging.warning("parselmouth not installed -> pitch/jitter/shimmer/HNR/formants "
                        "will be NaN. pip install praat-parselmouth")

    out_path = Path(args.out)
    if out_path.exists():
        logging.info(f"OUTPUT EXISTS, not overwriting: {out_path}"); return
    exts = tuple("." + e.lstrip(".") for e in args.exts.split(","))
    files = sorted(p for p in Path(args.indir).rglob("*") if p.suffix in exts)
    logging.info(f"{len(files)} files under {args.indir}")

    # optional before/after denoise sample from the first readable file
    if args.sample_out or args.sample_only:
        for f in files:
            try:
                export_denoise_sample(f, args.sample_out or "denoise_samples", args.sample_seconds)
                break
            except Exception as e:
                logging.error(f"sample failed on {f.name}: {e}")
        if args.sample_only:
            logging.info("sample_only set -> exiting before feature extraction."); return

    import time as _t
    signal.signal(signal.SIGALRM, _timeout_handler)   # per-file watchdog (Unix, main thread)
    frame_parts = []; call_parts = []; praat_rows = []; ok = 0; skipped = 0
    for i, f in enumerate(files, 1):
        try:
            t0 = _t.time()
            sdt = file_start_datetime(f)
            if sdt is None:
                logging.warning(f"NO TIMESTAMP skip {f.name}"); continue
            signal.alarm(args.file_timeout)           # abort this file if it runs too long
            fdf, calls = process_file(f, sdt)
            signal.alarm(0)
            frame_parts.append(fdf.drop(columns=[]).assign())
            if calls:
                call_parts.append(pd.DataFrame(calls))
            pr = fdf.attrs.get("praat", {})
            praat_rows.append(dict(hour=fdf["time"].dt.floor("h").iloc[0], **pr))
            ok += 1
            logging.info(f"[{i}/{len(files)}] {f.name} {_t.time()-t0:.1f}s frames={len(fdf)}")
        except _FileTimeout:
            signal.alarm(0); skipped += 1
            logging.error(f"[{i}/{len(files)}] TIMEOUT >{args.file_timeout}s, SKIPPED {f.name} "
                          f"(likely oversized/corrupt) — continuing")
        except Exception as e:
            signal.alarm(0); logging.error(f"FAILED {f.name}: {e}")

    if not frame_parts:
        logging.error("NO DATA — nothing written."); sys.exit(1)
    frame_df = pd.concat(frame_parts, ignore_index=True)
    call_df = pd.concat(call_parts, ignore_index=True) if call_parts else pd.DataFrame()
    hourly = aggregate_hourly(frame_df, call_df, praat_rows)
    hourly = hourly.sort_values("time").reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(out_path, index=False)
    logging.info(f"WROTE {out_path}  hours={len(hourly)}  cols={hourly.shape[1]}  files_ok={ok}/{len(files)}")


if __name__ == "__main__":
    main()
