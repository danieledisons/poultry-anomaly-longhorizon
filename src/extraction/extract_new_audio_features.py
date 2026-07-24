#!/usr/bin/env python3
"""
extract_new_audio_features.py — welfare-relevant hourly audio features.

Built on the PROVEN fast skeleton of extract_rich_audio.py: librosa.load, small
1024-pt STFT, single light denoise pass, PARALLEL (ProcessPoolExecutor across all
cores), and STREAMING per-hour aggregation so memory stays flat over thousands of
1-hour files. No Praat, no full-file spectral subtraction, no O(n*k) convolve — so
it will not hang the way extract_welfare_audio.py did.

Per-frame welfare descriptors (vectorised, no Python loop):
  rms, spectral centroid / bandwidth / rolloff / flatness / contrast,
  spectral entropy, zero-crossing rate, spectral flux,
  band-energy fractions (fan/mech, vocalisation, high, mid) + voc/mech ratio,
  MFCC 1-13.
Streamed as per-hour sum / sumsq / count  ->  hourly MEAN and STD per descriptor.

Call statistics (welfare biomarkers, streamed as counts per hour):
  call_activity_frac  — fraction of frames with vocalisation-band energy above an
                        adaptive noise floor AND tonal (harmonicity gate),
  call_rate_per_min   — vocalisation onset rate (rising edges of the active mask),
  transient_rate      — spectral-flux onset rate.

Output: audio_new_features_hourly_<room>_<tag>.csv  (one row per hour).

Run (per room-month, mirrors extract_rich_audio.py):
  python src/extraction/extract_new_audio_features.py \
      --input-root "/mnt/.../Audio/July/Room 2" \
      --output-dir features --room-label Room2 --month-tag 2025-07 --workers 8
Self-test:  python src/extraction/extract_new_audio_features.py --self-test
"""
from __future__ import annotations
import argparse, os, re, concurrent.futures as cf
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

# --- proven config (same as extract_rich_audio.py) ---
SR, N_FFT, HOP = 16000, 1024, 512
DENOISE, DENOISE_PROP, DENOISE_STATIONARY = True, 0.80, True
AUDIO_EXTENSIONS = (".wav", ".WAV", ".flac", ".FLAC")
# accept AudioMoth 20250831_083412 and Zoom 250811 style; fall back to mtime
FILENAME_TS_PATTERNS = [(r"(\d{8}_\d{6})", "%Y%m%d_%H%M%S"),
                        (r"(\d{6}_\d{6})", "%y%m%d_%H%M%S"),
                        (r"(\d{8})", "%Y%m%d"), (r"(\d{6})", "%y%m%d")]
# frequency bands (Hz)
FAN_BAND, VOC_BAND, HIGH_BAND, MID_BAND = (80, 500), (1500, 5000), (5000, 8000), (500, 1500)
CALL_SNR_DB, CALL_MIN_TONALITY = 6.0, 0.5     # voc-band SNR + (1-flatness) gate
N_MFCC = 13
FRAME_SEC = HOP / SR

# NEW descriptors (deliberately different from the existing hourly set, which already
# has rms/centroid/flatness/zcr/flux/band-fractions/voc_activity/transient). rms,
# centroid, rolloff kept only as anchors; the rest are new welfare biomarkers.
F0_MIN, F0_MAX = 200, 5000                     # chicken vocalisation fundamental range
CONT = (["rms", "centroid", "rolloff", "bandwidth", "contrast", "spec_entropy",
         "tonality", "voc_mech_ratio", "f0"]
        + [f"mfcc{j:02d}" for j in range(N_MFCC)]
        + [f"dmfcc{j:02d}" for j in range(N_MFCC)])
K = len(CONT)


def find_audio_files(root_dir):
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"input root does not exist: {root_dir!r}")
    return sorted(str(p) for p in root.rglob("*")
                  if p.suffix in AUDIO_EXTENSIONS and not p.name.startswith("._"))


def parse_start_time(path):
    b = os.path.basename(path)
    for pat, fmt in FILENAME_TS_PATTERNS:
        m = re.search(pat, b)
        if m:
            try:
                return datetime.strptime(m.group(1), fmt), "filename"
            except ValueError:
                continue
    return datetime.fromtimestamp(os.path.getmtime(path)), "mtime"


def _frame_features(y):
    """Return (F: n_frames x K continuous descriptors, active_mask, onset_mask)."""
    import librosa
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)) ** 2      # power (freq,frames)
    Smag = np.sqrt(S)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    nfr = S.shape[1]
    tot = S.sum(0) + 1e-12

    rms = np.sqrt(np.maximum(S.mean(0), 0.0))                              # anchor
    centroid = librosa.feature.spectral_centroid(S=Smag, sr=SR)[0]         # anchor
    rolloff = librosa.feature.spectral_rolloff(S=Smag, sr=SR, roll_percent=0.85)[0]  # anchor
    # --- NEW descriptors ---
    bandwidth = librosa.feature.spectral_bandwidth(S=Smag, sr=SR)[0]       # spectral spread
    contrast = librosa.feature.spectral_contrast(S=S, sr=SR).mean(0)       # peak-valley contrast
    flatness = librosa.feature.spectral_flatness(S=Smag)[0]
    p = S / tot
    spec_entropy = (-(p * np.log(p + 1e-12)).sum(0)) / np.log(S.shape[0])  # spectral disorder
    tonality = 1.0 - flatness                                              # harmonicity proxy
    fan_e = S[(freqs >= FAN_BAND[0]) & (freqs < FAN_BAND[1])].sum(0)
    voc_e = S[(freqs >= VOC_BAND[0]) & (freqs < VOC_BAND[1])].sum(0)
    voc_mech = voc_e / (fan_e + 1e-9)                                      # fan-normalised voc
    # F0 / pitch via YIN (fast, no Praat)
    try:
        f0 = librosa.yin(y, fmin=F0_MIN, fmax=F0_MAX, sr=SR,
                         frame_length=N_FFT, hop_length=HOP)
        f0 = f0[:nfr] if f0.shape[0] >= nfr else np.pad(f0, (0, nfr - f0.shape[0]), constant_values=np.nan)
    except Exception:
        f0 = np.full(nfr, np.nan)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=N_MFCC)   # (13, frames)
    dmfcc = librosa.feature.delta(mfcc)                                    # 1st-order deltas

    cols = [rms, centroid, rolloff, bandwidth, contrast, spec_entropy, tonality, voc_mech, f0]
    F = np.vstack(cols + [mfcc[j] for j in range(N_MFCC)]
                       + [dmfcc[j] for j in range(N_MFCC)]).T              # (frames, K)

    # call-activity: voc-band energy above adaptive floor AND tonal
    e_db = 10 * np.log10(voc_e + 1e-12)
    win = max(int(30 / FRAME_SEC), 5)
    floor = pd.Series(e_db).rolling(win, min_periods=5).median().to_numpy()
    active = (e_db > (floor + CALL_SNR_DB)) & (tonality > CALL_MIN_TONALITY)
    onset = active & ~np.concatenate(([False], active[:-1]))
    # spectral flux (local only) for transient-onset rate
    flux = np.sqrt((np.diff(Smag, axis=1, prepend=Smag[:, :1]) ** 2).sum(0))
    flux_thr = np.nanmedian(flux) + 1.5 * (np.nanstd(flux) + 1e-9)
    transient = (flux > flux_thr) & ~np.concatenate(([False], (flux > flux_thr)[:-1]))
    return np.nan_to_num(F), active, onset, transient


def file_partials(path):
    """One file -> per-hour partial stats (streaming, flat memory)."""
    import librosa
    y_raw, _ = librosa.load(path, sr=SR, mono=True)
    if y_raw.size == 0:
        return {}, "empty"
    start_time, ts_source = parse_start_time(path)
    if DENOISE:
        import noisereduce as nr
        y = nr.reduce_noise(y=y_raw, sr=SR, stationary=DENOISE_STATIONARY,
                            prop_decrease=DENOISE_PROP).astype(np.float32)
    else:
        y = y_raw.astype(np.float32)
    F, active, onset, transient = _frame_features(y)
    n = F.shape[0]
    secs = np.arange(n) * FRAME_SEC
    hours = (pd.Timestamp(start_time) + pd.to_timedelta(secs, unit="s")).floor("h")
    hcodes, huniq = pd.factorize(hours)
    partials = {}
    for hi, hr in enumerate(huniq):
        m = hcodes == hi
        Fb = F[m]
        partials[pd.Timestamp(hr)] = [Fb.sum(0), (Fb ** 2).sum(0), int(Fb.shape[0]),
                                      int(active[m].sum()), int(onset[m].sum()),
                                      int(transient[m].sum())]
    return partials, ts_source


def _extract_one(path):
    try:
        p, src = file_partials(path)
        return path, p, src, None
    except Exception as e:
        return path, None, None, (str(e) or type(e).__name__)


def merge_partial(acc, partials):
    for hr, v in partials.items():
        if hr in acc:
            a = acc[hr]
            a[0] += v[0]; a[1] += v[1]; a[2] += v[2]; a[3] += v[3]; a[4] += v[4]; a[5] += v[5]
        else:
            acc[hr] = [v[0].copy(), v[1].copy(), v[2], v[3], v[4], v[5]]


def finalize(acc):
    rows = []
    for hr in sorted(acc):
        s, sq, c, act, ons, tr = acc[hr]
        c = max(c, 1)
        mean = s / c
        std = np.sqrt(np.maximum(sq / c - mean ** 2, 0.0))
        row = {"time": hr, "n_frames": c}
        for j, name in enumerate(CONT):
            row[f"{name}_mean"] = mean[j]; row[f"{name}_std"] = std[j]
        minutes = c * FRAME_SEC / 60.0
        row["call_activity_frac"] = act / c
        row["call_rate_per_min"] = ons / minutes if minutes > 0 else 0.0
        row["transient_rate_per_min"] = tr / minutes if minutes > 0 else 0.0
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    df["gap_hours_since_prev"] = df["time"].diff().dt.total_seconds() / 3600
    return df


def run(input_root, output_dir, room_label, month_tag, workers):
    files = find_audio_files(input_root)
    if not files:
        raise FileNotFoundError(f"no audio under {input_root!r}")
    print(f"[scan] {len(files)} audio files under {input_root}", flush=True)
    n_workers = workers or os.cpu_count() or 1
    print(f"[run] {n_workers} worker(s), streaming aggregation (flat memory)", flush=True)
    acc = {}; failed = []; sources = {}; done = 0
    os.makedirs(output_dir, exist_ok=True)
    with cf.ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_extract_one, f): f for f in files}
        for fut in cf.as_completed(futs):
            path, partials, src, err = fut.result(); base = os.path.basename(futs[fut])
            done += 1
            if err or partials is None:
                failed.append({"file": base, "reason": err or "empty"})
            else:
                merge_partial(acc, partials); sources[src] = sources.get(src, 0) + 1
            if done % 25 == 0 or done == len(files):
                print(f"  [{done}/{len(files)}] files done  (hours {len(acc)}, failed {len(failed)})", flush=True)
    hourly = finalize(acc)
    out_path = os.path.join(output_dir, f"audio_new_features_hourly_{room_label}_{month_tag}.csv")
    hourly.to_csv(out_path, index=False)
    print(f"[write] {out_path}  ({hourly.shape[0]} hours x {hourly.shape[1]} cols)", flush=True)
    print(f"[ts source] {sources}", flush=True)
    if "mtime" in sources:
        print("  [!] some files used mtime fallback — verify filenames encode a date/time")
    if failed:
        fp = os.path.join(output_dir, f"audio_new_{room_label}_{month_tag}_failed.csv")
        pd.DataFrame(failed).to_csv(fp, index=False); print(f"[write] {fp} ({len(failed)} failed)")


def self_test():
    sr = SR
    t = np.arange(sr * 4) / sr
    y = (0.1 * np.random.randn(sr * 4) + 0.3 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)
    F, active, onset, transient = _frame_features(y)
    assert F.shape[1] == K, f"feature count {F.shape[1]} != {K}"
    hr = pd.Timestamp("2025-07-15 09:00:00")
    acc = {}
    merge_partial(acc, {hr: [F.sum(0), (F ** 2).sum(0), F.shape[0],
                             int(active.sum()), int(onset.sum()), int(transient.sum())]})
    df = finalize(acc)
    got = df[[f"{n}_mean" for n in CONT]].values[0]
    assert np.allclose(got, F.mean(0), atol=1e-6), "streaming mean mismatch"
    print(f"[self-test] {df.shape[1]} cols, {K} descriptors + call stats; streaming mean OK")
    print("[self-test] PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root"); ap.add_argument("--output-dir", default="features")
    ap.add_argument("--room-label", default="Room2"); ap.add_argument("--month-tag", default="all")
    ap.add_argument("--workers", type=int, default=None); ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test(); return
    if not a.input_root:
        ap.error("--input-root required (or --self-test)")
    run(a.input_root, a.output_dir, a.room_label, a.month_tag, a.workers)


if __name__ == "__main__":
    main()
