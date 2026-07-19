#!/usr/bin/env python3
"""
extract_rich_audio.py - RICH per-hour acoustic features (64-band log-mel).

Companion to extract_audio_features.py: keeps the full 64-band log-mel (that script
collapses it to scalars). Per hour -> ~130-dim vector [mel00_mean..mel63_mean,
mel00_std..mel63_std, n_frames, gap], aligned to room2_merged_hourly.csv's spine.

MEMORY-SAFE (streaming) aggregation: each worker reduces its file to compact per-hour
partial stats (sum, sumsq, count per band) instead of returning raw frames, so RAM
stays flat no matter how many thousands of files are processed. (An earlier version
held every frame in memory and OOM-ed on the full ~2900-file batch.)

Preprocessing REPLICATED from extract_audio_features.py (keep in sync):
  SR=16000 N_FFT=1024 HOP=512 N_MELS=64 ; noisereduce(stationary, prop=0.80);
  mel=melspectrogram(S=|stft(denoised)|^2); logmel=power_to_db(mel+1e-10);
  timestamps from filename 'YYYYMMDD_HHMMSS' (fallback mtime); recursive; skips '._'.

Usage:
  pip install librosa soundfile noisereduce numpy pandas
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # avoid thread blowup
  python extract_rich_audio.py --input-root "/mnt/.../Audio" \
      --output-dir ./rich_audio_features --month-tag all --room-label Room2 --workers 16
  python extract_rich_audio.py --self-test
"""
from __future__ import annotations
import argparse, os, re, concurrent.futures as cf
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

SR, N_FFT, HOP, N_MELS = 16000, 1024, 512, 64
DENOISE, DENOISE_PROP, DENOISE_STATIONARY = True, 0.80, True
AUDIO_EXTENSIONS = (".wav", ".WAV", ".flac", ".FLAC")
FILENAME_TS_REGEX, FILENAME_TS_FMT = r"(\d{8}_\d{6})", "%Y%m%d_%H%M%S"
MEL_COLS = [f"mel{j:02d}" for j in range(N_MELS)]


def find_audio_files(root_dir):
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"input root does not exist: {root_dir!r}")
    return sorted(str(p) for p in root.rglob("*")
                  if p.suffix in AUDIO_EXTENSIONS and not p.name.startswith("._"))


def parse_start_time(path):
    m = re.search(FILENAME_TS_REGEX, os.path.basename(path))
    if m:
        try:
            return datetime.strptime(m.group(1), FILENAME_TS_FMT), "filename"
        except ValueError:
            pass
    return datetime.fromtimestamp(os.path.getmtime(path)), "mtime"


def file_partials(path):
    """Load one file -> per-hour partial stats. Returns (partials, ts_source).
    partials: {hour_timestamp: [sum(64), sumsq(64), count]}  (compact, streaming)."""
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
        y = y_raw
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)) ** 2
    mel = librosa.feature.melspectrogram(S=S, sr=SR, n_mels=N_MELS)
    logmel = librosa.power_to_db(mel + 1e-10).T.astype(np.float64)   # (frames, 64)
    n = logmel.shape[0]
    # hour index per frame
    secs = np.arange(n) * HOP / SR
    base = pd.Timestamp(start_time)
    hours = (base + pd.to_timedelta(secs, unit="s")).floor("h")
    partials = {}
    # group frames by hour without building a big frame table
    hcodes, huniq = pd.factorize(hours)
    for hi, hr in enumerate(huniq):
        m = hcodes == hi
        block = logmel[m]
        partials[pd.Timestamp(hr)] = [block.sum(0), (block ** 2).sum(0), int(block.shape[0])]
    return partials, ts_source


def _extract_one(path):
    try:
        p, src = file_partials(path)
        return path, p, src, None
    except Exception as e:
        return path, None, None, (str(e) or type(e).__name__)


def merge_partial(acc, partials):
    for hr, (s, sq, c) in partials.items():
        if hr in acc:
            acc[hr][0] += s; acc[hr][1] += sq; acc[hr][2] += c
        else:
            acc[hr] = [s.copy(), sq.copy(), c]


def finalize(acc):
    rows = []
    for hr in sorted(acc):
        s, sq, c = acc[hr]
        mean = s / c
        var = np.maximum(sq / c - mean ** 2, 0.0)
        std = np.sqrt(var)
        row = {"time": hr, "n_frames": c}
        row.update({f"mel{j:02d}_mean": mean[j] for j in range(N_MELS)})
        row.update({f"mel{j:02d}_std": std[j] for j in range(N_MELS)})
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("time")
    df["gap_hours_since_prev"] = df["time"].diff().dt.total_seconds() / 3600
    return df


def run(input_root, output_dir, room_label, month_tag, workers):
    files = find_audio_files(input_root)
    if not files:
        raise FileNotFoundError(f"no audio under {input_root!r}")
    print(f"[scan] {len(files)} audio files under {input_root}", flush=True)
    n_workers = workers or os.cpu_count() or 1
    print(f"[run] {n_workers} worker process(es) (streaming aggregation - flat memory)", flush=True)
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
                print(f"  [{done}/{len(files)}] files done  "
                      f"(hours so far {len(acc)}, failed {len(failed)})", flush=True)
    hourly = finalize(acc)
    out_path = os.path.join(output_dir, f"audio_rich_features_hourly_{room_label}_{month_tag}.csv")
    hourly.to_csv(out_path, index=False)
    print(f"[write] {out_path}  ({hourly.shape[0]} hours x {hourly.shape[1]} cols)", flush=True)
    print(f"[ts source] {sources}", flush=True)
    if "mtime" in sources:
        print("  [!] some files used mtime fallback - verify filenames encode YYYYMMDD_HHMMSS")
    if failed:
        fp = os.path.join(output_dir, f"audio_rich_{room_label}_{month_tag}_failed.csv")
        pd.DataFrame(failed).to_csv(fp, index=False); print(f"[write] {fp} ({len(failed)} failed)")


def self_test():
    import librosa
    sr = SR
    y = 0.1*np.random.randn(sr*3) + 0.2*np.sin(2*np.pi*440*np.arange(sr*3)/sr)
    S = np.abs(librosa.stft(y.astype(np.float32), n_fft=N_FFT, hop_length=HOP))**2
    logmel = librosa.power_to_db(librosa.feature.melspectrogram(S=S, sr=sr, n_mels=N_MELS)+1e-10).T.astype(np.float64)
    n = logmel.shape[0]
    hr = pd.Timestamp("2025-07-15 09:00:00")
    acc = {}
    merge_partial(acc, {hr: [logmel.sum(0), (logmel**2).sum(0), n]})
    # merge a second identical block -> mean unchanged, std unchanged, count doubles
    merge_partial(acc, {hr: [logmel.sum(0), (logmel**2).sum(0), n]})
    df = finalize(acc)
    direct_mean = logmel.mean(0); direct_std = logmel.std(0)
    got_mean = df[[f"mel{j:02d}_mean" for j in range(N_MELS)]].values[0]
    got_std = df[[f"mel{j:02d}_std" for j in range(N_MELS)]].values[0]
    assert np.allclose(got_mean, direct_mean, atol=1e-6), "streaming mean mismatch"
    assert np.allclose(got_std, direct_std, atol=1e-4), "streaming std mismatch"
    assert df["n_frames"].iloc[0] == 2*n
    print(f"[self-test] streaming mean/std match direct computation; cols={df.shape[1]} (expect {N_MELS*2+2})")
    print("[self-test] PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root"); ap.add_argument("--output-dir", default="./rich_audio_features")
    ap.add_argument("--room-label", default="Room2"); ap.add_argument("--month-tag", default="all")
    ap.add_argument("--workers", type=int, default=None); ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test: self_test(); return
    if not a.input_root: ap.error("--input-root required (or --self-test)")
    run(a.input_root, a.output_dir, a.room_label, a.month_tag, a.workers)


if __name__ == "__main__":
    main()
