#!/usr/bin/env python3
"""
extract_rich_audio_crossbarn.py — RICH per-hour log-mel audio features for a
HELD-OUT barn, in the SAME format as Room 2 (for cross-barn validation).

Identical DSP to src/extraction/extract_rich_audio.py (SR=16000, N_FFT=1024,
HOP=512, N_MELS=64, stationary noisereduce prop=0.80) so the features are
directly comparable across barns. Output columns match exactly:
    time, n_frames, mel00_mean..mel63_mean, mel00_std..mel63_std, gap_hours_since_prev

THE ONLY DIFFERENCE — recorder-agnostic timestamps
--------------------------------------------------
Room 2 = AudioMoth: time is in the filename ('..._YYYYMMDD_HHMMSS.wav').
Room 6 = Zoom F6:   time is in the WAV's embedded BWF/iXML metadata
                    (FILE_UID 'ZOOM F6  YYYYMMDDHHMMSS...' / bext OriginationDate+Time).
`parse_start_time` tries filename first, then embedded metadata, then the
'YYMMDD_' filename date prefix, and only falls back to mtime as a last resort
(mtime is unreliable — on this dataset it is the copy date, not the recording date).

Usage (run overnight in tmux)
-----------------------------
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python crossbarn/extract_rich_audio_crossbarn.py \
        --input-root "/mnt/crucial_x10/Poultry Multimodal Data/Audio Data/Room 6" \
        --input-root "/mnt/drive_bf/Poultry_Multimodal_SeptDec/Audio data/Room6" \
        --room-label Room6 --month-tag all \
        --output-dir features/rich_audio_features --workers 12

    python crossbarn/extract_rich_audio_crossbarn.py --self-test
"""
from __future__ import annotations
import argparse, os, re, concurrent.futures as cf
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

# --- DSP config: MUST match src/extraction/extract_rich_audio.py exactly ---
SR, N_FFT, HOP, N_MELS = 16000, 1024, 512, 64
DENOISE, DENOISE_PROP, DENOISE_STATIONARY = True, 0.80, True
AUDIO_EXTENSIONS = (".wav", ".WAV", ".flac", ".FLAC")
N_MELS_COLS = [f"mel{j:02d}" for j in range(N_MELS)]


# ======================================================================
# Recorder-agnostic timestamp recovery
# ======================================================================
def read_embedded_timestamp(path, nbytes=1_048_576):
    """Read the recording start time from WAV BWF/iXML metadata (Zoom F6 etc.)."""
    try:
        with open(path, "rb") as fh:
            blob = fh.read(nbytes)
    except OSError:
        return None
    txt = blob.decode("latin-1", errors="ignore")
    # 1) FILE_UID / FAMILY_UID:  'ZOOM F6    20250706082750...'
    m = re.search(r"ZOOM\s*F6\s+(\d{14})", txt)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
        except ValueError:
            pass
    # 2) bext OriginationDate + OriginationTime, often concatenated: '2025-07-0608:27:50'
    m = re.search(r"(\d{4}-\d{2}-\d{2})[ T]?(\d{2}:\d{2}:\d{2})", txt)
    if m:
        try:
            return datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    return None


def parse_start_time(path):
    base = os.path.basename(path)
    # 1) AudioMoth-style filename: YYYYMMDD_HHMMSS
    m = re.search(r"(\d{8}_\d{6})", base)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S"), "filename"
        except ValueError:
            pass
    # 2) embedded metadata (Zoom F6 / BWF)
    ts = read_embedded_timestamp(path)
    if ts is not None:
        return ts, "metadata"
    # 3) 'YYMMDD_' filename date prefix (date only, assume midnight)
    m = re.match(r"(\d{6})[_\-]", base)
    if m:
        try:
            return datetime.strptime(m.group(1), "%y%m%d"), "filedate"
        except ValueError:
            pass
    # 4) last resort (unreliable on copied data)
    return datetime.fromtimestamp(os.path.getmtime(path)), "mtime"


# ======================================================================
# Feature extraction (streaming, flat memory) — unchanged from Room 2
# ======================================================================
def find_audio_files(roots):
    files = []
    for root_dir in roots:
        root = Path(root_dir)
        if not root.exists():
            print(f"[warn] input root not found (skipped): {root_dir!r}", flush=True)
            continue
        files += [str(p) for p in root.rglob("*")
                  if p.suffix in AUDIO_EXTENSIONS and not p.name.startswith("._")]
    return sorted(set(files))


def file_partials(path):
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
    logmel = librosa.power_to_db(mel + 1e-10).T.astype(np.float64)
    n = logmel.shape[0]
    secs = np.arange(n) * HOP / SR
    hours = (pd.Timestamp(start_time) + pd.to_timedelta(secs, unit="s")).floor("h")
    hcodes, huniq = pd.factorize(hours)
    partials = {}
    for hi, hr in enumerate(huniq):
        block = logmel[hcodes == hi]
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
        std = np.sqrt(np.maximum(sq / c - mean ** 2, 0.0))
        row = {"time": hr, "n_frames": c}
        row.update({f"mel{j:02d}_mean": mean[j] for j in range(N_MELS)})
        row.update({f"mel{j:02d}_std": std[j] for j in range(N_MELS)})
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("time")
    df["gap_hours_since_prev"] = df["time"].diff().dt.total_seconds() / 3600
    return df


def run(roots, output_dir, room_label, month_tag, workers):
    files = find_audio_files(roots)
    if not files:
        raise FileNotFoundError(f"no audio under {roots!r}")
    print(f"[scan] {len(files)} audio files", flush=True)
    n_workers = workers or os.cpu_count() or 1
    print(f"[run] {n_workers} workers (streaming aggregation, flat memory)", flush=True)
    os.makedirs(output_dir, exist_ok=True)
    acc, failed, sources, done = {}, [], {}, 0
    with cf.ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_extract_one, f): f for f in files}
        for fut in cf.as_completed(futs):
            path, partials, src, err = fut.result()
            done += 1
            if err or partials is None:
                failed.append({"file": os.path.basename(path), "reason": err or "empty"})
            else:
                merge_partial(acc, partials); sources[src] = sources.get(src, 0) + 1
            if done % 25 == 0 or done == len(files):
                print(f"  [{done}/{len(files)}] done  (hours {len(acc)}, failed {len(failed)})", flush=True)
    hourly = finalize(acc)
    out = os.path.join(output_dir, f"audio_rich_features_hourly_{room_label}_{month_tag}.csv")
    hourly.to_csv(out, index=False)
    print(f"[write] {out}  ({hourly.shape[0]} hours x {hourly.shape[1]} cols)", flush=True)
    print(f"[timestamp sources] {sources}", flush=True)
    if sources.get("mtime"):
        print("  [!] some files fell back to mtime — verify those recordings' metadata", flush=True)
    if failed:
        fp = os.path.join(output_dir, f"audio_rich_{room_label}_{month_tag}_failed.csv")
        pd.DataFrame(failed).to_csv(fp, index=False)
        print(f"[write] {fp}  ({len(failed)} failed)", flush=True)


def self_test():
    # timestamp parsing
    zoom_blob = b"....<FILE_UID>ZOOM F6    20250706082750050000A</FILE_UID>...."
    with open("/tmp/_zoomtest.WAV", "wb") as fh:
        fh.write(zoom_blob)
    ts, src = parse_start_time("/tmp/_zoomtest.WAV")
    assert ts == datetime(2025, 7, 6, 8, 27, 50) and src == "metadata", (ts, src)
    ts2, s2 = parse_start_time("S4A27290_20250729_122930.wav")
    assert ts2 == datetime(2025, 7, 29, 12, 29, 30) and s2 == "filename", (ts2, s2)
    # streaming mean/std correctness
    import librosa
    y = (0.1 * np.random.randn(SR * 3) + 0.2 * np.sin(2 * np.pi * 440 * np.arange(SR * 3) / SR)).astype(np.float32)
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)) ** 2
    lm = librosa.power_to_db(librosa.feature.melspectrogram(S=S, sr=SR, n_mels=N_MELS) + 1e-10).T.astype(np.float64)
    acc = {}
    hr = pd.Timestamp("2025-07-06 08:00:00")
    merge_partial(acc, {hr: [lm.sum(0), (lm ** 2).sum(0), lm.shape[0]]})
    df = finalize(acc)
    assert np.allclose(df[[f"mel{j:02d}_mean" for j in range(N_MELS)]].values[0], lm.mean(0), atol=1e-6)
    assert df.shape[1] == N_MELS * 2 + 3, df.shape
    print("[self-test] timestamp parsing (Zoom + AudioMoth) OK; streaming stats OK; "
          f"cols={df.shape[1]} (time+n_frames+128+gap)")
    print("[self-test] PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-root", action="append", default=[],
                    help="Raw-audio root for the barn (repeat for multiple drives).")
    ap.add_argument("--output-dir", default="features/rich_audio_features")
    ap.add_argument("--room-label", default="Room6")
    ap.add_argument("--month-tag", default="all")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test(); return
    if not a.input_root:
        ap.error("--input-root required (or --self-test)")
    run(a.input_root, a.output_dir, a.room_label, a.month_tag, a.workers)


if __name__ == "__main__":
    main()
