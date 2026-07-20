#!/usr/bin/env python3
"""Optional wav2vec2 audio embeddings, kept for a later comparison against the log-mel features.

Run: python src/extraction/wav2vec_audio_features.py --room 'Room 2' --month July --in <dir> --out <dir>
"""

import argparse
import logging
import re
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
from scipy.signal import butter, sosfilt
import noisereduce as nr
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

# ============================================================================
# PREPROCESSING CONFIG  —  SINGLE SOURCE OF TRUTH. Identical for every room.
# Change here only; never per-room, or leave-one-barn-out comparison breaks.
# ============================================================================
TARGET_SR      = 16_000     # wav2vec2 requirement (mono)
HPF_HZ         = 80         # gentle high-pass: removes subsonic/DC rumble only
DENOISE        = True       # stationary spectral gating for the steady fan
PROP_DECREASE  = 0.80       # <1.0 = UNDER-clean on purpose: keep 20% ambient bed
                            #        to protect biological signal from over-subtraction
STATIONARY     = True       # stationary mode does NOT eat transient chick calls
PEAK_TARGET    = 0.97       # peak-normalise for cross-room loudness consistency
CLIP_FRAC_THR  = 0.005      # >0.5% full-scale samples => flag file as clipped/damaged

WINDOW_S       = 10         # embed in 10 s windows, then pool
BATCH_WINDOWS  = 16         # windows per GPU forward pass (raise if VRAM allows)
MODEL_NAME     = "facebook/wav2vec2-base"   # swap if you use a different checkpoint

# Data folders are named by MONTH NAME (June, July, ...). Study year fixed here.
STUDY_YEAR     = 2025
MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

def resolve_month(name: str):
    """'July' -> (7, '2025-07'). Accepts any case; also accepts '2025-07' directly."""
    key = name.strip().lower()
    if key in MONTH_NAMES:
        mnum = MONTH_NAMES[key]
        return mnum, f"{STUDY_YEAR}-{mnum:02d}"
    m = re.fullmatch(r"(\d{4})-(\d{2})", key)     # tolerate explicit YYYY-MM too
    if m:
        return int(m.group(2)), key
    raise ValueError(f"Unrecognised --month '{name}'. Use a month name (e.g. July) "
                     f"or YYYY-MM.")

# Config fingerprint written into outputs so every artifact is traceable.
PREPROC_CONFIG = dict(
    target_sr=TARGET_SR, hpf_hz=HPF_HZ, denoise=DENOISE,
    prop_decrease=PROP_DECREASE, stationary=STATIONARY,
    peak_target=PEAK_TARGET, window_s=WINDOW_S, model=MODEL_NAME,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(logfile: Path):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(logfile), logging.StreamHandler(sys.stdout)],
    )


# ---------------------------------------------------------------------------
# Timestamp parsing
#   Room 2 (Zoom F6): filename like 250811_001_0026.WAV -> 2025-08-11
#   Also tries embedded BWF 'DateTimeOriginal' via soundfile if present.
#   Extend the regexes here for other rooms/recorders.
# ---------------------------------------------------------------------------
FNAME_DATE_PATTERNS = [
    (re.compile(r"(\d{8})_\d{6}"),  "%Y%m%d"),   # AudioMoth: 20250831_083412
    (re.compile(r"(\d{6})_\d{3}"),  "%y%m%d"),   # Zoom F6:   250811_001_0026
    (re.compile(r"(\d{6})"),        "%y%m%d"),   # bare YYMMDD fallback
]

def parse_date(path: Path):
    stem = path.stem
    for pat, fmt in FNAME_DATE_PATTERNS:
        m = pat.search(stem)
        if m:
            try:
                return datetime.strptime(m.group(1), fmt).date()
            except ValueError:
                continue
    return None


# ---------------------------------------------------------------------------
# Preprocessing  (mono -> clip-check -> 16k -> HPF -> denoise -> normalise)
# ---------------------------------------------------------------------------
def _highpass(y, hz):
    if hz <= 0:
        return y
    sos = butter(4, hz, btype="highpass", fs=TARGET_SR, output="sos")
    return sosfilt(sos, y).astype(np.float32)

def preprocess(y, sr):
    """Returns (audio_16k_mono_float32, quality_flags_dict)."""
    flags = {}
    if y.ndim > 1:
        y = y.mean(axis=1)                       # mix channels to mono
    flags["clip_frac"] = float(np.mean(np.abs(y) > 0.999))
    flags["clipped"]   = flags["clip_frac"] > CLIP_FRAC_THR

    if sr != TARGET_SR:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)

    y = _highpass(y, HPF_HZ)

    if DENOISE:
        y = nr.reduce_noise(
            y=y, sr=TARGET_SR,
            stationary=STATIONARY,
            prop_decrease=PROP_DECREASE,          # <1.0 keeps some ambient bed
        ).astype(np.float32)

    peak = float(np.max(np.abs(y))) + 1e-9
    y = (y * (PEAK_TARGET / peak)).astype(np.float32)
    return y, flags


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
class Embedder:
    def __init__(self, device):
        self.device = device
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
        self.model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device).eval()

    @torch.no_grad()
    def embed_file(self, y):
        """Window -> batched GPU forward -> mean-pool per window -> return [n_win, 768]."""
        win = WINDOW_S * TARGET_SR
        chunks = [y[s:s + win] for s in range(0, len(y), win)]
        chunks = [c for c in chunks if len(c) >= TARGET_SR]     # drop <1s tail
        if not chunks:
            return np.empty((0, self.model.config.hidden_size), dtype=np.float32)

        out = []
        for i in range(0, len(chunks), BATCH_WINDOWS):
            batch = chunks[i:i + BATCH_WINDOWS]
            inp = self.fe(batch, sampling_rate=TARGET_SR, return_tensors="pt",
                          padding=True).input_values.to(self.device)
            h = self.model(inp).last_hidden_state        # [B, T, 768]
            out.append(h.mean(dim=1).cpu().numpy())      # mean-pool over time
        return np.concatenate(out, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", required=True)
    ap.add_argument("--month", required=True, help="month name e.g. July (or YYYY-MM)")
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--logdir", default="results/logs")
    ap.add_argument("--exts", default="wav,WAV")
    args = ap.parse_args()

    month_num, month_iso = resolve_month(args.month)   # 'July' -> (7, '2025-07')
    month_name = args.month.strip().capitalize()

    room_tag = args.room.replace(" ", "")
    setup_logging(Path(args.logdir) / f"{room_tag}_{month_name}.log")
    logging.info(f"START room={args.room} month={month_name} ({month_iso})")
    logging.info(f"preproc={json.dumps(PREPROC_CONFIG)}")

    out_path = Path(args.out)
    quality_path = Path(str(out_path).replace(".parquet", "_quality.csv"))

    # ---- resumability: skip whole month if final output already exists ----
    if out_path.exists():
        logging.info(f"OUTPUT EXISTS, skipping month: {out_path}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logging.warning("CUDA NOT AVAILABLE — running on CPU. Fix torch/CUDA for speed "
                        "and for GPU-utilisation evidence.")
    else:
        logging.info(f"device=cuda ({torch.cuda.get_device_name(0)})")
    embedder = Embedder(device)

    exts = tuple("." + e.lstrip(".") for e in args.exts.split(","))
    files = sorted(p for p in Path(args.indir).rglob("*") if p.suffix in exts)
    logging.info(f"found {len(files)} audio files under {args.indir}")

    per_day = {}          # date_str -> list of [n_win,768] arrays
    quality_rows = []
    ok_files = 0

    for i, f in enumerate(files, 1):
        try:
            day = parse_date(f)
            if day is None:
                logging.warning(f"NO TIMESTAMP, skip: {f.name}")
                quality_rows.append(dict(file=f.name, issue="no_timestamp"))
                continue
            if day.month != month_num:
                # file's own date disagrees with the folder it sits in — log, keep, bin by true date
                logging.warning(f"MONTH MISMATCH: {f.name} dates to {day} "
                                f"but folder is {month_name}")
                quality_rows.append(dict(file=f.name, date=str(day),
                                         issue=f"month_mismatch_folder_{month_name}"))

            y, sr = sf.read(f)
            y, flags = preprocess(y, sr)
            if flags["clipped"]:
                logging.warning(f"CLIPPED {flags['clip_frac']:.3f}: {f.name}")

            emb = embedder.embed_file(y)              # [n_win, 768]
            if emb.shape[0] == 0:
                logging.warning(f"NO WINDOWS (too short?): {f.name}")
                quality_rows.append(dict(file=f.name, date=str(day), issue="no_windows"))
                continue

            per_day.setdefault(str(day), []).append(emb)
            ok_files += 1
            quality_rows.append(dict(
                file=f.name, date=str(day), n_windows=int(emb.shape[0]),
                clip_frac=round(flags["clip_frac"], 5), clipped=bool(flags["clipped"]),
            ))
            if i % 10 == 0 or i == len(files):
                logging.info(f"[{i}/{len(files)}] processed (ok={ok_files})")

        except Exception as e:
            logging.error(f"FAILED {f.name}: {e}")     # one bad file never kills the month
            quality_rows.append(dict(file=f.name, issue=f"error:{e}"))

    # ---- aggregate to DAILY rows: mean + std of window embeddings that day ----
    rows = []
    for day, arrs in sorted(per_day.items()):
        E = np.concatenate(arrs, axis=0)               # [total_windows, 768]
        row = dict(date=day, room=args.room, month=month_name, month_iso=month_iso,
                   n_files=len(arrs), n_windows=int(E.shape[0]))
        row.update({f"emb_mean_{k}": float(v) for k, v in enumerate(E.mean(0))})
        row.update({f"emb_std_{k}":  float(v) for k, v in enumerate(E.std(0))})
        rows.append(row)

    if not rows:
        logging.error("NO DATA AGGREGATED — writing nothing. Check inputs/timestamps.")
        pd.DataFrame(quality_rows).to_csv(quality_path, index=False)
        sys.exit(1)                                    # non-zero => driver keeps raw, stops

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.attrs["preproc"] = PREPROC_CONFIG               # traceable config
    df.to_parquet(out_path)
    pd.DataFrame(quality_rows).to_csv(quality_path, index=False)

    logging.info(f"WROTE {out_path}  days={len(df)}  files_ok={ok_files}/{len(files)}")
    logging.info(f"WROTE {quality_path}  (per-file quality -> inventory 'missing data' column)")
    logging.info("DONE")


if __name__ == "__main__":
    main()