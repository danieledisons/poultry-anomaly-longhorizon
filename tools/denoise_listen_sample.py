"""Denoise a short audio clip and save a listenable sample.

Run: python tools/denoise_listen_sample.py <clip>
"""

import argparse
import os

import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr

# --- MUST MATCH extract_audio_features.py ---
SR                 = 16000
DENOISE_PROP       = 0.80
DENOISE_STATIONARY = True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", help="path to a single audio file")
    ap.add_argument("--start", type=float, default=0.0, help="start second (default 0)")
    ap.add_argument("--dur", type=float, default=60.0, help="seconds to save (default 60)")
    ap.add_argument("--outdir", default="listen_samples")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.clip))[0]

    # load just the requested segment
    y, _ = librosa.load(args.clip, sr=SR, mono=True,
                        offset=args.start, duration=args.dur)
    if y.size == 0:
        raise SystemExit("Empty segment — check --start/--dur against clip length.")

    y_dn = nr.reduce_noise(y=y, sr=SR,
                           stationary=DENOISE_STATIONARY,
                           prop_decrease=DENOISE_PROP).astype(np.float32)

    raw_path = os.path.join(args.outdir, f"{base}_{int(args.start)}s_raw.wav")
    dn_path  = os.path.join(args.outdir, f"{base}_{int(args.start)}s_denoised.wav")
    sf.write(raw_path, y, SR)
    sf.write(dn_path, y_dn, SR)

    # quick before/after numbers
    def band_frac(sig, lo, hi):
        S = np.abs(librosa.stft(sig, n_fft=1024, hop_length=512)) ** 2
        f = librosa.fft_frequencies(sr=SR, n_fft=1024)
        idx = (f >= lo) & (f < hi)
        return float(S[idx].sum() / (S.sum() + 1e-10))

    print(f"Saved:\n  {raw_path}\n  {dn_path}")
    print("\nBand energy (fraction of total):")
    print(f"  fan   0-500 Hz : raw {band_frac(y,0,500):.3f}  -> denoised {band_frac(y_dn,0,500):.3f}")
    print(f"  voc 2-6 kHz    : raw {band_frac(y,2000,6000):.3f}  -> denoised {band_frac(y_dn,2000,6000):.3f}")
    print("\nListen to BOTH. Denoised should have less hum and clearer calls.")
    print("If calls sound watery/gated, lower DENOISE_PROP and re-run.")


if __name__ == "__main__":
    main()