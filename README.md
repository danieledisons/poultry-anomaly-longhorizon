# poultry-anomaly-longhorizon

Multimodal (audio · video · thermal · environment) long-horizon anomaly
detection for poultry barns. Core contribution: a slow/fast band decomposition
feeding a minimal, stateful **alpha_t trust gate** that responds to the
*persistence* of unexplained residual energy rather than instantaneous
magnitude.

## Cross-machine setup (laptop ↔ server)

Paths are **never hardcoded**. `config.py` derives every path from a single
`PROJECT_ROOT` read from a `.env` file. Each machine keeps its own `.env`:

```bash
# on the laptop
ln -sf .env.laptop .env
# on the server
ln -sf .env.server .env

# confirm the paths resolved correctly:
python config.py
```

Edit `PROJECT_ROOT` (and optionally `RAW_DIR`, etc.) inside `.env.laptop` /
`.env.server`. Nothing else in the code changes between machines. You can also
override any path inline: `RESULTS_DIR=/tmp/run7 python run_pipeline.py`.

## Run the pipeline

The end-to-end slow/fast + gate analysis (merge → decompose → gate →
synthetic injection test → figures) is one command:

```bash
python run_pipeline.py                 # reads FEATURES_DIR / RESULTS_DIR from .env
python run_pipeline.py --out-dir /tmp/run7
```

Expects these feature CSVs in `FEATURES_DIR`: `hourly_features_all_folders_room_2.csv`,
`hourly_features_all_folders_room_6.csv`, `audio_features_hourly_Room2_2025-0{6,7,8}.csv`,
`audio_features_hourly_Room6_2025-07.csv`, `env_features_Room2.csv`.

## Layout

```
config.py            # single source of truth for all paths (reads .env)
run_pipeline.py      # entrypoint → src/pipeline/analysis.py
requirements.txt

src/
  extraction/        # raw media → feature CSVs (audio, video, thermal, wav2vec)
  models/            # alpha_gate.py, slow_fast.py, baseline.py
  pipeline/          # analysis.py (unified run), dl_model_comparison.py
  validation/        # room2_validation.py, cross_modal_analysis.py
  viz/               # plotting (env trajectory, etc.)
tools/               # one-off inspection utilities (parquet, timestamps, denoise)

data/  features/  results/  figures/    # data + outputs (shared, committed)
```

Install: `pip install -r requirements.txt` (install `torch` separately with the
CUDA wheel matching your GPU — see `requirements.txt`).
