#!/usr/bin/env python3
"""
run_pipeline.py — single entrypoint for the slow/fast + alpha_t gate pipeline.

Run from the repo root so `config` and `src` resolve:

    python run_pipeline.py                      # uses FEATURES_DIR / RESULTS_DIR from .env
    python run_pipeline.py --out-dir /tmp/run7  # override output location

This is a thin launcher; the pipeline itself lives in src/pipeline/analysis.py.
"""
from src.pipeline.analysis import main

if __name__ == "__main__":
    main()
