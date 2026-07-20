#!/usr/bin/env python3
"""Entry point for the slow/fast + gate analysis; just calls the pipeline in src/pipeline/analysis.py.

Run: python run_pipeline.py
"""
from src.pipeline.analysis import main

if __name__ == "__main__":
    main()
