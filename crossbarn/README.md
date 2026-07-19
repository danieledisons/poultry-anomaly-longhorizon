# Cross-barn validation — feature extraction

Extracts rich audio features for the **held-out barn (Room 6)** in the exact same
format as Room 2, so the locked model (`src/pipeline/final_model.py`) can be fit
on Room 2 and evaluated, unchanged, on Room 6.

Room 6 audio is **Zoom F6** (timestamp embedded in WAV metadata), unlike Room 2's
AudioMoth (timestamp in the filename). The extractor recovers either automatically.

## Run overnight (server, tmux)

```bash
# 0. sanity-check dependencies + timestamp/DSP logic (seconds)
python crossbarn/extract_rich_audio_crossbarn.py --self-test

# 1. start a tmux session so it survives disconnects
tmux new -s room6

# 2. run it (logs to crossbarn/room6_audio_*.log)
bash crossbarn/run_room6_audio.sh

# detach: Ctrl-b then d   |   reattach later: tmux attach -t room6
```

Deps (same as Room 2 extraction): `pip install librosa soundfile noisereduce numpy pandas`.

## Output

`features/rich_audio_features/audio_rich_features_hourly_Room6_all.csv`
— columns identical to Room 2: `time, n_frames, mel00_mean..mel63_mean,
mel00_std..mel63_std, gap_hours_since_prev`.

## After it finishes — check the timestamp sources

The run prints `[timestamp sources] {...}`. For Room 6 expect mostly `metadata`
(Zoom F6). A few `filedate` is fine (date-only fallback). **`mtime` should be ~0** —
if it's high, those files' metadata didn't parse and their hour alignment is the
copy date, not the recording date (flag them before trusting the spine).

## Next steps (after audio + video features exist)

1. Extract Room 6 **video** features (needs the offline video drive connected).
2. Build the Room 6 spine (`build_spine.py` pointed at Room 6 CSVs).
3. Cross-barn: `final_model` fit on Room 2 → `.calibrate()` + `.alarm()` on Room 6
   → magnitude/duration sweep charts for both barns.
