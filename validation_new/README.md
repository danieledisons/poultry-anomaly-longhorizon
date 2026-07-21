# validation_new

Causal, reproducible validation of the PhD contribution (Approach C: coupled
multi-rate state-space with residual-gated slow-state assimilation), prepared
in response to the July 17 progress-report review.

## Structure
- `data/`         Frozen COPIES of source CSVs. Do not edit. Originals live in
                  `../features/` and `../results/` and remain untouched.
- `causal/`       Strictly-causal (past-only) preprocessing + before/after figs.
- `gate_ablation/`Gated vs ungated online slow-state assimilation experiment.
- `repro/`        Reproducibility artifacts.
- `repro/docs/`   findings.md, implemented_vs_conceptual.md, repro_table.csv.

## Frozen inputs (copied 2026-07-21)
| File | Source | Notes |
|------|--------|-------|
| room2_merged_hourly.csv | features/room2_merged_hourly.csv | Room 2 hourly fused table (2092 hrs) |
| recommended_features.csv | results/recommended_features.csv | selected feature list |
| spine_room2_rich.csv | results/spine_room2_rich.csv | Room 2 rich spine |
| spine_room6_rich.csv | results/spine_room6_rich.csv | Room 6 rich spine (held-out) |

Branch: feature/causal-validation
