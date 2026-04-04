# Thesis Figure and Table Plan

Status on 2026-04-03:
- Main-body figures 1--4 are generated and checked into `tex/figures/`.
- Main-body tables 1--2 are already embedded in `tex/sections.tex`.
- Remaining visual gaps are optional qualitative panels and the Track~3 adaptive figure, which should only be added after the adaptive lane becomes canonical.

## Main-body figures

1. `tex/figures/thesis_system_overview_track2_track3.{pdf,png}`
   - purpose: show Track 2 main claim lane and Track 3 support lane
   - current source: `scripts/build_thesis_figures.py`
   - editable diagram source: `tex/figures/thesis_system_overview_track2_track3.mmd`
2. `tex/figures/track2_mainline_progression.{pdf,png}`
   - purpose: Stage0 -> Stage1 recovered -> Stage1 confirm -> Stage2 A0
   - current source: `scripts/build_thesis_figures.py`
3. `tex/figures/track3_operating_points.{pdf,png}`
   - purpose: B0/B1 at `posQ={16,64}` on `true_bpp` vs `map3d_mod_mean`
   - current source: `scripts/build_thesis_figures.py`
4. `tex/figures/track3_bref_drift.{pdf,png}`
   - purpose: show B0/B1 paired values and why adaptive gate remains open
   - current source: `scripts/build_thesis_figures.py`

## Main-body tables

1. Track 2 mainline result table
   - Stage0 winner
   - Stage1 recovered
   - Stage1 confirm
   - Stage2 A0
   - metrics: `AP3D@0.3`, `AP3D@0.5`, `meanBestIoU3D@0.3`
   - inserted in `tex/sections.tex` as `Table~\\ref{tab:track2-main}`
   - numeric sources:
     - `docs/report/results/260402_t2_maskA_stage1_uniform_q6_confirm_summary.md`
     - `docs/report/results/260403_t2_maskA_stage2_adaptive_a0_summary.md`
     - `notebooks/260402_t2_maskA_stage1_uniform_q6_confirm_ep80_compare_fixed_archive_car_ap_summary.csv`
     - `notebooks/260403_t2_maskA_stage2_adaptive_a0_ep80_compare_archive_car_ap_summary.csv`
2. Track 3 support baseline table
   - `B0@16`, `B0@64`, `B1@16`, `B1@64`
   - metrics: `true_bpp`, `AP3D_car_mod`, `map3d_mod_mean`
   - inserted in `tex/sections.tex` as `Table~\\ref{tab:track3-baselines}`
   - numeric sources:
     - `docs/report/results/260402_t3_b0_pp16_summary.md`
     - `docs/report/results/260402_t3_b0_pp64_summary.md`
     - `docs/report/results/260403_t3_b1_pp16_summary.md`
     - `docs/report/results/260403_t3_b1_pp64_summary.md`
     - `docs/report/results/260403_t3_bref_drift_summary.md`

## Optional / future figures

1. qualitative reconstruction panel
2. Track 3 adaptive result figure once canonical

## Current blockers

1. no final canonical adaptive Track 3 result yet
2. no qualitative reconstruction panel asset yet
