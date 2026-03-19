# Research State Audit — 2026-03-18

## Scope and evidence

This audit summarizes the current LiDAR compression research state from:

- the current worker checkout at commit `d57b484`
- the richer local mirror at `/Users/shawn/Documents/sjsu/thesis/lidar_pointcloud_compression` (same commit `d57b484`)
- in-repo summaries and notes:
  - `README.md`
  - `docs/repo_guide.md`
  - `docs/notes/research_journal_en.md`
  - `docs/notes/research_progress.md`
  - `notebooks/results_summary_current.md`
  - `/Users/shawn/Documents/sjsu/thesis/lidar_pointcloud_compression/docs/report/pointpillar_finetune_kitti.md`

Note: the task-specified HPC path `/fs/atipa/home/018219422/lidar_pointcloud_compression` was not mounted in this environment on 2026-03-18, so the richer local mirror above was used as the best available substitute.

## Repo-state audit

The worker checkout is a trimmed analysis snapshot, not the full experiment workspace.

| Path | Worker checkout | Rich mirror |
|---|---:|---:|
| `data/` files | missing | `66,847` |
| `logs/` files | missing | `101` |
| `docs/` files | `62` | `80` |
| `notebooks/` files | `20` | `20` |
| `src/` files | `108` | `135` |

Implication: this checkout is sufficient for reading the synthesized research state, but not for re-auditing every raw experiment artifact locally.

## Current project state

### High-level conclusion

As of the latest accessible evidence dated **March 16, 2026**, both evaluation tracks still identify the **codec / reconstruction path** as the main bottleneck, not the downstream detector baselines.

### Track 1 state

Stable identity-domain detector reference:

- PointPillars on `KITTI_Identity` stays near **~73 mAP3D(mod)** across the March 5-13, 2026 fine-tune runs.
- Best listed identity-domain run: `pp_ft_t1nq_geo_fr_geo_fr_260305_215136` at **73.6671 mAP3D(mod)**.

Reconstructed endpoint remains weak:

- Most reconstructed runs are near zero.
- The first materially better run is `t1nq_pillar_b_pillar_b_260309_201414` at **2.192242 reconstructed mAP3D(mod)**.
- Even that best completed result remains far below the identity-domain reference (`52.7378` reference vs `2.192242` reconstructed in the paired summary table).

Interpretation:

- Detector fine-tuning is not the limiting factor.
- Adding pillar/BEV side information helped directionally.
- The remaining failure mode is still geometry destruction in the codec/decoder path.
- The research journal explicitly points to stripe/banding artifacts and weak decoder structure as the next bottleneck, motivating the skip-decoder follow-up runs `25892` and `25894`.

### Track 2 state

Track 2 is more decisive after the March 16, 2026 repair cycle.

Accessible summary values:

| Setting | `AP3D@0.3` | `AP3D@0.5` | `AP3D@0.7` | `meanBestIoU3D` |
|---|---:|---:|---:|---:|
| `raw/basic` | `0.5700` | `0.4979` | `0.2435` | `0.6098` |
| `Stage0 baseline` | `0.0215` | `0.0028` | `0.0000` | `0.1202` |
| `Stage0 enhanced` | `0.0099` | `0.0010` | `0.0000` | `0.1040` |
| `Stage1 baseline` | `0.0083` | `0.0007` | `0.0000` | `0.0874` |
| `Stage1 enhanced` | `0.0050` | `0.0003` | `0.0000` | `0.0704` |

Interpretation:

- The repaired `RangeDet` raw/basic baseline is now credible.
- `Stage0 baseline` loses about **96.2%** of `AP3D@0.3` versus raw/basic (`0.0215` vs `0.5700`).
- `Stage1 enhanced` loses about **99.1%** of `AP3D@0.3` versus raw/basic (`0.0050` vs `0.5700`).
- `Stage1` is worse than `Stage0`, and `enhanced` is worse than `baseline` on this repaired detector path.

So the latest evidence supports a strong claim: **Track 2 failure is now primarily codec-induced range-image distortion, not a broken detector baseline**.

## Research narrative consistency check

The accessible summaries are internally consistent across files:

- `README.md` says Track 2's repaired baseline is no longer the blocker.
- `notebooks/results_summary_current.md` says the codec is the Track 2 bottleneck.
- `docs/notes/research_journal_en.md` says the March 16, 2026 corrected raw/basic chain isolated the codec as the dominant failure.
- `docs/report/pointpillar_finetune_kitti.md` confirms the Track 1 identity-domain detector remains stable around ~73 mAP3D(mod), so Track 1 collapse is also downstream of reconstruction.

## Most defensible current thesis-state statement

A conservative, evidence-backed statement is:

> As of March 16, 2026, the project has credible detector baselines in both tracks, but the learned LiDAR codec still fails to preserve detector-relevant geometry. Track 1 shows only a small recovery from pillar/BEV side information, while Track 2 shows near-total collapse once reconstructed outputs replace raw/basic inputs.

## Recommended next feasible work

1. Finish and read out the skip-decoder Track 1 / Track 2 follow-up jobs (`25892`, `25894`, and dependent Track 2 Stage0 evaluations).
2. Treat decoder artifact suppression and geometry-preserving supervision as the highest-priority technical direction.
3. Avoid spending more time on detector-baseline debugging unless new evidence contradicts the March 16, 2026 repaired-baseline result.
4. If a richer artifact audit is needed later, run it from the full workspace or an environment where the original `/fs/atipa/...` mount is available.
