# Figure validation notes — worker 2 (2026-03-19)

Scope: validate current Track 1 / Track 2 figure assets for correctness, readability, aspect ratio, labeling, and communication quality using only committed local assets and notebook-backed exports. The dirty remote HPC worktree was not touched.

## Reproducible local outputs checked

Generated with:

```bash
python3 scripts/export_project_page_assets.py --output-dir <tmpdir>
```

Confirmed notebook-backed outputs:

- `track1-identity-bev-panel-web.png`
- `track1-codec-bev-panel-web.png`
- `track1-pointpillar-endpoint-panel-web.png`
- `track2-rangedet-overview-panel-web.png`
- `track2-rangedet-zoom-panel-web.png`
- `track2-cell5-latest-web.png`
- `track2-artifact-profiles-web.png`
- `track2-rootcause-worst-web.png`
- `track1-identity-vs-codec.gif`
- `track2-rangedet-overview-zoom.gif`
- `track2-rootcause-spectrum.gif`

Matching checks:

- Byte-identical to committed assets: `track1-*` web PNGs, `track2-rangedet-overview-panel-web.png`, `track2-rangedet-zoom-panel-web.png`, `track2-rootcause-worst-web.png`, `track1-identity-vs-codec.gif`, `track2-rangedet-overview-zoom.gif`
- Same rendered content / same canvas size from local export: `track2-cell5-latest-web.png` (`1800x456`), `track2-artifact-profiles-web.png` (`1600x486`), `track2-rootcause-spectrum.gif` (`1400x553`)

## Strong figures

| File | Why it works |
| --- | --- |
| `docs/assets/track1-pointpillar-endpoint-panel-web.png` | Best Track 1 figure for the detector-facing story. Six-panel layout is dense but still legible; the decoded-vs-raw comparison and box overlays make the failure mode easy to explain. |
| `docs/assets/track2-rangedet-analysis-preview-web.png` | Strong overview figure. It combines scene context, GT/predicted envelopes, and the raw-vs-reconstructed split without obvious aspect-ratio distortion. |
| `docs/assets/track2-rangedet-overview-panel-web.png` | Strong support figure. The side-by-side structure is easy to read and the labeling is concise. |
| `docs/assets/track2-rangedet-zoom-panel-web.png` | Strong companion to the overview panel because it isolates the detector-critical crop without losing label clarity. |
| `docs/assets/track2-artifact-profiles-web.png` | Strong quantitative support figure. The row-profile plot plus range-image strip communicates banding / row corruption clearly and uses the wider aspect ratio well. |
| `docs/assets/track2-rootcause-worst-web.png` | Strong “make it concrete” diagnostic. Titles are readable at web size and the five-way comparison makes the deterioration pattern obvious. |

## Medium / usable figures

| File | Judgment | Notes |
| --- | --- | --- |
| `docs/assets/track1-identity-bev-panel-web.png` | Usable | Correct and readable, but strongest when paired with the codec panel rather than shown alone. |
| `docs/assets/track1-codec-bev-panel-web.png` | Usable | Same as above; the single panel is less persuasive than the identity-vs-codec pair. |
| `docs/assets/track2-cell5-latest-web.png` | Usable | Correct and reproducible, but communication is narrower: heavy box overlap and limited surrounding context make it better as a supporting figure than a lead figure. |
| `docs/assets/track2-rootcause-spectrum.gif` | Usable | Good for showing recurrence across cases, but better as appendix motion support than as a primary explanatory asset. |

## Weak / likely redundant figures

| File | Judgment | Notes |
| --- | --- | --- |
| `docs/assets/track2-exactgrid-preview-web.png` | Weak | Technically readable, but visually close to the LUT variant and relies heavily on caption text to explain why the matching rule matters. |
| `docs/assets/track2-exactlut-preview-web.png` | Weak | Same issue as exact-grid: useful as a backup asset, but not strong enough to stand alone in a compact gallery. |
| `docs/assets/track2-grid-vs-lut.gif` | Weak-to-usable | Works only if the audience already understands the grid/LUT distinction; otherwise the motion compares two panels that look too similar. |
| `docs/assets/track2-rootcause-cycle.gif` | Weak-to-usable | Redundant next to `track2-rootcause-worst-web.png` and `track2-rootcause-spectrum.gif`; the cycle adds motion but not much new explanatory content. |

## Labeling / aspect-ratio notes

- No obvious aspect-ratio distortion was observed in the checked web PNGs.
- The best-labeled assets are the Track 1 endpoint panel and the Track 2 overview / zoom / root-cause panels.
- `track2-cell5-latest-web.png`, `track2-exactgrid-preview-web.png`, and `track2-exactlut-preview-web.png` are correct but visually depend more on the surrounding caption than on self-contained in-image labeling.

## Recommendation

If the gallery needs trimming, keep these as the core set:

1. `docs/assets/track1-pointpillar-endpoint-panel-web.png`
2. `docs/assets/track2-rangedet-analysis-preview-web.png`
3. `docs/assets/track2-rangedet-overview-panel-web.png`
4. `docs/assets/track2-rangedet-zoom-panel-web.png`
5. `docs/assets/track2-artifact-profiles-web.png`
6. `docs/assets/track2-rootcause-worst-web.png`

Keep the exact-grid / exact-LUT pair and motion GIFs as appendix-only support.
