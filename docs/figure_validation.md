# Figure validation audit

Date: 2026-03-18

## Scope

- Validated public-facing figure embeds in `docs/index.html` and `docs/gallery.html`
- Checked declared `<img width height>` pairs against real files in `docs/assets/`
- Reviewed representative figure outputs for readability and communication quality

## Approved homepage/public set

- `docs/assets/track2-rangedet-analysis-preview-web.png`
  - Homepage lead figure; strongest compact Track 2 summary for public readers.
- `docs/assets/track1-pointpillar-endpoint-panel-web.png`
  - Homepage Track 1 anchor; clearest detector-facing evidence.
- `docs/assets/track2-rootcause-worst-web.png`
  - Homepage support figure for concrete failure analysis.
- `docs/assets/track2-artifact-profiles-web.png`
  - Homepage support figure for analytic credibility beyond qualitative screenshots.

## Strong figures

- `docs/assets/track1-pointpillar-endpoint-panel-web.png`
  - Strongest Track 1 figure; directly links reconstruction quality to downstream detector failure.
- `docs/assets/track2-rangedet-analysis-preview-web.png`
  - Strongest overview figure; shows raw vs reconstructed behavior in one panel with clear before/after structure.
- `docs/assets/track2-rootcause-worst-web.png`
  - Strong failure-analysis figure; artifact pattern is obvious enough to support the captioned root-cause story.
- `docs/assets/track2-artifact-profiles-web.png`
  - Strong analytic support figure; the row-profile plot and staged comparison make recurring corruption interpretable.

## Weaker but still usable figures

- `docs/assets/track1-identity-bev-panel-web.png`
  - Useful as reference context, but weaker as a standalone hero figure because it needs the codec comparison beside it.
- `docs/assets/track1-codec-bev-panel-web.png`
  - Useful as the pair to the identity panel, but weaker than the endpoint panel for non-expert readers.
- `docs/assets/track2-cell5-latest-web.png`
  - Real and informative, but visually dense and less self-explanatory than the broader Track 2 overview/root-cause panels.

## Appendix-only / de-emphasized figures

- `docs/assets/track1-identity-vs-codec.gif`
  - Useful as gallery support, but weaker than the endpoint panel on the homepage.
- `docs/assets/track2-rangedet-overview-zoom.gif`
  - Motion support only; weaker than the static analysis preview as a homepage lead.
- `docs/assets/track2-rootcause-cycle.gif`
  - Redundant next to the stronger static root-cause and spectrum assets.
- `docs/assets/track2-exactgrid-preview-web.png`
  - Narrower detector-matching support figure; too caption-dependent for the homepage.
- `docs/assets/track2-exactlut-preview-web.png`
  - Same limitation as exact-grid.
- `docs/assets/track2-grid-vs-lut.gif`
  - Useful only when the audience already understands the grid/LUT distinction.

## Generated outputs used for validation

- `docs/assets/track1-identity-vs-codec.gif`
- `docs/assets/track2-rangedet-overview-zoom.gif`
- `docs/assets/track2-rootcause-cycle.gif`
- `docs/assets/track2-grid-vs-lut.gif`
- `docs/assets/track2-rootcause-spectrum.gif`

## Current embed status

Fresh local verification shows no remaining `width` / `height` mismatches in `docs/index.html` or `docs/gallery.html`.

## Verification command

```bash
python3 - <<'PY'
from pathlib import Path
from PIL import Image
from html.parser import HTMLParser

class P(HTMLParser):
    def __init__(self):
        super().__init__()
        self.items=[]
    def handle_starttag(self, tag, attrs):
        if tag == 'img':
            self.items.append(dict(attrs))

for html in Path('docs').glob('*.html'):
    p = P()
    p.feed(html.read_text())
    for item in p.items:
        src = item.get('src')
        if not src or src.startswith('http') or src.endswith('.svg'):
            continue
        with Image.open(html.parent / src) as im:
            assert str(im.size[0]) == item.get('width')
            assert str(im.size[1]) == item.get('height')
PY
```
