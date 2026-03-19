from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "export_project_page_assets.py"
DOCS_DIR = REPO_ROOT / "docs"


class ImgParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.items: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        if tag == "img":
            self.items.append(dict(attrs))


def load_module():
    spec = importlib.util.spec_from_file_location("export_project_page_assets", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_local_images(html_path: Path) -> list[dict[str, str]]:
    parser = ImgParser()
    parser.feed(html_path.read_text(encoding="utf-8"))
    return [
        item
        for item in parser.items
        if (src := item.get("src")) and not src.startswith("http") and not src.endswith(".svg")
    ]


class ExportProjectPageAssetsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_track2_specs_include_notebook_backed_assets(self) -> None:
        filenames = {spec.filename for spec in self.module.EXPORT_SPECS}
        self.assertIn("track2-cell5-latest.png", filenames)
        self.assertIn("track2-artifact-profiles.png", filenames)
        self.assertIn("track2-rootcause-worst.png", filenames)

    def test_export_selects_expected_multi_output_track2_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            exported = self.module.export_assets(output_dir)
            exported_map = {spec.filename: path for spec, path in exported}

            with Image.open(exported_map["track2-cell5-latest.png"]) as image:
                self.assertEqual(image.size, (3655, 925))

            with Image.open(exported_map["track2-artifact-profiles.png"]) as image:
                self.assertEqual(image.size, (2615, 795))

            with Image.open(exported_map["track2-rootcause-worst.png"]) as image:
                self.assertEqual(image.size, (5060, 2000))

            with Image.open(exported_map["track2-rootcause-median-a.png"]) as image:
                self.assertEqual(image.size, (5060, 2000))

    def test_gif_specs_cover_rootcause_spectrum(self) -> None:
        spectrum = next(
            spec for spec in self.module.GIF_VARIANTS if spec.filename == "track2-rootcause-spectrum.gif"
        )
        self.assertEqual(
            spectrum.frame_names,
            (
                "track2-rootcause-worst.png",
                "track2-rootcause-worst-secondary.png",
                "track2-rootcause-median-a.png",
                "track2-rootcause-median-b.png",
                "track2-rootcause-median-c.png",
            ),
        )
        self.assertEqual(spectrum.max_width, 1400)

    def test_index_and_gallery_image_dimensions_match_assets(self) -> None:
        for html_path in (DOCS_DIR / "index.html", DOCS_DIR / "gallery.html"):
            for item in parse_local_images(html_path):
                with self.subTest(html=html_path.name, src=item["src"]):
                    with Image.open(DOCS_DIR / item["src"]) as image:
                        self.assertEqual(item.get("width"), str(image.size[0]))
                        self.assertEqual(item.get("height"), str(image.size[1]))

    def test_homepage_uses_approved_public_assets(self) -> None:
        index_html = (DOCS_DIR / "index.html").read_text(encoding="utf-8")

        self.assertIn('src="assets/downstream-gap-summary.png"', index_html)
        self.assertNotIn('src="assets/project-hero-montage.png"', index_html)

        for asset_name in (
            "track2-rangedet-analysis-preview-web.png",
            "track1-pointpillar-endpoint-panel-web.png",
            "track2-rootcause-worst-web.png",
            "track2-artifact-profiles-web.png",
        ):
            with self.subTest(required_asset=asset_name):
                self.assertIn(asset_name, index_html)

        for appendix_only in (
            "track1-identity-vs-codec.gif",
            "track2-exactgrid-preview-web.png",
            "track2-exactlut-preview-web.png",
            "track2-grid-vs-lut.gif",
        ):
            with self.subTest(disallowed_asset=appendix_only):
                self.assertNotIn(appendix_only, index_html)


if __name__ == "__main__":
    unittest.main()
