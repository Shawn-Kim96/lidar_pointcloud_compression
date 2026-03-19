from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_project_page_visuals.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_project_page_visuals", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BuildProjectPageVisualsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_metrics_summary_loads_current_repo_values(self) -> None:
        summary = self.module.load_metrics_summary(
            REPO_ROOT / "notebooks" / "results_summary_current.csv",
            REPO_ROOT / "docs" / "notes" / "research_progress.md",
        )
        self.assertGreater(summary.pointcloud_reference, 70.0)
        self.assertGreater(summary.pointcloud_reconstructed, 2.0)
        self.assertGreater(summary.range_raw_basic, 0.4)
        self.assertLess(summary.range_reconstructed_best, 0.01)
        self.assertAlmostEqual(summary.projection_retention, 41.69, places=2)

    def test_main_build_generates_chart_and_montage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            self.module.main.__wrapped__ if hasattr(self.module.main, "__wrapped__") else None
            summary = self.module.load_metrics_summary(
                REPO_ROOT / "notebooks" / "results_summary_current.csv",
                REPO_ROOT / "docs" / "notes" / "research_progress.md",
            )
            chart_path = output_dir / "downstream-gap-summary.png"
            self.module.build_downstream_gap_summary(summary, chart_path)
            self.module.build_hero_montage(REPO_ROOT / "docs" / "assets", chart_path, output_dir / "project-hero-montage.png")

            with Image.open(output_dir / "downstream-gap-summary.png") as image:
                self.assertGreaterEqual(image.size[0], 1800)
                self.assertGreaterEqual(image.size[1], 800)

            with Image.open(output_dir / "project-hero-montage.png") as image:
                self.assertEqual(image.size, (1800, 1320))


if __name__ == "__main__":
    unittest.main()
