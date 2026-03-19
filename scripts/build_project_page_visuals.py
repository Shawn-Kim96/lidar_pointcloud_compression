#!/usr/bin/env python3
"""Build data-driven visuals for the project page.

This script creates two public-page assets from repository data:
- `downstream-gap-summary.png`: quantitative summary chart from current results
- `project-hero-montage.png`: hero montage composed from the strongest approved figures

Usage:
    python3 scripts/build_project_page_visuals.py
    python3 scripts/build_project_page_visuals.py --output-dir docs/assets
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class MetricsSummary:
    pointcloud_reference: float
    pointcloud_reconstructed: float
    range_raw_basic: float
    range_reconstructed_best: float
    projection_retention: float


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build derived project-page visuals from repo data.")
    parser.add_argument(
        "--output-dir",
        default="docs/assets",
        help="Output directory, relative to the repository root by default.",
    )
    return parser.parse_args()


def _to_float(value: str) -> float | None:
    value = (value or "").strip()
    if not value:
        return None
    return float(value)


def load_metrics_summary(results_csv: Path, research_progress_md: Path) -> MetricsSummary:
    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    track1_identity = max(
        _to_float(row["reference_metric"]) or float("-inf")
        for row in rows
        if row["track"] == "track1" and row["group"] == "identity_pointpillars"
    )
    track1_reconstructed = max(
        _to_float(row["reconstructed_metric"]) or float("-inf")
        for row in rows
        if row["track"] == "track1" and row["group"] == "noquant_reconstructed_endpoint"
    )
    track2_raw_basic = max(
        _to_float(row["ap3d_05"]) or float("-inf")
        for row in rows
        if row["track"] == "track2" and row["experiment"] == "raw_basic"
    )
    track2_reconstructed_best = max(
        _to_float(row["ap3d_05"]) or float("-inf")
        for row in rows
        if row["track"] == "track2" and row["experiment"] != "raw_basic"
    )

    progress_text = research_progress_md.read_text(encoding="utf-8")
    retention_match = re.search(r"mean point-retention after projection:\s*`?([0-9.]+)`?", progress_text)
    if retention_match is None:
        raise ValueError("Could not find projection retention metric in research_progress.md.")
    projection_retention = float(retention_match.group(1)) * 100.0

    return MetricsSummary(
        pointcloud_reference=track1_identity,
        pointcloud_reconstructed=track1_reconstructed,
        range_raw_basic=track2_raw_basic,
        range_reconstructed_best=track2_reconstructed_best,
        projection_retention=projection_retention,
    )


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def build_downstream_gap_summary(summary: MetricsSummary, output_path: Path) -> None:
    labels = [
        "Point-cloud path\nreference detector",
        "Point-cloud path\nbest reconstructed",
        "Range-image path\nraw/basic detector",
        "Range-image path\nbest reconstructed",
    ]
    values = [
        summary.pointcloud_reference,
        summary.pointcloud_reconstructed,
        summary.range_raw_basic,
        summary.range_reconstructed_best,
    ]
    colors = ["#0f766e", "#b42318", "#334155", "#b45309"]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.9), dpi=170, constrained_layout=True)
    fig.patch.set_facecolor("#f7f2ea")

    axes[0].barh(labels[:2], values[:2], color=colors[:2], height=0.6)
    axes[0].set_title("Point-cloud reconstruction path", fontsize=14, weight="bold")
    axes[0].set_xlabel("mAP3D(mod)")
    axes[0].set_xlim(0, max(values[:2]) * 1.08)
    axes[0].grid(axis="x", alpha=0.2)
    for idx, value in enumerate(values[:2]):
        axes[0].text(
            value + 0.8,
            idx,
            f"{value:.2f}",
            va="center",
            fontsize=12,
            weight="bold",
            bbox={"boxstyle": "round,pad=0.25", "fc": "#fffdfa", "ec": "none", "alpha": 0.9},
        )
    axes[0].text(
        0.98,
        0.1,
        f"{summary.pointcloud_reference / max(summary.pointcloud_reconstructed, 1e-6):.1f}× gap",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        color="#6b7280",
    )

    axes[1].barh(labels[2:], values[2:], color=colors[2:], height=0.6)
    axes[1].set_title("Direct range-image detector path", fontsize=14, weight="bold")
    axes[1].set_xlabel("AP3D@0.5")
    axes[1].set_xlim(0, max(values[2:]) * 1.16)
    axes[1].grid(axis="x", alpha=0.2)
    for idx, value in enumerate(values[2:]):
        offset = 0.014 if value > 0.01 else 0.045
        axes[1].text(
            value + offset,
            idx,
            f"{value:.4f}",
            va="center",
            fontsize=12,
            weight="bold",
            bbox={"boxstyle": "round,pad=0.25", "fc": "#fffdfa", "ec": "none", "alpha": 0.9},
        )
    axes[1].text(
        0.98,
        0.1,
        f"{summary.range_raw_basic / max(summary.range_reconstructed_best, 1e-6):.0f}× gap",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        color="#6b7280",
    )

    for ax in axes:
        ax.set_facecolor("#fffdfa")
        ax.tick_params(labelsize=10)
        for spine in ax.spines.values():
            spine.set_alpha(0.15)

    fig.suptitle("Current downstream gap", fontsize=18, weight="bold", y=1.01)
    fig.text(
        0.5,
        0.01,
        f"Projection-only retention baseline: {summary.projection_retention:.2f}% of raw points kept after project→unproject.",
        ha="center",
        fontsize=11,
        color="#4b5563",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def _fit_panel(image: Image.Image, size: tuple[int, int], background: str = "#fffdfa") -> Image.Image:
    canvas = Image.new("RGB", size, background)
    fitted = image.copy()
    fitted.thumbnail((size[0] - 24, size[1] - 24), Image.Resampling.LANCZOS)
    offset = ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2)
    canvas.paste(fitted.convert("RGB"), offset)
    return canvas


def build_hero_montage(assets_dir: Path, chart_path: Path, output_path: Path) -> None:
    top = Image.open(assets_dir / "track2-rangedet-analysis-preview-web.png").convert("RGB")
    bottom = Image.open(chart_path).convert("RGB")

    canvas = Image.new("RGB", (1800, 1320), "#f7f2ea")
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(36)
    label_font = _load_font(24)

    top_panel = _fit_panel(top, (1740, 700))
    bottom_panel = _fit_panel(bottom, (1740, 470))

    canvas.paste(top_panel, (30, 90))
    canvas.paste(bottom_panel, (30, 820))

    draw.text((34, 22), "Current detector-facing evidence", fill="#1f2937", font=title_font)
    draw.rounded_rectangle((48, 110, 520, 150), radius=16, fill="#ffffffdd")
    draw.text((64, 118), "Direct range-image path: raw/basic vs reconstructed", fill="#0f172a", font=label_font)
    draw.rounded_rectangle((48, 840, 420, 880), radius=16, fill="#ffffffdd")
    draw.text((64, 848), "Quantitative gap from experiment tables", fill="#0f172a", font=label_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, optimize=True)


def main() -> int:
    args = parse_args()
    root = repo_root()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_metrics_summary(
        root / "notebooks" / "results_summary_current.csv",
        root / "docs" / "notes" / "research_progress.md",
    )

    chart_path = output_dir / "downstream-gap-summary.png"
    build_downstream_gap_summary(summary, chart_path)
    build_hero_montage(output_dir, chart_path, output_dir / "project-hero-montage.png")

    print(output_dir / "downstream-gap-summary.png")
    print(output_dir / "project-hero-montage.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
