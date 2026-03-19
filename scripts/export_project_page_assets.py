#!/usr/bin/env python3
"""Export notebook PNG outputs used by the project page.

This keeps the current Track 1 and Track 2 notebook-derived visuals
reproducible instead of relying on one-off manual extraction.

Usage:
    python3 scripts/export_project_page_assets.py
    python3 scripts/export_project_page_assets.py --output-dir docs/assets
"""

from __future__ import annotations

import argparse
import base64
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class ExportSpec:
    notebook_relpath: str
    cell_index: int
    image_index: int
    filename: str
    description: str
    web_max_width: int = 1800


@dataclass(frozen=True)
class GifSpec:
    filename: str
    frame_names: tuple[str, ...]
    duration_ms: int
    max_width: int


EXPORT_SPECS = (
    ExportSpec(
        notebook_relpath="notebooks/track1_identity_vs_codec_visualization.ipynb",
        cell_index=7,
        image_index=0,
        filename="track1-identity-bev-panel.png",
        description="Track 1 identity reconstruction BEV panel",
    ),
    ExportSpec(
        notebook_relpath="notebooks/track1_identity_vs_codec_visualization.ipynb",
        cell_index=8,
        image_index=0,
        filename="track1-codec-bev-panel.png",
        description="Track 1 codec reconstruction BEV panel",
    ),
    ExportSpec(
        notebook_relpath="notebooks/stage0_stage1_kitti_pointpillar_visualization.executed.ipynb",
        cell_index=6,
        image_index=0,
        filename="track1-pointpillar-endpoint-panel.png",
        description="Track 1 PointPillars endpoint comparison panel",
    ),
    ExportSpec(
        notebook_relpath="notebooks/rangedet_analysis.executed.ipynb",
        cell_index=4,
        image_index=0,
        filename="track2-rangedet-overview-panel.png",
        description="Track 2 RangeDet raw versus reconstructed overview panel",
    ),
    ExportSpec(
        notebook_relpath="notebooks/rangedet_analysis.executed.ipynb",
        cell_index=6,
        image_index=0,
        filename="track2-rangedet-zoom-panel.png",
        description="Track 2 RangeDet zoomed raw versus reconstructed comparison",
    ),
    ExportSpec(
        notebook_relpath="notebooks/rangedet_analysis.executed.ipynb",
        cell_index=5,
        image_index=0,
        filename="track2-cell5-latest.png",
        description="Track 2 GT-versus-RangeDet envelope comparison panel",
    ),
    ExportSpec(
        notebook_relpath="notebooks/rangedet_analysis.executed.ipynb",
        cell_index=10,
        image_index=0,
        filename="track2-artifact-profiles.png",
        description="Track 2 artifact profile summary panel",
        web_max_width=1600,
    ),
    ExportSpec(
        notebook_relpath="notebooks/track2_codec_root_cause.executed.ipynb",
        cell_index=6,
        image_index=0,
        filename="track2-rootcause-worst.png",
        description="Track 2 worst-case root-cause panel",
    ),
    ExportSpec(
        notebook_relpath="notebooks/track2_codec_root_cause.executed.ipynb",
        cell_index=6,
        image_index=1,
        filename="track2-rootcause-worst-secondary.png",
        description="Track 2 secondary worst-case root-cause panel",
    ),
    ExportSpec(
        notebook_relpath="notebooks/track2_codec_root_cause.executed.ipynb",
        cell_index=6,
        image_index=2,
        filename="track2-rootcause-median-a.png",
        description="Track 2 median root-cause panel A",
    ),
    ExportSpec(
        notebook_relpath="notebooks/track2_codec_root_cause.executed.ipynb",
        cell_index=6,
        image_index=3,
        filename="track2-rootcause-median-b.png",
        description="Track 2 median root-cause panel B",
    ),
    ExportSpec(
        notebook_relpath="notebooks/track2_codec_root_cause.executed.ipynb",
        cell_index=6,
        image_index=4,
        filename="track2-rootcause-median-c.png",
        description="Track 2 median root-cause panel C",
    ),
)

WEB_VARIANTS = (
    "track1-identity-bev-panel.png",
    "track1-codec-bev-panel.png",
    "track1-pointpillar-endpoint-panel.png",
    "track2-rangedet-overview-panel.png",
    "track2-rangedet-zoom-panel.png",
    "track2-cell5-latest.png",
    "track2-artifact-profiles.png",
    "track2-rootcause-worst.png",
)

GIF_VARIANTS = (
    GifSpec(
        filename="track1-identity-vs-codec.gif",
        frame_names=("track1-identity-bev-panel.png", "track1-codec-bev-panel.png"),
        duration_ms=900,
        max_width=1600,
    ),
    GifSpec(
        filename="track2-rangedet-overview-zoom.gif",
        frame_names=("track2-rangedet-overview-panel.png", "track2-rangedet-zoom-panel.png"),
        duration_ms=1050,
        max_width=1600,
    ),
    GifSpec(
        filename="track2-rootcause-spectrum.gif",
        frame_names=(
            "track2-rootcause-worst.png",
            "track2-rootcause-worst-secondary.png",
            "track2-rootcause-median-a.png",
            "track2-rootcause-median-b.png",
            "track2-rootcause-median-c.png",
        ),
        duration_ms=1150,
        max_width=1400,
    ),
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the current Track 1 PNG notebook outputs used by the project "
            "page into docs/assets/."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="docs/assets",
        help="Output directory, relative to the repository root by default.",
    )
    return parser.parse_args()


def load_notebook(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_png_data(png_data: object) -> str:
    if isinstance(png_data, str):
        return png_data
    if isinstance(png_data, list):
        return "".join(chunk for chunk in png_data if isinstance(chunk, str))
    raise TypeError(f"Unsupported image/png payload type: {type(png_data)!r}")


def extract_png_bytes(cell: dict, spec: ExportSpec) -> bytes:
    png_payloads = []

    for output in cell.get("outputs", []):
        if not isinstance(output, dict):
            continue

        data = output.get("data")
        if not isinstance(data, dict) or "image/png" not in data:
            continue

        png_payloads.append(normalize_png_data(data["image/png"]))

    if not png_payloads:
        raise ValueError(
            f"No PNG outputs found in {spec.notebook_relpath} cell {spec.cell_index}."
        )

    if spec.image_index >= len(png_payloads):
        raise IndexError(
            f"{spec.notebook_relpath} cell {spec.cell_index} exposes {len(png_payloads)} PNG outputs; "
            f"cannot read image index {spec.image_index}."
        )

    return base64.b64decode(png_payloads[spec.image_index])


def export_assets(output_dir: Path) -> list[tuple[ExportSpec, Path]]:
    root = repo_root()
    output_dir.mkdir(parents=True, exist_ok=True)

    notebook_cache: dict[Path, dict] = {}
    exported: list[tuple[ExportSpec, Path]] = []

    for spec in EXPORT_SPECS:
        notebook_path = root / spec.notebook_relpath
        notebook = notebook_cache.setdefault(notebook_path, load_notebook(notebook_path))
        cells = notebook.get("cells", [])

        if spec.cell_index >= len(cells):
            raise IndexError(
                f"{spec.notebook_relpath} has {len(cells)} cells; "
                f"cannot read cell {spec.cell_index}."
            )

        png_bytes = extract_png_bytes(cells[spec.cell_index], spec)
        output_path = output_dir / spec.filename
        output_path.write_bytes(png_bytes)
        exported.append((spec, output_path))

    return exported


def build_web_variant(image_path: Path, max_width: int) -> Path:
    web_path = image_path.with_name(f"{image_path.stem}-web{image_path.suffix}")
    with Image.open(image_path) as image:
        image = resize_to_max_width(image, max_width)
        image.save(web_path, optimize=True, compress_level=9)
    return web_path


def build_gif(
    output_dir: Path,
    filename: str,
    frame_names: tuple[str, ...],
    duration_ms: int,
    max_width: int,
) -> Path:
    frames = []
    for frame_name in frame_names:
        frame_path = output_dir / frame_name
        with Image.open(frame_path) as image:
            image = resize_to_max_width(image, max_width)
            frames.append(image.convert("P", palette=Image.ADAPTIVE))

    output_path = output_dir / filename
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return output_path


def resize_to_max_width(image: Image.Image, max_width: int) -> Image.Image:
    if image.width <= max_width:
        return image.copy()
    new_height = round(image.height * (max_width / image.width))
    return image.resize((max_width, new_height), Image.Resampling.LANCZOS)


def main() -> int:
    args = parse_args()
    root = repo_root()
    output_dir = Path(args.output_dir)

    if not output_dir.is_absolute():
        output_dir = root / output_dir

    exported_paths = {}
    for spec, output_path in export_assets(output_dir):
        try:
            display_path = output_path.relative_to(root)
        except ValueError:
            display_path = output_path

        exported_paths[spec.filename] = output_path
        print(
            f"exported {display_path} "
            f"<- {spec.notebook_relpath} cell[{spec.cell_index}] "
            f"({spec.description})"
        )

    spec_by_filename = {spec.filename: spec for spec in EXPORT_SPECS}

    for filename in WEB_VARIANTS:
        output_path = output_dir / filename
        if not output_path.exists():
            continue
        web_path = build_web_variant(output_path, spec_by_filename[filename].web_max_width)
        try:
            display_path = web_path.relative_to(root)
        except ValueError:
            display_path = web_path
        print(f"optimized {display_path} <- {output_path.name}")

    for gif_spec in GIF_VARIANTS:
        if not all((output_dir / frame_name).exists() for frame_name in gif_spec.frame_names):
            continue
        gif_path = build_gif(
            output_dir,
            gif_spec.filename,
            gif_spec.frame_names,
            gif_spec.duration_ms,
            gif_spec.max_width,
        )
        try:
            display_path = gif_path.relative_to(root)
        except ValueError:
            display_path = gif_path
        print(f"animated {display_path} <- {', '.join(gif_spec.frame_names)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
