#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "tex" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def savefig(fig, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_system_overview() -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    def box(x, y, w, h, label, fc="#f4f4f4", ec="#222222", size=11, weight="normal"):
        rect = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=size, weight=weight)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, linewidth=1.4))

    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)

    ax.text(4.2, 5.45, "Track 2 Main Claim", fontsize=13, weight="bold")
    box(0.5, 4.1, 2.1, 0.9, "KITTI\nPoint Clouds", fc="#e8f1ff")
    box(3.0, 4.1, 2.1, 0.9, "Range Image\nProjection", fc="#eef6ea")
    box(5.5, 4.1, 2.3, 0.9, "Task-Driven\nCodec", fc="#fff2cc", weight="bold")
    box(8.3, 4.1, 2.3, 0.9, "Reconstructed\nRange Image", fc="#fce5cd")
    box(11.1, 4.1, 2.2, 0.9, "RangeDet\nExport", fc="#eadcf8")
    box(13.8, 4.1, 1.7, 0.9, "Frozen\nCompare", fc="#f4cccc", weight="bold")
    for xs in [(2.6, 4.55, 3.0, 4.55), (5.1, 4.55, 5.5, 4.55), (7.8, 4.55, 8.3, 4.55), (10.6, 4.55, 11.1, 4.55), (13.3, 4.55, 13.8, 4.55)]:
        arrow(*xs)

    ax.text(4.4, 2.1, "Track 3 Support Lane", fontsize=13, weight="bold")
    box(0.8, 0.8, 2.1, 0.9, "KITTI\nPoint Clouds", fc="#e8f1ff")
    box(3.4, 0.8, 2.3, 0.9, "External Codec\n(RENO)", fc="#fff2cc", weight="bold")
    box(6.3, 0.8, 2.5, 0.9, "Reconstructed\nPoint Cloud", fc="#fce5cd")
    box(9.4, 0.8, 2.5, 0.9, "PointPillars\nEndpoint", fc="#eadcf8")
    box(12.5, 0.8, 2.5, 0.9, "Support Baseline\n/ Adaptive Check", fc="#f4cccc", weight="bold")
    for xs in [(2.9, 1.25, 3.4, 1.25), (5.7, 1.25, 6.3, 1.25), (8.8, 1.25, 9.4, 1.25), (11.9, 1.25, 12.5, 1.25)]:
        arrow(*xs)

    ax.text(14.65, 2.9, "Frozen downstream\nprotocol is part of\n the method.", fontsize=11, ha="center", va="center")

    savefig(fig, "thesis_system_overview_track2_track3")


def draw_track2_progression() -> None:
    labels = ["Stage0\nwinner", "Stage1\nrecovered", "Stage1\nconfirm", "Stage2\nA0"]
    ap03 = [0.1877, 0.2565, 0.2492, 0.2717]
    ap05 = [0.0741, 0.1293, 0.1266, 0.1539]
    iou = [0.2418, 0.3308, 0.3170, 0.3611]

    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(x, ap03, marker="o", linewidth=2.0, label="AP3D@0.3")
    ax.plot(x, ap05, marker="s", linewidth=2.0, label="AP3D@0.5")
    ax.plot(x, iou, marker="^", linewidth=2.0, label="meanBestIoU3D@0.3")
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Metric value")
    ax.set_title("Track 2 Mainline Progression")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncols=3, loc="upper left")
    ax.set_ylim(0, 0.42)
    savefig(fig, "track2_mainline_progression")


def _load_track3_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ref = next(r for r in rows if r["mode"] != "reconstructed")
    recon = next(r for r in rows if r["mode"] == "reconstructed")
    return ref, recon


def draw_track3_operating_points() -> None:
    points = [
        ("B0@16", 8.6340619559, 36.2923010157, "#1f77b4"),
        ("B0@64", 3.3648332411, 35.9167071444, "#1f77b4"),
        ("B1@16", 8.6323362044, 36.4478640924, "#d62728"),
        ("B1@64", 3.3263946149, 36.0382274699, "#d62728"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for label, bpp, score, color in points:
        ax.scatter([bpp], [score], s=90, color=color)
        ax.annotate(label, (bpp, score), xytext=(6, 6), textcoords="offset points", fontsize=10)
    ax.plot([points[1][1], points[0][1]], [points[1][2], points[0][2]], color="#1f77b4", alpha=0.6)
    ax.plot([points[3][1], points[2][1]], [points[3][2], points[2][2]], color="#d62728", alpha=0.6)
    ax.set_xlabel("true_bpp")
    ax.set_ylabel("map3d_mod_mean")
    ax.set_title("Track 3 B0/B1 Operating Points")
    ax.grid(alpha=0.25)
    savefig(fig, "track3_operating_points")


def draw_track3_bref_drift() -> None:
    labels = ["posQ16", "posQ64"]
    b0 = [36.2923010157, 35.9167071444]
    b1 = [36.4478640924, 36.0382274699]
    drift = [0.0043, 0.0034]
    x = [0, 1]

    fig, ax1 = plt.subplots(figsize=(7.4, 4.8))
    width = 0.28
    ax1.bar([i - width / 2 for i in x], b0, width, label="B0", color="#4c78a8")
    ax1.bar([i + width / 2 for i in x], b1, width, label="B1", color="#e45756")
    ax1.set_xticks(x, labels)
    ax1.set_ylabel("map3d_mod_mean")
    ax1.set_title("Track 3 Baseline Reference Stability")
    ax1.legend(frameon=False, loc="upper left")
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, drift, marker="o", color="#54a24b", linewidth=2, label="drift")
    ax2.set_ylabel("drift")
    ax2.set_ylim(0, 0.02)
    for i, d in zip(x, drift):
        ax2.annotate(f"{d:.4f}", (i, d), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)

    savefig(fig, "track3_bref_drift")


def main() -> None:
    draw_system_overview()
    draw_track2_progression()
    draw_track3_operating_points()
    draw_track3_bref_drift()
    print("Generated thesis figures in", FIG_DIR)


if __name__ == "__main__":
    main()
