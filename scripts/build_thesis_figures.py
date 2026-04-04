#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "tex" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.5,
        "figure.dpi": 160,
        "savefig.dpi": 240,
    }
)

BLUE = "#2C6DB2"
ORANGE = "#D97706"
RED = "#B8332A"
GREEN = "#2E8B57"
GRAY = "#5B6573"
LIGHT_BLUE = "#EAF2FB"
LIGHT_ORANGE = "#FFF1DF"
LIGHT_RED = "#FBE6E3"
LIGHT_GREEN = "#E8F4EC"
LIGHT_GRAY = "#F4F6F8"


def savefig(fig, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / f"{stem}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def stylize_axes(ax, *, ygrid: bool = True) -> None:
    ax.tick_params(length=3.5, width=0.8)
    if ygrid:
        ax.grid(axis="y")
    else:
        ax.grid(False)


def add_panel_tag(ax, text: str) -> None:
    ax.text(
        0.0,
        1.04,
        text,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def draw_system_overview() -> None:
    fig, ax = plt.subplots(figsize=(12.8, 4.9))
    ax.axis("off")

    def lane(x, y, w, h, title, accent, fill):
        panel = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.16",
            linewidth=1.0,
            edgecolor=accent,
            facecolor=fill,
        )
        ax.add_patch(panel)
        ax.text(x + 0.18, y + h - 0.32, title, fontsize=11, fontweight="bold", color=accent, va="top")

    def box(x, y, w, h, label, fc=LIGHT_GRAY, ec=GRAY, size=9.5, weight="normal"):
        rect = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.0,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=size, weight=weight)

    def arrow(x1, y1, x2, y2, color=GRAY):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.2,
                color=color,
            )
        )

    ax.set_xlim(0, 16.5)
    ax.set_ylim(0, 6.6)

    lane(0.25, 3.4, 15.8, 2.6, "Track 2  Main claim lane", BLUE, "#F7FBFF")
    box(0.7, 4.15, 2.0, 0.95, "KITTI\npoint clouds", fc=LIGHT_BLUE, ec=BLUE)
    box(3.05, 4.15, 2.15, 0.95, "Range-image\nprojection", fc=LIGHT_GREEN, ec=GREEN)
    box(5.62, 4.15, 2.35, 0.95, "Mask-aware\nadaptive codec", fc=LIGHT_ORANGE, ec=ORANGE, weight="bold")
    box(8.42, 4.15, 2.35, 0.95, "Reconstructed\nrange image", fc=LIGHT_ORANGE, ec=ORANGE)
    box(11.2, 4.15, 2.05, 0.95, "RangeDet\nexport", fc="#F4EBFA", ec="#7A52A1")
    box(13.6, 4.15, 1.85, 0.95, "Frozen\ncompare", fc=LIGHT_RED, ec=RED, weight="bold")
    for xs in [(2.7, 4.63, 3.05, 4.63), (5.2, 4.63, 5.62, 4.63), (7.97, 4.63, 8.42, 4.63), (10.77, 4.63, 11.2, 4.63), (13.25, 4.63, 13.6, 4.63)]:
        arrow(*xs, color=BLUE)
    ax.text(13.65, 3.72, "Detector-facing endpoint is fixed.\nOnly codec-side choices move.", fontsize=8.8, color=GRAY)

    lane(0.25, 0.45, 15.8, 2.3, "Track 3  Support lane", GREEN, "#F8FCF8")
    box(0.9, 1.1, 2.0, 0.9, "KITTI\npoint clouds", fc=LIGHT_BLUE, ec=BLUE)
    box(3.35, 1.1, 2.2, 0.9, "RENO\nbaseline family", fc=LIGHT_ORANGE, ec=ORANGE, weight="bold")
    box(6.1, 1.1, 2.35, 0.9, "Reconstructed\npoint cloud", fc=LIGHT_ORANGE, ec=ORANGE)
    box(8.95, 1.1, 2.35, 0.9, "PointPillars\nendpoint", fc="#F4EBFA", ec="#7A52A1")
    box(11.85, 1.1, 2.8, 0.9, "Baseline set / bounded\nadaptive follow-up", fc=LIGHT_GREEN, ec=GREEN, weight="bold")
    for xs in [(2.9, 1.55, 3.35, 1.55), (5.55, 1.55, 6.1, 1.55), (8.45, 1.55, 8.95, 1.55), (11.3, 1.55, 11.85, 1.55)]:
        arrow(*xs, color=GREEN)
    ax.text(12.1, 2.18, "Adaptive claim only opens after\nB0/B1 references are stable.", fontsize=8.8, color=GRAY)

    savefig(fig, "thesis_system_overview_track2_track3")


def draw_track2_progression() -> None:
    labels = ["Stage0\nwinner", "Stage1\nrecovered", "Stage1\nconfirm", "Stage2\nA0"]
    ap03 = [0.1877, 0.2565, 0.2492, 0.2717]
    ap05 = [0.0741, 0.1293, 0.1266, 0.1539]
    iou = [0.2418, 0.3308, 0.3170, 0.3611]

    rate_labels = ["Stage1\nconfirm", "Stage2\nA0"]
    rate_proxy = [63.0, 18.2034]
    eq_bits = [6.0, 4.1843]

    x = list(range(len(labels)))
    fig = plt.figure(figsize=(10.2, 5.6))
    gs = GridSpec(2, 1, height_ratios=[2.2, 1.0], hspace=0.28)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    ax1.plot(x, ap03, marker="o", markersize=5.8, linewidth=2.1, color=BLUE, label=r"$AP3D@0.3$")
    ax1.plot(x, ap05, marker="s", markersize=5.2, linewidth=2.0, color=ORANGE, label=r"$AP3D@0.5$")
    ax1.plot(x, iou, marker="^", markersize=5.6, linewidth=2.0, color=GREEN, label=r"$meanBestIoU3D@0.3$")
    ax1.set_xticks(x, labels)
    ax1.set_ylabel("Detector-facing metric")
    ax1.set_ylim(0.05, 0.40)
    stylize_axes(ax1)
    add_panel_tag(ax1, "(a)")
    ax1.legend(frameon=False, loc="upper left", ncols=3, columnspacing=1.0, handlelength=2.2)
    ax1.axvspan(2.75, 3.25, color=LIGHT_RED, alpha=0.55, zorder=0)
    ax1.annotate(
        "Best current thesis result",
        xy=(3, ap03[-1]),
        xytext=(2.25, 0.355),
        arrowprops={"arrowstyle": "->", "linewidth": 0.9, "color": RED},
        fontsize=8.7,
        color=RED,
    )

    ax2.bar([0, 1], rate_proxy, width=0.42, color=[GRAY, RED], alpha=0.88, label="rate proxy")
    ax2.plot([0, 1], eq_bits, color=BLUE, marker="o", linewidth=1.8, markersize=5, label="eq bits")
    ax2.set_xticks([0, 1], rate_labels)
    ax2.set_ylabel("Internal rate\nindicator")
    stylize_axes(ax2)
    add_panel_tag(ax2, "(b)")
    ax2.legend(frameon=False, loc="upper right")
    ax2.annotate(
        "lower than confirm baseline",
        xy=(1, rate_proxy[1]),
        xytext=(0.32, 43),
        arrowprops={"arrowstyle": "->", "linewidth": 0.9, "color": GRAY},
        fontsize=8.5,
        color=GRAY,
    )

    savefig(fig, "track2_mainline_progression")


def _load_track3_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ref = next(r for r in rows if r["mode"] != "reconstructed")
    recon = next(r for r in rows if r["mode"] == "reconstructed")
    return ref, recon


def draw_track3_operating_points() -> None:
    points = [
        ("B0@16", 8.6340619559, 36.2923010157, 71.8146, BLUE),
        ("B0@64", 3.3648332411, 35.9167071444, 70.5157, BLUE),
        ("B1@16", 8.6323362044, 36.4478640924, 72.2564, RED),
        ("B1@64", 3.3263946149, 36.0382274699, 70.7054, RED),
    ]
    fig = plt.figure(figsize=(10.2, 4.8))
    gs = GridSpec(1, 2, width_ratios=[1.0, 1.0], wspace=0.28)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    for family, color in [("B0", BLUE), ("B1", RED)]:
        family_points = [p for p in points if p[0].startswith(family)]
        family_points = sorted(family_points, key=lambda row: row[1])
        ax1.plot([p[1] for p in family_points], [p[2] for p in family_points], color=color, linewidth=2.0, marker="o")
        ax2.plot([p[1] for p in family_points], [p[3] for p in family_points], color=color, linewidth=2.0, marker="o", label=family)
        for label, bpp, score, ap3d, _ in family_points:
            ax1.annotate(label, (bpp, score), xytext=(5, 5), textcoords="offset points", fontsize=8.6)
            ax2.annotate(label, (bpp, ap3d), xytext=(5, 5), textcoords="offset points", fontsize=8.6)

    ax1.set_xlabel("True bitrate (bpp)")
    ax1.set_ylabel(r"$map3d\_mod\_mean$")
    stylize_axes(ax1)
    add_panel_tag(ax1, "(a)")
    ax1.set_xlim(2.8, 9.2)
    ax1.set_ylim(35.75, 36.55)

    ax2.set_xlabel("True bitrate (bpp)")
    ax2.set_ylabel(r"$AP3D_{car,mod}$")
    stylize_axes(ax2)
    add_panel_tag(ax2, "(b)")
    ax2.set_xlim(2.8, 9.2)
    ax2.set_ylim(70.2, 72.6)
    ax2.legend(frameon=False, loc="lower right")

    savefig(fig, "track3_operating_points")


def draw_track3_bref_drift() -> None:
    labels = ["posQ16", "posQ64"]
    b0 = [36.2923010157, 35.9167071444]
    b1 = [36.4478640924, 36.0382274699]
    drift = [0.0043, 0.0034]
    x = [0, 1]

    fig = plt.figure(figsize=(9.6, 4.8))
    gs = GridSpec(1, 2, width_ratios=[1.45, 1.0], wspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    width = 0.32
    ax1.bar([i - width / 2 for i in x], b0, width, label="B0", color=BLUE, alpha=0.92)
    ax1.bar([i + width / 2 for i in x], b1, width, label="B1", color=RED, alpha=0.92)
    ax1.set_xticks(x, labels)
    ax1.set_ylabel(r"$map3d\_mod\_mean$")
    stylize_axes(ax1)
    add_panel_tag(ax1, "(a)")
    ax1.legend(frameon=False, loc="upper left")
    ax1.set_ylim(35.7, 36.6)

    ax2.bar(x, drift, width=0.44, color=GREEN, alpha=0.9)
    ax2.axhline(0.15, color=RED, linestyle="--", linewidth=1.2)
    ax2.text(1.45, 0.152, "gate threshold = 0.15", color=RED, fontsize=8.4, va="bottom", ha="right")
    ax2.set_xticks(x, labels)
    ax2.set_ylabel("Reference drift")
    stylize_axes(ax2)
    add_panel_tag(ax2, "(b)")
    ax2.set_ylim(0, 0.17)
    for i, d in zip(x, drift):
        ax2.annotate(f"{d:.4f}", (i, d), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8.5)

    savefig(fig, "track3_bref_drift")


def main() -> None:
    draw_system_overview()
    draw_track2_progression()
    draw_track3_operating_points()
    draw_track3_bref_drift()
    print("Generated thesis figures in", FIG_DIR)


if __name__ == "__main__":
    main()
