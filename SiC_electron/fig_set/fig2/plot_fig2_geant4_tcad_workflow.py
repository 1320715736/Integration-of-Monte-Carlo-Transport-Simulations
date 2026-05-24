#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Wedge


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
    }
)


FIG_DIR = Path(__file__).resolve().parent
ASSET_DIR = FIG_DIR / "assets"
OUTPUT_PNG = FIG_DIR / "fig2_geant4_tcad_workflow.png"
OUTPUT_SVG = FIG_DIR / "fig2_geant4_tcad_workflow.svg"

TEXT = "#111827"
MUTED = "#475569"
ARROW = "#334155"
SOURCE_EDGE = "#0284C7"
SOURCE_FACE = "#E0F2FE"
SOURCE_TEXT = "#075985"
PURPLE_EDGE = "#4F46E5"
PURPLE_FACE = "#EEF2FF"
GREEN_EDGE = "#0F766E"
GREEN_FACE = "#F0FDFA"
SENTAURUS_EDGE = "#059669"
SENTAURUS_FACE = "#ECFDF5"
OUTPUT_EDGE = "#2563EB"
OUTPUT_FACE = "#EFF6FF"


def add_box(ax: plt.Axes, x: float, y: float, width: float, height: float, edge: str, face: str) -> tuple[float, float, float, float]:
    x0 = x - width / 2.0
    y0 = y - height / 2.0
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            width,
            height,
            boxstyle="round,pad=0.018,rounding_size=0.06",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.45,
        )
    )
    return x0, y0, width, height


def add_arrow(ax: plt.Axes, left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> tuple[float, float]:
    y = left[1] + left[3] / 2.0
    start = (left[0] + left[2] + 0.10, y)
    end = (right[0] - 0.10, y)
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.35,
            color=ARROW,
        )
    )
    return (start[0] + end[0]) / 2.0, y


def add_c14_source(ax: plt.Axes, box: tuple[float, float, float, float]) -> None:
    x0, y0, w, h = box
    cx = x0 + w * 0.38
    cy = y0 + h * 0.55
    ax.add_patch(Circle((cx, cy), 0.25, facecolor="#F8FAFC", edgecolor=SOURCE_EDGE, linewidth=1.25))
    for start in (90, 210, 330):
        ax.add_patch(Wedge((cx, cy), 0.21, start - 28, start + 28, width=0.095, facecolor=SOURCE_EDGE, edgecolor="none"))
    ax.add_patch(Circle((cx, cy), 0.050, facecolor=SOURCE_EDGE, edgecolor="none"))
    ax.text(x0 + w * 0.65, cy + 0.09, "14C", ha="center", va="center", color=SOURCE_TEXT, fontsize=13.2, fontweight="black")
    ax.text(x0 + w * 0.65, cy - 0.17, r"$\mathbf{\beta^-}$", ha="center", va="center", color=SOURCE_TEXT, fontsize=11.2, fontweight="black")


def add_electron(ax: plt.Axes, x: float, y: float) -> None:
    ax.add_patch(Circle((x, y + 0.30), 0.135, facecolor=SOURCE_FACE, edgecolor=SOURCE_EDGE, linewidth=1.05))
    ax.text(x, y + 0.30, "e-", ha="center", va="center", color=SOURCE_TEXT, fontsize=8.8, fontweight="bold")


def add_geant4_logo(ax: plt.Axes, box: tuple[float, float, float, float]) -> None:
    x0, y0, w, h = box
    img = plt.imread(ASSET_DIR / "geant4_logo.png")
    margin_x = 0.16
    logo_h = 0.50
    logo_w = logo_h * img.shape[1] / img.shape[0]
    cx = x0 + w / 2.0
    cy = y0 + h / 2.0
    ax.imshow(
        img,
        extent=(cx - logo_w / 2.0, cx + logo_w / 2.0, cy - logo_h / 2.0, cy + logo_h / 2.0),
        zorder=4,
    )
    ax.text(cx, y0 + margin_x, "Monte Carlo", ha="center", va="center", color=PURPLE_EDGE, fontsize=8.2, fontweight="bold")


def add_grid(ax: plt.Axes, box: tuple[float, float, float, float]) -> None:
    x0, y0, w, h = box
    gx0 = x0 + w * 0.30
    gx1 = x0 + w * 0.70
    gy0 = y0 + h * 0.34
    gy1 = y0 + h * 0.74
    for frac in (0.0, 1 / 3, 2 / 3, 1.0):
        x = gx0 + frac * (gx1 - gx0)
        y = gy0 + frac * (gy1 - gy0)
        ax.plot([x, x], [gy0, gy1], color=GREEN_EDGE, linewidth=1.05, alpha=0.95)
        ax.plot([gx0, gx1], [y, y], color=GREEN_EDGE, linewidth=1.05, alpha=0.95)

    values = [["3", "5", "2"], ["6", "7", "4"], ["1", "8", "9"]]
    for row_index, row in enumerate(values):
        for col_index, value in enumerate(row):
            cx = gx0 + (col_index + 0.5) * (gx1 - gx0) / 3.0
            cy = gy1 - (row_index + 0.5) * (gy1 - gy0) / 3.0
            ax.text(cx, cy, value, ha="center", va="center", color=GREEN_EDGE, fontsize=7.4, fontweight="bold")
    ax.text(x0 + w / 2.0, y0 + h * 0.16, "mesh data", ha="center", va="center", color=MUTED, fontsize=8.2, fontweight="bold")


def add_sentaurus_logo(ax: plt.Axes, box: tuple[float, float, float, float]) -> None:
    x0, y0, w, h = box
    cx = x0 + w / 2.0
    img = plt.imread(ASSET_DIR / "synopsys_logo.png")
    logo_h = 0.24
    logo_w = logo_h * img.shape[1] / img.shape[0]
    ax.imshow(
        img,
        extent=(cx - logo_w / 2.0, cx + logo_w / 2.0, y0 + h * 0.58, y0 + h * 0.58 + logo_h),
        zorder=4,
    )
    ax.text(cx, y0 + h * 0.42, "Sentaurus", ha="center", va="center", color="#5A2A82", fontsize=10.0, fontweight="bold")
    ax.text(cx, y0 + h * 0.24, "TCAD", ha="center", va="center", color="#5A2A82", fontsize=9.5, fontweight="bold")


def add_it_curve(ax: plt.Axes, box: tuple[float, float, float, float]) -> None:
    x0, y0, w, h = box
    px0 = x0 + w * 0.18
    px1 = x0 + w * 0.82
    py0 = y0 + h * 0.24
    py1 = y0 + h * 0.78
    ax.plot([px0, px0, px1], [py1, py0, py0], color=TEXT, linewidth=1.45)
    t = np.linspace(0.0, 1.0, 100)
    pulse = (t / 0.20) ** 2 * np.exp(2.0 - t / 0.20)
    pulse /= pulse.max()
    ax.plot(px0 + t * (px1 - px0), py0 + pulse * (py1 - py0) * 0.88, color=TEXT, linewidth=1.65)
    ax.text(x0 + w / 2.0, y0 + h * 0.13, "i-t", ha="center", va="center", color=MUTED, fontsize=8.8, fontweight="bold")


def add_arrow_label(ax: plt.Axes, x: float, y: float, label: str) -> None:
    ax.text(x, y + 0.25, label, ha="center", va="center", color=TEXT, fontsize=10.2, fontweight="bold")


def main() -> int:
    fig, ax = plt.subplots(figsize=(10.8, 2.35), dpi=260)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    y = 1.18
    box_w = 1.48
    box_h = 1.05
    xs = [0.95, 3.02, 5.09, 7.16, 9.23]

    boxes = [
        add_box(ax, xs[0], y, box_w, box_h, SOURCE_EDGE, SOURCE_FACE),
        add_box(ax, xs[1], y, box_w, box_h, PURPLE_EDGE, PURPLE_FACE),
        add_box(ax, xs[2], y, box_w, box_h, GREEN_EDGE, GREEN_FACE),
        add_box(ax, xs[3], y, box_w, box_h, SENTAURUS_EDGE, SENTAURUS_FACE),
        add_box(ax, xs[4], y, box_w, box_h, OUTPUT_EDGE, OUTPUT_FACE),
    ]

    add_c14_source(ax, boxes[0])
    add_geant4_logo(ax, boxes[1])
    add_grid(ax, boxes[2])
    add_sentaurus_logo(ax, boxes[3])
    add_it_curve(ax, boxes[4])

    arrow_centers = [add_arrow(ax, boxes[i], boxes[i + 1]) for i in range(len(boxes) - 1)]
    add_electron(ax, *arrow_centers[0])
    add_arrow_label(ax, *arrow_centers[1], "Edep")
    add_arrow_label(ax, *arrow_centers[2], "G")
    add_arrow_label(ax, *arrow_centers[3], "i-t")

    ax.set_xlim(0.05, 10.15)
    ax.set_ylim(0.38, 2.12)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout(pad=0.08)
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUTPUT_SVG, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_SVG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
