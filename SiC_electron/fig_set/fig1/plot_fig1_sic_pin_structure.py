#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon


OUTPUT_PNG = Path(__file__).resolve().parent / "fig1_sic_pin_structure.png"
OUTPUT_SVG = Path(__file__).resolve().parent / "fig1_sic_pin_structure.svg"

TEXT = "#111827"
DIMENSION = "#374151"
METAL = "#1F2937"
METAL_EDGE = "#111827"
N_COLOR = "#5B8CCB"
N_EDGE = "#1D4ED8"
N_TEXT = "#0F2F6B"
I_COLOR = "#F2D394"
I_EDGE = "#B7791F"
I_TEXT = "#6F4B0D"
P_COLOR = "#D97757"
P_EDGE = "#9A3412"
P_TEXT = "#7C2D12"
ELECTRON_FILL = "#2563EB"
ELECTRON_EDGE = "#1E3A8A"
ELECTRON_ARROW = "#2563EB"
ELECTRON_TEXT = "white"


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def shade(hex_color: str, factor: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    r = max(0, min(255, int(r * factor)))
    g = max(0, min(255, int(g * factor)))
    b = max(0, min(255, int(b * factor)))
    return f"#{r:02x}{g:02x}{b:02x}"


def add_layer(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    depth_x: float,
    depth_y: float,
    color: str,
    edge: str,
    label: str | None = None,
    label_color: str = TEXT,
    label_size: float = 10.0,
    label_weight: str = "bold",
) -> None:
    front = [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]
    right = [
        (x + width, y),
        (x + width + depth_x, y + depth_y),
        (x + width + depth_x, y + height + depth_y),
        (x + width, y + height),
    ]
    top = [
        (x, y + height),
        (x + width, y + height),
        (x + width + depth_x, y + height + depth_y),
        (x + depth_x, y + height + depth_y),
    ]
    ax.add_patch(Polygon(right, closed=True, facecolor=shade(color, 0.86), edgecolor=edge, linewidth=0.9))
    ax.add_patch(Polygon(front, closed=True, facecolor=color, edgecolor=edge, linewidth=0.9))
    ax.add_patch(Polygon(top, closed=True, facecolor=shade(color, 1.08), edgecolor=edge, linewidth=0.9))

    if label:
        ax.text(
            x + width / 2.0,
            y + height / 2.0,
            label,
            ha="center",
            va="center",
            color=label_color,
            fontsize=label_size,
            fontweight=label_weight,
            linespacing=1.22,
        )


def add_dimension(ax: plt.Axes, x: float, y0: float, y1: float, text: str) -> None:
    ax.plot([x, x], [y0, y1], color=DIMENSION, linewidth=1.25)
    ax.plot([x - 0.14, x + 0.14], [y0, y0], color=DIMENSION, linewidth=1.25)
    ax.plot([x - 0.14, x + 0.14], [y1, y1], color=DIMENSION, linewidth=1.25)
    ax.text(x + 0.22, (y0 + y1) / 2.0, text, ha="left", va="center", fontsize=10.5, color=TEXT)


def add_electron(ax: plt.Axes, x: float, y: float, radius: float = 0.23) -> None:
    ax.add_patch(Circle((x, y), radius=radius, facecolor=ELECTRON_FILL, edgecolor=ELECTRON_EDGE, linewidth=1.15))
    ax.text(x, y - 0.006, "e-", ha="center", va="center", fontsize=12.5, color=ELECTRON_TEXT, fontweight="bold")


def main() -> int:
    fig, ax = plt.subplots(figsize=(8.4, 5.8), dpi=260)

    x0 = 2.05
    y0 = 0.45
    width = 4.4
    dx = 1.15
    dy = 0.55

    # Visual heights are not to scale; numeric thicknesses are carried by labels.
    h_bottom_metal = 0.16
    h_n = 0.42
    h_i = 4.35
    h_p = 0.36
    h_top_metal = 0.16

    z = y0
    add_layer(ax, x=x0, y=z, width=width, height=h_bottom_metal, depth_x=dx, depth_y=dy, color=METAL, edge=METAL_EDGE)
    z += h_bottom_metal
    n_bottom = z
    add_layer(
        ax,
        x=x0,
        y=z,
        width=width,
        height=h_n,
        depth_x=dx,
        depth_y=dy,
        color=N_COLOR,
        edge=N_EDGE,
        label="n+  0.5 μm",
        label_color=N_TEXT,
        label_size=9.6,
    )
    z += h_n
    i_bottom = z
    add_layer(
        ax,
        x=x0,
        y=z,
        width=width,
        height=h_i,
        depth_x=dx,
        depth_y=dy,
        color=I_COLOR,
        edge=I_EDGE,
        label="4H-SiC i-region\n120 μm\nN_D = 5.6e12 cm^-3",
        label_color=I_TEXT,
        label_size=11.4,
    )
    z += h_i
    i_top = z
    add_layer(
        ax,
        x=x0,
        y=z,
        width=width,
        height=h_p,
        depth_x=dx,
        depth_y=dy,
        color=P_COLOR,
        edge=P_EDGE,
        label="p+  0.2 μm",
        label_color=P_TEXT,
        label_size=10.0,
    )
    z += h_p
    top_active = z
    add_layer(ax, x=x0, y=z, width=width, height=h_top_metal, depth_x=dx, depth_y=dy, color=METAL, edge=METAL_EDGE)
    top_total = z + h_top_metal

    # Incident electrons: three repeated vertical injection marks above the top electrode.
    electron_y = top_total + dy + 0.56
    target_y = top_total + dy + 0.03
    for frac in (0.28, 0.50, 0.72):
        electron_x = x0 + dx * 0.50 + width * frac
        electron_center = (electron_x, electron_y)
        electron_target = (electron_x, target_y)
        ax.plot(
            [electron_center[0], electron_target[0]],
            [electron_center[1] - 0.20, electron_target[1] + 0.03],
            color=ELECTRON_ARROW,
            linewidth=1.8,
            linestyle=(0, (3, 2)),
            alpha=0.9,
        )
        ax.add_patch(
            FancyArrowPatch(
                (electron_center[0], electron_center[1] - 0.18),
                electron_target,
                arrowstyle="-|>",
                mutation_scale=17,
                linewidth=1.8,
                color=ELECTRON_ARROW,
            )
        )
        add_electron(ax, *electron_center)

    # i-region thickness marker aligned to the right-side i-layer boundary.
    add_dimension(ax, x0 + width + dx + 0.25, i_bottom + dy, i_top + dy, "i = 120 μm")

    footprint_y = y0 - 0.20
    ax.plot([x0, x0 + width], [footprint_y, footprint_y], color=DIMENSION, linewidth=1.15)
    ax.plot([x0, x0], [footprint_y - 0.09, footprint_y + 0.09], color=DIMENSION, linewidth=1.15)
    ax.plot([x0 + width, x0 + width], [footprint_y - 0.09, footprint_y + 0.09], color=DIMENSION, linewidth=1.15)
    ax.text(x0 + width / 2.0, footprint_y - 0.16, "240 μm × 240 μm", ha="center", va="top", fontsize=9.5, color=TEXT)


    ax.set_xlim(0.0, 9.2)
    ax.set_ylim(-0.05, 7.55)
    ax.set_aspect("equal")
    ax.axis("off")

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(OUTPUT_SVG, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)

    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_SVG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
