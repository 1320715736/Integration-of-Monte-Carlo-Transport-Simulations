#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.transforms import Affine2D


plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


FIG_DIR = Path(__file__).resolve().parent
OUTPUT_PNG = FIG_DIR / "fig2_geant4_tcad_workflow.png"
OUTPUT_SVG = FIG_DIR / "fig2_geant4_tcad_workflow.svg"
OUTPUT_PDF = FIG_DIR / "fig2_geant4_tcad_workflow.pdf"

TEXT = "#20252D"
MUTED = "#4B5563"
INK = "#141820"
ARROW = "#111827"
ACCENT_RED = "#E63946"
ACCENT_BLUE = "#2D6CDF"
ACCENT_GREEN = "#159A74"
ACCENT_AMBER = "#C7772C"

PANEL_COLORS = {
    "source": "#F7D1B7",
    "transport": "#F7F3CF",
    "device": "#DCEED7",
    "signal": "#D8E3F3",
}


def add_panel(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    color: str,
    title: str,
    title_offset: float = 0.34,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.30",
            facecolor=color,
            edgecolor="#FFFFFF",
            linewidth=1.2,
            zorder=0,
        )
    )
    ax.text(
        x + title_offset,
        y + h - 0.32,
        title,
        ha="left",
        va="center",
        color=INK,
        fontsize=14.0,
        fontweight="bold",
        zorder=2,
    )


def add_text(ax: plt.Axes, x: float, y: float, text: str, size: float = 10.85) -> None:
    ax.text(x, y, text, ha="center", va="top", color=TEXT, fontsize=size, fontweight="bold", linespacing=1.82, zorder=5)


def add_step_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.30,
            color=ARROW,
            shrinkA=2,
            shrinkB=3,
            zorder=3,
        )
    )


def source_icon(ax: plt.Axes, x: float, y: float) -> None:
    ax.add_patch(Circle((x - 0.06, y + 0.05), 0.31, facecolor="#FFFFFF", edgecolor=INK, linewidth=1.55, zorder=3))
    for start in (90, 210, 330):
        ax.add_patch(
            Polygon(
                [
                    (x - 0.06, y + 0.05),
                    (x - 0.06 + 0.25 * np.cos(np.deg2rad(start - 24)), y + 0.05 + 0.25 * np.sin(np.deg2rad(start - 24))),
                    (x - 0.06 + 0.25 * np.cos(np.deg2rad(start + 24)), y + 0.05 + 0.25 * np.sin(np.deg2rad(start + 24))),
                ],
                closed=True,
                facecolor=ACCENT_AMBER,
                edgecolor="none",
                zorder=4,
            )
        )
    ax.add_patch(Circle((x - 0.06, y + 0.05), 0.055, facecolor=INK, edgecolor="none", zorder=5))
    ax.text(x + 0.49, y + 0.08, r"$^{\mathbf{14}}\mathbf{C}$", ha="center", va="center", fontsize=12.4, fontweight="bold", color=INK, zorder=4)


def spectrum_icon(ax: plt.Axes, x: float, y: float) -> None:
    ax.plot([x - 0.43, x - 0.43, x + 0.47], [y - 0.22, y + 0.45, y + 0.45], color=INK, linewidth=1.5, zorder=3)
    xs = np.linspace(-0.34, 0.40, 60)
    curve = 0.18 + 0.42 * (1.0 - ((xs - 0.03) / 0.43) ** 2)
    curve = np.clip(curve, 0.15, None)
    ax.plot(x + xs, y - 0.22 + curve, color=ACCENT_RED, linewidth=1.8, zorder=4)
    for px, py in [(-0.17, 0.03), (0.08, 0.20), (0.28, 0.13)]:
        ax.add_patch(Circle((x + px, y + py), 0.05, facecolor=ACCENT_BLUE, edgecolor="white", linewidth=0.55, zorder=5))


def source_setup_icon(ax: plt.Axes, x: float, y: float) -> None:
    ax.text(x - 0.36, y - 0.06, "GPS", ha="center", va="center", fontsize=10.0, fontweight="bold", color=INK, zorder=4)
    ax.add_patch(Circle((x - 0.02, y + 0.12), 0.045, facecolor=ACCENT_RED, edgecolor="none", zorder=5))
    ax.add_patch(Circle((x - 0.02, y - 0.07), 0.045, facecolor=ACCENT_RED, edgecolor="none", zorder=5))
    ax.add_patch(Circle((x - 0.02, y - 0.26), 0.045, facecolor=ACCENT_RED, edgecolor="none", zorder=5))
    for yoff in (0.12, -0.07, -0.26):
        ax.add_patch(
            FancyArrowPatch(
                (x + 0.06, y + yoff),
                (x + 0.42, y - 0.07),
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.25,
                color=ACCENT_BLUE,
                zorder=5,
            )
        )


def track_icon(ax: plt.Axes, x: float, y: float) -> None:
    t = np.linspace(0.0, 1.0, 100)
    ax.plot(x - 0.45 + 0.90 * t, y + 0.28 * np.sin(2.6 * np.pi * t), color=INK, linewidth=1.6, zorder=3)
    rng = np.random.default_rng(14)
    for _ in range(16):
        ax.add_patch(
            Circle(
                (x - 0.36 + 0.72 * rng.random(), y - 0.30 + 0.50 * rng.random()),
                0.026,
                facecolor=ACCENT_BLUE,
                edgecolor="none",
                alpha=0.8,
                zorder=4,
            )
        )


def deposit_icon(ax: plt.Axes, x: float, y: float) -> None:
    ax.plot([x - 0.44, x + 0.44], [y - 0.16, y - 0.16], color=INK, linewidth=1.5, zorder=3)
    heights = [0.24, 0.50, 0.32, 0.42]
    for index, height in enumerate(heights):
        bx = x - 0.34 + index * 0.22
        ax.add_patch(Rectangle((bx, y - 0.16), 0.13, height, facecolor=ACCENT_AMBER, edgecolor=INK, linewidth=1.0, zorder=4))
    ax.text(x + 0.02, y + 0.55, r"$\mathbf{E}_{\mathbf{dep}}$", ha="center", va="center", fontsize=9.2, color=INK, fontweight="bold", zorder=5)


def map_icon(ax: plt.Axes, x: float, y: float) -> None:
    x0 = x - 0.285
    y0 = y - 0.30
    cell = 0.19
    values = np.array([[0.2, 0.5, 0.3], [0.4, 0.9, 0.6], [0.1, 0.3, 0.8]])
    for row in range(3):
        for col in range(3):
            shade = 0.25 + 0.65 * values[row, col]
            face = (1.0 - 0.42 * shade, 1.0 - 0.15 * shade, 1.0 - 0.70 * shade)
            ax.add_patch(
                Rectangle((x0 + col * cell, y0 + row * cell), cell, cell, facecolor=face, edgecolor=INK, linewidth=0.7, zorder=4)
            )
    ax.text(x, y + 0.45, "CSV", ha="center", va="center", color=INK, fontsize=9.0, fontweight="bold", zorder=5)


def mesh_icon(ax: plt.Axes, x: float, y: float) -> None:
    for dx in (-0.14, 0.14):
        ax.plot([x + dx, x + dx], [y - 0.32, y + 0.32], color=INK, linewidth=1.25, zorder=4)
    for dy in (-0.11, 0.11):
        ax.plot([x - 0.42, x + 0.42], [y + dy, y + dy], color=INK, linewidth=1.25, zorder=4)


def trap_icon(ax: plt.Axes, x: float, y: float) -> None:
    ax.add_patch(Arc((x, y + 0.04), 0.82, 0.54, angle=0, theta1=205, theta2=335, color=INK, linewidth=1.6, zorder=3))
    ax.add_patch(Arc((x, y + 0.04), 0.82, 0.54, angle=0, theta1=25, theta2=155, color=INK, linewidth=1.6, zorder=3))
    for px, py in [(-0.22, 0.12), (0.00, -0.06), (0.23, 0.11)]:
        ax.add_patch(Circle((x + px, y + py), 0.055, facecolor=ACCENT_RED, edgecolor="white", linewidth=0.45, zorder=5))
    ax.text(x, y + 0.48, r"$\mathbf{N}_{\mathbf{t}}$", ha="center", va="center", fontsize=9.5, color=INK, fontweight="bold", zorder=5)


def solver_icon(ax: plt.Axes, x: float, y: float) -> None:
    ax.add_patch(Rectangle((x - 0.35, y - 0.24), 0.70, 0.48, facecolor="#FFFFFF", edgecolor=INK, linewidth=1.35, zorder=3))
    ax.plot([x - 0.20, x + 0.20], [y + 0.11, y + 0.11], color=INK, linewidth=1.1, zorder=4)
    ax.plot([x - 0.20, x + 0.20], [y - 0.01, y - 0.01], color=INK, linewidth=1.1, zorder=4)
    ax.plot([x - 0.20, x + 0.08], [y - 0.13, y - 0.13], color=INK, linewidth=1.1, zorder=4)


def current_icon(ax: plt.Axes, x: float, y: float) -> None:
    ax.plot([x - 0.44, x - 0.44, x + 0.44], [y + 0.34, y - 0.27, y - 0.27], color=INK, linewidth=1.35, zorder=3)
    t = np.linspace(0.0, 1.0, 100)
    pulse = (t / 0.20) ** 2 * np.exp(2.0 - t / 0.20)
    pulse /= pulse.max()
    ax.plot(x - 0.38 + 0.76 * t, y - 0.27 + 0.52 * pulse, color=ACCENT_BLUE, linewidth=1.85, zorder=4)


def charge_icon(ax: plt.Axes, x: float, y: float) -> None:
    ax.plot([x - 0.43, x - 0.43, x + 0.43], [y + 0.30, y - 0.27, y - 0.27], color=INK, linewidth=1.35, zorder=3)
    t = np.linspace(0.0, 1.0, 100)
    pulse = (t / 0.24) ** 2 * np.exp(2.0 - t / 0.24)
    pulse /= pulse.max()
    xs = x - 0.36 + 0.74 * t
    ys = y - 0.27 + 0.45 * pulse
    ax.fill_between(xs, ys, y - 0.27, color=ACCENT_GREEN, alpha=0.30, zorder=2)
    ax.plot(xs, ys, color=ACCENT_BLUE, linewidth=1.65, zorder=4)
    ax.text(x, y + 0.48, r"$\mathbf{Q}_{\mathbf{col}}=\int \mathbf{i}(\mathbf{t})\,\mathbf{d}\mathbf{t}$", ha="center", va="center", fontsize=9.4, color=INK, fontweight="bold", zorder=5)


def cce_icon(ax: plt.Axes, x: float, y: float) -> None:
    ax.add_patch(Rectangle((x - 0.40, y - 0.28), 0.80, 0.56, facecolor="#FFFFFF", edgecolor=INK, linewidth=1.3, zorder=3))
    block_size = 0.16
    block_step = 0.22
    block_group_width = block_size + 2 * block_step
    block_group_height = block_size + block_step
    x_start = x - block_group_width / 2.0
    y_start = y - block_group_height / 2.0
    for row in range(2):
        for col in range(3):
            color = ["#2D6CDF", "#159A74", "#F2A65A", "#E63946", "#7A5CCB", "#4A5568"][row * 3 + col]
            ax.add_patch(Rectangle((x_start + col * block_step, y_start + row * block_step), block_size, block_size, facecolor=color, edgecolor="none", zorder=4))


def add_center_loop(ax: plt.Axes, cx: float, cy: float) -> None:
    ax.add_patch(Circle((cx, cy), 0.66, facecolor="#FFFFFF", edgecolor="#D7DCE4", linewidth=1.15, zorder=8))
    ax.text(cx, cy + 0.12, "MC-TCAD", ha="center", va="center", color=TEXT, fontsize=13.0, fontweight="bold", zorder=11)
    ax.text(cx, cy - 0.18, "coupling", ha="center", va="center", color=TEXT, fontsize=12.8, fontweight="bold", zorder=11)
    specs = [
        (140, 40, "#F1B986", -0.30),
        (35, -55, "#F2EEB5", -0.28),
        (-45, -135, "#BCDDAE", -0.30),
        (-140, 130, "#BFD0EA", -0.28),
    ]
    radius = 0.95
    for theta1, theta2, color, rad in specs:
        p1 = (cx + radius * np.cos(np.deg2rad(theta1)), cy + radius * np.sin(np.deg2rad(theta1)))
        p2 = (cx + radius * np.cos(np.deg2rad(theta2)), cy + radius * np.sin(np.deg2rad(theta2)))
        ax.add_patch(
            FancyArrowPatch(
                p1,
                p2,
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                mutation_scale=17,
                linewidth=4.4,
                color=color,
                alpha=0.95,
                zorder=9,
            )
        )


def add_step_group(
    ax: plt.Axes,
    centers: list[tuple[float, float]],
    labels: list[str],
    icon_funcs: list,
    label_y: float,
    icon_scale: float = 1.18,
) -> None:
    for index, (center, label, icon_func) in enumerate(zip(centers, labels, icon_funcs)):
        start_counts = (len(ax.patches), len(ax.lines), len(ax.collections), len(ax.texts))
        icon_func(ax, center[0], center[1])
        scale_transform = (
            Affine2D()
            .translate(-center[0], -center[1])
            .scale(icon_scale)
            .translate(center[0], center[1])
            + ax.transData
        )
        for artist in ax.patches[start_counts[0] :]:
            artist.set_transform(scale_transform)
        for artist in ax.lines[start_counts[1] :]:
            artist.set_transform(scale_transform)
        for artist in ax.collections[start_counts[2] :]:
            artist.set_transform(scale_transform)
        for artist in ax.texts[start_counts[3] :]:
            artist.set_transform(scale_transform)
            artist.set_fontsize(artist.get_fontsize() * icon_scale)
        add_text(ax, center[0], label_y, label)
        if index < len(centers) - 1:
            midpoint_x = 0.5 * (center[0] + centers[index + 1][0])
            if icon_func is deposit_icon and icon_funcs[index + 1] is map_icon:
                midpoint_x += 0.10
            arrow_length = 0.30
            add_step_arrow(
                ax,
                (midpoint_x - arrow_length / 2.0, center[1] + 0.02),
                (midpoint_x + arrow_length / 2.0, centers[index + 1][1] + 0.02),
            )


def main() -> int:
    fig, ax = plt.subplots(figsize=(10.8, 6.0), dpi=260)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    add_panel(ax, 0.50, 3.55, 5.18, 2.42, PANEL_COLORS["source"], "(a) Source setup")
    add_panel(ax, 6.28, 3.55, 5.18, 2.42, PANEL_COLORS["transport"], "(b) Geant4 scoring")
    add_panel(ax, 6.28, 0.50, 5.18, 2.42, PANEL_COLORS["device"], "(c) Device calculation", title_offset=0.50)
    add_panel(ax, 0.50, 0.50, 5.18, 2.42, PANEL_COLORS["signal"], "(d) Signal evaluation")

    add_step_group(
        ax,
        [(1.34, 4.62), (3.14, 4.62), (4.68, 4.62)],
        [r"$^{\mathbf{14}}\mathbf{C}$ / mono" "\n" "electron", "Energy\nsampling", "Tagged GPS\nbeam"],
        [source_icon, spectrum_icon, source_setup_icon],
        4.03,
    )
    add_step_group(
        ax,
        [(7.24, 4.62), (8.78, 4.62), (10.12, 4.62)],
        ["Transport\nin SiC", r"Step-wise" "\n" r"$\mathbf{E}_{\mathbf{dep}}$", r"1D + 3D" "\n" r"$\mathbf{E}_{\mathbf{dep}}$ CSV"],
        [track_icon, deposit_icon, map_icon],
        4.03,
    )
    add_step_group(
        ax,
        [(7.24, 1.57), (8.78, 1.57), (10.26, 1.57)],
        ["Device\nmesh", "Trap\nmodel", "Transient\nsolution"],
        [mesh_icon, trap_icon, solver_icon],
        0.98,
    )
    add_step_group(
        ax,
        [(1.46, 1.57), (3.00, 1.57), (4.50, 1.57)],
        ["i-t\ncurve", "Collected\ncharge", "CCE\nmap"],
        [current_icon, charge_icon, cce_icon],
        0.98,
    )

    add_center_loop(ax, 5.98, 3.23)

    ax.set_xlim(0.15, 11.82)
    ax.set_ylim(0.12, 6.28)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout(pad=0.08)
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, bbox_inches="tight", pad_inches=0.05, dpi=600)
    fig.savefig(OUTPUT_SVG, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_SVG}")
    print(f"Saved: {OUTPUT_PDF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
