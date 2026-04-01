#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize the Step 4 2D intermediate picture."
    )
    parser.add_argument(
        "--input",
        default=str(Path("step4_output") / "intermediate_picture_2d.npz"),
        help="Step 4 NPZ file.",
    )
    parser.add_argument(
        "--output-dir",
        default="step4_output",
        help="Directory for generated figures.",
    )
    return parser


def find_nonzero_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    nonzero = np.argwhere(grid > 0.0)
    if nonzero.size == 0:
        return 0, grid.shape[0] - 1, 0, grid.shape[1] - 1
    x0 = int(nonzero[:, 0].min())
    x1 = int(nonzero[:, 0].max())
    z0 = int(nonzero[:, 1].min())
    z1 = int(nonzero[:, 1].max())
    return x0, x1, z0, z1


def expand_bbox(
    x0: int, x1: int, z0: int, z1: int, shape: tuple[int, int], margin: int
) -> tuple[int, int, int, int]:
    return (
        max(0, x0 - margin),
        min(shape[0] - 1, x1 + margin),
        max(0, z0 - margin),
        min(shape[1] - 1, z1 + margin),
    )


def save_heatmap(
    output_path: Path,
    x_centers: np.ndarray,
    z_centers: np.ndarray,
    grid: np.ndarray,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required to save the heatmap PNG.")
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=180)
    image = ax.imshow(
        grid.T,
        origin="lower",
        aspect="auto",
        extent=[x_centers[0], x_centers[-1], z_centers[0], z_centers[-1]],
        cmap="turbo",
    )
    ax.set_xlabel("Lateral position x (um)")
    ax.set_ylabel("Depth (um)")
    ax.set_title("Step 4 Intermediate Picture")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("OpticalGeneration (cm^-3 s^-1)")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_mesh_preview(
    output_path: Path,
    x_centers: np.ndarray,
    z_centers: np.ndarray,
    grid: np.ndarray,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required to save the mesh preview PNG.")
    x0, x1, z0, z1 = find_nonzero_bbox(grid)
    x0, x1, z0, z1 = expand_bbox(x0, x1, z0, z1, grid.shape, margin=15)

    x_slice = slice(x0, x1 + 1, 8)
    z_slice = slice(z0, z1 + 1, 8)
    x_plot = x_centers[x_slice]
    z_plot = z_centers[z_slice]
    g_plot = grid[x_slice, z_slice]

    if x_plot.size < 2 or z_plot.size < 2:
        x_plot = x_centers[max(0, x0) : min(grid.shape[0], x0 + 20)]
        z_plot = z_centers[max(0, z0) : min(grid.shape[1], z0 + 20)]
        g_plot = grid[max(0, x0) : min(grid.shape[0], x0 + 20), max(0, z0) : min(grid.shape[1], z0 + 20)]

    dx = float(np.median(np.diff(x_plot))) if x_plot.size > 1 else 0.2
    dz = float(np.median(np.diff(z_plot))) if z_plot.size > 1 else 0.2
    x_edges = np.concatenate(([x_plot[0] - dx / 2], x_plot + dx / 2))
    z_edges = np.concatenate(([z_plot[0] - dz / 2], z_plot + dz / 2))
    xx, zz = np.meshgrid(x_edges, z_edges, indexing="ij")

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=180)
    mesh = ax.pcolormesh(
        xx,
        zz,
        g_plot,
        cmap="turbo",
        shading="flat",
        edgecolors=(0, 0, 0, 0.18),
        linewidth=0.15,
    )
    ax.set_xlabel("Lateral position x (um)")
    ax.set_ylabel("Depth (um)")
    ax.set_title("Step 4 Intermediate Picture Mesh Preview")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("OpticalGeneration (cm^-3 s^-1)")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_histogram(output_path: Path, grid: np.ndarray) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required to save the histogram PNG.")
    nonzero = np.asarray(grid[grid > 0.0], dtype=float)

    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=180)
    if nonzero.size == 0:
        ax.text(
            0.5,
            0.5,
            "No nonzero optical generation values found.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
    else:
        vmin = float(nonzero.min())
        vmax = float(nonzero.max())
        if np.isclose(vmin, vmax):
            bins = 20
        else:
            bins = np.logspace(np.log10(vmin), np.log10(vmax), 50)

        ax.hist(nonzero, bins=bins, color="#1f77b4", alpha=0.9, edgecolor="white", linewidth=0.4)
        if not np.isscalar(bins):
            ax.set_xscale("log")
        ax.set_xlabel("OpticalGeneration (cm^-3 s^-1)")
        ax.set_ylabel("Cell count")
        ax.set_title("Step 4 Nonzero Optical Generation Histogram")
        ax.grid(True, which="both", axis="x", alpha=0.25, linewidth=0.5)

        stats_text = "\n".join(
            [
                f"count = {nonzero.size}",
                f"min = {vmin:.3e}",
                f"median = {np.median(nonzero):.3e}",
                f"mean = {np.mean(nonzero):.3e}",
                f"max = {vmax:.3e}",
            ]
        )
        ax.text(
            0.98,
            0.98,
            stats_text,
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.88, "edgecolor": "#cccccc"},
        )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_histogram_svg(output_path: Path, grid: np.ndarray) -> None:
    nonzero = np.asarray(grid[grid > 0.0], dtype=float)

    width = 1100
    height = 700
    left = 90
    right = 40
    top = 50
    bottom = 90
    plot_w = width - left - right
    plot_h = height - top - bottom

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial, sans-serif" font-size="24">Step 4 Nonzero Optical Generation Histogram</text>',
    ]

    if nonzero.size == 0:
        parts.append(
            f'<text x="{width/2}" y="{height/2}" text-anchor="middle" font-family="Arial, sans-serif" font-size="22">No nonzero optical generation values found.</text>'
        )
        parts.append("</svg>")
        output_path.write_text("\n".join(parts), encoding="utf-8")
        return

    vmin = float(nonzero.min())
    vmax = float(nonzero.max())
    if math.isclose(vmin, vmax):
        bins = np.linspace(vmin * 0.95, vmax * 1.05 if vmax > 0.0 else 1.0, 21)
    else:
        bins = np.logspace(np.log10(vmin), np.log10(vmax), 50)

    counts, edges = np.histogram(nonzero, bins=bins)
    max_count = max(int(counts.max()), 1)

    parts.append(
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#333" stroke-width="1.2"/>'
    )

    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + plot_h - frac * plot_h
        value = int(round(max_count * frac))
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#d9d9d9" stroke-width="1"/>')
        parts.append(
            f'<text x="{left - 12}" y="{y + 5:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="14">{value}</text>'
        )

    log_min = float(np.log10(edges[0]))
    log_max = float(np.log10(edges[-1]))
    if math.isclose(log_min, log_max):
        log_max = log_min + 1.0

    tick_logs = np.arange(math.floor(log_min), math.ceil(log_max) + 1, 1)
    for tick_log in tick_logs:
        frac = (tick_log - log_min) / (log_max - log_min)
        x = left + frac * plot_w
        label = f"1e{tick_log}"
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#ececec" stroke-width="1"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{top + plot_h + 28}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">{label}</text>'
        )

    for idx, count in enumerate(counts):
        x0 = left + (np.log10(edges[idx]) - log_min) / (log_max - log_min) * plot_w
        x1 = left + (np.log10(edges[idx + 1]) - log_min) / (log_max - log_min) * plot_w
        bar_h = 0.0 if max_count == 0 else (count / max_count) * plot_h
        y = top + plot_h - bar_h
        parts.append(
            f'<rect x="{x0:.2f}" y="{y:.2f}" width="{max(x1 - x0 - 1.0, 0.8):.2f}" height="{bar_h:.2f}" fill="#1f77b4" fill-opacity="0.9"/>'
        )

    parts.append(
        f'<text x="{left + plot_w/2}" y="{height - 22}" text-anchor="middle" font-family="Arial, sans-serif" font-size="16">OpticalGeneration (cm^-3 s^-1, log scale)</text>'
    )
    parts.append(
        f'<text x="24" y="{top + plot_h/2}" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" transform="rotate(-90 24 {top + plot_h/2})">Cell count</text>'
    )

    stats_lines = [
        f"count = {nonzero.size}",
        f"min = {vmin:.3e}",
        f"median = {np.median(nonzero):.3e}",
        f"mean = {np.mean(nonzero):.3e}",
        f"max = {vmax:.3e}",
    ]
    box_x = left + plot_w - 220
    box_y = top + 18
    box_h = 24 + 22 * len(stats_lines)
    parts.append(
        f'<rect x="{box_x}" y="{box_y}" width="200" height="{box_h}" rx="8" ry="8" fill="white" fill-opacity="0.88" stroke="#cccccc"/>'
    )
    for i, line in enumerate(stats_lines):
        parts.append(
            f'<text x="{box_x + 12}" y="{box_y + 26 + i * 22}" font-family="Arial, sans-serif" font-size="15">{line}</text>'
        )

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    data = np.load(Path(args.input))
    x_centers = np.asarray(data["x_centers_um"], dtype=float)
    z_centers = np.asarray(data["z_centers_um"], dtype=float)
    grid = np.asarray(data["optical_generation_cm3_s"], dtype=float)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if plt is not None:
        save_heatmap(output_dir / "intermediate_picture_heatmap.png", x_centers, z_centers, grid)
        save_mesh_preview(output_dir / "intermediate_picture_mesh_preview.png", x_centers, z_centers, grid)
        save_histogram(output_dir / "intermediate_picture_histogram.png", grid)
        print(f"Saved PNG figures to {output_dir}")
    else:
        print("matplotlib is not installed; skipping PNG figure generation.")

    save_histogram_svg(output_dir / "intermediate_picture_histogram.svg", grid)
    print(f"Saved histogram SVG to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
