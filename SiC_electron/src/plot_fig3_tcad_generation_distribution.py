#!/usr/bin/env python3
# Generate Fig. 3 from the C-14 optical-generation grid.
"""Plot the smoothed TCAD carrier-generation distribution for Fig. 3."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.ndimage import gaussian_filter
except ImportError:  # pragma: no cover - project environment includes scipy.
    gaussian_filter = None

from journal_style import set_export_defaults

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
DATA_PATH = ROOT_DIR / "output" / "c14" / "step4_output" / "intermediate_picture_2d.npz"
OUT_BASE = ROOT_DIR / "fig_set" / "fig3" / "fig3_tcad_generation_distribution"


def sentaurus_reference_cmap() -> mpl.colors.LinearSegmentedColormap:
    """Approximate the blue-to-red palette used by fig_set/fig2/image.png."""

    return mpl.colors.LinearSegmentedColormap.from_list(
        "sentaurus_reference",
        [
            (0.00, "#001EFF"),
            (0.10, "#0048FF"),
            (0.22, "#00B8FF"),
            (0.38, "#00F05A"),
            (0.56, "#D8FF00"),
            (0.76, "#FF9A00"),
            (1.00, "#E00000"),
        ],
    )


def crop_and_smooth(
    x_centers: np.ndarray,
    z_centers: np.ndarray,
    generation_grid: np.ndarray,
) -> tuple[np.ndarray, list[float]]:
    """Reuse the established crop and smoothing logic from the workflow plot."""

    clean_grid = generation_grid.copy()
    clean_grid[clean_grid < 0.005 * float(clean_grid.max())] = 0.0

    nonzero = np.argwhere(clean_grid > 0.0)
    xi0, xi1 = int(nonzero[:, 0].min()), int(nonzero[:, 0].max())
    zi0, zi1 = int(nonzero[:, 1].min()), int(nonzero[:, 1].max())

    margin_x = max(1, int(0.1 * (xi1 - xi0 + 1)))
    margin_z = max(2, int(0.05 * (zi1 - zi0 + 1)))
    xi0 = max(0, xi0 - margin_x)
    xi1 = min(clean_grid.shape[0] - 1, xi1 + margin_x)
    zi0 = max(0, zi0 - margin_z)
    zi1 = min(clean_grid.shape[1] - 1, zi1 + margin_z)

    grid_crop = clean_grid[xi0 : xi1 + 1, zi0 : zi1 + 1]
    x_crop = x_centers[xi0 : xi1 + 1]
    z_crop = z_centers[zi0 : zi1 + 1]

    if x_crop.size > 1:
        x_bin = float(x_centers[1] - x_centers[0])
        x_extreme = max(abs(float(x_crop[0])), abs(float(x_crop[-1])))
        left_pad = int(round((x_extreme - (-x_crop[0])) / x_bin)) if -x_crop[0] < x_extreme else 0
        right_pad = int(round((x_extreme - x_crop[-1]) / x_bin)) if x_crop[-1] < x_extreme else 0
        if left_pad > 0 or right_pad > 0:
            grid_crop = np.vstack(
                [
                    np.zeros((left_pad, grid_crop.shape[1])),
                    grid_crop,
                    np.zeros((right_pad, grid_crop.shape[1])),
                ]
            )
            x_crop = np.linspace(
                float(x_crop[0]) - left_pad * x_bin,
                float(x_crop[-1]) + right_pad * x_bin,
                grid_crop.shape[0],
            )

    if gaussian_filter is not None:
        grid_crop = gaussian_filter(grid_crop, sigma=(0.8, 1.2), mode="nearest")

    dx = float(x_centers[1] - x_centers[0]) / 2.0
    dz = float(z_centers[1] - z_centers[0]) / 2.0
    extent = [
        float(x_crop[0]) - dx,
        float(x_crop[-1]) + dx,
        float(z_crop[0]) - dz,
        float(z_crop[-1]) + dz,
    ]
    return grid_crop.T, extent


def main() -> int:
    set_export_defaults(width=5.6, height=3.55, font_size=8.6)

    data = np.load(DATA_PATH)
    x_centers = np.asarray(data["x_centers_um"], dtype=float)
    z_centers = np.asarray(data["z_centers_um"], dtype=float)
    generation_grid = np.asarray(data["optical_generation_cm3_s"], dtype=float)
    plot_grid, extent = crop_and_smooth(x_centers, z_centers, generation_grid)

    fig, ax = plt.subplots()
    image = ax.imshow(
        plot_grid,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=sentaurus_reference_cmap(),
        vmin=0.0,
        vmax=float(np.nanmax(plot_grid)),
        interpolation="bilinear",
    )

    ax.set_xlim(-40, 40)
    ax.set_ylim(0, 50)
    ax.set_xlabel(r"Lateral position x ($\mu$m)")
    ax.set_ylabel(r"Depth ($\mu$m)")
    ax.set_xticks(np.arange(-40, 41, 10))
    ax.set_yticks(np.arange(0, 51, 10))
    ax.tick_params(direction="out", length=3.0, width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    cbar = fig.colorbar(image, ax=ax, pad=0.018)
    cbar.set_label(r"Carrier generation rate (cm$^{-3}$ s$^{-1}$)")
    cbar.ax.tick_params(labelsize=7.5)

    OUT_BASE.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{OUT_BASE}.png")
    fig.savefig(f"{OUT_BASE}.svg")
    fig.savefig(f"{OUT_BASE}.pdf")
    plt.close(fig)
    print(f"wrote {OUT_BASE}.png/.svg/.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
