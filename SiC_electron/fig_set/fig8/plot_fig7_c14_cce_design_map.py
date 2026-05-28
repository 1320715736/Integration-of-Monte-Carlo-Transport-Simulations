#!/usr/bin/env python3
"""Plot Fig. 7: discrete C-14 CCE map versus thickness and trap density."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from matplotlib.patches import Rectangle

THIS_DIR = Path(__file__).resolve().parent
FIG_SET_DIR = THIS_DIR.parent
sys.path.insert(0, str(FIG_SET_DIR))

from tcad_it_tools import NT_LABEL, NT_ORDER, THICKNESS_ORDER, extract_metrics, write_csv

OUT_BASE = THIS_DIR / "fig7_c14_cce_design_map"
OUT_CSV = THIS_DIR / "fig7_c14_cce_design_map.csv"
REPRESENTATIVE_THICKNESS = [30.0, 40.0, 60.0, 100.0, 150.0]


def main() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.labelsize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 8,
            "axes.grid": False,
            "axes.unicode_minus": False,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "svg.fonttype": "none",
        }
    )

    rows = [
        row
        for row in extract_metrics()
        if row["source"] == "c14"
        and bool(row["qc_pass"])
        and float(row["thickness_um"]) in REPRESENTATIVE_THICKNESS
    ]
    rows = sorted(rows, key=lambda row: (NT_ORDER.index(str(row["nt"])), float(row["thickness_um"])))
    write_csv(OUT_CSV, rows)

    cce_grid = np.full((len(NT_ORDER), len(REPRESENTATIVE_THICKNESS)), np.nan, dtype=float)
    by_key = {
        (str(row["nt"]), float(row["thickness_um"])): float(row["cce_percent"])
        for row in rows
    }
    for row_index, nt in enumerate(NT_ORDER):
        for col_index, thickness_um in enumerate(REPRESENTATIVE_THICKNESS):
            cce_grid[row_index, col_index] = by_key[(nt, thickness_um)]

    fig, ax = plt.subplots(figsize=(4.9, 3.0))
    image = ax.imshow(
        cce_grid,
        origin="lower",
        aspect="auto",
        cmap="YlOrRd",
        norm=PowerNorm(gamma=1.45, vmin=35.0, vmax=float(np.nanmax(cce_grid))),
        interpolation="nearest",
    )

    ax.set_xlabel(r"$W_i$ ($\mu$m)")
    ax.set_ylabel(r"$N_t$ (cm$^{-3}$)")
    ax.set_xticks(
        range(len(REPRESENTATIVE_THICKNESS)),
        [f"{thickness:g}" for thickness in REPRESENTATIVE_THICKNESS],
    )
    ax.set_yticks(range(len(NT_ORDER)), [NT_LABEL[nt] for nt in NT_ORDER])

    for row_index in range(cce_grid.shape[0]):
        best_col = int(np.nanargmax(cce_grid[row_index]))
        ax.add_patch(
            Rectangle(
                (best_col - 0.5, row_index - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor="#111827",
                linewidth=1.2,
            )
        )
        for col_index in range(cce_grid.shape[1]):
            value = cce_grid[row_index, col_index]
            text_color = "white" if value >= 86.0 else "#111827"
            ax.text(
                col_index,
                row_index,
                f"{value:.1f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=7,
            )

    ax.set_xticks(np.arange(-0.5, len(REPRESENTATIVE_THICKNESS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(NT_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(direction="out", length=3.0, width=0.8)
    ax.set_title("")

    cbar = fig.colorbar(image, ax=ax, pad=0.015)
    cbar.set_label("CCE (%)")

    fig.tight_layout()
    fig.savefig(f"{OUT_BASE}.png")
    fig.savefig(f"{OUT_BASE}.svg")
    plt.close(fig)
    print(f"wrote {OUT_BASE}.png/.svg and {OUT_CSV}")


if __name__ == "__main__":
    main()
