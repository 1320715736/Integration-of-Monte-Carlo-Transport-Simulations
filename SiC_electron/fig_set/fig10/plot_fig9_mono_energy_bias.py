#!/usr/bin/env python3
"""Plot Fig. 10: 2-D design-bias matrix for mono-energetic designs."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
FIG_SET_DIR = THIS_DIR.parent
SRC_DIR = FIG_SET_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from journal_style import set_export_defaults
from tcad_it_tools import NT_ORDER, SOURCE_CONFIG, SOURCE_ORDER, best_by_source_nt, extract_metrics

OUT_BASE = THIS_DIR / "fig10_design_bias_matrix"
OUT_CSV = THIS_DIR / "fig10_design_bias_matrix.csv"


def write_cce_csv(rows: list[dict[str, object]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "nt",
        "design_source",
        "actual_source",
        "design_thickness_um",
        "actual_optimal_thickness_um",
        "actual_cce_at_design_percent",
        "actual_optimal_cce_percent",
        "cce_loss_percent_point",
    ]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    set_export_defaults(width=5.6, height=3.25, font_size=8.0)

    metrics = extract_metrics()
    best_rows = best_by_source_nt(metrics)
    actual_source = "c14"

    by_key = {
        (str(row["source"]), str(row["nt"]), float(row["thickness_um"])): row
        for row in metrics
    }
    best_by_key = {(str(row["source"]), str(row["nt"])): row for row in best_rows}

    design_sources = [source for source in SOURCE_ORDER if source != actual_source]
    rows: list[dict[str, object]] = []
    cce_grid = np.full((len(design_sources), len(NT_ORDER)), np.nan, dtype=float)
    loss_grid = np.full_like(cce_grid, np.nan)

    for nt_index, nt in enumerate(NT_ORDER):
        actual_best = best_by_key[(actual_source, nt)]
        for source_index, design_source in enumerate(design_sources):
            design_best = best_by_key[(design_source, nt)]
            design_thickness = float(design_best["thickness_um"])
            actual_at_design = by_key[(actual_source, nt, design_thickness)]
            actual_cce = float(actual_at_design["cce_percent"])
            optimal_cce = float(actual_best["cce_percent"])
            loss = optimal_cce - actual_cce
            cce_grid[source_index, nt_index] = actual_cce
            loss_grid[source_index, nt_index] = loss
            rows.append(
                {
                    "nt": nt,
                    "design_source": design_source,
                    "actual_source": actual_source,
                    "design_thickness_um": design_thickness,
                    "actual_optimal_thickness_um": float(actual_best["thickness_um"]),
                    "actual_cce_at_design_percent": actual_cce,
                    "actual_optimal_cce_percent": optimal_cce,
                    "cce_loss_percent_point": loss,
                }
            )

    write_cce_csv(rows)

    fig, ax = plt.subplots()
    base_cmap = mpl.colormaps["YlOrRd"]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "YlOrRd_soft_low",
        base_cmap(np.linspace(0.18, 1.0, 256)),
    )
    norm = mpl.colors.PowerNorm(gamma=1.15, vmin=0.0, vmax=max(45.0, float(np.nanmax(loss_grid))))
    image = ax.imshow(loss_grid, origin="lower", aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    nt_labels = ["0", r"$10^{12}$", r"$10^{13}$", r"$2.5\times10^{13}$", r"$5\times10^{13}$"]
    source_labels = [str(SOURCE_CONFIG[source]["label"]) for source in design_sources]
    ax.set_xticks(range(len(NT_ORDER)), nt_labels)
    ax.set_yticks(range(len(design_sources)), source_labels)
    ax.set_xlabel(r"$N_t$ (cm$^{-3}$)")
    ax.set_ylabel("Design source")

    ax.set_xticks(np.arange(-0.5, len(NT_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(design_sources), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row_index in range(loss_grid.shape[0]):
        for col_index in range(loss_grid.shape[1]):
            loss = loss_grid[row_index, col_index]
            text_color = "white" if loss >= 34.0 else "#111827"
            ax.text(
                col_index,
                row_index,
                f"{loss:.1f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9.0,
                fontweight="bold",
            )

    cbar = fig.colorbar(image, ax=ax, pad=0.018)
    cbar.set_label("CCE loss relative to C-14 optimum (pp)")
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(f"{OUT_BASE}.png")
    fig.savefig(f"{OUT_BASE}.svg")
    fig.savefig(f"{OUT_BASE}.pdf")
    plt.close(fig)
    print(f"wrote {OUT_BASE}.png/.svg/.pdf and {OUT_CSV}")


if __name__ == "__main__":
    main()
