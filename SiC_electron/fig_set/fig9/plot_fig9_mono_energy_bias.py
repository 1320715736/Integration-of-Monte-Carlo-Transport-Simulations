#!/usr/bin/env python3
"""Plot Fig. 9: CCE loss caused by mono-energetic design approximations."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
FIG_SET_DIR = THIS_DIR.parent
sys.path.insert(0, str(FIG_SET_DIR))

from journal_style import finalize_axes, save_figure, use_ieee_style
from tcad_it_tools import NT_ORDER, SOURCE_CONFIG, SOURCE_ORDER, best_by_source_nt, extract_metrics

OUT_BASE = THIS_DIR / "fig9_design_bias_matrix"
OUT_CSV = THIS_DIR / "fig9_design_bias_matrix.csv"


def write_matrix_csv(rows: list[dict[str, object]]) -> None:
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
    use_ieee_style(single_column=False)
    metrics = extract_metrics()
    best_rows = best_by_source_nt(metrics)

    by_key = {
        (str(row["source"]), str(row["nt"]), float(row["thickness_um"])): row
        for row in metrics
    }
    best_by_key = {(str(row["source"]), str(row["nt"])): row for row in best_rows}

    rows: list[dict[str, object]] = []
    matrices: dict[str, np.ndarray] = {}
    for nt in NT_ORDER:
        matrix = np.zeros((len(SOURCE_ORDER), len(SOURCE_ORDER)), dtype=float)
        for i, design_source in enumerate(SOURCE_ORDER):
            design_best = best_by_key[(design_source, nt)]
            design_thickness = float(design_best["thickness_um"])
            for j, actual_source in enumerate(SOURCE_ORDER):
                actual_best = best_by_key[(actual_source, nt)]
                actual_at_design = by_key[(actual_source, nt, design_thickness)]
                loss = float(actual_best["cce_percent"]) - float(actual_at_design["cce_percent"])
                matrix[i, j] = loss
                rows.append(
                    {
                        "nt": nt,
                        "design_source": design_source,
                        "actual_source": actual_source,
                        "design_thickness_um": design_thickness,
                        "actual_optimal_thickness_um": float(actual_best["thickness_um"]),
                        "actual_cce_at_design_percent": float(actual_at_design["cce_percent"]),
                        "actual_optimal_cce_percent": float(actual_best["cce_percent"]),
                        "cce_loss_percent_point": loss,
                    }
                )
        matrices[nt] = matrix

    write_matrix_csv(rows)

    vmax = max(float(np.max(matrix)) for matrix in matrices.values())
    vmax = max(vmax, 0.1)
    source_labels = [str(SOURCE_CONFIG[source]["label"]).replace(" spectrum", "") for source in SOURCE_ORDER]
    nt_titles = ["0", r"$10^{11}$", r"$10^{12}$", r"$10^{13}$", r"$5\times10^{13}$"]

    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.8), constrained_layout=True)
    flat_axes = list(axes.ravel())
    image = None
    for ax, nt, title in zip(flat_axes, NT_ORDER, nt_titles):
        image = ax.imshow(matrices[nt], vmin=0.0, vmax=vmax, cmap="YlOrRd")
        ax.set_title(title, fontsize=8)
        ax.set_xticks(range(len(SOURCE_ORDER)), source_labels, rotation=45, ha="right")
        ax.set_yticks(range(len(SOURCE_ORDER)), source_labels)
        ax.set_xlabel("Actual source")
        ax.set_ylabel("Design source")
        for i in range(len(SOURCE_ORDER)):
            for j in range(len(SOURCE_ORDER)):
                value = matrices[nt][i, j]
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=6, color="#111827")
        ax.grid(False)

    cbar_ax = flat_axes[-1]
    cbar_ax.clear()
    if image is not None:
        cbar = fig.colorbar(image, cax=cbar_ax)
        cbar.set_label("CCE loss (percentage points)")
    for ax in flat_axes[:-1]:
        ax.grid(False)
    save_figure(fig, OUT_BASE)
    plt.close(fig)
    print(f"wrote {OUT_BASE}.png/.svg and {OUT_CSV}")


if __name__ == "__main__":
    main()
