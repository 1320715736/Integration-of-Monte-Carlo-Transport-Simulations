#!/usr/bin/env python3
"""Plot Fig. 9: 3-D CCE scatter for C-14 under mono-energetic designs."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  Registers 3-D projection.

THIS_DIR = Path(__file__).resolve().parent
FIG_SET_DIR = THIS_DIR.parent
sys.path.insert(0, str(FIG_SET_DIR))

from tcad_it_tools import NT_ORDER, SOURCE_CONFIG, SOURCE_ORDER, best_by_source_nt, extract_metrics

OUT_BASE = THIS_DIR / "fig9_design_bias_matrix"
OUT_CSV = THIS_DIR / "fig9_design_bias_matrix.csv"


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
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.unicode_minus": False,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "svg.fonttype": "none",
        }
    )

    metrics = extract_metrics()
    best_rows = best_by_source_nt(metrics)
    actual_source = "c14"

    by_key = {
        (str(row["source"]), str(row["nt"]), float(row["thickness_um"])): row
        for row in metrics
    }
    best_by_key = {(str(row["source"]), str(row["nt"])): row for row in best_rows}

    rows: list[dict[str, object]] = []
    cce_grid = np.zeros((len(SOURCE_ORDER), len(NT_ORDER)), dtype=float)
    for nt_index, nt in enumerate(NT_ORDER):
        actual_best = best_by_key[(actual_source, nt)]
        for source_index, design_source in enumerate(SOURCE_ORDER):
            design_best = best_by_key[(design_source, nt)]
            design_thickness = float(design_best["thickness_um"])
            actual_at_design = by_key[(actual_source, nt, design_thickness)]
            actual_cce = float(actual_at_design["cce_percent"])
            optimal_cce = float(actual_best["cce_percent"])
            loss = optimal_cce - actual_cce
            cce_grid[source_index, nt_index] = actual_cce
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

    x_pos, y_pos = np.meshgrid(np.arange(len(NT_ORDER)), np.arange(len(SOURCE_ORDER)))
    x_flat = x_pos.ravel()
    y_flat = y_pos.ravel()
    z_flat = cce_grid.ravel()

    norm = mpl.colors.PowerNorm(gamma=1.2, vmin=35.0, vmax=100.0)
    cmap = mpl.colormaps["YlOrRd"]
    z_floor = 35.0

    fig = plt.figure(figsize=(5.7, 4.25))
    ax = fig.add_subplot(111, projection="3d")

    for x, y, z in zip(x_flat, y_flat, z_flat):
        ax.plot(
            [x, x],
            [y, y],
            [z_floor, z],
            color="#9CA3AF",
            linewidth=0.35,
            alpha=0.45,
            zorder=1,
        )

    ax.scatter(
        x_flat,
        y_flat,
        np.full_like(z_flat, z_floor),
        c=z_flat,
        cmap=cmap,
        norm=norm,
        s=12,
        marker="s",
        alpha=0.22,
        linewidths=0,
        depthshade=False,
        zorder=2,
    )
    scatter = ax.scatter(
        x_flat,
        y_flat,
        z_flat,
        c=z_flat,
        cmap=cmap,
        norm=norm,
        s=30,
        marker="o",
        edgecolors="#374151",
        linewidths=0.25,
        alpha=0.95,
        depthshade=False,
        zorder=3,
    )

    nt_labels = ["0", r"$10^{12}$", r"$10^{13}$", r"$2.5\times10^{13}$", r"$5\times10^{13}$"]
    source_labels = [str(SOURCE_CONFIG[source]["label"]).replace(" spectrum", "") for source in SOURCE_ORDER]
    ax.set_xticks(range(len(NT_ORDER)), nt_labels, rotation=18, ha="right")
    ax.set_yticks(range(len(SOURCE_ORDER)), source_labels)
    ax.set_xlabel(r"$N_t$ (cm$^{-3}$)", labelpad=7)
    ax.set_ylabel("Design source", labelpad=8)
    ax.set_zlabel("CCE (%)", labelpad=6)
    ax.set_zlim(z_floor, 100.0)
    ax.view_init(elev=22, azim=-48)
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1.45, 1.05, 0.82))
    ax.set_title("")
    ax.tick_params(axis="both", which="major", pad=1)
    ax.tick_params(axis="z", which="major", pad=2)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor("#E5E7EB")
        axis._axinfo["grid"]["color"] = (0.82, 0.82, 0.82, 0.65)
        axis._axinfo["grid"]["linewidth"] = 0.45

    cbar = fig.colorbar(scatter, ax=ax, pad=0.08, shrink=0.62, fraction=0.035, aspect=24)
    cbar.ax.tick_params(labelsize=7, length=2.2, width=0.6)
    cbar.outline.set_linewidth(0.6)

    fig.subplots_adjust(left=0.0, right=0.9, bottom=0.02, top=0.98)
    fig.savefig(f"{OUT_BASE}.png")
    fig.savefig(f"{OUT_BASE}.svg")
    plt.close(fig)
    print(f"wrote {OUT_BASE}.png/.svg and {OUT_CSV}")


if __name__ == "__main__":
    main()
