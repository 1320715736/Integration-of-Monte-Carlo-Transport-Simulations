#!/usr/bin/env python3
"""Plot Fig. 7: optimal i-region thickness versus trap density."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent
FIG_SET_DIR = THIS_DIR.parent
sys.path.insert(0, str(FIG_SET_DIR))

from journal_style import finalize_axes, save_figure, use_ieee_style
from tcad_it_tools import NT_ORDER, SOURCE_CONFIG, SOURCE_ORDER, best_by_source_nt, extract_metrics, write_csv

OUT_BASE = THIS_DIR / "fig7_optimal_thickness_vs_Nt"
OUT_CSV = THIS_DIR / "fig7_optimal_thickness_vs_Nt.csv"


def main() -> None:
    use_ieee_style(single_column=True)
    best_rows = best_by_source_nt(extract_metrics())
    write_csv(OUT_CSV, best_rows)

    x_pos = list(range(len(NT_ORDER)))
    x_labels = ["0", r"$10^{11}$", r"$10^{12}$", r"$10^{13}$", r"$5\times10^{13}$"]
    markers = ["o", "s", "^", "D", "v"]

    fig, ax = plt.subplots()
    for source, marker in zip(SOURCE_ORDER, markers):
        selected = [row for row in best_rows if row["source"] == source]
        y_values = [float(next(row for row in selected if row["nt"] == nt)["thickness_um"]) for nt in NT_ORDER]
        ax.plot(
            x_pos,
            y_values,
            marker=marker,
            markersize=3.2,
            linewidth=1.15,
            color=str(SOURCE_CONFIG[source]["color"]),
            label=str(SOURCE_CONFIG[source]["label"]),
        )

    ax.set_xlabel(r"$N_t$ (cm$^{-3}$)")
    ax.set_ylabel(r"Optimal $W_i$ ($\mu$m)")
    ax.set_xticks(x_pos, x_labels)
    ax.set_ylim(0, 140)
    ax.legend(loc="best")
    ax.minorticks_on()
    finalize_axes(ax)
    fig.tight_layout()
    save_figure(fig, OUT_BASE)
    plt.close(fig)
    print(f"wrote {OUT_BASE}.png/.svg and {OUT_CSV}")


if __name__ == "__main__":
    main()
