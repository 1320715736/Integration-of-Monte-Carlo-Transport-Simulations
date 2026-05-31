#!/usr/bin/env python3
"""Plot Fig. 9: optimal i-region thickness versus trap density."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent
FIG_SET_DIR = THIS_DIR.parent
SRC_DIR = FIG_SET_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from journal_style import finalize_axes, save_figure, set_export_defaults
from tcad_it_tools import NT_ORDER, SOURCE_CONFIG, SOURCE_ORDER, best_by_source_nt, extract_metrics, write_csv

OUT_BASE = THIS_DIR / "fig9_optimal_thickness_vs_Nt"
OUT_CSV = THIS_DIR / "fig9_optimal_thickness_vs_Nt.csv"


def main() -> None:
    set_export_defaults(width=3.5, height=2.55, font_size=8.0)
    best_rows = best_by_source_nt(extract_metrics())
    write_csv(OUT_CSV, best_rows)

    x_pos = list(range(len(NT_ORDER)))
    x_labels = ["0", r"$10^{12}$", r"$10^{13}$", r"$2.5\times10^{13}$", r"$5\times10^{13}$"]
    markers = ["o", "s", "^", "D", "v"]

    fig, ax = plt.subplots()
    for source, marker in zip(SOURCE_ORDER, markers):
        selected = [row for row in best_rows if row["source"] == source]
        y_values = [float(next(row for row in selected if row["nt"] == nt)["thickness_um"]) for nt in NT_ORDER]
        ax.plot(
            x_pos,
            y_values,
            marker=marker,
            markersize=5.6,
            markerfacecolor="white" if source != "c14" else "#111827",
            markeredgewidth=1.0,
            linewidth=1.45,
            color=str(SOURCE_CONFIG[source]["color"]),
            label=str(SOURCE_CONFIG[source]["label"]),
        )

    ax.set_xlabel(r"$N_t$ (cm$^{-3}$)")
    ax.set_ylabel(r"Optimal $W_i$ ($\mu$m)")
    ax.set_xticks(x_pos, x_labels)
    ax.set_ylim(0, 190)
    ax.legend(loc="best", frameon=False)
    ax.minorticks_on()
    ax.tick_params(direction="in", top=True, right=True)
    finalize_axes(ax)
    fig.tight_layout()
    save_figure(fig, OUT_BASE)
    plt.close(fig)
    print(f"wrote {OUT_BASE}.png/.svg/.pdf and {OUT_CSV}")


if __name__ == "__main__":
    main()
