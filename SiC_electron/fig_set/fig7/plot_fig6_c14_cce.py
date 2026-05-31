#!/usr/bin/env python3
"""Plot Fig. 7: C-14 CCE versus i-region thickness at different Nt."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent
FIG_SET_DIR = THIS_DIR.parent
SRC_DIR = FIG_SET_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from journal_style import finalize_axes, save_figure, set_export_defaults
from tcad_it_tools import NT_LABEL, NT_ORDER, extract_metrics, write_csv

OUT_BASE = THIS_DIR / "fig7_c14_cce_vs_thickness_by_Nt"
OUT_CSV = THIS_DIR / "fig7_c14_cce_vs_thickness_by_Nt.csv"


def main() -> None:
    set_export_defaults(width=5.6, height=3.25, font_size=8.0)
    rows = [row for row in extract_metrics() if row["source"] == "c14"]
    write_csv(OUT_CSV, rows)

    colors = {
        "0": "#000000",
        "1e12": "#0072B2",
        "1e13": "#D55E00",
        "2.5e13": "#009E73",
        "5e13": "#CC79A7",
    }
    markers = {"0": "o", "1e12": "s", "1e13": "^", "2.5e13": "D", "5e13": "v"}

    fig, ax = plt.subplots()
    for nt in NT_ORDER:
        selected = sorted(
            [row for row in rows if row["nt"] == nt and bool(row["qc_pass"])],
            key=lambda row: float(row["thickness_um"]),
        )
        ax.plot(
            [float(row["thickness_um"]) for row in selected],
            [float(row["cce_percent"]) for row in selected],
            marker=markers[nt],
            markersize=7.4,
            markerfacecolor="white" if nt != "0" else "#000000",
            markeredgewidth=1.25,
            linewidth=1.7,
            color=colors[nt],
            label=NT_LABEL[nt],
        )

    ax.set_xlabel(r"$W_i$ ($\mu$m)")
    ax.set_ylabel("CCE (%)")
    ax.set_xlim(0, 185)
    ax.legend(title=r"$N_t$ (cm$^{-3}$)", loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1, frameon=False)
    ax.minorticks_on()
    ax.tick_params(direction="in", top=True, right=True)
    finalize_axes(ax)
    fig.tight_layout()
    save_figure(fig, OUT_BASE)
    plt.close(fig)
    print(f"wrote {OUT_BASE}.png/.svg/.pdf and {OUT_CSV}")


if __name__ == "__main__":
    main()
