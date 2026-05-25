#!/usr/bin/env python3
"""Plot Fig. 6: C-14 CCE versus i-region thickness at different Nt."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent
FIG_SET_DIR = THIS_DIR.parent
sys.path.insert(0, str(FIG_SET_DIR))

from journal_style import finalize_axes, save_figure, use_ieee_style
from tcad_it_tools import NT_LABEL, NT_ORDER, extract_metrics, write_csv

OUT_BASE = THIS_DIR / "fig6_c14_cce_vs_thickness_by_Nt"
OUT_CSV = THIS_DIR / "fig6_c14_cce_vs_thickness_by_Nt.csv"


def main() -> None:
    use_ieee_style(single_column=False)
    rows = [row for row in extract_metrics() if row["source"] == "c14"]
    write_csv(OUT_CSV, rows)

    colors = {
        "0": "#111827",
        "1e11": "#4C78A8",
        "1e12": "#59A14F",
        "1e13": "#F28E2B",
        "5e13": "#E15759",
    }
    markers = {"0": "o", "1e11": "s", "1e12": "^", "1e13": "D", "5e13": "v"}

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
            markersize=3.0,
            linewidth=1.1,
            color=colors[nt],
            label=NT_LABEL[nt],
        )

    ax.set_xlabel(r"$W_i$ ($\mu$m)")
    ax.set_ylabel("CCE (%)")
    ax.set_xlim(5, 135)
    ax.legend(title=r"$N_t$ (cm$^{-3}$)", loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1)
    ax.minorticks_on()
    finalize_axes(ax)
    fig.tight_layout()
    save_figure(fig, OUT_BASE)
    plt.close(fig)
    print(f"wrote {OUT_BASE}.png/.svg and {OUT_CSV}")


if __name__ == "__main__":
    main()
