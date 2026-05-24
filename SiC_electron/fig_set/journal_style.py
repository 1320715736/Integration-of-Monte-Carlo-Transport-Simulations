"""Shared IEEE plotting style for SiC_electron figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  Required before plt.style.use().


def use_ieee_style(single_column: bool = True) -> None:
    """Apply SciencePlots IEEE style and export settings."""

    plt.style.use(["science", "ieee", "no-latex"])

    width = 3.5 if single_column else 7.16
    height = 2.55 if single_column else 4.6
    plt.rcParams.update(
        {
            "figure.figsize": (width, height),
            "font.family": "Arial",
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig, output_base, *, png: bool = True, svg: bool = True, pdf: bool = True) -> None:
    """Save a figure as PNG, SVG, and PDF."""

    output_base = str(output_base)
    if png:
        fig.savefig(f"{output_base}.png")
    if svg:
        fig.savefig(f"{output_base}.svg")
    if pdf:
        fig.savefig(f"{output_base}.pdf")
