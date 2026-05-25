"""Shared plotting style for SiC_electron figures.

Figure titles belong in manuscript captions, not inside the plot area.
Background grids are disabled by default for TNS-style figures.
SciencePlots is a required dependency:
    pip install SciencePlots
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  Required before plt.style.use().


def use_ieee_style(single_column: bool = True) -> None:
    """Apply the required SciencePlots IEEE style and export settings."""

    plt.style.use(["science", "ieee", "no-latex"])

    width = 3.5 if single_column else 7.16
    height = 2.55 if single_column else 4.6
    plt.rcParams.update(
        {
            "figure.figsize": (width, height),
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.title_fontsize": 7,
            "figure.titlesize": 8,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "axes.unicode_minus": False,
            "axes.grid": False,
            "axes.titlelocation": "center",
            "axes.labelpad": 2.0,
            "xtick.major.pad": 2.0,
            "ytick.major.pad": 2.0,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def finalize_axes(*axes) -> None:
    """Apply final TNS figure rules to axes.

    Use this after plotting and before saving. It removes in-plot titles and
    disables background grids, including for inset axes.
    """

    for ax in axes:
        ax.set_title("")
        ax.grid(False)


def save_figure(fig, output_base, *, png: bool = True, svg: bool = True, pdf: bool = False) -> None:
    """Save a figure as PNG and SVG by default."""

    output_base = str(output_base)
    if png:
        fig.savefig(f"{output_base}.png")
    if svg:
        fig.savefig(f"{output_base}.svg")
    if pdf:
        fig.savefig(f"{output_base}.pdf")
