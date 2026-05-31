# Shared neutral plotting/export helpers moved from fig_set to src.
"""Small plotting utilities shared by generated figure scripts.

This module intentionally does not define a journal-wide visual preset. Each
figure script chooses its own colors, line styles, axes, and layout according
to the information it needs to show.
"""

from __future__ import annotations

import matplotlib.pyplot as plt


def set_export_defaults(*, width: float, height: float, font_size: float = 8.0) -> None:
    """Reset Matplotlib and set only neutral export parameters."""

    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (width, height),
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "xtick.labelsize": max(font_size - 1.0, 6.0),
            "ytick.labelsize": max(font_size - 1.0, 6.0),
            "legend.fontsize": max(font_size - 1.0, 6.0),
            "legend.title_fontsize": max(font_size - 1.0, 6.0),
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "axes.unicode_minus": False,
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
    """Remove plot titles; captions carry figure titles in the manuscript."""

    for ax in axes:
        ax.set_title("")


def save_figure(fig, output_base, *, png: bool = True, svg: bool = True, pdf: bool = True) -> None:
    """Save a figure as PNG, SVG, and PDF by default."""

    output_base = str(output_base)
    if png:
        fig.savefig(f"{output_base}.png")
    if svg:
        fig.savefig(f"{output_base}.svg")
    if pdf:
        fig.savefig(f"{output_base}.pdf")
