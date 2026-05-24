#!/usr/bin/env python3
"""Plot Fig. 5: 120 um baseline 1/C^2-V characteristic with C-V inset."""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt


THIS_DIR = Path(__file__).resolve().parent
FIG_SET_DIR = THIS_DIR.parent
sys.path.insert(0, str(FIG_SET_DIR))

from journal_style import save_figure, use_ieee_style

INPUT_CSV = THIS_DIR / "tcad_cv.csv"
OUT_CSV = THIS_DIR / "fig5_cv_processed.csv"
OUT_BASE = THIS_DIR / "fig5_cv_1overc2_baseline"

BASELINE_THICKNESS_UM = 120.0
EPS0_F_PER_CM = 8.8541878128e-14
EPSR_4H_SIC = 9.7
Q = 1.602176634e-19
ND_CM3 = 5.6e12


def depletion_voltage(wi_um: float) -> float:
    wi_cm = wi_um * 1e-4
    eps_sic = EPSR_4H_SIC * EPS0_F_PER_CM
    return Q * ND_CM3 * wi_cm * wi_cm / (2.0 * eps_sic)


def parse_thickness(header: str) -> float:
    match = re.search(r"\((\d+(?:\.\d+)?)_Nt", header)
    if not match:
        raise ValueError(f"Cannot parse thickness from column header: {header}")
    return float(match.group(1))


def read_curves(path: Path) -> dict[float, list[tuple[float, float]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)

    curves: dict[float, list[tuple[float, float]]] = {}
    for col in range(0, len(header), 2):
        if col + 1 >= len(header):
            continue
        thickness_um = parse_thickness(header[col])
        points = []
        for row in rows:
            if col + 1 >= len(row) or not row[col] or not row[col + 1]:
                continue
            points.append((float(row[col]), float(row[col + 1])))
        if points:
            curves[thickness_um] = points

    if BASELINE_THICKNESS_UM not in curves:
        available = ", ".join(f"{item:g}" for item in sorted(curves))
        raise RuntimeError(f"{BASELINE_THICKNESS_UM:g} um curve not found. Available: {available}")
    return curves


def write_processed(thickness_um: float, points: list[tuple[float, float]]) -> None:
    vdep = depletion_voltage(thickness_um)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["thickness_um", "reverse_bias_v", "capacitance_f", "one_over_c2_f_minus2", "analytical_vdep_v"])
        for voltage, capacitance in points:
            writer.writerow(
                [
                    f"{thickness_um:.12g}",
                    f"{voltage:.12g}",
                    f"{capacitance:.12e}",
                    f"{1.0 / (capacitance * capacitance):.12e}",
                    f"{vdep:.12g}",
                ]
            )


def plot(thickness_um: float, points: list[tuple[float, float]]) -> None:
    use_ieee_style(single_column=False)

    voltage = [v for v, _ in points]
    capacitance = [c for _, c in points]
    one_over_c2 = [1.0 / (c * c) for _, c in points]
    vdep = depletion_voltage(thickness_um)

    fig, ax = plt.subplots()
    ax.plot(voltage, one_over_c2, color="#1f5aa6", linewidth=1.6)
    ax.axvline(vdep, color="#c0392b", linestyle="--", linewidth=1.0)
    ax.text(
        vdep + 6,
        min(one_over_c2) + 0.15 * (max(one_over_c2) - min(one_over_c2)),
        f"Vdep = {vdep:.1f} V\nWi = {thickness_um:.0f} um",
        color="#922b21",
        fontsize=10,
        va="bottom",
    )
    ax.set_xlabel("Reverse bias (V)")
    ax.set_ylabel(r"$1/C^2$ (F$^{-2}$)")
    ax.set_title(r"Baseline 4H-SiC PIN $1/C^2$-V characteristic ($N_t=10^{13}$ cm$^{-3}$)")
    ax.minorticks_on()
    ax.grid(True, color="#d7dce2", linewidth=0.5)
    ax.set_axisbelow(True)

    inset = ax.inset_axes([0.56, 0.53, 0.38, 0.36])
    inset.plot(voltage, [c * 1e12 for c in capacitance], color="#2f855a", linewidth=1.0)
    inset.axvline(vdep, color="#c0392b", linestyle="--", linewidth=0.8)
    inset.set_xlabel("V", fontsize=8)
    inset.set_ylabel("C (pF)", fontsize=8)
    inset.tick_params(labelsize=8)
    inset.minorticks_on()
    inset.grid(True, color="#e5e7eb", linewidth=0.45)

    fig.tight_layout()
    save_figure(fig, OUT_BASE)
    plt.close(fig)


def main() -> None:
    curves = read_curves(INPUT_CSV)
    points = curves[BASELINE_THICKNESS_UM]
    write_processed(BASELINE_THICKNESS_UM, points)
    plot(BASELINE_THICKNESS_UM, points)

    caps = [c for _, c in points]
    print(f"baseline thickness: {BASELINE_THICKNESS_UM:g} um")
    print(f"points: {len(points)}")
    print(f"Vdep: {depletion_voltage(BASELINE_THICKNESS_UM):.3f} V")
    print(f"C range: {min(caps):.3e}..{max(caps):.3e} F")
    print(f"wrote: {OUT_BASE}.png/.svg/.pdf")


if __name__ == "__main__":
    main()
