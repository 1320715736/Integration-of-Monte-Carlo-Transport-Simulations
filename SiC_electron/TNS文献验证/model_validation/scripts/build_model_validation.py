#!/usr/bin/env python3
"""Build model-validation tables for the Geant4-to-TCAD workflow.

Outputs are written under SiC_electron/文献验证/model_validation.
The script uses only existing project data and downloads the official NIST
ESTAR table for a user-defined SiC material.
"""

from __future__ import annotations

import csv
import html
import json
import math
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
VALIDATION_DIR = THIS_DIR.parent
SIC_DIR = VALIDATION_DIR.parents[1]
DATA_DIR = VALIDATION_DIR / "data"
FIGURE_DIR = VALIDATION_DIR / "figures"
REPORT_DIR = VALIDATION_DIR / "reports"

FIG_SET_DIR = SIC_DIR / "fig_set"
RAW_GEANT4_DIR = SIC_DIR / "raw_data" / "geant4_csv"
RAW_GEANT4_FULLDATA_DIR = RAW_GEANT4_DIR / "OutPut_fulldata"
sys.path.insert(0, str(FIG_SET_DIR))

from journal_style import finalize_axes, save_figure, use_ieee_style  # noqa: E402
from tcad_it_tools import extract_metrics  # noqa: E402


ELEMENTARY_CHARGE_C = 1.602176634e-19
SIC_DENSITY_G_CM3 = 3.21
ENERGY_BENCHMARKS = [
    ("20keV", "depth_profile_20keV_all.csv", "20 keV", 20.0),
    ("49keV", "depth_profile_49keV_all.csv", "49 keV", 49.0),
    ("100keV", "depth_profile_100keV_all.csv", "100 keV", 100.0),
    ("156p5keV", "depth_profile_156p5keV_all.csv", "156.5 keV", 156.5),
]
MAPPING_SOURCES = [
    ("20keV", "grid3d_20keV.csv", "20 keV", 20.0),
    ("49keV", "grid3d_49keV.csv", "49 keV", 49.0),
    ("100keV", "grid3d_100keV.csv", "100 keV", 100.0),
    ("156p5keV", "grid3d_156p5keV.csv", "156.5 keV", 156.5),
    ("c14", "grid3d_c14.csv", "C-14 spectrum", None),
]


def ensure_dirs() -> None:
    for path in (DATA_DIR, FIGURE_DIR, REPORT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_grid3d_csv(path: Path) -> tuple[np.ndarray, np.ndarray, int, float, float]:
    """Return depth and Edep arrays from the raw sparse Geant4 3D grid."""

    n_events: int | None = None
    xy_bin_mm: float | None = None
    z_bin_um: float | None = None
    depths: list[float] = []
    edep: list[float] = []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("# NEvents:"):
                n_events = int(line.split(":", 1)[1].strip())
                continue
            if line.startswith("# xy_bin_mm:"):
                tokens = line.split()
                xy_bin_mm = float(tokens[2])
                z_bin_um = float(tokens[4])
                continue
            if line.startswith("#") or line.startswith("x_mm"):
                continue
            parts = line.split(",")
            if len(parts) != 4:
                continue
            depths.append(float(parts[2]))
            edep.append(float(parts[3]))

    if n_events is None or xy_bin_mm is None or z_bin_um is None:
        raise ValueError(f"Missing grid3d metadata in {path}")
    if not depths:
        raise ValueError(f"No grid3d deposition rows found in {path}")
    return (
        np.asarray(depths, dtype=float),
        np.asarray(edep, dtype=float),
        n_events,
        xy_bin_mm,
        z_bin_um,
    )


def parse_depth_profile_csv(path: Path) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Return depth centers and per-bin Edep from a 1D Geant4 depth profile."""

    n_events: int | None = None
    bin_width_um: float | None = None
    depths: list[float] = []
    dedx_ev_per_um: list[float] = []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("# NEvents:"):
                n_events = int(line.split(":", 1)[1].strip())
                continue
            if line.startswith("# BinWidth_um:"):
                bin_width_um = float(line.split(":", 1)[1].strip())
                continue
            if line.startswith("#") or line.startswith("depth_um"):
                continue
            parts = line.split(",")
            if len(parts) != 2:
                continue
            depths.append(float(parts[0]))
            dedx_ev_per_um.append(float(parts[1]))

    if n_events is None or bin_width_um is None:
        raise ValueError(f"Missing depth-profile metadata in {path}")
    if not depths:
        raise ValueError(f"No depth-profile rows found in {path}")

    depth_um = np.asarray(depths, dtype=float)
    edep_ev_per_bin = np.asarray(dedx_ev_per_um, dtype=float) * float(bin_width_um)
    return depth_um, edep_ev_per_bin, n_events, float(bin_width_um)


def step4_paths(source_key: str, thickness_um: float = 120.0) -> tuple[Path, Path]:
    root = SIC_DIR / "generation" / f"{thickness_um:g}" / "output" / source_key / "step4_output"
    return root / "step4_summary.json", root / "intermediate_picture_2d.npz"


def fetch_nist_estar_sic_table() -> list[dict[str, float]]:
    """Download and parse the default NIST ESTAR table for SiC."""

    url = "https://physics.nist.gov/cgi-bin/Star/e_table-ut.pl"
    body = urllib.parse.urlencode(
        {
            "I": "150.1",
            "Energies": "",
            "ShowDefault": "on",
            "Name": "Silicon Carbide",
            "Density": f"{SIC_DENSITY_G_CM3:g}",
            "pairnum": "0",
            "line0": "SiC 1",
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"User-Agent": "SiC-electron-validation/1.0"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        raw_html = response.read().decode("utf-8", errors="replace")

    raw_path = DATA_DIR / "nist_estar_sic_raw.html"
    raw_path.write_text(raw_html, encoding="utf-8")

    text = raw_html.replace("<br>", "\n").replace("<BR>", "\n")
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)

    rows: list[dict[str, float]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not re.match(r"^\d+\.\d+E[+-]\d+", stripped):
            continue
        parts = stripped.split()
        if len(parts) < 7:
            continue
        energy_mev, collision, radiative, total, csda, rad_yield, density_effect = map(
            float, parts[:7]
        )
        rows.append(
            {
                "energy_MeV": energy_mev,
                "collision_stopping_power_MeV_cm2_g": collision,
                "radiative_stopping_power_MeV_cm2_g": radiative,
                "total_stopping_power_MeV_cm2_g": total,
                "csda_range_g_cm2": csda,
                "csda_range_um": csda / SIC_DENSITY_G_CM3 * 1.0e4,
                "radiation_yield": rad_yield,
                "density_effect": density_effect,
            }
        )
    if not rows:
        raise RuntimeError("Failed to parse NIST ESTAR table.")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def log_interp(x_target: float, x_values: np.ndarray, y_values: np.ndarray) -> float:
    return float(np.exp(np.interp(np.log(x_target), np.log(x_values), np.log(y_values))))


def depth_at_fraction(z_um: np.ndarray, profile: np.ndarray, fraction: float) -> float:
    total = float(profile.sum())
    if total <= 0.0:
        return math.nan
    cumulative = np.cumsum(profile) / total
    return float(np.interp(fraction, cumulative, z_um))


def build_depth_metrics(estar_rows: list[dict[str, float]]) -> list[dict[str, object]]:
    estar_energy = np.array([row["energy_MeV"] for row in estar_rows], dtype=float)
    estar_range_um = np.array([row["csda_range_um"] for row in estar_rows], dtype=float)

    rows: list[dict[str, object]] = []
    for source_key, depth_profile_name, label, energy_kev in ENERGY_BENCHMARKS:
        z_um, profile, n_events, z_bin_um = parse_depth_profile_csv(
            RAW_GEANT4_FULLDATA_DIR / source_key / depth_profile_name
        )
        nonzero = np.where(profile > 0.0)[0]

        edep_ev_per_primary = float(profile.sum())
        rows.append(
            {
                "source": source_key,
                "raw_depth_profile_file": str(Path(source_key) / depth_profile_name),
                "energy_keV": energy_kev,
                "estar_csda_range_um": log_interp(energy_kev / 1000.0, estar_energy, estar_range_um),
                "geant4_z50_um": depth_at_fraction(z_um, profile, 0.50),
                "geant4_z90_um": depth_at_fraction(z_um, profile, 0.90),
                "geant4_z99_um": depth_at_fraction(z_um, profile, 0.99),
                "geant4_nonzero_tail_um": float(z_um[nonzero[-1]]) if nonzero.size else math.nan,
                "edep_keV_per_primary": edep_ev_per_primary / 1000.0,
                "edep_over_ein": edep_ev_per_primary / (energy_kev * 1000.0),
                "n_events": int(n_events),
                "z_bin_um": float(z_bin_um),
            }
        )
    return rows


def build_raw3d_to_step4_conservation() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source_key, grid3d_name, label, _energy_kev in MAPPING_SOURCES:
        depth_values, edep_values, n_events, _xy_bin_mm, _z_bin_um = parse_grid3d_csv(
            RAW_GEANT4_DIR / grid3d_name
        )
        summary_path, _npz_path = step4_paths(source_key)
        summary = read_json(summary_path)
        raw_edep_total_ev = float(edep_values.sum())
        step4_edep_total_ev = float(summary["raw_edep_total_eV"])
        raw_nevents = float(n_events)
        step4_nevents = float(summary["n_events"])
        rel_error = (
            (step4_edep_total_ev - raw_edep_total_ev) / raw_edep_total_ev * 100.0
            if raw_edep_total_ev > 0.0
            else math.nan
        )
        rows.append(
            {
                "source": source_key,
                "source_label": label,
                "raw_grid3d_file": grid3d_name,
                "raw_grid3d_events": int(raw_nevents),
                "step4_events": int(step4_nevents),
                "raw_grid3d_edep_eV_total": raw_edep_total_ev,
                "step4_edep_eV_total": step4_edep_total_ev,
                "relative_error_percent": rel_error,
                "raw_nonzero_cells": int(depth_values.size),
            }
        )
    return rows


def build_mapping_conservation() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source_key, _grid3d_name, label, energy_kev in MAPPING_SOURCES:
        summary_path, _npz_path = step4_paths(source_key)
        summary = read_json(summary_path)
        n_events = float(summary["n_events"])
        eeh_ev = float(summary["eh_pair_energy_ev"])
        voxel_volume_cm3 = float(summary["voxel_volume_cm3"])
        integration_time_s = float(summary["integration_time_s"])

        neh_geant4 = float(summary["raw_edep_total_eV"]) / n_events / eeh_ev
        neh_generation_grid = (
            float(summary["optical_generation_sum"]) * voxel_volume_cm3 * integration_time_s
        )
        rel_error = (
            (neh_generation_grid - neh_geant4) / neh_geant4 * 100.0
            if neh_geant4 > 0.0
            else math.nan
        )
        rows.append(
            {
                "source": source_key,
                "source_label": label,
                "energy_keV": "" if energy_kev is None else energy_kev,
                "Neh_Geant4_pairs_per_primary": neh_geant4,
                "Neh_generation_grid_pairs_per_primary": neh_generation_grid,
                "relative_error_percent": rel_error,
                "raw_edep_eV_per_primary": float(summary["raw_edep_total_eV"]) / n_events,
                "voxel_volume_cm3": voxel_volume_cm3,
            }
        )
    return rows


def build_cce_limit_check() -> list[dict[str, object]]:
    metrics = extract_metrics()
    rows: list[dict[str, object]] = [
        {
            "test_case": "Ideal uniform generation, no trap",
            "thickness_um": "",
            "bias_V": "",
            "trap_density_cm3": 0.0,
            "expected_CCE_percent": "~100",
            "TCAD_CCE_percent": "",
            "status": "needs dedicated control simulation",
            "interpretation": "Use this to confirm the transient-current integration limit.",
        }
    ]

    selected_cases = [
        ("Geant4 mapped C-14, no trap", "0", "~97.8"),
        ("Geant4 mapped C-14, Nt=1e13", "1e13", "lower than no-trap case"),
    ]
    for test_case, nt_label, expected in selected_cases:
        candidates = [
            row
            for row in metrics
            if row["source"] == "c14"
            and abs(float(row["thickness_um"]) - 120.0) < 1.0e-9
            and row["nt"] == nt_label
        ]
        if not candidates:
            rows.append(
                {
                    "test_case": test_case,
                    "thickness_um": 120.0,
                    "bias_V": 75.0,
                    "trap_density_cm3": "",
                    "expected_CCE_percent": expected,
                    "TCAD_CCE_percent": "",
                    "status": "missing",
                    "interpretation": "No matching row found in raw_data/tcad_it.",
                }
            )
            continue
        row = candidates[0]
        rows.append(
            {
                "test_case": test_case,
                "thickness_um": float(row["thickness_um"]),
                "bias_V": float(row["bias_v"]),
                "trap_density_cm3": float(row["trap_density_cm3"]),
                "expected_CCE_percent": expected,
                "TCAD_CCE_percent": float(row["cce_percent"]),
                "status": "available",
                "interpretation": "Existing C-14 transient response at 120 um full-depletion bias.",
            }
        )
    return rows


def plot_depth_benchmark(rows: list[dict[str, object]]) -> None:
    use_ieee_style(single_column=True)
    fig, ax = plt.subplots()
    energy = np.array([float(row["energy_keV"]) for row in rows])
    order = np.argsort(energy)
    energy = energy[order]
    estar = np.array([float(row["estar_csda_range_um"]) for row in rows])[order]
    z50 = np.array([float(row["geant4_z50_um"]) for row in rows])[order]
    z90 = np.array([float(row["geant4_z90_um"]) for row in rows])[order]

    ax.plot(energy, estar, color="#222222", marker="o", linewidth=1.1, markersize=3.0, label="ESTAR CSDA")
    ax.plot(energy, z90, color="#C73E1D", marker="s", linewidth=1.1, markersize=3.0, label="Geant4 $z_{90}$")
    ax.plot(energy, z50, color="#F2A541", marker="^", linewidth=1.1, markersize=3.0, label="Geant4 $z_{50}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Electron energy (keV)")
    ax.set_ylabel(r"Depth or range ($\mu$m)")
    ax.legend(loc="upper left")
    ax.minorticks_on()
    finalize_axes(ax)
    save_figure(fig, FIGURE_DIR / "estar_geant4_depth_benchmark")
    plt.close(fig)


def fmt(value: object, digits: int = 3) -> str:
    if value == "":
        return ""
    if isinstance(value, str):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return ""
    if abs(numeric) >= 1.0e4 or (0.0 < abs(numeric) < 1.0e-3):
        return f"{numeric:.{digits}e}"
    return f"{numeric:.{digits}f}"


def md_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(header for header, _key in columns) + " |",
        "| " + " | ".join("---" for _header, _key in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(key, "")) for _header, key in columns) + " |")
    return "\n".join(lines)


def write_report(
    depth_rows: list[dict[str, object]],
    raw3d_rows: list[dict[str, object]],
    mapping_rows: list[dict[str, object]],
    cce_rows: list[dict[str, object]],
) -> None:
    depth_columns = [
        ("Energy", "energy_keV"),
        ("ESTAR R_CSDA (um)", "estar_csda_range_um"),
        ("Geant4 z50 (um)", "geant4_z50_um"),
        ("Geant4 z90 (um)", "geant4_z90_um"),
        ("Edep/Ein", "edep_over_ein"),
    ]
    mapping_columns = [
        ("Source", "source_label"),
        ("Neh Geant4", "Neh_Geant4_pairs_per_primary"),
        ("Neh generation grid", "Neh_generation_grid_pairs_per_primary"),
        ("Rel. error (%)", "relative_error_percent"),
    ]
    raw3d_columns = [
        ("Source", "source_label"),
        ("Raw file", "raw_grid3d_file"),
        ("Raw events", "raw_grid3d_events"),
        ("Step4 events", "step4_events"),
        ("Raw Edep total (eV)", "raw_grid3d_edep_eV_total"),
        ("Step4 Edep total (eV)", "step4_edep_eV_total"),
        ("Rel. error (%)", "relative_error_percent"),
    ]
    cce_columns = [
        ("Test case", "test_case"),
        ("Wi (um)", "thickness_um"),
        ("Bias (V)", "bias_V"),
        ("Nt (cm^-3)", "trap_density_cm3"),
        ("Expected CCE (%)", "expected_CCE_percent"),
        ("TCAD CCE (%)", "TCAD_CCE_percent"),
        ("Status", "status"),
    ]

    max_mapping_error = max(abs(float(row["relative_error_percent"])) for row in mapping_rows)
    max_raw3d_error = max(abs(float(row["relative_error_percent"])) for row in raw3d_rows)
    report = f"""# Model Validation Results

This folder implements the first two validation items in `markdown/验证方法.md`.

## 1. Geant4-ESTAR Low-Energy Electron Range Benchmark

NIST ESTAR was queried as a user-defined silicon carbide material with density {SIC_DENSITY_G_CM3:g} g/cm3 and formula `SiC 1`. ESTAR reports CSDA range as mass thickness; it was converted to projected length by dividing by density. The 49 keV and 156.5 keV entries are log-log interpolated from the default ESTAR energy grid because NIST only reports range for default energies in this text-table mode.

The Geant4 depths below are extracted directly from the one-dimensional depth-deposition profiles under `raw_data/geant4_csv/OutPut_fulldata/*/depth_profile_*_all.csv`, not from the Step-4 2D generation map.

{md_table(depth_rows, depth_columns)}

Interpretation: `z50` and `z90` are deposited-energy cumulative depths, whereas ESTAR `R_CSDA` is a continuous-slowing-down path-length range. They should agree in order and monotonic trend, not point-by-point equality.

Figure: `../figures/estar_geant4_depth_benchmark.png`

## 2. Raw Geant4 3D Grid to Step-4 Conservation

This check verifies that the original sparse 3D Geant4 deposition files are carried into Step 4 without changing total deposited energy or event count.

{md_table(raw3d_rows, raw3d_columns)}

Maximum absolute relative error: {max_raw3d_error:.3e}%.

## 3. Step-4 Energy-to-Generation Conservation

The check compares `raw_edep_total_eV / (N_events * E_eh)` with `sum(G) * V_voxel * T_int` from Step 4. This verifies the normalization used before writing `OpticalGeneration` into the TCAD input.

{md_table(mapping_rows, mapping_columns)}

Maximum absolute relative error: {max_mapping_error:.3e}%.

## 4. CCE Limit Check

The existing C-14 data support the mapped-source no-trap and trapped cases at the 120 um full-depletion baseline. The ideal uniform-generation control case is not present in the current `raw_data/tcad_it` set and should be run separately if we want to close this validation item exactly as written in `验证方法.md`.

{md_table(cce_rows, cce_columns)}

## Files

- `../data/nist_estar_sic_table.csv`
- `../data/estar_geant4_depth_metrics.csv`
- `../data/raw3d_to_step4_conservation_table.csv`
- `../data/mapping_conservation_table.csv`
- `../data/cce_limit_check.csv`
- `../figures/estar_geant4_depth_benchmark.png`
- `../figures/estar_geant4_depth_benchmark.svg`
- `moscatelli_benchmark_next_steps.md`
"""
    (REPORT_DIR / "model_validation_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    estar_rows = fetch_nist_estar_sic_table()
    write_csv(DATA_DIR / "nist_estar_sic_table.csv", estar_rows)

    depth_rows = build_depth_metrics(estar_rows)
    raw3d_rows = build_raw3d_to_step4_conservation()
    mapping_rows = build_mapping_conservation()
    cce_rows = build_cce_limit_check()

    write_csv(DATA_DIR / "estar_geant4_depth_metrics.csv", depth_rows)
    write_csv(DATA_DIR / "raw3d_to_step4_conservation_table.csv", raw3d_rows)
    write_csv(DATA_DIR / "mapping_conservation_table.csv", mapping_rows)
    write_csv(DATA_DIR / "cce_limit_check.csv", cce_rows)
    plot_depth_benchmark(depth_rows)
    write_report(depth_rows, raw3d_rows, mapping_rows, cce_rows)

    print(f"Wrote validation outputs to {VALIDATION_DIR}")


if __name__ == "__main__":
    main()
