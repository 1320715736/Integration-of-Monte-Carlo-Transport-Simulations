#!/usr/bin/env python3
"""Step 4 (grid3d variant): convert a pre-binned sparse 3D energy-deposition
CSV (produced by the Geant4 analysis macro) into a 2D optical-generation grid
compatible with the Step-5 / Step-6 pipeline.

Input CSV format
----------------
Comment lines carry metadata:
    # NEvents: 100000
    # xy_bin_mm: 0.01  z_bin_um: 0.2
Data columns (bin centres, sparse – non-zero voxels only):
    x_mm, y_mm, depth_um, edep_eV
    (edep_eV = total energy deposited in that voxel over **all** N events)

Output (identical schema to mc_to_tcad.py / Step 4)
-----------------------------------------------------
    <output-dir>/intermediate_picture_2d.npz
    <output-dir>/intermediate_picture_nonzero.csv
    <output-dir>/intermediate_picture_full_grid.csv
    <output-dir>/step4_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

# 4H-SiC average energy required to create one electron-hole pair.
# Value: 7.8 eV (commonly used value for 4H-SiC, consistent with project design spec).
EH_PAIR_ENERGY_EV: float = 7.8


# ─── CSV parsing ──────────────────────────────────────────────────────────────

def parse_grid3d_csv(
    csv_path: Path,
) -> tuple[list[tuple[float, float, float, float]], int, float, float]:
    """Read a sparse pre-binned 3D grid CSV.

    Returns
    -------
    rows       : list of (x_mm, y_mm, depth_um, edep_eV)
    n_events   : number of simulated primary particles
    xy_bin_mm  : lateral bin size in mm
    z_bin_um   : depth bin size in µm
    """
    n_events: int | None = None
    xy_bin_mm: float | None = None
    z_bin_um: float | None = None
    rows: list[tuple[float, float, float, float]] = []

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("# NEvents:"):
                n_events = int(line.split(":", 1)[1].strip())
            elif line.startswith("# xy_bin_mm:"):
                # Expected format: "# xy_bin_mm: 0.01  z_bin_um: 0.2"
                # tokens: ['#', 'xy_bin_mm:', '0.01', 'z_bin_um:', '0.2']
                tokens = line.split()
                xy_bin_mm = float(tokens[2])
                z_bin_um = float(tokens[4])
            elif line.startswith("#") or line.startswith("x_mm"):
                continue
            else:
                parts = line.split(",")
                if len(parts) == 4:
                    try:
                        rows.append(
                            (
                                float(parts[0]),
                                float(parts[1]),
                                float(parts[2]),
                                float(parts[3]),
                            )
                        )
                    except ValueError:
                        continue

    if n_events is None or xy_bin_mm is None or z_bin_um is None:
        raise ValueError(
            f"Could not parse required header fields (NEvents, xy_bin_mm, z_bin_um) "
            f"from {csv_path}."
        )
    if not rows:
        raise ValueError(f"No data rows found in {csv_path}.")

    return rows, n_events, xy_bin_mm, z_bin_um


# ─── 2-D grid builder ─────────────────────────────────────────────────────────

def build_2d_grid(
    rows: list[tuple[float, float, float, float]],
    n_events: int,
    xy_bin_mm: float,
    z_bin_um: float,
    *,
    integration_time_s: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Collapse the y dimension and compute optical generation G [cm⁻³ s⁻¹].

    The 3-D voxel has lateral dimensions xy_bin × xy_bin and depth z_bin.
    Collapsing y means each 2-D cell at (x, z) accumulates energy from all y
    bins; its effective volume is xy_bin × z_bin × (total_y_span).

    G(x, z) = ΣEdep(x,z) [eV] / (N_events × E_eh [eV] × T_int [s] × V [cm³])

    Parameters
    ----------
    rows            : [(x_mm, y_mm, depth_um, edep_eV), ...]
    n_events        : number of simulated primary particles
    xy_bin_mm       : lateral bin width [mm]
    z_bin_um        : depth bin width [µm]
    integration_time_s : normalisation time [s]; default 1.0 s

    Returns
    -------
    x_centers_um    : 1-D array, lateral bin centres [µm]
    z_centers_um    : 1-D array, depth bin centres [µm]
    G_2d            : 2-D array, shape (n_x, n_z), units cm⁻³ s⁻¹
    summary         : dict with key metadata
    """
    xy_bin_um = xy_bin_mm * 1000.0

    # Unique bin centres (sorted)
    x_vals = sorted(set(r[0] for r in rows))
    y_vals = sorted(set(r[1] for r in rows))
    z_vals = sorted(set(r[2] for r in rows))

    x_centers_um = np.array(x_vals, dtype=float) * 1000.0  # mm → µm
    z_centers_um = np.array(z_vals, dtype=float)            # already µm

    n_x = len(x_vals)
    n_z = len(z_vals)

    # Full y span: bin edges extend half-bin beyond the outermost bin centres
    y_min_mm = float(min(y_vals)) - 0.5 * xy_bin_mm
    y_max_mm = float(max(y_vals)) + 0.5 * xy_bin_mm
    collapsed_thickness_um = (y_max_mm - y_min_mm) * 1000.0

    # 2-D voxel volume [cm³]: Δx × Δz × collapsed_y_thickness
    voxel_volume_um3 = xy_bin_um * z_bin_um * collapsed_thickness_um
    voxel_volume_cm3 = voxel_volume_um3 * 1.0e-12

    # Pre-compute conversion factor
    # G = edep_total_eV / (N_events × E_eh × T_int × V_cm3)
    optical_generation_factor = 1.0 / (
        float(n_events) * integration_time_s * EH_PAIR_ENERGY_EV * voxel_volume_cm3
    )

    # Collapse y: accumulate edep_eV for each (x, z)
    x_idx = {x: i for i, x in enumerate(x_vals)}
    z_idx = {z: i for i, z in enumerate(z_vals)}
    edep_2d = np.zeros((n_x, n_z), dtype=float)
    for x_mm, _y_mm, depth_um, edep_ev in rows:
        edep_2d[x_idx[x_mm], z_idx[depth_um]] += edep_ev

    G_2d = edep_2d * optical_generation_factor

    summary: dict = {
        "material": "4H-SiC",
        "eh_pair_energy_ev": EH_PAIR_ENERGY_EV,
        "eh_pair_energy_source": (
            "4H-SiC design specification (E_eh = 7.8 eV)"
        ),
        "n_events": n_events,
        "xy_bin_mm": xy_bin_mm,
        "xy_bin_um": xy_bin_um,
        "z_bin_um": z_bin_um,
        "x_min_um": float(x_centers_um.min()),
        "x_max_um": float(x_centers_um.max()),
        "lateral_size_um": float(x_centers_um.max() - x_centers_um.min() + xy_bin_um),
        "x_bins": n_x,
        "z_min_um": float(z_centers_um.min()),
        "z_max_um": float(z_centers_um.max()),
        "depth_um": float(z_centers_um.max() - z_centers_um.min() + z_bin_um),
        "z_bins": n_z,
        "collapsed_axis": "y",
        "collapsed_y_min_mm": float(y_min_mm),
        "collapsed_y_max_mm": float(y_max_mm),
        "collapsed_thickness_um": float(collapsed_thickness_um),
        "voxel_volume_um3": float(voxel_volume_um3),
        "voxel_volume_cm3": float(voxel_volume_cm3),
        "integration_time_s": float(integration_time_s),
        "optical_generation_factor": float(optical_generation_factor),
        "raw_edep_total_eV": float(edep_2d.sum()),
        "optical_generation_sum": float(G_2d.sum()),
    }

    return x_centers_um, z_centers_um, G_2d, summary


# ─── Output helpers ───────────────────────────────────────────────────────────

def save_nonzero_csv(
    output_path: Path,
    x_centers: np.ndarray,
    z_centers: np.ndarray,
    G_2d: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nonzero = np.argwhere(G_2d > 0.0)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x_um", "depth_um", "optical_generation_cm^-3_s^-1"])
        for ix, iz in nonzero:
            writer.writerow(
                [float(x_centers[ix]), float(z_centers[iz]), float(G_2d[ix, iz])]
            )


def save_full_grid_csv(
    output_path: Path,
    x_centers: np.ndarray,
    z_centers: np.ndarray,
    G_2d: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["depth_um", *[f"{x:.6f}" for x in x_centers]])
        for iz, depth in enumerate(z_centers):
            writer.writerow([float(depth), *G_2d[:, iz].tolist()])


def save_summary_json(output_path: Path, summary: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def save_outputs(
    output_dir: Path,
    x_centers: np.ndarray,
    z_centers: np.ndarray,
    G_2d: np.ndarray,
    summary: dict,
) -> None:
    """Save all Step-4 outputs to *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)
    save_nonzero_csv(
        output_dir / "intermediate_picture_nonzero.csv", x_centers, z_centers, G_2d
    )
    save_full_grid_csv(
        output_dir / "intermediate_picture_full_grid.csv", x_centers, z_centers, G_2d
    )
    np.savez_compressed(
        output_dir / "intermediate_picture_2d.npz",
        x_centers_um=x_centers,
        z_centers_um=z_centers,
        optical_generation_cm3_s=G_2d,
    )
    save_summary_json(output_dir / "step4_summary.json", summary)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Step 4 (grid3d): convert a sparse pre-binned 3-D energy-deposition CSV "
            "to a 2-D optical-generation grid for TCAD."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the sparse 3-D grid CSV file (grid3d_*.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("output") / "step4_output"),
        help="Directory for the generated Step-4 outputs. Default: output/step4_output",
    )
    parser.add_argument(
        "--integration-time-s",
        type=float,
        default=1.0,
        help="Normalisation integration time in seconds. Default: 1.0",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    rows, n_events, xy_bin_mm, z_bin_um = parse_grid3d_csv(input_path)
    x_centers, z_centers, G_2d, summary = build_2d_grid(
        rows,
        n_events,
        xy_bin_mm,
        z_bin_um,
        integration_time_s=args.integration_time_s,
    )
    save_outputs(output_dir, x_centers, z_centers, G_2d, summary)

    print(f"Input:        {input_path}")
    print(f"Output dir:   {output_dir}")
    print(f"Grid:         {summary['x_bins']} × {summary['z_bins']} (x × z)")
    print(f"Lateral range: [{summary['x_min_um']:.1f}, {summary['x_max_um']:.1f}] µm")
    print(f"Depth range:   [{summary['z_min_um']:.1f}, {summary['z_max_um']:.1f}] µm")
    print(f"G_max:        {G_2d.max():.4e} cm⁻³ s⁻¹")
    print(f"G_sum:        {summary['optical_generation_sum']:.4e} cm⁻³ s⁻¹")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
