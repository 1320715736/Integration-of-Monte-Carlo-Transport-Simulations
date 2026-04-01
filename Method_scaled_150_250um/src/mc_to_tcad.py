#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


MM_TO_UM = 1000.0
EH_PAIR_ENERGY_EV = 3.6
DEFAULT_GRID_STEP_UM = 0.2


def floor_to_step(value: float, step: float) -> float:
    return math.floor(value / step) * step


def ceil_to_step(value: float, step: float) -> float:
    return math.ceil(value / step) * step


def read_step_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    projected_points: list[tuple[float, float]] = []
    weights_eh_pairs: list[float] = []
    primary_histories: set[tuple[int, int]] = set()

    x_min = x_max = None
    y_min = y_max = None
    z_min = z_max = None

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            x_um = 0.5 * (float(row["PreX_mm"]) + float(row["PostX_mm"])) * MM_TO_UM
            y_um = 0.5 * (float(row["PreY_mm"]) + float(row["PostY_mm"])) * MM_TO_UM
            z_um = float(row["Depth_um"])

            projected_points.append((x_um, z_um))
            weights_eh_pairs.append(float(row["Edep_keV"]) * 1000.0 / EH_PAIR_ENERGY_EV)

            if row.get("ParentID") == "0" and row.get("ParticleName") == "e-":
                primary_histories.add((int(row["EventID"]), int(row["TrackID"])))

            if x_min is None:
                x_min = x_max = x_um
                y_min = y_max = y_um
                z_min = z_max = z_um
            else:
                x_min = min(x_min, x_um)
                x_max = max(x_max, x_um)
                y_min = min(y_min, y_um)
                y_max = max(y_max, y_um)
                z_min = min(z_min, z_um)
                z_max = max(z_max, z_um)

    if not projected_points:
        raise ValueError(f"No step data found in {csv_path}.")

    stats = {
        "x_min_um": float(x_min),
        "x_max_um": float(x_max),
        "y_min_um": float(y_min),
        "y_max_um": float(y_max),
        "z_min_um": float(z_min),
        "z_max_um": float(z_max),
        "row_count": float(len(projected_points)),
        "simulated_primary_histories": float(len(primary_histories) or 1),
    }
    return np.asarray(projected_points, dtype=float), np.asarray(weights_eh_pairs, dtype=float), stats


def build_2d_intermediate_picture(
    projected_points: np.ndarray,
    weights_eh_pairs: np.ndarray,
    stats: dict[str, float],
    *,
    step_um: float,
    incoming_electrons: float | None,
    integration_time_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    x_min = floor_to_step(stats["x_min_um"], step_um)
    x_max = ceil_to_step(stats["x_max_um"], step_um)
    y_min = floor_to_step(stats["y_min_um"], step_um)
    y_max = ceil_to_step(stats["y_max_um"], step_um)
    z_min = floor_to_step(stats["z_min_um"], step_um)
    z_max = ceil_to_step(stats["z_max_um"], step_um)

    x_edges = np.arange(x_min, x_max + step_um, step_um, dtype=float)
    z_edges = np.arange(z_min, z_max + step_um, step_um, dtype=float)
    if x_edges.size < 2 or z_edges.size < 2:
        raise ValueError("Grid edges could not be constructed.")

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    raw_pairs_grid, _, _ = np.histogram2d(
        projected_points[:, 0],
        projected_points[:, 1],
        bins=(x_edges, z_edges),
        weights=weights_eh_pairs,
    )

    simulated_primary_histories = stats["simulated_primary_histories"]
    incoming_electrons = (
        float(incoming_electrons)
        if incoming_electrons is not None
        else float(simulated_primary_histories)
    )
    if integration_time_s <= 0.0:
        raise ValueError("integration_time_s must be positive.")

    collapsed_thickness_um = y_max - y_min
    voxel_volume_um3 = step_um * step_um * collapsed_thickness_um
    voxel_volume_cm3 = voxel_volume_um3 * 1.0e-12

    optical_generation_factor = (
        incoming_electrons / simulated_primary_histories
    ) / (integration_time_s * voxel_volume_cm3)

    optical_generation_grid = raw_pairs_grid * optical_generation_factor

    summary = {
        "grid_step_um": float(step_um),
        "x_min_um": float(x_min),
        "x_max_um": float(x_max),
        "lateral_size_um": float(x_max - x_min),
        "z_min_um": float(z_min),
        "z_max_um": float(z_max),
        "depth_um": float(z_max - z_min),
        "collapsed_axis": "y",
        "collapsed_y_min_um": float(y_min),
        "collapsed_y_max_um": float(y_max),
        "collapsed_thickness_um": float(collapsed_thickness_um),
        "voxel_volume_um3": float(voxel_volume_um3),
        "voxel_volume_cm3": float(voxel_volume_cm3),
        "integration_time_s": float(integration_time_s),
        "simulated_primary_histories": float(simulated_primary_histories),
        "incoming_electrons_assumed": float(incoming_electrons),
        "optical_generation_factor": float(optical_generation_factor),
        "raw_pairs_sum": float(raw_pairs_grid.sum()),
        "optical_generation_sum": float(optical_generation_grid.sum()),
        "x_bins": int(raw_pairs_grid.shape[0]),
        "z_bins": int(raw_pairs_grid.shape[1]),
    }
    return x_centers, z_centers, optical_generation_grid, summary


def save_nonzero_csv(
    output_path: Path,
    x_centers: np.ndarray,
    z_centers: np.ndarray,
    optical_generation_grid: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nonzero_indices = np.argwhere(optical_generation_grid > 0.0)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x_um", "depth_um", "optical_generation_cm^-3_s^-1"])
        for ix, iz in nonzero_indices:
            writer.writerow(
                [
                    float(x_centers[ix]),
                    float(z_centers[iz]),
                    float(optical_generation_grid[ix, iz]),
                ]
            )


def save_full_grid_csv(
    output_path: Path,
    x_centers: np.ndarray,
    z_centers: np.ndarray,
    optical_generation_grid: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["depth_um", *[f"{x:.6f}" for x in x_centers]])
        for iz, depth in enumerate(z_centers):
            row = [float(depth), *optical_generation_grid[:, iz].tolist()]
            writer.writerow(row)


def save_summary(output_path: Path, summary: dict[str, float]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 4: collapse a 3D Geant4 step CSV into a 2D intermediate picture."
    )
    parser.add_argument(
        "--input",
        default="step_5um_168504_data.csv",
        help="3D Geant4 step CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default="step4_output",
        help="Directory for the generated Step 4 outputs.",
    )
    parser.add_argument(
        "--grid-step-um",
        type=float,
        default=DEFAULT_GRID_STEP_UM,
        help="Uniform 2D grid size in um. Default: 0.2",
    )
    parser.add_argument(
        "--integration-time-s",
        type=float,
        default=1.0,
        help="Assumed integration time used for OpticalGeneration. Default: 1.0 s",
    )
    parser.add_argument(
        "--incoming-electrons",
        type=float,
        default=None,
        help="Actual incoming electrons during the assumed integration time. "
        "Default: equal to the simulated primary histories.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    projected_points, weights_eh_pairs, stats = read_step_csv(input_path)
    x_centers, z_centers, optical_generation_grid, summary = build_2d_intermediate_picture(
        projected_points,
        weights_eh_pairs,
        stats,
        step_um=args.grid_step_um,
        incoming_electrons=args.incoming_electrons,
        integration_time_s=args.integration_time_s,
    )

    save_nonzero_csv(
        output_dir / "intermediate_picture_nonzero.csv",
        x_centers,
        z_centers,
        optical_generation_grid,
    )
    save_full_grid_csv(
        output_dir / "intermediate_picture_full_grid.csv",
        x_centers,
        z_centers,
        optical_generation_grid,
    )
    np.savez_compressed(
        output_dir / "intermediate_picture_2d.npz",
        x_centers_um=x_centers,
        z_centers_um=z_centers,
        optical_generation_cm3_s=optical_generation_grid,
    )
    save_summary(output_dir / "step4_summary.json", summary)

    print(f"Input CSV: {input_path}")
    print(f"Output directory: {output_dir}")
    print("Projection: keep x and depth, collapse y")
    print(f"Lateral size (um): {summary['lateral_size_um']}")
    print(f"Depth (um): {summary['depth_um']}")
    print(f"Optical generation factor: {summary['optical_generation_factor']:.12e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
