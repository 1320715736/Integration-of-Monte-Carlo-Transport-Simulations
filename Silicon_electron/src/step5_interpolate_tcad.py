#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import interpn


def parse_vertices_from_grd(grd_path: Path) -> np.ndarray:
    vertices: list[tuple[float, float]] = []
    inside_vertices = False

    with grd_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not inside_vertices:
                if line.startswith("Vertices ("):
                    inside_vertices = True
                continue

            if line == "}":
                break
            if not line:
                continue

            parts = line.split()
            if len(parts) % 2 != 0:
                raise ValueError(f"Unexpected vertex line: {raw_line.rstrip()}")
            for index in range(0, len(parts), 2):
                vertices.append((float(parts[index]), float(parts[index + 1])))

    if not vertices:
        raise ValueError(f"No vertices found in {grd_path}.")
    return np.asarray(vertices, dtype=float)


def load_step4_grid(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    return (
        np.asarray(data["x_centers_um"], dtype=float),
        np.asarray(data["z_centers_um"], dtype=float),
        np.asarray(data["optical_generation_cm3_s"], dtype=float),
    )


def interpolate_to_tcad_vertices(
    vertices: np.ndarray,
    x_centers_um: np.ndarray,
    z_centers_um: np.ndarray,
    optical_generation_grid: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    tcad_depth_um = vertices[:, 0] - float(vertices[:, 0].min())
    tcad_lateral_center_um = 0.5 * (float(vertices[:, 1].min()) + float(vertices[:, 1].max()))
    tcad_lateral_um = vertices[:, 1] - tcad_lateral_center_um

    query_points = np.column_stack([tcad_lateral_um, tcad_depth_um])
    interpolated = interpn(
        points=(x_centers_um, z_centers_um),
        values=optical_generation_grid,
        xi=query_points,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    summary = {
        "vertex_count": int(vertices.shape[0]),
        "tcad_depth_min_um": float(tcad_depth_um.min()),
        "tcad_depth_max_um": float(tcad_depth_um.max()),
        "tcad_lateral_min_um": float(tcad_lateral_um.min()),
        "tcad_lateral_max_um": float(tcad_lateral_um.max()),
        "tcad_lateral_center_um": float(tcad_lateral_center_um),
        "interpolated_min": float(interpolated.min()),
        "interpolated_max": float(interpolated.max()),
        "interpolated_nonzero_count": int(np.count_nonzero(interpolated)),
    }
    return interpolated, summary


def save_vertex_table(
    output_path: Path,
    vertices: np.ndarray,
    interpolated: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tcad_depth_um = vertices[:, 0] - float(vertices[:, 0].min())
    tcad_lateral_center_um = 0.5 * (float(vertices[:, 1].min()) + float(vertices[:, 1].max()))
    tcad_lateral_um = vertices[:, 1] - tcad_lateral_center_um

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "vertex_id",
                "tcad_coord0_um",
                "tcad_coord1_um",
                "mapped_lateral_um",
                "mapped_depth_um",
                "optical_generation_cm^-3_s^-1",
            ]
        )
        for vertex_id, (vertex, lateral_um, depth_um, value) in enumerate(
            zip(vertices, tcad_lateral_um, tcad_depth_um, interpolated)
        ):
            writer.writerow(
                [
                    vertex_id,
                    float(vertex[0]),
                    float(vertex[1]),
                    float(lateral_um),
                    float(depth_um),
                    float(value),
                ]
            )


def save_values_only(output_path: Path, interpolated: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["optical_generation_cm^-3_s^-1"])
        for value in interpolated:
            writer.writerow([float(value)])


def save_summary(output_path: Path, summary: dict[str, float]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 5: interpolate the 2D intermediate picture onto the TCAD mesh vertices."
    )
    parser.add_argument(
        "--grd",
        default="n51_.grd",
        help="TCAD DF-ISE grid file.",
    )
    parser.add_argument(
        "--step4-grid",
        default=str(Path("step4_output") / "intermediate_picture_2d.npz"),
        help="Step 4 intermediate picture .npz file.",
    )
    parser.add_argument(
        "--output-dir",
        default="step5_output",
        help="Directory for the generated Step 5 outputs.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    grd_path = Path(args.grd)
    step4_grid_path = Path(args.step4_grid)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vertices = parse_vertices_from_grd(grd_path)
    x_centers_um, z_centers_um, optical_generation_grid = load_step4_grid(step4_grid_path)
    interpolated, summary = interpolate_to_tcad_vertices(
        vertices,
        x_centers_um,
        z_centers_um,
        optical_generation_grid,
    )

    save_vertex_table(output_dir / "tcad_vertices_interpolated.csv", vertices, interpolated)
    save_values_only(output_dir / "optical_generation_values.csv", interpolated)
    np.savez_compressed(
        output_dir / "step5_interpolated_vertices.npz",
        tcad_vertices_um=vertices,
        optical_generation_cm3_s=interpolated,
    )
    save_summary(output_dir / "step5_summary.json", summary)

    print(f"Grid file: {grd_path}")
    print(f"Step4 grid: {step4_grid_path}")
    print(f"Output directory: {output_dir}")
    print(f"Vertex count: {summary['vertex_count']}")
    print(f"Non-zero interpolated values: {summary['interpolated_nonzero_count']}")
    print(f"Mapped lateral range (um): {summary['tcad_lateral_min_um']} to {summary['tcad_lateral_max_um']}")
    print(f"Mapped depth range (um): {summary['tcad_depth_min_um']} to {summary['tcad_depth_max_um']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
