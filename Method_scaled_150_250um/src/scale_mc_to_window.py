#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MM_TO_UM = 1000.0
UM_TO_MM = 1.0e-3


def combined_bounds(rows: list[dict[str, str]], columns: tuple[str, str]) -> tuple[float, float]:
    values_um: list[float] = []
    for row in rows:
        for column in columns:
            values_um.append(float(row[column]) * MM_TO_UM)
    return min(values_um), max(values_um)


def scaled_value_mm(
    value_mm: float,
    *,
    source_center_um: float,
    scale_factor: float,
    target_center_um: float,
) -> float:
    value_um = value_mm * MM_TO_UM
    scaled_um = (value_um - source_center_um) * scale_factor + target_center_um
    return scaled_um * UM_TO_MM


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scale Monte Carlo x/y coordinates into a target TCAD lateral window while preserving deposited energy."
    )
    parser.add_argument(
        "--input",
        default="step_5um_168504_data.csv",
        help="Input Monte Carlo step CSV.",
    )
    parser.add_argument(
        "--output",
        default="step_5um_168504_data_scaled_150_250um.csv",
        help="Scaled Monte Carlo step CSV.",
    )
    parser.add_argument(
        "--summary",
        default=str(Path("output") / "scale_summary.json"),
        help="Scaling summary JSON.",
    )
    parser.add_argument(
        "--target-abs-min-um",
        type=float,
        default=150.0,
        help="Target absolute lateral window minimum in um.",
    )
    parser.add_argument(
        "--target-abs-max-um",
        type=float,
        default=250.0,
        help="Target absolute lateral window maximum in um.",
    )
    parser.add_argument(
        "--tcad-lateral-center-um",
        type=float,
        default=200.0,
        help="TCAD lateral center used by Step 5 for relative coordinate mapping.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary)

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not rows or not fieldnames:
        raise ValueError(f"No Monte Carlo rows found in {input_path}.")

    x_min_um, x_max_um = combined_bounds(rows, ("PreX_mm", "PostX_mm"))
    y_min_um, y_max_um = combined_bounds(rows, ("PreY_mm", "PostY_mm"))
    x_center_um = 0.5 * (x_min_um + x_max_um)
    y_center_um = 0.5 * (y_min_um + y_max_um)
    x_span_um = x_max_um - x_min_um
    y_span_um = y_max_um - y_min_um

    target_center_abs_um = 0.5 * (args.target_abs_min_um + args.target_abs_max_um)
    target_center_rel_um = target_center_abs_um - args.tcad_lateral_center_um
    target_width_um = args.target_abs_max_um - args.target_abs_min_um
    if target_width_um <= 0.0:
        raise ValueError("Target absolute window must have a positive width.")

    source_max_span_um = max(x_span_um, y_span_um)
    if source_max_span_um <= 0.0:
        raise ValueError("Source x/y span must be positive.")
    scale_factor = target_width_um / source_max_span_um

    scaled_rows: list[dict[str, str]] = []
    total_edep_keV = 0.0
    for row in rows:
        updated = dict(row)
        updated["PreX_mm"] = f"{scaled_value_mm(float(row['PreX_mm']), source_center_um=x_center_um, scale_factor=scale_factor, target_center_um=target_center_rel_um):.12f}"
        updated["PostX_mm"] = f"{scaled_value_mm(float(row['PostX_mm']), source_center_um=x_center_um, scale_factor=scale_factor, target_center_um=target_center_rel_um):.12f}"
        updated["PreY_mm"] = f"{scaled_value_mm(float(row['PreY_mm']), source_center_um=y_center_um, scale_factor=scale_factor, target_center_um=0.0):.12f}"
        updated["PostY_mm"] = f"{scaled_value_mm(float(row['PostY_mm']), source_center_um=y_center_um, scale_factor=scale_factor, target_center_um=0.0):.12f}"
        scaled_rows.append(updated)
        total_edep_keV += float(row["Edep_keV"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scaled_rows)

    scaled_x_min_um, scaled_x_max_um = combined_bounds(scaled_rows, ("PreX_mm", "PostX_mm"))
    scaled_y_min_um, scaled_y_max_um = combined_bounds(scaled_rows, ("PreY_mm", "PostY_mm"))

    summary = {
        "input_csv": str(input_path),
        "output_csv": str(output_path),
        "row_count": len(rows),
        "target_absolute_window_um": [float(args.target_abs_min_um), float(args.target_abs_max_um)],
        "tcad_lateral_center_um": float(args.tcad_lateral_center_um),
        "target_relative_center_um": float(target_center_rel_um),
        "uniform_scale_factor_xy": float(scale_factor),
        "source_x_range_um": [float(x_min_um), float(x_max_um)],
        "source_y_range_um": [float(y_min_um), float(y_max_um)],
        "scaled_x_range_relative_um": [float(scaled_x_min_um), float(scaled_x_max_um)],
        "scaled_y_range_relative_um": [float(scaled_y_min_um), float(scaled_y_max_um)],
        "mapped_absolute_x_range_um": [
            float(scaled_x_min_um + args.tcad_lateral_center_um),
            float(scaled_x_max_um + args.tcad_lateral_center_um),
        ],
        "total_edep_keV_preserved": float(total_edep_keV),
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Input CSV: {input_path}")
    print(f"Output CSV: {output_path}")
    print(f"Uniform x/y scale factor: {scale_factor:.12f}")
    print(
        "Mapped absolute x range (um): "
        f"{summary['mapped_absolute_x_range_um'][0]:.6f} to {summary['mapped_absolute_x_range_um'][1]:.6f}"
    )
    print(f"Total deposited energy kept (keV): {total_edep_keV:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
