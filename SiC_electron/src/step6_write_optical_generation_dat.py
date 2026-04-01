#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_optical_generation_values(csv_path: Path) -> list[float]:
    values: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header:
            raise ValueError(f"No header found in {csv_path}.")
        for row in reader:
            if not row:
                continue
            values.append(float(row[0]))
    if not values:
        raise ValueError(f"No optical generation values found in {csv_path}.")
    return values


def format_values_block(values: list[float], values_per_line: int = 10) -> list[str]:
    lines: list[str] = []
    for start in range(0, len(values), values_per_line):
        chunk = values[start : start + values_per_line]
        lines.append(" " + " ".join(f"{value:.15e}" for value in chunk) + "\n")
    return lines


def extract_dataset_validity(lines: list[str], dataset_name: str) -> list[str]:
    inside_dataset = False

    for raw_line in lines:
        line = raw_line.strip()
        if not inside_dataset:
            if line == f'Dataset ("{dataset_name}") {{':
                inside_dataset = True
            continue

        if line.startswith("validity"):
            start = line.find("[")
            end = line.rfind("]")
            if start == -1 or end == -1 or end <= start:
                raise ValueError(f"Could not parse validity for dataset {dataset_name}.")
            content = line[start + 1 : end].strip()
            if not content:
                raise ValueError(f"Dataset {dataset_name} has an empty validity list.")
            return [token.strip().strip('"') for token in content.split() if token.strip()]

        if line == "}":
            break

    raise ValueError(f"Could not find validity for dataset {dataset_name}.")


def build_optical_generation_dataset(values: list[float], validity: list[str]) -> list[str]:
    validity_str = " ".join(f'"{region}"' for region in validity)
    dataset_lines = [
        '  Dataset ("OpticalGeneration") {\n',
        "    function  = OpticalGeneration\n",
        "    type      = scalar\n",
        "    dimension = 1\n",
        "    location  = vertex\n",
        f"    validity  = [ {validity_str} ]\n",
        f"    Values ({len(values)}) {{\n",
    ]
    dataset_lines.extend(format_values_block(values))
    dataset_lines.extend(
        [
            "    }\n",
            "  }\n",
            "\n",
        ]
    )
    return dataset_lines


def update_header_lists(lines: list[str]) -> list[str]:
    updated = list(lines)
    dataset_line_index = None
    function_line_index = None

    for index, line in enumerate(updated):
        if "datasets    = [" in line:
            dataset_line_index = index
        if "functions   = [" in line:
            function_line_index = index

    if dataset_line_index is None or function_line_index is None:
        raise ValueError("Could not find datasets/functions header lines.")

    if '"OpticalGeneration"' not in updated[dataset_line_index]:
        updated[dataset_line_index] = updated[dataset_line_index].replace(
            " ]",
            ' "OpticalGeneration" ]',
        )
    if " OpticalGeneration " not in f" {updated[function_line_index]} ":
        updated[function_line_index] = updated[function_line_index].replace(
            " ]",
            " OpticalGeneration ]",
        )
    return updated


def insert_dataset_before_data_end(lines: list[str], dataset_lines: list[str]) -> list[str]:
    updated = list(lines)
    insert_index = None
    for index in range(len(updated) - 1, -1, -1):
        if updated[index].strip() == "}":
            insert_index = index
            break
    if insert_index is None:
        raise ValueError("Could not find final closing brace in DAT file.")

    return updated[:insert_index] + dataset_lines + updated[insert_index:]


def count_pmi_values(lines: list[str], dataset_name: str) -> int:
    inside_dataset = False
    inside_values = False
    count = 0

    for raw_line in lines:
        line = raw_line.strip()
        if not inside_dataset:
            if line == f'Dataset ("{dataset_name}") {{':
                inside_dataset = True
            continue

        if inside_dataset and not inside_values:
            if line.startswith("Values ("):
                inside_values = True
            continue

        if inside_values:
            if line == "}":
                break
            count += len(line.split())

    if count == 0:
        raise ValueError(f"Could not count values for dataset {dataset_name}.")
    return count


def build_summary(
    *,
    source_dat: Path,
    source_values: Path,
    output_dat: Path,
    value_count: int,
    pmi_reference_count: int,
    validity: list[str],
) -> dict[str, object]:
    return {
        "source_dat": str(source_dat),
        "source_values": str(source_values),
        "output_dat": str(output_dat),
        "optical_generation_value_count": value_count,
        "pmi_reference_value_count": pmi_reference_count,
        "dataset_added": "OpticalGeneration",
        "validity": validity,
        "source_file_untouched": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 6: write a new DAT file with an OpticalGeneration dataset."
    )
    parser.add_argument(
        "--input-dat",
        default="n4_.dat",
        help="Source DF-ISE DAT file.",
    )
    parser.add_argument(
        "--input-values",
        default=str(Path("output") / "step5_output" / "optical_generation_values.csv"),
        help="Interpolated optical generation values from Step 5.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("output") / "step6_output"),
        help="Directory for generated Step 6 outputs.",
    )
    parser.add_argument(
        "--output-name",
        default="n4_optical_generation.dat",
        help="Name of the new DAT file.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    input_dat = Path(args.input_dat)
    input_values = Path(args.input_values)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dat = output_dir / args.output_name

    lines = input_dat.read_text(encoding="utf-8").splitlines(keepends=True)
    values = load_optical_generation_values(input_values)
    pmi_reference_count = count_pmi_values(lines, "PMIUserField0")
    validity = extract_dataset_validity(lines, "PMIUserField0")
    if len(values) != pmi_reference_count:
        raise ValueError(
            f"Step5 values ({len(values)}) do not match PMIUserField0 value count ({pmi_reference_count})."
        )

    updated_lines = update_header_lists(lines)
    dataset_lines = build_optical_generation_dataset(values, validity)
    updated_lines = insert_dataset_before_data_end(updated_lines, dataset_lines)
    output_dat.write_text("".join(updated_lines), encoding="utf-8")

    summary = build_summary(
        source_dat=input_dat,
        source_values=input_values,
        output_dat=output_dat,
        value_count=len(values),
        pmi_reference_count=pmi_reference_count,
        validity=validity,
    )
    (output_dir / "step6_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Source DAT: {input_dat}")
    print(f"Input values: {input_values}")
    print(f"Output DAT: {output_dat}")
    print(f"OpticalGeneration values written: {len(values)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
