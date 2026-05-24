from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
THICKNESS_CSV_DIR = ROOT_DIR / "raw_data" / "tcad_i-t_csv" / "thickness"
THICKNESS_GENERATION_DIR = ROOT_DIR / "thickness_generation"

BASELINE_SAMPLE_COUNT = 5
ELEMENTARY_CHARGE_C = 1.602176634e-19
MIN_PLOT_THICKNESS_UM = 0.1

DISPLAY_LABEL_ALIASES = {
    "c14": "C-14 spectrum",
}

FIXED_SERIES_COLORS = {
    "10kev": "#4C78A8",
    "20kev": "#F58518",
    "30kev": "#54A24B",
    "49kev": "#B279A2",
    "c14": "#111111",
}


@dataclass
class ThicknessCurve:
    key: str
    thickness_um: float
    x_s: np.ndarray
    y_a: np.ndarray
    raw_point_count: int
    skipped_row_count: int
    merged_duplicate_time_count: int


@dataclass
class ThicknessDataset:
    key: str
    label: str
    energy_keV: float | None
    csv_path: Path
    curves: dict[str, ThicknessCurve]
    metrics: dict[str, dict[str, float | int | str | None]]


def normalize_key(raw_key: str) -> str:
    return raw_key.strip().replace(" ", "").lower()


def thickness_key(thickness_um: float) -> str:
    return f"{float(thickness_um):g}"


def parse_float(value: str) -> float | None:
    text = value.strip()
    if not text or text == "-":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def merge_duplicate_times(x_values: np.ndarray, y_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    order = np.argsort(x_values, kind="mergesort")
    sorted_x = x_values[order]
    sorted_y = y_values[order]
    unique_x, inverse, counts = np.unique(sorted_x, return_inverse=True, return_counts=True)
    y_sum = np.zeros(unique_x.shape, dtype=float)
    np.add.at(y_sum, inverse, sorted_y)
    return unique_x, y_sum / counts, int(sorted_x.size - unique_x.size)


def integrate_trapezoid(y_values: np.ndarray, x_values: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y_values, x_values))
    return float(np.trapz(y_values, x_values))


def energy_from_key(key: str) -> float | None:
    match = re.fullmatch(r"([0-9]+(?:p[0-9]+)?)kev", key)
    if match is None:
        return None
    return float(match.group(1).replace("p", "."))


def display_label(key: str) -> str:
    if key in DISPLAY_LABEL_ALIASES:
        return DISPLAY_LABEL_ALIASES[key]

    energy = energy_from_key(key)
    if energy is None:
        return key
    if energy.is_integer():
        return f"{int(energy)} keV"
    return f"{energy:g} keV"


def series_color(key: str) -> str:
    return FIXED_SERIES_COLORS.get(key, "#5F6B7A")


def dataset_key_from_path(csv_path: Path) -> str:
    stem = normalize_key(csv_path.stem)
    if stem.startswith("thickness_"):
        stem = stem[len("thickness_") :]
    return stem


def dataset_sort_key(dataset_key: str) -> tuple[int, float, str]:
    if dataset_key == "c14":
        return (1, 14.0, dataset_key)
    energy = energy_from_key(dataset_key)
    if energy is not None:
        return (0, energy, dataset_key)
    return (2, math.inf, dataset_key)


def parse_thickness_csv(csv_path: Path) -> dict[str, ThicknessCurve]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        headers = next(reader)
        rows = list(reader)

    if len(headers) % 2 != 0:
        raise ValueError(f"Expected X/Y column pairs in {csv_path}, got {len(headers)} headers")
    if not rows:
        raise ValueError(f"No data rows found in {csv_path}")

    columns = list(zip(*rows))
    curves: dict[str, ThicknessCurve] = {}
    for index in range(0, len(headers), 2):
        match = re.search(r"\(([^)]+)\)", headers[index])
        if match is None:
            continue

        curve_key = thickness_key(float(match.group(1).strip()))
        thickness_um = float(curve_key)
        x_values_list: list[float] = []
        y_values_list: list[float] = []
        skipped_rows = 0
        for x_raw, y_raw in zip(columns[index], columns[index + 1]):
            x_value = parse_float(x_raw)
            y_value = parse_float(y_raw)
            if x_value is None or y_value is None:
                skipped_rows += 1
                continue
            x_values_list.append(x_value)
            y_values_list.append(y_value)

        x_values = np.asarray(x_values_list, dtype=float)
        y_values = np.asarray(y_values_list, dtype=float)
        if x_values.size == 0 or y_values.size == 0:
            continue

        raw_point_count = int(min(x_values.size, y_values.size))
        x_values = x_values[:raw_point_count]
        y_values = y_values[:raw_point_count]
        x_values, y_values, merged_duplicate_time_count = merge_duplicate_times(x_values, y_values)

        baseline_count = min(BASELINE_SAMPLE_COUNT, y_values.size)
        baseline = float(np.mean(y_values[:baseline_count]))
        curves[curve_key] = ThicknessCurve(
            key=curve_key,
            thickness_um=thickness_um,
            x_s=x_values,
            y_a=y_values - baseline,
            raw_point_count=raw_point_count,
            skipped_row_count=skipped_rows,
            merged_duplicate_time_count=merged_duplicate_time_count,
        )

    if not curves:
        raise ValueError(f"No thickness curves parsed from {csv_path}")
    return curves


def thickness_folder_candidates(thickness_um: float) -> list[str]:
    return list(
        dict.fromkeys(
            [
                thickness_key(thickness_um),
                f"{float(thickness_um):.1f}",
                f"{float(thickness_um):.2f}",
            ]
        )
    )


def energy_folder_candidates(dataset_key: str) -> list[str]:
    if dataset_key == "c14":
        return ["c14"]

    energy = energy_from_key(dataset_key)
    if energy is None:
        return [dataset_key]

    candidates = [dataset_key]
    if energy.is_integer():
        candidates.extend([f"{int(energy)}keV", f"{int(energy)}kev", f"{energy:g}keV"])
    else:
        candidates.extend([f"{energy:g}keV", f"{energy:g}kev"])
    return list(dict.fromkeys(candidates))


def load_step4_summary(thickness_um: float, dataset_key: str) -> dict | None:
    for folder_name in thickness_folder_candidates(thickness_um):
        output_root = THICKNESS_GENERATION_DIR / folder_name / "output"
        if not output_root.exists():
            continue

        folder_by_lower = {path.name.lower(): path for path in output_root.iterdir() if path.is_dir()}
        for candidate in energy_folder_candidates(dataset_key):
            folder = folder_by_lower.get(candidate.lower())
            if folder is None:
                continue
            summary_path = folder / "step4_output" / "step4_summary.json"
            if summary_path.exists():
                return json.loads(summary_path.read_text(encoding="utf-8"))
    return None


def build_metrics(curve: ThicknessCurve, summary: dict | None) -> dict[str, float | int | str | None]:
    peak_index = int(np.argmax(curve.y_a))
    peak_current_a = float(curve.y_a[peak_index])
    peak_time_s = float(curve.x_s[peak_index])
    collected_charge_2d_c = integrate_trapezoid(curve.y_a, curve.x_s)

    metrics: dict[str, float | int | str | None] = {
        "label": f"{curve.thickness_um:g} um",
        "thickness_um": curve.thickness_um,
        "peak_current_a": peak_current_a,
        "peak_time_s": peak_time_s,
        "collected_charge_2d_c": collected_charge_2d_c,
        "collected_charge_c": None,
        "collected_charge_restored_c": None,
        "raw_point_count": curve.raw_point_count,
        "unique_time_point_count": int(curve.x_s.size),
        "skipped_row_count": curve.skipped_row_count,
        "merged_duplicate_time_count": curve.merged_duplicate_time_count,
        "collapsed_thickness_um": None,
        "generated_pairs_per_event": None,
        "deposited_energy_ev_per_event": None,
        "cce": None,
        "cce_percent": None,
        "relative_to_0p1um_percent": None,
        "delta_vs_0p2um_cce_points": None,
    }

    if summary is None:
        return metrics

    collapsed_thickness_um = float(summary.get("collapsed_thickness_um", 1.0))
    collected_charge_restored_c = collected_charge_2d_c * collapsed_thickness_um
    generated_pairs_per_event = (
        float(summary["raw_edep_total_eV"])
        / float(summary["n_events"])
        / float(summary["eh_pair_energy_ev"])
    )
    expected_charge_c = generated_pairs_per_event * ELEMENTARY_CHARGE_C
    cce = collected_charge_restored_c / expected_charge_c if expected_charge_c > 0.0 else None
    metrics.update(
        {
            "collected_charge_c": collected_charge_restored_c,
            "collected_charge_restored_c": collected_charge_restored_c,
            "collapsed_thickness_um": collapsed_thickness_um,
            "generated_pairs_per_event": generated_pairs_per_event,
            "deposited_energy_ev_per_event": float(summary["raw_edep_total_eV"]) / float(summary["n_events"]),
            "cce": cce,
            "cce_percent": 100.0 * cce if cce is not None else None,
        }
    )
    return metrics


def add_relative_metrics(metrics: dict[str, dict[str, float | int | str | None]], curves: list[ThicknessCurve]) -> None:
    if not curves:
        return

    thinnest_key = curves[0].key
    reference_key = min(curves, key=lambda curve: abs(curve.thickness_um - 0.2)).key
    thinnest_cce = metrics[thinnest_key].get("cce")
    reference_cce = metrics[reference_key].get("cce")

    for order, curve in enumerate(curves):
        line = metrics[curve.key]
        line["plot_order"] = order
        line["relative_to_0p1um_percent"] = (
            100.0 * float(line["cce"]) / float(thinnest_cce)
            if thinnest_cce is not None and line.get("cce") is not None
            else None
        )
        line["delta_vs_0p2um_cce_points"] = (
            100.0 * (float(line["cce"]) - float(reference_cce))
            if reference_cce is not None and line.get("cce") is not None
            else None
        )


def load_dataset(csv_path: Path) -> ThicknessDataset:
    dataset_key = dataset_key_from_path(csv_path)
    curves = parse_thickness_csv(csv_path)
    ordered_curves = sorted(curves.values(), key=lambda curve: curve.thickness_um)
    metrics: dict[str, dict[str, float | int | str | None]] = {}
    for curve in ordered_curves:
        summary = load_step4_summary(curve.thickness_um, dataset_key)
        metrics[curve.key] = build_metrics(curve, summary)

    add_relative_metrics(metrics, ordered_curves)
    return ThicknessDataset(
        key=dataset_key,
        label=display_label(dataset_key),
        energy_keV=energy_from_key(dataset_key),
        csv_path=csv_path,
        curves=curves,
        metrics=metrics,
    )


def discover_thickness_csvs(csv_dir: Path = THICKNESS_CSV_DIR) -> list[Path]:
    csv_files = [path for path in csv_dir.glob("thickness_*.csv") if path.is_file()]
    if not csv_files:
        raise FileNotFoundError(f"No thickness CSV files found under {csv_dir}")
    return sorted(csv_files, key=lambda path: dataset_sort_key(dataset_key_from_path(path)))


def load_all_datasets(csv_dir: Path = THICKNESS_CSV_DIR) -> list[ThicknessDataset]:
    return [load_dataset(path) for path in discover_thickness_csvs(csv_dir)]


def dataset_cce_points(
    dataset: ThicknessDataset,
    min_thickness_um: float = MIN_PLOT_THICKNESS_UM,
) -> list[dict[str, float | str | None]]:
    points: list[dict[str, float | str | None]] = []
    for metric in sorted(dataset.metrics.values(), key=lambda item: float(item["thickness_um"])):
        thickness_um = float(metric["thickness_um"])
        cce_percent = metric.get("cce_percent")
        if cce_percent is None or thickness_um < min_thickness_um:
            continue
        points.append(
            {
                "thickness_um": thickness_um,
                "cce_percent": float(cce_percent),
                "relative_to_0p1um_percent": (
                    float(metric["relative_to_0p1um_percent"])
                    if metric.get("relative_to_0p1um_percent") is not None
                    else None
                ),
                "peak_current_nA": (
                    float(metric["peak_current_a"]) * 1e9 if metric.get("peak_current_a") is not None else None
                ),
            }
        )
    return points


def metric_for_thickness(dataset: ThicknessDataset, thickness_um: float) -> dict[str, float | int | str | None] | None:
    return dataset.metrics.get(thickness_key(thickness_um))


def numeric_energy_datasets(datasets: list[ThicknessDataset]) -> list[ThicknessDataset]:
    return sorted(
        [dataset for dataset in datasets if dataset.energy_keV is not None],
        key=lambda dataset: (float(dataset.energy_keV), dataset.key),
    )


def shared_thicknesses(
    datasets: list[ThicknessDataset],
    min_thickness_um: float = MIN_PLOT_THICKNESS_UM,
) -> list[float]:
    if not datasets:
        return []

    thickness_key_sets: list[set[str]] = []
    for dataset in datasets:
        keys = {
            thickness_key(float(metric["thickness_um"]))
            for metric in dataset.metrics.values()
            if metric.get("cce_percent") is not None and float(metric["thickness_um"]) >= min_thickness_um
        }
        thickness_key_sets.append(keys)

    shared_keys = set.intersection(*thickness_key_sets) if thickness_key_sets else set()
    return [float(key) for key in sorted(shared_keys, key=float)]


def interpolate_threshold_energy(
    points: list[tuple[float, float]],
    target_cce_percent: float,
) -> tuple[float | None, str, tuple[float, float] | None]:
    if len(points) < 2:
        return None, "insufficient-points", None

    for energy_keV, cce_percent in points:
        if abs(cce_percent - target_cce_percent) <= 1e-12:
            return energy_keV, "exact", (energy_keV, energy_keV)

    for index in range(len(points) - 1):
        energy_0, cce_0 = points[index]
        energy_1, cce_1 = points[index + 1]
        if (cce_0 - target_cce_percent) * (cce_1 - target_cce_percent) <= 0.0 and cce_0 != cce_1:
            fraction = (target_cce_percent - cce_0) / (cce_1 - cce_0)
            energy_keV = energy_0 + fraction * (energy_1 - energy_0)
            return energy_keV, "interpolated", (energy_0, energy_1)

    if target_cce_percent < points[0][1]:
        energy_0, cce_0 = points[0]
        energy_1, cce_1 = points[1]
        mode = "extrapolated-below-range"
    else:
        energy_0, cce_0 = points[-2]
        energy_1, cce_1 = points[-1]
        mode = "extrapolated-above-range"

    slope = (cce_1 - cce_0) / (energy_1 - energy_0)
    if slope == 0.0:
        return None, "flat-segment", (energy_0, energy_1)

    energy_keV = energy_0 + (target_cce_percent - cce_0) / slope
    return max(0.0, energy_keV), mode, (energy_0, energy_1)


def threshold_energy_for_thickness(
    datasets: list[ThicknessDataset],
    thickness_um: float,
    target_cce_percent: float,
) -> dict[str, object]:
    points: list[tuple[float, float]] = []
    detailed_points: list[dict[str, float | str]] = []
    for dataset in numeric_energy_datasets(datasets):
        metric = metric_for_thickness(dataset, thickness_um)
        if metric is None or metric.get("cce_percent") is None or dataset.energy_keV is None:
            continue
        cce_percent = float(metric["cce_percent"])
        energy_keV = float(dataset.energy_keV)
        points.append((energy_keV, cce_percent))
        detailed_points.append(
            {
                "key": dataset.key,
                "label": dataset.label,
                "energy_keV": energy_keV,
                "cce_percent": cce_percent,
            }
        )

    points.sort(key=lambda item: item[0])
    detailed_points.sort(key=lambda item: float(item["energy_keV"]))
    threshold_keV, mode, bracket = interpolate_threshold_energy(points, target_cce_percent)

    return {
        "thickness_um": thickness_um,
        "target_cce_percent": target_cce_percent,
        "threshold_energy_keV": threshold_keV,
        "mode": mode,
        "bracket_energy_keV": list(bracket) if bracket is not None else None,
        "numeric_points": detailed_points,
    }
