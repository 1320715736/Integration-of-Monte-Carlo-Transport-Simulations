#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[3]
VALIDATION_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = VALIDATION_DIR / "data"
REPORT_DIR = VALIDATION_DIR / "report"
FIGURE_DIR = VALIDATION_DIR / "figures"
TREND1_DIR = FIGURE_DIR / "trend1_charge_vs_Nt"
TREND2_DIR = FIGURE_DIR / "trend2_loss_vs_thickness"
TREND3_DIR = FIGURE_DIR / "trend3_cce_vs_thickness"

TCAD_IT_DIR = ROOT_DIR / "raw_data" / "tcad_it"
GENERATION_DIR = ROOT_DIR / "generation"

BASELINE_SAMPLE_COUNT = 5
ELEMENTARY_CHARGE_C = 1.602176634e-19

NT_ORDER = ["0", "1e11", "1e12", "1e13", "5e13"]
NT_VALUE = {
    "0": 0.0,
    "1e11": 1.0e11,
    "1e12": 1.0e12,
    "1e13": 1.0e13,
    "5e13": 5.0e13,
}
NT_LABEL = {
    "0": "0",
    "1e11": "1e11",
    "1e12": "1e12",
    "1e13": "1e13",
    "5e13": "5e13",
}

SOURCE_ORDER = ["10kev", "20kev", "30kev", "49kev", "100kev", "156p5", "c14"]
SOURCE_CONFIG = {
    "10kev": {
        "label": "10 keV",
        "csv": "10kev.csv",
        "folder": "10keV",
        "energy_keV": 10.0,
        "color": "#8C564B",
    },
    "20kev": {
        "label": "20 keV",
        "csv": "20kev.csv",
        "folder": "20keV",
        "energy_keV": 20.0,
        "color": "#4C78A8",
    },
    "30kev": {
        "label": "30 keV",
        "csv": "30kev.csv",
        "folder": "30keV",
        "energy_keV": 30.0,
        "color": "#9467BD",
    },
    "49kev": {
        "label": "49 keV",
        "csv": "49kev.csv",
        "folder": "49keV",
        "energy_keV": 49.0,
        "color": "#59A14F",
    },
    "100kev": {
        "label": "100 keV",
        "csv": "100kev.csv",
        "folder": "100keV",
        "energy_keV": 100.0,
        "color": "#F28E2B",
    },
    "156p5": {
        "label": "156.5 keV",
        "csv": "156p5.csv",
        "folder": "156p5keV",
        "energy_keV": 156.5,
        "color": "#E15759",
    },
    "c14": {
        "label": "C-14 spectrum",
        "csv": "c14.csv",
        "folder": "c14",
        "energy_keV": "",
        "color": "#111827",
    },
}

HEADER_RE = re.compile(r"TotalCurrent\((?P<tag>[^)]*)\)\s+(?P<axis>[XY])$")
TAG_RE = re.compile(
    r"(?P<thickness>\d+(?:\.\d+)?)um_"
    r"(?P<bias>\d+(?:\.\d+)?)V_"
    r"Nt(?P<nt>0|1e11|1e12|1e13|5e13)"
    r"n(?P<node>\d+)_des"
)

TEXT = "#111827"
MUTED = "#64748B"
GRID = "#CBD5E1"
RED = "#D62728"
BLUE = "#1F77B4"


@dataclass(frozen=True)
class Curve:
    source: str
    source_label: str
    source_csv: str
    source_folder: str
    tag: str
    thickness_um: float
    bias_v: float
    nt_label: str
    trap_density_cm3: float
    node: int
    time_s: np.ndarray
    current_a: np.ndarray
    baseline_current_a: float
    baseline_method: str
    raw_point_count: int
    unique_time_point_count: int
    skipped_row_count: int
    merged_duplicate_time_count: int
    initial_outlier_flag: bool


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text or text == "-":
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def integrate_trapezoid(y_values: np.ndarray, x_values: np.ndarray) -> float:
    if y_values.size < 2:
        return 0.0
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y_values, x_values))
    return float(np.trapz(y_values, x_values))


def merge_duplicate_times(x_values: np.ndarray, y_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    order = np.argsort(x_values, kind="mergesort")
    sorted_x = x_values[order]
    sorted_y = y_values[order]
    unique_x, inverse, counts = np.unique(sorted_x, return_inverse=True, return_counts=True)
    y_sum = np.zeros(unique_x.shape, dtype=float)
    np.add.at(y_sum, inverse, sorted_y)
    return unique_x, y_sum / counts, int(sorted_x.size - unique_x.size)


def detect_initial_outlier(current_a: np.ndarray) -> bool:
    if current_a.size < 10:
        return False
    tail = np.abs(current_a[1 : min(current_a.size, 10)])
    scale = float(np.median(tail)) if tail.size else 0.0
    if scale <= 0.0:
        scale = float(np.max(tail)) if tail.size else 0.0
    if scale <= 0.0:
        return False
    return abs(float(current_a[0])) > 1000.0 * scale


def charge_time_from_signal(time_s: np.ndarray, signal_a: np.ndarray, fraction: float) -> float:
    if time_s.size < 2:
        return float("nan")
    increments = 0.5 * (signal_a[:-1] + signal_a[1:]) * np.diff(time_s)
    increments = np.clip(increments, 0.0, None)
    cumulative_q = np.concatenate(([0.0], np.cumsum(increments)))
    final_q = float(cumulative_q[-1])
    if final_q <= 0.0:
        return float("nan")
    target = fraction * final_q
    index = int(np.searchsorted(cumulative_q, target, side="left"))
    if index <= 0:
        return float(time_s[0])
    if index >= time_s.size:
        return float(time_s[-1])
    q0 = float(cumulative_q[index - 1])
    q1 = float(cumulative_q[index])
    t0 = float(time_s[index - 1])
    t1 = float(time_s[index])
    if q1 == q0:
        return t1
    return t0 + (target - q0) * (t1 - t0) / (q1 - q0)


def fwhm_from_signal(time_s: np.ndarray, signal_a: np.ndarray) -> tuple[float, float, float, float, str]:
    positive = np.clip(signal_a, 0.0, None)
    negative = np.clip(-signal_a, 0.0, None)
    pos_peak = float(np.max(positive)) if positive.size else 0.0
    neg_peak = float(np.max(negative)) if negative.size else 0.0
    if time_s.size < 2 or (pos_peak <= 0.0 and neg_peak <= 0.0):
        return float("nan"), float("nan"), float("nan"), float("nan"), "no_peak"

    if neg_peak > pos_peak:
        pulse = negative
        peak = neg_peak
        polarity = "negative"
    else:
        pulse = positive
        peak = pos_peak
        polarity = "positive"

    half = 0.5 * peak
    above = np.where(pulse >= half)[0]
    if above.size == 0:
        return float("nan"), peak, half, float("nan"), "no_half_crossing"

    left_index = int(above[0])
    right_index = int(above[-1])

    def crossing(index0: int, index1: int) -> float:
        x0 = float(time_s[index0])
        x1 = float(time_s[index1])
        y0 = float(pulse[index0])
        y1 = float(pulse[index1])
        if y1 == y0:
            return x1
        return x0 + (half - y0) * (x1 - x0) / (y1 - y0)

    left = crossing(left_index - 1, left_index) if left_index > 0 else float(time_s[left_index])
    right = (
        crossing(right_index, right_index + 1)
        if right_index < time_s.size - 1
        else float(time_s[right_index])
    )
    width = right - left
    status = "ok" if width >= 0.0 and math.isfinite(width) else "invalid_width"
    return width, peak, half, float(time_s[int(np.argmax(pulse))]), f"{status};{polarity}"


def read_source_curves(source: str) -> list[Curve]:
    config = SOURCE_CONFIG[source]
    csv_path = TCAD_IT_DIR / str(config["csv"])
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        headers = next(reader)
        rows = list(reader)

    curves: list[Curve] = []
    for column_index in range(0, len(headers), 2):
        if column_index + 1 >= len(headers):
            continue
        match_x = HEADER_RE.search(headers[column_index].strip())
        match_y = HEADER_RE.search(headers[column_index + 1].strip())
        if match_x is None or match_y is None:
            continue
        if match_x.group("tag") != match_y.group("tag"):
            continue
        tag = match_x.group("tag")
        tag_match = TAG_RE.search(tag)
        if tag_match is None:
            continue

        x_values: list[float] = []
        y_values: list[float] = []
        skipped = 0
        for row in rows:
            x_value = parse_float(row[column_index] if column_index < len(row) else None)
            y_value = parse_float(row[column_index + 1] if column_index + 1 < len(row) else None)
            if x_value is None or y_value is None:
                skipped += 1
                continue
            x_values.append(x_value)
            y_values.append(y_value)

        if not x_values:
            continue

        raw_current = np.asarray(y_values, dtype=float)
        initial_outlier = detect_initial_outlier(raw_current)
        time_s, current_a, merged = merge_duplicate_times(
            np.asarray(x_values, dtype=float),
            raw_current,
        )
        baseline_count = min(BASELINE_SAMPLE_COUNT, current_a.size)
        if initial_outlier:
            robust_count = min(10, current_a.size)
            baseline = float(np.median(current_a[:robust_count]))
            baseline_method = f"median_first_{robust_count}_points_initial_outlier"
        else:
            baseline = float(np.mean(current_a[:baseline_count]))
            baseline_method = f"mean_first_{baseline_count}_points"
        current_a = current_a - baseline

        nt_label = tag_match.group("nt")
        curves.append(
            Curve(
                source=source,
                source_label=str(config["label"]),
                source_csv=str(config["csv"]),
                source_folder=str(config["folder"]),
                tag=tag,
                thickness_um=float(tag_match.group("thickness")),
                bias_v=float(tag_match.group("bias")),
                nt_label=nt_label,
                trap_density_cm3=NT_VALUE[nt_label],
                node=int(tag_match.group("node")),
                time_s=time_s,
                current_a=current_a,
                baseline_current_a=baseline,
                baseline_method=baseline_method,
                raw_point_count=len(x_values),
                unique_time_point_count=int(time_s.size),
                skipped_row_count=skipped,
                merged_duplicate_time_count=merged,
                initial_outlier_flag=initial_outlier,
            )
        )
    return sorted(curves, key=lambda item: (item.thickness_um, item.trap_density_cm3))


def load_generation_summary(curve: Curve) -> dict[str, float]:
    path = (
        GENERATION_DIR
        / f"{curve.thickness_um:g}"
        / "output"
        / curve.source_folder
        / "step4_output"
        / "step4_summary.json"
    )
    summary = json.loads(path.read_text(encoding="utf-8"))
    deposited_energy_ev_per_event = float(summary["raw_edep_total_eV"]) / float(summary["n_events"])
    generated_pairs_per_event = deposited_energy_ev_per_event / float(summary["eh_pair_energy_ev"])
    return {
        "collapsed_thickness_um": float(summary.get("collapsed_thickness_um", 1.0)),
        "deposited_energy_ev_per_event": deposited_energy_ev_per_event,
        "generated_pairs_per_event": generated_pairs_per_event,
        "q_gen_c": generated_pairs_per_event * ELEMENTARY_CHARGE_C,
    }


def build_metric_row(curve: Curve) -> dict[str, object]:
    summary = load_generation_summary(curve)
    fwhm_s, peak_a, half_a, peak_time_s, fwhm_status = fwhm_from_signal(curve.time_s, curve.current_a)
    positive = np.clip(curve.current_a, 0.0, None)
    negative = np.clip(-curve.current_a, 0.0, None)
    pos_peak = float(np.max(positive)) if positive.size else 0.0
    neg_peak = float(np.max(negative)) if negative.size else 0.0
    if neg_peak > pos_peak:
        main_signal = negative
        polarity = "negative"
    else:
        main_signal = positive
        polarity = "positive"

    q_signed_2d_c = integrate_trapezoid(curve.current_a, curve.time_s)
    q_main_2d_c = integrate_trapezoid(main_signal, curve.time_s)
    q_signed_c = q_signed_2d_c * summary["collapsed_thickness_um"]
    q_main_c = q_main_2d_c * summary["collapsed_thickness_um"]
    cce_signed_percent = 100.0 * q_signed_c / summary["q_gen_c"]
    cce_main_percent = 100.0 * q_main_c / summary["q_gen_c"]

    return {
        "source": curve.source,
        "source_label": curve.source_label,
        "source_csv": curve.source_csv,
        "tag": curve.tag,
        "thickness_um": curve.thickness_um,
        "bias_v": curve.bias_v,
        "nt_label": curve.nt_label,
        "trap_density_cm3": curve.trap_density_cm3,
        "node": curve.node,
        "dominant_polarity": polarity,
        "Imax_A": peak_a,
        "Imax_nA": peak_a * 1.0e9 if math.isfinite(peak_a) else "",
        "peak_time_ns": peak_time_s * 1.0e9 if math.isfinite(peak_time_s) else "",
        "half_peak_current_nA": half_a * 1.0e9 if math.isfinite(half_a) else "",
        "FWHM_ns": fwhm_s * 1.0e9 if math.isfinite(fwhm_s) else "",
        "t50_ns": charge_time_from_signal(curve.time_s, main_signal, 0.5) * 1.0e9,
        "t90_ns": charge_time_from_signal(curve.time_s, main_signal, 0.9) * 1.0e9,
        "Qcol_signed_C": q_signed_c,
        "Qcol_main_C": q_main_c,
        "CCE_signed_percent": cce_signed_percent,
        "CCE_main_percent": cce_main_percent,
        "q_gen_c": summary["q_gen_c"],
        "generated_pairs_per_event": summary["generated_pairs_per_event"],
        "deposited_energy_ev_per_event": summary["deposited_energy_ev_per_event"],
        "collapsed_thickness_um": summary["collapsed_thickness_um"],
        "baseline_current_a": curve.baseline_current_a,
        "baseline_method": curve.baseline_method,
        "raw_point_count": curve.raw_point_count,
        "unique_time_point_count": curve.unique_time_point_count,
        "skipped_row_count": curve.skipped_row_count,
        "merged_duplicate_time_count": curve.merged_duplicate_time_count,
        "initial_outlier_flag": curve.initial_outlier_flag,
        "fwhm_status": fwhm_status,
        "time_end_ns": float(curve.time_s[-1]) * 1.0e9 if curve.time_s.size else "",
    }


def add_ideal_ratios(rows: list[dict[str, object]]) -> None:
    ideal_by_key: dict[tuple[str, float], dict[str, object]] = {}
    for row in rows:
        if row["nt_label"] == "0":
            ideal_by_key[(str(row["source"]), float(row["thickness_um"]))] = row

    for row in rows:
        key = (str(row["source"]), float(row["thickness_um"]))
        ideal = ideal_by_key.get(key)
        if ideal is None:
            row["Qcol_ideal_C"] = ""
            row["CCE_ideal_percent"] = ""
            row["charge_ratio_to_ideal"] = ""
            row["charge_loss_percent"] = ""
            row["charge_ratio_main_to_ideal"] = ""
            row["charge_loss_main_percent"] = ""
            continue
        q_ideal = float(ideal["Qcol_signed_C"])
        cce_ideal = float(ideal["CCE_signed_percent"])
        q_main_ideal = float(ideal["Qcol_main_C"])
        row["Qcol_ideal_C"] = q_ideal
        row["CCE_ideal_percent"] = cce_ideal
        if q_ideal != 0.0:
            ratio = float(row["Qcol_signed_C"]) / q_ideal
            row["charge_ratio_to_ideal"] = ratio
            row["charge_loss_percent"] = (1.0 - ratio) * 100.0
        else:
            row["charge_ratio_to_ideal"] = ""
            row["charge_loss_percent"] = ""
        if q_main_ideal != 0.0:
            ratio_main = float(row["Qcol_main_C"]) / q_main_ideal
            row["charge_ratio_main_to_ideal"] = ratio_main
            row["charge_loss_main_percent"] = (1.0 - ratio_main) * 100.0
        else:
            row["charge_ratio_main_to_ideal"] = ""
            row["charge_loss_main_percent"] = ""


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def build_coverage(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    coverage_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    by_source: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_source.setdefault(str(row["source"]), []).append(row)

    for source in SOURCE_ORDER:
        source_rows = by_source.get(source, [])
        thicknesses = sorted({float(row["thickness_um"]) for row in source_rows})
        for thickness in thicknesses:
            combo_rows = [row for row in source_rows if math.isclose(float(row["thickness_um"]), thickness)]
            present = sorted({str(row["nt_label"]) for row in combo_rows}, key=NT_ORDER.index)
            missing = [nt for nt in NT_ORDER if nt not in present]
            bias_values = sorted({float(row["bias_v"]) for row in combo_rows})
            coverage_rows.append(
                {
                    "source": source,
                    "source_label": SOURCE_CONFIG[source]["label"],
                    "thickness_um": thickness,
                    "bias_v": bias_values[0] if bias_values else "",
                    "present_nt": ";".join(present),
                    "missing_nt": ";".join(missing),
                    "complete_nt_set": not missing,
                    "curve_count": len(combo_rows),
                }
            )
            for nt in missing:
                missing_rows.append(
                    {
                        "source": source,
                        "source_label": SOURCE_CONFIG[source]["label"],
                        "thickness_um": thickness,
                        "bias_v": bias_values[0] if bias_values else "",
                        "missing_nt": nt,
                    }
                )
    return coverage_rows, missing_rows


def finite_values(rows: list[dict[str, object]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(field, "")
        if value == "":
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    return values


def median(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.median(np.asarray(values, dtype=float)))


def build_trend_tables(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    trend1: list[dict[str, object]] = []
    trend2: list[dict[str, object]] = []
    trend3: list[dict[str, object]] = []

    for source in SOURCE_ORDER:
        source_rows = [row for row in rows if row["source"] == source]
        for nt in NT_ORDER:
            nt_rows = [row for row in source_rows if row["nt_label"] == nt]
            ratios = finite_values(nt_rows, "charge_ratio_to_ideal")
            losses = finite_values(nt_rows, "charge_loss_percent")
            cces = finite_values(nt_rows, "CCE_signed_percent")
            trend1.append(
                {
                    "source": source,
                    "source_label": SOURCE_CONFIG[source]["label"],
                    "nt_label": nt,
                    "trap_density_cm3": NT_VALUE[nt],
                    "curve_count": len(nt_rows),
                    "median_charge_ratio_to_ideal": median(ratios),
                    "min_charge_ratio_to_ideal": min(ratios) if ratios else "",
                    "max_charge_ratio_to_ideal": max(ratios) if ratios else "",
                    "median_charge_loss_percent": median(losses),
                    "min_charge_loss_percent": min(losses) if losses else "",
                    "max_charge_loss_percent": max(losses) if losses else "",
                    "median_CCE_percent": median(cces),
                    "min_CCE_percent": min(cces) if cces else "",
                    "max_CCE_percent": max(cces) if cces else "",
                }
            )

        for nt in NT_ORDER[1:]:
            nt_rows = [row for row in source_rows if row["nt_label"] == nt]
            thin_rows = [row for row in nt_rows if float(row["thickness_um"]) <= 30.0]
            thick_rows = [row for row in nt_rows if float(row["thickness_um"]) >= 80.0]
            thin_losses = finite_values(thin_rows, "charge_loss_percent")
            thick_losses = finite_values(thick_rows, "charge_loss_percent")
            trend2.append(
                {
                    "source": source,
                    "source_label": SOURCE_CONFIG[source]["label"],
                    "nt_label": nt,
                    "trap_density_cm3": NT_VALUE[nt],
                    "thin_range": "<=30 um",
                    "thick_range": ">=80 um",
                    "thin_count": len(thin_losses),
                    "thick_count": len(thick_losses),
                    "median_thin_charge_loss_percent": median(thin_losses),
                    "median_thick_charge_loss_percent": median(thick_losses),
                    "thick_minus_thin_loss_pp": median(thick_losses) - median(thin_losses)
                    if thin_losses and thick_losses
                    else "",
                }
            )

        for nt in NT_ORDER:
            nt_rows = [row for row in source_rows if row["nt_label"] == nt]
            if not nt_rows:
                continue
            best = max(nt_rows, key=lambda row: float(row["CCE_signed_percent"]))
            thicknesses = sorted(float(row["thickness_um"]) for row in nt_rows)
            at_boundary = math.isclose(float(best["thickness_um"]), min(thicknesses)) or math.isclose(
                float(best["thickness_um"]), max(thicknesses)
            )
            trend3.append(
                {
                    "source": source,
                    "source_label": SOURCE_CONFIG[source]["label"],
                    "nt_label": nt,
                    "trap_density_cm3": NT_VALUE[nt],
                    "curve_count": len(nt_rows),
                    "best_thickness_um": float(best["thickness_um"]),
                    "best_bias_v": float(best["bias_v"]),
                    "best_CCE_percent": float(best["CCE_signed_percent"]),
                    "best_charge_ratio_to_ideal": best.get("charge_ratio_to_ideal", ""),
                    "is_boundary_best": at_boundary,
                    "available_min_thickness_um": min(thicknesses),
                    "available_max_thickness_um": max(thicknesses),
                }
            )

    return trend1, trend2, trend3


def setup_axis(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=10.5)
    ax.set_ylabel(ylabel, fontsize=10.5)
    ax.grid(True, which="major", linestyle=":", linewidth=0.65, color=GRID)
    ax.tick_params(axis="both", colors=TEXT, labelsize=8.8)
    for spine in ax.spines.values():
        spine.set_color(TEXT)


def source_filename(source: str) -> str:
    return source.replace(".", "p").replace("-", "_")


def plot_trend1(rows: list[dict[str, object]]) -> None:
    x = np.arange(len(NT_ORDER))
    for source in SOURCE_ORDER:
        source_rows = [row for row in rows if row["source"] == source]
        if not source_rows:
            continue
        fig, ax = plt.subplots(figsize=(7.8, 5.0), dpi=220)
        thicknesses = sorted({float(row["thickness_um"]) for row in source_rows})
        for thickness in thicknesses:
            t_rows = {str(row["nt_label"]): row for row in source_rows if math.isclose(float(row["thickness_um"]), thickness)}
            y = []
            for nt in NT_ORDER:
                row = t_rows.get(nt)
                y.append(float(row["CCE_signed_percent"]) if row and row.get("CCE_signed_percent") != "" else np.nan)
            ax.plot(x, y, marker="o", linewidth=1.2, markersize=3.5, alpha=0.72, label=f"{thickness:g} um")
        ax.set_xticks(x)
        ax.set_xticklabels([NT_LABEL[nt] for nt in NT_ORDER])
        setup_axis(
            ax,
            f"Trend 1: CCE vs trap density ({SOURCE_CONFIG[source]['label']})",
            "Trap density Nt (cm^-3)",
            "CCE from signed integrated current (%)",
        )
        ax.legend(loc="best", fontsize=7.0, ncol=2, frameon=True, framealpha=0.92)
        fig.tight_layout()
        fig.savefig(TREND1_DIR / f"trend1_charge_ratio_vs_Nt_{source_filename(source)}.png", bbox_inches="tight")
        fig.savefig(TREND1_DIR / f"trend1_charge_ratio_vs_Nt_{source_filename(source)}.svg", bbox_inches="tight")
        plt.close(fig)


def plot_trend2(rows: list[dict[str, object]]) -> None:
    for source in SOURCE_ORDER:
        source_rows = [row for row in rows if row["source"] == source]
        if not source_rows:
            continue
        fig, ax = plt.subplots(figsize=(7.8, 5.0), dpi=220)
        for nt in NT_ORDER[1:]:
            nt_rows = sorted(
                [row for row in source_rows if row["nt_label"] == nt and row.get("charge_loss_percent") != ""],
                key=lambda row: float(row["thickness_um"]),
            )
            if not nt_rows:
                continue
            ax.plot(
                [float(row["thickness_um"]) for row in nt_rows],
                [float(row["charge_loss_percent"]) for row in nt_rows],
                marker="o",
                linewidth=1.8,
                markersize=4.0,
                label=f"Nt={nt}",
            )
        ax.axhline(0.0, color=MUTED, linewidth=1.0, linestyle="--")
        setup_axis(
            ax,
            f"Trend 2: charge loss vs thickness ({SOURCE_CONFIG[source]['label']})",
            "i-region thickness (um)",
            "Charge loss relative to Nt=0 (%)",
        )
        ax.legend(loc="best", fontsize=8.0, frameon=True, framealpha=0.92)
        fig.tight_layout()
        fig.savefig(TREND2_DIR / f"trend2_charge_loss_vs_thickness_{source_filename(source)}.png", bbox_inches="tight")
        fig.savefig(TREND2_DIR / f"trend2_charge_loss_vs_thickness_{source_filename(source)}.svg", bbox_inches="tight")
        plt.close(fig)


def plot_trend3(rows: list[dict[str, object]]) -> None:
    for source in SOURCE_ORDER:
        source_rows = [row for row in rows if row["source"] == source]
        if not source_rows:
            continue
        fig, ax = plt.subplots(figsize=(7.8, 5.0), dpi=220)
        for nt in NT_ORDER:
            nt_rows = sorted(
                [row for row in source_rows if row["nt_label"] == nt],
                key=lambda row: float(row["thickness_um"]),
            )
            if not nt_rows:
                continue
            ax.plot(
                [float(row["thickness_um"]) for row in nt_rows],
                [float(row["CCE_signed_percent"]) for row in nt_rows],
                marker="o",
                linewidth=1.8,
                markersize=4.0,
                label=f"Nt={nt}",
            )
        setup_axis(
            ax,
            f"Trend 3: CCE vs thickness by Nt ({SOURCE_CONFIG[source]['label']})",
            "i-region thickness (um)",
            "CCE from signed integrated current (%)",
        )
        ax.legend(loc="best", fontsize=8.0, frameon=True, framealpha=0.92)
        fig.tight_layout()
        fig.savefig(TREND3_DIR / f"trend3_CCE_vs_thickness_by_Nt_{source_filename(source)}.png", bbox_inches="tight")
        fig.savefig(TREND3_DIR / f"trend3_CCE_vs_thickness_by_Nt_{source_filename(source)}.svg", bbox_inches="tight")
        plt.close(fig)


def plot_overview(trend1: list[dict[str, object]], trend2: list[dict[str, object]], trend3: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=220)

    for source in SOURCE_ORDER:
        rows = [row for row in trend1 if row["source"] == source]
        if not rows:
            continue
        axes[0].plot(
            np.arange(len(rows)),
            [float(row["median_CCE_percent"]) for row in rows],
            marker="o",
            linewidth=1.4,
            markersize=3.6,
            label=SOURCE_CONFIG[source]["label"],
        )
    axes[0].set_xticks(np.arange(len(NT_ORDER)))
    axes[0].set_xticklabels(NT_ORDER)
    setup_axis(axes[0], "Trend 1 overview", "Nt (cm^-3)", "Median CCE (%)")

    for source in SOURCE_ORDER:
        rows = [row for row in trend2 if row["source"] == source and row["nt_label"] == "5e13"]
        if not rows:
            continue
        row = rows[0]
        if row["thick_minus_thin_loss_pp"] == "":
            continue
        axes[1].bar(
            SOURCE_CONFIG[source]["label"],
            float(row["thick_minus_thin_loss_pp"]),
            color=SOURCE_CONFIG[source]["color"],
            alpha=0.82,
        )
    axes[1].tick_params(axis="x", labelrotation=45)
    setup_axis(axes[1], "Trend 2 overview (Nt=5e13)", "Source", "Thick loss - thin loss (pp)")

    for source in SOURCE_ORDER:
        rows = [row for row in trend3 if row["source"] == source]
        if not rows:
            continue
        axes[2].plot(
            np.arange(len(rows)),
            [float(row["best_thickness_um"]) for row in rows],
            marker="o",
            linewidth=1.4,
            markersize=3.6,
            label=SOURCE_CONFIG[source]["label"],
        )
    axes[2].set_xticks(np.arange(len(NT_ORDER)))
    axes[2].set_xticklabels(NT_ORDER)
    setup_axis(axes[2], "Trend 3 overview", "Nt (cm^-3)", "Best thickness in available data (um)")

    axes[0].legend(loc="best", fontsize=6.8, frameon=True)
    axes[2].legend(loc="best", fontsize=6.8, frameon=True)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "kim_validation_overview.png", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "kim_validation_overview.svg", bbox_inches="tight")
    plt.close(fig)


def format_float(value: object, digits: int = 3) -> str:
    if value == "" or value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "nan"
    return f"{number:.{digits}f}"


def write_report(
    rows: list[dict[str, object]],
    coverage_rows: list[dict[str, object]],
    missing_rows: list[dict[str, object]],
    trend1: list[dict[str, object]],
    trend2: list[dict[str, object]],
    trend3: list[dict[str, object]],
) -> None:
    total_curves = len(rows)
    outlier_count = sum(1 for row in rows if row.get("initial_outlier_flag"))
    completed_count = sum(1 for row in coverage_rows if row.get("complete_nt_set"))
    incomplete_count = len(coverage_rows) - completed_count

    lines: list[str] = []
    lines.append("# Kim 2025 文献趋势验证初步结果")
    lines.append("")
    lines.append("本报告基于 `SiC_electron/raw_data/tcad_it/*.csv` 中当前已经完成的 TCAD i-t 数据生成。由于厚度扫描尚未全部完成，所有结论均只针对当前已有数据。")
    lines.append("")
    lines.append("## 输出文件")
    lines.append("")
    lines.append("- 指标总表：`../data/validation_metrics.csv`")
    lines.append("- 覆盖矩阵：`../data/coverage_matrix.csv`")
    lines.append("- 缺失组合：`../data/missing_combinations.csv`")
    lines.append("- 趋势 1 统计：`../data/trend1_charge_vs_Nt_summary.csv`")
    lines.append("- 趋势 2 统计：`../data/trend2_loss_vs_thickness_summary.csv`")
    lines.append("- 趋势 3 统计：`../data/trend3_best_thickness_summary.csv`")
    lines.append("- 图件目录：`../figures/`")
    lines.append("")
    lines.append("## 数据覆盖")
    lines.append("")
    lines.append(f"- 已解析曲线数：`{total_curves}`")
    lines.append(f"- 完整 `Nt=0,1e11,1e12,1e13,5e13` 组合数：`{completed_count}`")
    lines.append(f"- 不完整厚度组合数：`{incomplete_count}`")
    lines.append(f"- 初始点疑似异常曲线数：`{outlier_count}`")
    lines.append("")
    if outlier_count:
        lines.append("首点疑似异常曲线没有丢弃；主验证计算对这些曲线使用 `median_first_10_points_initial_outlier` 基线，其余曲线保持前 5 点均值基线。")
        lines.append("")
    if missing_rows:
        lines.append("未完成组合已写入 `missing_combinations.csv`。趋势图会保留缺失处的断点，不做插值。")
        lines.append("")

    lines.append("## 趋势 1：trap density 增大时积分电荷变化")
    lines.append("")
    lines.append("对照 Kim 2025 Fig.7：trap density 增大时，JSC/VOC/Pout_max 下降。本项目用瞬态积分得到的 `CCE = Qcol/Qgen` 对应这个趋势。`Qcol(Nt)/Qcol(Nt=0)` 仍保留在统计表中用于归一化检查。")
    lines.append("")
    lines.append("| Source | Nt=0 | Nt=1e11 | Nt=1e12 | Nt=1e13 | Nt=5e13 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for source in SOURCE_ORDER:
        values = {}
        for nt in NT_ORDER:
            row = next((item for item in trend1 if item["source"] == source and item["nt_label"] == nt), None)
            values[nt] = format_float(row["median_CCE_percent"], 2) if row else ""
        lines.append(
            f"| {SOURCE_CONFIG[source]['label']} | {values['0']} | {values['1e11']} | {values['1e12']} | {values['1e13']} | {values['5e13']} |"
        )
    lines.append("")

    lines.append("## 趋势 2：厚器件更受 trap 影响")
    lines.append("")
    lines.append("对照 Kim 2025 Fig.6：薄 i-layer 下 trap/no-trap 差别小，厚 i-layer 下差别扩大。这里比较 `<=30 um` 和 `>=80 um` 的中位 charge loss。")
    lines.append("")
    lines.append("注意：overview 第二幅图只画 `thick loss - thin loss` 的中位差，把完整厚度曲线压缩成一个柱子，所以不同源项之间的视觉差异会被弱化。判断趋势 2 时应优先看各源项子图 `trend2_loss_vs_thickness/*.png`，而不是只看 overview 柱状图。")
    lines.append("")
    lines.append("| Source | Nt | thin loss % | thick loss % | thick-thin pp |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for source in SOURCE_ORDER:
        for nt in ["1e13", "5e13"]:
            row = next((item for item in trend2 if item["source"] == source and item["nt_label"] == nt), None)
            if not row:
                continue
            lines.append(
                f"| {SOURCE_CONFIG[source]['label']} | {nt} | "
                f"{format_float(row['median_thin_charge_loss_percent'], 2)} | "
                f"{format_float(row['median_thick_charge_loss_percent'], 2)} | "
                f"{format_float(row['thick_minus_thin_loss_pp'], 2)} |"
            )
    lines.append("")

    lines.append("## 趋势 3：有缺陷后 CCE vs thickness 的最优厚度")
    lines.append("")
    lines.append("对照 Kim 2025 Fig.6：无 trap 时性能随 i-layer 厚度增加并趋于饱和；有 trap 时存在最优厚度，厚端可能下降。下表列出当前已有数据中 CCE 最大的厚度。若 `boundary=True`，说明最大值落在当前扫描边界，不能断言已经看到 rollover。")
    lines.append("")
    lines.append("| Source | Nt=0 | Nt=1e12 | Nt=1e13 | Nt=5e13 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for source in SOURCE_ORDER:
        values = {}
        for nt in ["0", "1e12", "1e13", "5e13"]:
            row = next((item for item in trend3 if item["source"] == source and item["nt_label"] == nt), None)
            if row:
                suffix = " boundary" if row["is_boundary_best"] else ""
                values[nt] = f"{format_float(row['best_thickness_um'], 0)} um{suffix}"
            else:
                values[nt] = ""
        lines.append(
            f"| {SOURCE_CONFIG[source]['label']} | {values['0']} | {values['1e12']} | {values['1e13']} | {values['5e13']} |"
        )
    lines.append("")

    lines.append("## 解释口径")
    lines.append("")
    lines.append("本次验证使用 `Qcol` 和 `CCE`，不使用 `Imax` 作为主要退化指标。`Imax` 可能受电场重分布、脉冲变窄和感应电流形状影响，不能直接等同于总收集电荷。")
    lines.append("")
    lines.append("建议在论文中表述为：虽然 Kim 2025 使用 betavoltaic J-V 提取，而本文使用瞬态电流积分，但积分收集电荷随 trap density 增加而下降、厚 i 区退化更明显、以及高缺陷条件下最优厚度前移的趋势，可作为一致的 trap-assisted recombination 证据链。")
    lines.append("")

    (REPORT_DIR / "Kim2025_trend_validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    for directory in [DATA_DIR, REPORT_DIR, FIGURE_DIR, TREND1_DIR, TREND2_DIR, TREND3_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    curves: list[Curve] = []
    for source in SOURCE_ORDER:
        curves.extend(read_source_curves(source))

    rows = [build_metric_row(curve) for curve in curves]
    add_ideal_ratios(rows)
    rows.sort(key=lambda row: (SOURCE_ORDER.index(str(row["source"])), float(row["thickness_um"]), NT_ORDER.index(str(row["nt_label"]))))

    coverage_rows, missing_rows = build_coverage(rows)
    trend1, trend2, trend3 = build_trend_tables(rows)

    metric_fields = [
        "source",
        "source_label",
        "source_csv",
        "tag",
        "thickness_um",
        "bias_v",
        "nt_label",
        "trap_density_cm3",
        "node",
        "dominant_polarity",
        "Imax_A",
        "Imax_nA",
        "peak_time_ns",
        "half_peak_current_nA",
        "FWHM_ns",
        "t50_ns",
        "t90_ns",
        "Qcol_signed_C",
        "Qcol_main_C",
        "Qcol_ideal_C",
        "CCE_signed_percent",
        "CCE_main_percent",
        "CCE_ideal_percent",
        "charge_ratio_to_ideal",
        "charge_loss_percent",
        "charge_ratio_main_to_ideal",
        "charge_loss_main_percent",
        "q_gen_c",
        "generated_pairs_per_event",
        "deposited_energy_ev_per_event",
        "collapsed_thickness_um",
        "baseline_current_a",
        "baseline_method",
        "raw_point_count",
        "unique_time_point_count",
        "skipped_row_count",
        "merged_duplicate_time_count",
        "initial_outlier_flag",
        "fwhm_status",
        "time_end_ns",
    ]
    write_csv(DATA_DIR / "validation_metrics.csv", rows, metric_fields)
    write_csv(DATA_DIR / "coverage_matrix.csv", coverage_rows)
    write_csv(DATA_DIR / "missing_combinations.csv", missing_rows, ["source", "source_label", "thickness_um", "bias_v", "missing_nt"])
    write_csv(DATA_DIR / "trend1_charge_vs_Nt_summary.csv", trend1)
    write_csv(DATA_DIR / "trend2_loss_vs_thickness_summary.csv", trend2)
    write_csv(DATA_DIR / "trend3_best_thickness_summary.csv", trend3)

    (DATA_DIR / "validation_metrics.json").write_text(
        json.dumps(
            {
                "summary": {
                    "input_dir": str(TCAD_IT_DIR),
                    "baseline_sample_count": BASELINE_SAMPLE_COUNT,
                    "sources": SOURCE_ORDER,
                    "nt_order": NT_ORDER,
                    "metric_used_for_primary_validation": "Qcol_signed_C, CCE_signed_percent, charge_ratio_to_ideal, charge_loss_percent",
                    "note": "Current files do not cover all thicknesses and Nt combinations; missing combinations are listed separately.",
                },
                "coverage": coverage_rows,
                "missing": missing_rows,
                "metrics": rows,
                "trend1": trend1,
                "trend2": trend2,
                "trend3": trend3,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    plot_trend1(rows)
    plot_trend2(rows)
    plot_trend3(rows)
    plot_overview(trend1, trend2, trend3)
    write_report(rows, coverage_rows, missing_rows, trend1, trend2, trend3)

    print(f"Saved metrics: {DATA_DIR / 'validation_metrics.csv'}")
    print(f"Saved report: {REPORT_DIR / 'Kim2025_trend_validation_report.md'}")
    print(f"Saved figures: {FIGURE_DIR}")
    print(f"Parsed curves: {len(rows)}")
    print(f"Missing combinations: {len(missing_rows)}")
    print(f"Initial outlier flags: {sum(1 for row in rows if row.get('initial_outlier_flag'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
