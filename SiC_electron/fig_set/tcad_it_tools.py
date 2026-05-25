"""Utilities for extracting CCE metrics from Sentaurus transient currents."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
TCAD_IT_DIR = ROOT_DIR / "raw_data" / "tcad_it"
GENERATION_DIR = ROOT_DIR / "generation"

ELEMENTARY_CHARGE_C = 1.602176634e-19
INITIAL_SAMPLE_DROP_COUNT = 6
BASELINE_SAMPLE_COUNT = 5

THICKNESS_ORDER = [float(item) for item in range(10, 131, 10)]
NT_ORDER = ["0", "1e11", "1e12", "1e13", "5e13"]
NT_VALUE = {"0": 0.0, "1e11": 1e11, "1e12": 1e12, "1e13": 1e13, "5e13": 5e13}
NT_LABEL = {
    "0": "0",
    "1e11": r"$10^{11}$",
    "1e12": r"$10^{12}$",
    "1e13": r"$10^{13}$",
    "5e13": r"$5\times10^{13}$",
}

SOURCE_ORDER = ["20kev", "49kev", "100kev", "156p5kev", "c14"]
SOURCE_CONFIG = {
    "20kev": {"label": "20 keV", "csv": "20kev.csv", "folder": "20keV", "color": "#4C78A8"},
    "49kev": {"label": "49 keV", "csv": "49kev.csv", "folder": "49keV", "color": "#59A14F"},
    "100kev": {"label": "100 keV", "csv": "100kev.csv", "folder": "100keV", "color": "#F28E2B"},
    "156p5kev": {
        "label": "156.5 keV",
        "csv": "156p5kev.csv",
        "folder": "156p5keV",
        "color": "#E15759",
    },
    "c14": {"label": "C-14 spectrum", "csv": "c14.csv", "folder": "c14", "color": "#111827"},
}

HEADER_RE = re.compile(r"TotalCurrent\((?P<tag>[^)]*)\)\s+(?P<axis>[XY])$")
TAG_RE = re.compile(
    r"(?P<thickness>\d+(?:\.\d+)?)um_"
    r"(?P<bias>\d+(?:\.\d+)?)V_"
    r"Nt(?P<nt>0|1e11|1e12|1e13|5e13)"
    r"n(?P<node>\d+)_des"
)


@dataclass(frozen=True)
class Curve:
    source: str
    tag: str
    thickness_um: float
    bias_v: float
    nt: str
    node: int
    time_s: np.ndarray
    current_a: np.ndarray
    raw_points: int
    unique_points: int
    dropped_initial_points: int


def _parse_float(value: str | None) -> float | None:
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


def _integrate(y_values: np.ndarray, x_values: np.ndarray) -> float:
    if y_values.size < 2:
        return 0.0
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y_values, x_values))
    return float(np.trapz(y_values, x_values))


def _merge_duplicate_times(x_values: np.ndarray, y_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x_values, kind="mergesort")
    sorted_x = x_values[order]
    sorted_y = y_values[order]
    unique_x, inverse, counts = np.unique(sorted_x, return_inverse=True, return_counts=True)
    y_sum = np.zeros(unique_x.shape, dtype=float)
    np.add.at(y_sum, inverse, sorted_y)
    return unique_x, y_sum / counts


def _read_source_curves(source: str) -> list[Curve]:
    config = SOURCE_CONFIG[source]
    csv_path = TCAD_IT_DIR / str(config["csv"])
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        headers = next(reader)
        rows = list(reader)

    curves: list[Curve] = []
    for col in range(0, len(headers), 2):
        if col + 1 >= len(headers):
            continue
        match_x = HEADER_RE.search(headers[col])
        match_y = HEADER_RE.search(headers[col + 1])
        if not match_x or not match_y or match_x.group("tag") != match_y.group("tag"):
            continue
        tag = match_x.group("tag")
        tag_match = TAG_RE.fullmatch(tag)
        if not tag_match:
            continue

        x_values: list[float] = []
        y_values: list[float] = []
        for row in rows:
            if col + 1 >= len(row):
                continue
            x_value = _parse_float(row[col])
            y_value = _parse_float(row[col + 1])
            if x_value is None or y_value is None:
                continue
            x_values.append(x_value)
            y_values.append(y_value)
        if len(x_values) < 2:
            continue

        time_s, current_a = _merge_duplicate_times(np.array(x_values), np.array(y_values))
        dropped_initial_points = min(INITIAL_SAMPLE_DROP_COUNT, max(int(time_s.size) - 1, 0))
        if dropped_initial_points:
            time_s = time_s[dropped_initial_points:]
            current_a = current_a[dropped_initial_points:]
        if time_s.size >= BASELINE_SAMPLE_COUNT:
            baseline = float(np.mean(current_a[:BASELINE_SAMPLE_COUNT]))
            current_a = current_a - baseline

        curves.append(
            Curve(
                source=source,
                tag=tag,
                thickness_um=float(tag_match.group("thickness")),
                bias_v=float(tag_match.group("bias")),
                nt=tag_match.group("nt"),
                node=int(tag_match.group("node")),
                time_s=time_s,
                current_a=current_a,
                raw_points=len(x_values),
                unique_points=int(time_s.size),
                dropped_initial_points=dropped_initial_points,
            )
        )
    return sorted(curves, key=lambda item: (item.thickness_um, NT_VALUE[item.nt]))


def _generation_summary(source: str, thickness_um: float) -> dict[str, float]:
    config = SOURCE_CONFIG[source]
    path = (
        GENERATION_DIR
        / f"{thickness_um:g}"
        / "output"
        / str(config["folder"])
        / "step4_output"
        / "step4_summary.json"
    )
    summary = json.loads(path.read_text(encoding="utf-8"))
    deposited_energy_ev = float(summary["raw_edep_total_eV"]) / float(summary["n_events"])
    generated_pairs = deposited_energy_ev / float(summary["eh_pair_energy_ev"])
    return {
        "collapsed_thickness_um": float(summary.get("collapsed_thickness_um", 1.0)),
        "deposited_energy_ev": deposited_energy_ev,
        "generated_pairs": generated_pairs,
        "q_gen_c": generated_pairs * ELEMENTARY_CHARGE_C,
    }


def extract_metrics() -> list[dict[str, object]]:
    """Return one CCE metric row for each source/thickness/Nt curve."""

    rows: list[dict[str, object]] = []
    for source in SOURCE_ORDER:
        for curve in _read_source_curves(source):
            summary = _generation_summary(source, curve.thickness_um)
            positive = np.clip(curve.current_a, 0.0, None)
            negative = np.clip(-curve.current_a, 0.0, None)
            pos_peak = float(np.max(positive)) if positive.size else 0.0
            neg_peak = float(np.max(negative)) if negative.size else 0.0
            q_signed_2d_c = _integrate(curve.current_a, curve.time_s)
            q_pos_2d_c = _integrate(positive, curve.time_s)
            q_neg_2d_c = _integrate(negative, curve.time_s)
            q_2d_c = abs(q_signed_2d_c)
            q_col_c = q_2d_c * summary["collapsed_thickness_um"]
            cce_percent = 100.0 * q_col_c / summary["q_gen_c"]
            q_pos_c = q_pos_2d_c * summary["collapsed_thickness_um"]
            q_neg_c = q_neg_2d_c * summary["collapsed_thickness_um"]
            reverse_peak_ratio = neg_peak / pos_peak if pos_peak > 0.0 else 0.0
            qc_reasons: list[str] = []
            if cce_percent < 0.0 or cce_percent > 100.05:
                qc_reasons.append("cce_out_of_physical_range")
            if reverse_peak_ratio > 0.10:
                qc_reasons.append("large_reverse_spike")
            qc_pass = not qc_reasons

            rows.append(
                {
                    "source": source,
                    "source_label": SOURCE_CONFIG[source]["label"],
                    "thickness_um": curve.thickness_um,
                    "bias_v": curve.bias_v,
                    "nt": curve.nt,
                    "trap_density_cm3": NT_VALUE[curve.nt],
                    "node": curve.node,
                    "q_col_c": q_col_c,
                    "q_pos_c": q_pos_c,
                    "q_neg_c": q_neg_c,
                    "q_gen_c": summary["q_gen_c"],
                    "cce_percent": cce_percent,
                    "cce_pos_percent": 100.0 * q_pos_c / summary["q_gen_c"],
                    "cce_neg_percent": 100.0 * q_neg_c / summary["q_gen_c"],
                    "positive_peak_nA": pos_peak * 1e9,
                    "negative_peak_nA": neg_peak * 1e9,
                    "reverse_peak_ratio": reverse_peak_ratio,
                    "qc_pass": qc_pass,
                    "qc_reason": ";".join(qc_reasons),
                    "deposited_energy_ev": summary["deposited_energy_ev"],
                    "generated_pairs": summary["generated_pairs"],
                    "collapsed_thickness_um": summary["collapsed_thickness_um"],
                    "raw_points": curve.raw_points,
                    "unique_points": curve.unique_points,
                    "dropped_initial_points": curve.dropped_initial_points,
                }
            )
    return rows


def best_by_source_nt(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    best_rows: list[dict[str, object]] = []
    for source in SOURCE_ORDER:
        for nt in NT_ORDER:
            candidates = [
                row for row in rows if row["source"] == source and row["nt"] == nt and bool(row.get("qc_pass", True))
            ]
            if not candidates:
                continue
            best = max(candidates, key=lambda row: float(row["cce_percent"]))
            best_rows.append({**best})
    return best_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
