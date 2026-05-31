#!/usr/bin/env python3
"""Trend 3 best-CCE statistics for paper-selected sources.

Selected sources:
  20 keV, 49 keV, 100 keV, 156.5 keV, C-14 spectrum.

The best CCE is defined within each source and Nt group as the maximum
CCE_signed_percent among all currently available thickness points.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path


SELECTED_SOURCES = ["20kev", "49kev", "100kev", "156p5", "c14"]
SOURCE_LABELS = {
    "20kev": "20 keV",
    "49kev": "49 keV",
    "100kev": "100 keV",
    "156p5": "156.5 keV",
    "c14": "C-14 spectrum",
}
NT_ORDER = ["0", "1e11", "1e12", "1e13", "5e13"]
NT_VALUE = {
    "0": 0.0,
    "1e11": 1e11,
    "1e12": 1e12,
    "1e13": 1e13,
    "5e13": 5e13,
}
CCE_COLUMN = "CCE_signed_percent"


def as_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def format_num(value: float, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def source_sort_key(source: str) -> int:
    return SELECTED_SOURCES.index(source) if source in SELECTED_SOURCES else 999


def build_best_rows(metrics_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in metrics_rows:
        source = row.get("source", "")
        nt_label = row.get("nt_label", "")
        if source in SELECTED_SOURCES and nt_label in NT_ORDER:
            grouped[(source, nt_label)].append(row)

    best_rows: list[dict] = []
    for source in SELECTED_SOURCES:
        for nt_label in NT_ORDER:
            group_rows = grouped.get((source, nt_label), [])
            if not group_rows:
                continue

            best = max(group_rows, key=lambda row: as_float(row.get(CCE_COLUMN, "")))
            thicknesses = sorted(as_float(row["thickness_um"]) for row in group_rows)
            best_thickness = as_float(best["thickness_um"])
            is_boundary = math.isclose(best_thickness, thicknesses[0]) or math.isclose(best_thickness, thicknesses[-1])

            best_rows.append(
                {
                    "source": source,
                    "source_label": SOURCE_LABELS[source],
                    "nt_label": nt_label,
                    "trap_density_cm3": f"{NT_VALUE[nt_label]:.12g}",
                    "curve_count": str(len(group_rows)),
                    "best_thickness_um": format_num(best_thickness, 6),
                    "best_bias_v": format_num(as_float(best["bias_v"]), 6),
                    "best_CCE_percent": format_num(as_float(best[CCE_COLUMN]), 9),
                    "best_Qcol_signed_C": f"{as_float(best['Qcol_signed_C']):.12e}",
                    "best_charge_ratio_to_same_thickness_ideal": format_num(as_float(best["charge_ratio_to_ideal"]), 9),
                    "is_boundary_best": str(is_boundary),
                    "available_min_thickness_um": format_num(thicknesses[0], 6),
                    "available_max_thickness_um": format_num(thicknesses[-1], 6),
                    "missing_highest_thickness_for_this_nt": str(thicknesses[-1] < 100.0 and source in {"156p5", "c14"}),
                }
            )

    baseline_by_source = {
        row["source"]: row
        for row in best_rows
        if row["nt_label"] == "0"
    }
    for row in best_rows:
        base = baseline_by_source.get(row["source"])
        if not base:
            row["shift_from_Nt0_best_um"] = ""
            row["best_CCE_drop_from_Nt0_best_pp"] = ""
            continue
        row["shift_from_Nt0_best_um"] = format_num(
            as_float(row["best_thickness_um"]) - as_float(base["best_thickness_um"]),
            6,
        )
        row["best_CCE_drop_from_Nt0_best_pp"] = format_num(
            as_float(base["best_CCE_percent"]) - as_float(row["best_CCE_percent"]),
            9,
        )

    return best_rows


def build_pivot_rows(best_rows: list[dict]) -> list[dict]:
    by_source_nt = {(row["source"], row["nt_label"]): row for row in best_rows}
    pivot_rows: list[dict] = []
    for source in SELECTED_SOURCES:
        out = {
            "source": source,
            "source_label": SOURCE_LABELS[source],
        }
        for nt in NT_ORDER:
            row = by_source_nt.get((source, nt))
            if row:
                suffix = " boundary" if row["is_boundary_best"] == "True" else ""
                out[f"best_thickness_Nt{nt}_um"] = row["best_thickness_um"] + suffix
                out[f"best_CCE_Nt{nt}_percent"] = row["best_CCE_percent"]
            else:
                out[f"best_thickness_Nt{nt}_um"] = ""
                out[f"best_CCE_Nt{nt}_percent"] = ""
        pivot_rows.append(out)
    return pivot_rows


def median(values: list[float]) -> float:
    values = sorted(values)
    if not values:
        return math.nan
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def build_aggregate_rows(best_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for nt in NT_ORDER:
        nt_rows = [row for row in best_rows if row["nt_label"] == nt]
        best_thicknesses = [as_float(row["best_thickness_um"]) for row in nt_rows]
        best_cces = [as_float(row["best_CCE_percent"]) for row in nt_rows]
        shifts = [as_float(row["shift_from_Nt0_best_um"]) for row in nt_rows]
        drops = [as_float(row["best_CCE_drop_from_Nt0_best_pp"]) for row in nt_rows]
        rows.append(
            {
                "nt_label": nt,
                "trap_density_cm3": f"{NT_VALUE[nt]:.12g}",
                "source_count": str(len(nt_rows)),
                "median_best_thickness_um": format_num(median(best_thicknesses), 6),
                "mean_best_thickness_um": format_num(sum(best_thicknesses) / len(best_thicknesses), 6),
                "median_shift_from_Nt0_best_um": format_num(median(shifts), 6),
                "mean_shift_from_Nt0_best_um": format_num(sum(shifts) / len(shifts), 6),
                "median_best_CCE_percent": format_num(median(best_cces), 9),
                "mean_best_CCE_percent": format_num(sum(best_cces) / len(best_cces), 9),
                "median_best_CCE_drop_from_Nt0_best_pp": format_num(median(drops), 9),
                "mean_best_CCE_drop_from_Nt0_best_pp": format_num(sum(drops) / len(drops), 9),
                "boundary_best_count": str(sum(row["is_boundary_best"] == "True" for row in nt_rows)),
                "internal_best_count": str(sum(row["is_boundary_best"] == "False" for row in nt_rows)),
            }
        )
    return rows


def write_report(path: Path, best_rows: list[dict], pivot_rows: list[dict], aggregate_rows: list[dict]) -> None:
    lines = [
        "# Trend 3 selected-source best CCE statistics",
        "",
        "Selected sources: 20 keV, 49 keV, 100 keV, 156.5 keV, and C-14 spectrum.",
        "",
        "Best CCE definition: within each source and Nt group, choose the available thickness with maximum `CCE_signed_percent`.",
        "A `boundary` label means the maximum lies at the scanned thickness edge, so the true optimum may lie outside the current scan range.",
        "",
        "## Best thickness and best CCE",
        "",
        "| Source | Nt=0 | Nt=1e11 | Nt=1e12 | Nt=1e13 | Nt=5e13 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for row in pivot_rows:
        values = []
        for nt in NT_ORDER:
            thickness = row[f"best_thickness_Nt{nt}_um"]
            cce = row[f"best_CCE_Nt{nt}_percent"]
            values.append(f"{thickness}, CCE={as_float(cce):.2f}%")
        lines.append(f"| {row['source_label']} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Aggregate trend across selected sources",
            "",
            "| Nt | median best thickness um | mean best thickness um | median shift from Nt=0 um | median best CCE % | median best-CCE drop pp | boundary/internal |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in aggregate_rows:
        lines.append(
            f"| {row['nt_label']} | {row['median_best_thickness_um']} | {row['mean_best_thickness_um']} | "
            f"{row['median_shift_from_Nt0_best_um']} | {row['median_best_CCE_percent']} | "
            f"{row['median_best_CCE_drop_from_Nt0_best_pp']} | "
            f"{row['boundary_best_count']}/{row['internal_best_count']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The selected-source set keeps the cases most relevant to the manuscript argument and excludes 10 keV / 30 keV from the main Trend 3 statistics.",
            "- In the current data window, 100 keV and C-14 show clear internal optimum shifts under high Nt.",
            "- 20 keV and 49 keV move to the low-thickness edge under high Nt, which still supports trap-limited thinning but should be described as a boundary optimum.",
            "- 156.5 keV still has boundary sensitivity at Nt=5e13 because the current high-Nt dataset ends at 90 um; this point should be treated cautiously until the missing thick points are completed.",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(path_png: Path, path_svg: Path, best_rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"Skipping plot: {exc}")
        return

    x = list(range(len(NT_ORDER)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)

    for source in SELECTED_SOURCES:
        source_rows = {
            row["nt_label"]: row
            for row in best_rows
            if row["source"] == source
        }
        y_thickness = [as_float(source_rows[nt]["best_thickness_um"]) for nt in NT_ORDER]
        y_cce = [as_float(source_rows[nt]["best_CCE_percent"]) for nt in NT_ORDER]
        axes[0].plot(x, y_thickness, marker="o", linewidth=1.8, label=SOURCE_LABELS[source])
        axes[1].plot(x, y_cce, marker="o", linewidth=1.8, label=SOURCE_LABELS[source])

        boundary_x = [
            idx
            for idx, nt in enumerate(NT_ORDER)
            if source_rows[nt]["is_boundary_best"] == "True"
        ]
        axes[0].scatter(
            boundary_x,
            [y_thickness[idx] for idx in boundary_x],
            marker="s",
            facecolors="none",
            edgecolors="black",
            linewidths=1.0,
            s=72,
            zorder=4,
        )

    axes[0].set_ylabel("Best thickness (um)")
    axes[0].set_title("Best thickness from maximum CCE")
    axes[1].set_ylabel("Best CCE (%)")
    axes[1].set_title("Best CCE at selected thickness")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(NT_ORDER)
        ax.set_xlabel("Trap density Nt (cm^-3)")
        ax.grid(True, color="#d1d5db", linewidth=0.8)
        ax.set_axisbelow(True)

    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=220, bbox_inches="tight")
    fig.savefig(path_svg, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    validation_root = Path(__file__).resolve().parents[1]
    data_dir = validation_root / "data"
    report_dir = validation_root / "report"
    figure_dir = validation_root / "figures" / "trend3_cce_vs_thickness"

    metrics_rows = read_csv(data_dir / "validation_metrics.csv")
    best_rows = build_best_rows(metrics_rows)
    pivot_rows = build_pivot_rows(best_rows)
    aggregate_rows = build_aggregate_rows(best_rows)

    best_fields = [
        "source",
        "source_label",
        "nt_label",
        "trap_density_cm3",
        "curve_count",
        "best_thickness_um",
        "best_bias_v",
        "best_CCE_percent",
        "best_Qcol_signed_C",
        "best_charge_ratio_to_same_thickness_ideal",
        "is_boundary_best",
        "available_min_thickness_um",
        "available_max_thickness_um",
        "missing_highest_thickness_for_this_nt",
        "shift_from_Nt0_best_um",
        "best_CCE_drop_from_Nt0_best_pp",
    ]
    write_csv(data_dir / "trend3_selected_best_CCE_summary.csv", best_rows, best_fields)
    write_csv(data_dir / "trend3_selected_best_CCE_pivot.csv", pivot_rows)
    write_csv(data_dir / "trend3_selected_best_CCE_aggregate.csv", aggregate_rows)
    write_report(report_dir / "Trend3_selected_best_CCE_report.md", best_rows, pivot_rows, aggregate_rows)
    write_plot(
        figure_dir / "trend3_selected_best_CCE_statistics.png",
        figure_dir / "trend3_selected_best_CCE_statistics.svg",
        best_rows,
    )

    print(f"Wrote selected Trend 3 best-CCE statistics for {len(SELECTED_SOURCES)} sources.")
    print(f"Rows: best={len(best_rows)}, aggregate={len(aggregate_rows)}.")


if __name__ == "__main__":
    main()
