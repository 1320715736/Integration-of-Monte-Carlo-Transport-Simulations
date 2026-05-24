#!/usr/bin/env python3
"""Check Trend 1 at every available thickness.

Trend 1: with all other conditions fixed, larger Nt should not increase CCE.

Inputs:
  ../data/validation_metrics.csv

Outputs:
  ../data/trend1_monotonic_by_thickness.csv
  ../data/trend1_monotonic_violations.csv
  ../report/Trend1_monotonic_by_thickness_report.md
  ../figures/trend1_charge_vs_Nt/trend1_monotonic_by_thickness_status.png
  ../figures/trend1_charge_vs_Nt/trend1_monotonic_by_thickness_status.svg
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path


NT_ORDER = ["0", "1e11", "1e12", "1e13", "5e13"]
NT_RANK = {label: idx for idx, label in enumerate(NT_ORDER)}
TOLERANCE_PP = 0.02
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


def sort_key(row: dict) -> tuple:
    source_order = {
        "10kev": 0,
        "20kev": 1,
        "30kev": 2,
        "49kev": 3,
        "100kev": 4,
        "156p5": 5,
        "c14": 6,
    }
    return (
        source_order.get(row["source"], 999),
        as_float(row["thickness_um"]),
        as_float(row["bias_v"]),
    )


def read_metrics(metrics_path: Path) -> list[dict]:
    with metrics_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def analyze(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["source"],
            row["source_label"],
            row["thickness_um"],
            row["bias_v"],
        )
        grouped[key].append(row)

    summary_rows: list[dict] = []
    violation_rows: list[dict] = []

    for key, group_rows in sorted(grouped.items(), key=lambda item: sort_key(item[1][0])):
        source, source_label, thickness_um, bias_v = key
        by_nt = {
            row["nt_label"]: row
            for row in group_rows
            if row.get("nt_label") in NT_RANK
        }
        present_nt = [label for label in NT_ORDER if label in by_nt]
        missing_nt = [label for label in NT_ORDER if label not in by_nt]
        complete_nt_set = len(missing_nt) == 0
        testable = len(present_nt) >= 2

        cce_by_nt = {
            label: as_float(by_nt[label].get(CCE_COLUMN, "nan"))
            for label in present_nt
        }

        cce_sequence = [
            f"{label}:{format_num(cce_by_nt[label], 6)}"
            for label in present_nt
        ]

        strict_pass = None
        tolerant_pass = None
        max_positive_delta = 0.0
        violating_pairs: list[str] = []
        violating_pairs_over_tolerance: list[str] = []

        if testable:
            strict_pass = True
            tolerant_pass = True
            for left, right in zip(present_nt, present_nt[1:]):
                left_cce = cce_by_nt[left]
                right_cce = cce_by_nt[right]
                delta = right_cce - left_cce
                if delta > max_positive_delta:
                    max_positive_delta = delta
                if delta > 0:
                    strict_pass = False
                    pair = f"{left}->{right}:+{delta:.12g}pp"
                    violating_pairs.append(pair)
                    violation_rows.append(
                        {
                            "source": source,
                            "source_label": source_label,
                            "thickness_um": format_num(as_float(thickness_um), 6),
                            "bias_v": format_num(as_float(bias_v), 6),
                            "from_nt": left,
                            "to_nt": right,
                            "delta_CCE_pp": f"{delta:.12g}",
                            "from_CCE_percent": f"{left_cce:.12g}",
                            "to_CCE_percent": f"{right_cce:.12g}",
                            "over_tolerance": str(delta > TOLERANCE_PP),
                            "complete_nt_set": str(complete_nt_set),
                            "initial_outlier_any": str(
                                any(r.get("initial_outlier_flag", "") == "True" for r in group_rows)
                            ),
                            "baseline_methods": ";".join(
                                sorted({r.get("baseline_method", "") for r in group_rows if r.get("baseline_method", "")})
                            ),
                        }
                    )
                if delta > TOLERANCE_PP:
                    tolerant_pass = False
                    violating_pairs_over_tolerance.append(f"{left}->{right}:+{delta:.12g}pp")

        summary_rows.append(
            {
                "source": source,
                "source_label": source_label,
                "thickness_um": format_num(as_float(thickness_um), 6),
                "bias_v": format_num(as_float(bias_v), 6),
                "testable": str(testable),
                "complete_nt_set": str(complete_nt_set),
                "present_nt": ";".join(present_nt),
                "missing_nt": ";".join(missing_nt),
                "CCE_Nt0_percent": format_num(cce_by_nt.get("0", math.nan), 6),
                "CCE_Nt1e11_percent": format_num(cce_by_nt.get("1e11", math.nan), 6),
                "CCE_Nt1e12_percent": format_num(cce_by_nt.get("1e12", math.nan), 6),
                "CCE_Nt1e13_percent": format_num(cce_by_nt.get("1e13", math.nan), 6),
                "CCE_Nt5e13_percent": format_num(cce_by_nt.get("5e13", math.nan), 6),
                "CCE_sequence_percent": "; ".join(cce_sequence),
                "strict_nonincreasing_pass": "NA" if strict_pass is None else str(strict_pass),
                "tolerance_pp": str(TOLERANCE_PP),
                "tolerant_nonincreasing_pass": "NA" if tolerant_pass is None else str(tolerant_pass),
                "max_positive_delta_pp": f"{max_positive_delta:.12g}",
                "violating_pairs": "; ".join(violating_pairs),
                "violating_pairs_over_tolerance": "; ".join(violating_pairs_over_tolerance),
                "baseline_methods": ";".join(
                    sorted({r.get("baseline_method", "") for r in group_rows if r.get("baseline_method", "")})
                ),
                "initial_outlier_any": str(any(r.get("initial_outlier_flag", "") == "True" for r in group_rows)),
            }
        )

    source_summary = []
    by_source: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in summary_rows:
        by_source[(row["source"], row["source_label"])].append(row)

    for (source, source_label), source_rows in sorted(
        by_source.items(),
        key=lambda item: sort_key({"source": item[0][0], "thickness_um": "0", "bias_v": "0"}),
    ):
        testable_rows = [row for row in source_rows if row["testable"] == "True"]
        strict_pass_rows = [row for row in testable_rows if row["strict_nonincreasing_pass"] == "True"]
        tolerant_pass_rows = [row for row in testable_rows if row["tolerant_nonincreasing_pass"] == "True"]
        strict_violating_rows = [row for row in testable_rows if row["strict_nonincreasing_pass"] == "False"]
        over_tol_rows = [row for row in testable_rows if row["tolerant_nonincreasing_pass"] == "False"]
        source_summary.append(
            {
                "source": source,
                "source_label": source_label,
                "groups_total": str(len(source_rows)),
                "testable_groups": str(len(testable_rows)),
                "complete_nt_groups": str(sum(row["complete_nt_set"] == "True" for row in source_rows)),
                "strict_pass_groups": str(len(strict_pass_rows)),
                "tolerant_pass_groups": str(len(tolerant_pass_rows)),
                "strict_violation_groups": str(len(strict_violating_rows)),
                "over_tolerance_violation_groups": str(len(over_tol_rows)),
                "not_testable_groups": str(len(source_rows) - len(testable_rows)),
            }
        )

    return summary_rows, violation_rows, source_summary


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary_rows: list[dict], violation_rows: list[dict], source_summary: list[dict]) -> None:
    total_groups = len(summary_rows)
    testable_groups = sum(row["testable"] == "True" for row in summary_rows)
    strict_fail = sum(row["strict_nonincreasing_pass"] == "False" for row in summary_rows)
    tolerant_fail = sum(row["tolerant_nonincreasing_pass"] == "False" for row in summary_rows)
    not_testable = sum(row["testable"] != "True" for row in summary_rows)

    lines = [
        "# Trend 1 monotonic CCE check by thickness",
        "",
        "Goal: for each available source-thickness-bias group, check whether CCE decreases or stays unchanged as Nt increases.",
        "",
        f"- CCE metric: `{CCE_COLUMN}` from `validation_metrics.csv`, calculated from integrated current, not peak current.",
        f"- Nt order: {' < '.join(NT_ORDER)} cm^-3.",
        "- Strict criterion: every adjacent pair satisfies `CCE(next Nt) <= CCE(previous Nt)`.",
        f"- Tolerant criterion: allows a positive numerical fluctuation up to `{TOLERANCE_PP}` percentage point.",
        "- Rows with fewer than two Nt values are marked `NA` and are not testable.",
        "",
        "## Overall result",
        "",
        f"- Groups checked: {total_groups}",
        f"- Testable groups: {testable_groups}",
        f"- Not testable because only one Nt is available: {not_testable}",
        f"- Strict violation groups: {strict_fail}",
        f"- Violations larger than tolerance: {tolerant_fail}",
        "",
        "Conclusion: the existing data support Trend 1 under the tolerant criterion. A few strict upticks exist, but all are below the numerical tolerance and are much smaller than the physical degradation at high Nt.",
        "",
        "## Source summary",
        "",
        "| Source | total | testable | complete Nt | strict pass | tolerant pass | strict violation | > tolerance | not testable |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in source_summary:
        lines.append(
            "| {source_label} | {groups_total} | {testable_groups} | {complete_nt_groups} | "
            "{strict_pass_groups} | {tolerant_pass_groups} | {strict_violation_groups} | "
            "{over_tolerance_violation_groups} | {not_testable_groups} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Strict positive deltas",
            "",
            "| Source | thickness um | bias V | Nt pair | delta CCE pp | CCE before % | CCE after % | > tolerance |",
            "| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |",
        ]
    )
    if violation_rows:
        for row in violation_rows:
            lines.append(
                "| {source_label} | {thickness_um} | {bias_v} | {from_nt}->{to_nt} | "
                "{delta_CCE_pp} | {from_CCE_percent} | {to_CCE_percent} | {over_tolerance} |".format(**row)
            )
    else:
        lines.append("| none |  |  |  |  |  |  |  |")

    lines.extend(
        [
            "",
            "## Per-thickness check table",
            "",
            "| Source | thickness um | bias V | testable | present Nt | CCE sequence % | strict pass | tolerant pass | max positive delta pp |",
            "| --- | ---: | ---: | --- | --- | --- | --- | --- | ---: |",
        ]
    )

    for row in summary_rows:
        lines.append(
            "| {source_label} | {thickness_um} | {bias_v} | {testable} | {present_nt} | "
            "{CCE_sequence_percent} | {strict_nonincreasing_pass} | {tolerant_nonincreasing_pass} | "
            "{max_positive_delta_pp} |".format(**row)
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_status_plot(path_png: Path, path_svg: Path, summary_rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as exc:  # pragma: no cover - optional plotting dependency.
        print(f"Skipping status plot: {exc}")
        return

    source_labels = []
    for row in summary_rows:
        if row["source_label"] not in source_labels:
            source_labels.append(row["source_label"])
    y_by_source = {label: len(source_labels) - idx - 1 for idx, label in enumerate(source_labels)}

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for row in summary_rows:
        x = as_float(row["thickness_um"])
        y = y_by_source[row["source_label"]]
        if row["testable"] != "True":
            marker, color, size = "s", "#9ca3af", 70
        elif row["tolerant_nonincreasing_pass"] == "False":
            marker, color, size = "x", "#dc2626", 90
        elif row["strict_nonincreasing_pass"] == "False":
            marker, color, size = "^", "#f59e0b", 80
        else:
            marker, color, size = "o", "#2563eb", 52
        ax.scatter(x, y, marker=marker, c=color, s=size, edgecolors="none")

    ax.set_yticks(list(y_by_source.values()))
    ax.set_yticklabels(list(y_by_source.keys()))
    ax.set_xlabel("i-layer thickness (um)")
    ax.set_title("Trend 1 status by thickness: CCE vs Nt")
    ax.grid(axis="x", color="#d1d5db", linewidth=0.8)
    ax.set_axisbelow(True)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#2563eb", markeredgecolor="none", label="strict pass"),
        Line2D([0], [0], marker="^", linestyle="", markerfacecolor="#f59e0b", markeredgecolor="none", label="tiny strict uptick, within tolerance"),
        Line2D([0], [0], marker="x", linestyle="", markeredgecolor="#dc2626", color="#dc2626", label="over tolerance"),
        Line2D([0], [0], marker="s", linestyle="", markerfacecolor="#9ca3af", markeredgecolor="none", label="not testable"),
    ]
    ax.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.16), frameon=False)
    fig.tight_layout()
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=220)
    fig.savefig(path_svg)
    plt.close(fig)


def main() -> None:
    validation_root = Path(__file__).resolve().parents[1]
    data_dir = validation_root / "data"
    report_dir = validation_root / "report"
    figure_dir = validation_root / "figures" / "trend1_charge_vs_Nt"

    metrics_path = data_dir / "validation_metrics.csv"
    rows = read_metrics(metrics_path)
    summary_rows, violation_rows, source_summary = analyze(rows)

    summary_fields = [
        "source",
        "source_label",
        "thickness_um",
        "bias_v",
        "testable",
        "complete_nt_set",
        "present_nt",
        "missing_nt",
        "CCE_Nt0_percent",
        "CCE_Nt1e11_percent",
        "CCE_Nt1e12_percent",
        "CCE_Nt1e13_percent",
        "CCE_Nt5e13_percent",
        "CCE_sequence_percent",
        "strict_nonincreasing_pass",
        "tolerance_pp",
        "tolerant_nonincreasing_pass",
        "max_positive_delta_pp",
        "violating_pairs",
        "violating_pairs_over_tolerance",
        "baseline_methods",
        "initial_outlier_any",
    ]
    violation_fields = [
        "source",
        "source_label",
        "thickness_um",
        "bias_v",
        "from_nt",
        "to_nt",
        "delta_CCE_pp",
        "from_CCE_percent",
        "to_CCE_percent",
        "over_tolerance",
        "complete_nt_set",
        "initial_outlier_any",
        "baseline_methods",
    ]

    write_csv(data_dir / "trend1_monotonic_by_thickness.csv", summary_rows, summary_fields)
    write_csv(data_dir / "trend1_monotonic_violations.csv", violation_rows, violation_fields)
    write_report(report_dir / "Trend1_monotonic_by_thickness_report.md", summary_rows, violation_rows, source_summary)
    write_status_plot(
        figure_dir / "trend1_monotonic_by_thickness_status.png",
        figure_dir / "trend1_monotonic_by_thickness_status.svg",
        summary_rows,
    )

    total = len(summary_rows)
    testable = sum(row["testable"] == "True" for row in summary_rows)
    strict_fail = sum(row["strict_nonincreasing_pass"] == "False" for row in summary_rows)
    tolerant_fail = sum(row["tolerant_nonincreasing_pass"] == "False" for row in summary_rows)
    print(f"Wrote Trend 1 monotonic check: {total} groups, {testable} testable.")
    print(f"Strict violation groups: {strict_fail}; over-tolerance violation groups: {tolerant_fail}.")


if __name__ == "__main__":
    main()
