#!/usr/bin/env python3
"""Check coverage and numerical quality of the TCAD transient-current dataset."""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
FIG_SET_DIR = THIS_DIR.parent
sys.path.insert(0, str(FIG_SET_DIR))

from tcad_it_tools import NT_ORDER, SOURCE_CONFIG, SOURCE_ORDER, THICKNESS_ORDER, extract_metrics, write_csv

OUT_COVERAGE = THIS_DIR / "tcad_it_data_availability.csv"
OUT_QC = THIS_DIR / "tcad_it_qc_report.csv"


def main() -> None:
    rows = extract_metrics()
    by_key = {
        (str(row["source"]), float(row["thickness_um"]), str(row["nt"])): row
        for row in rows
    }

    coverage_rows: list[dict[str, object]] = []
    missing_rows: list[tuple[str, float, str]] = []
    for source in SOURCE_ORDER:
        for thickness_um in THICKNESS_ORDER:
            for nt in NT_ORDER:
                row = by_key.get((source, thickness_um, nt))
                if row is None:
                    missing_rows.append((source, thickness_um, nt))
                    coverage_rows.append(
                        {
                            "source": source,
                            "source_label": SOURCE_CONFIG[source]["label"],
                            "thickness_um": thickness_um,
                            "nt": nt,
                            "available": False,
                            "qc_pass": False,
                            "qc_reason": "missing",
                        }
                    )
                    continue
                coverage_rows.append(
                    {
                        "source": source,
                        "source_label": row["source_label"],
                        "thickness_um": thickness_um,
                        "nt": nt,
                        "available": True,
                        "qc_pass": row["qc_pass"],
                        "qc_reason": row["qc_reason"],
                        "cce_percent": row["cce_percent"],
                        "positive_peak_nA": row["positive_peak_nA"],
                        "negative_peak_nA": row["negative_peak_nA"],
                        "reverse_peak_ratio": row["reverse_peak_ratio"],
                        "dropped_initial_points": row["dropped_initial_points"],
                    }
                )

    write_csv(OUT_COVERAGE, coverage_rows)
    qc_rows = [row for row in rows if not bool(row["qc_pass"])]
    write_csv(OUT_QC, qc_rows)

    counter = Counter(str(row["qc_reason"]) for row in qc_rows)
    print(f"total expected combinations: {len(SOURCE_ORDER) * len(THICKNESS_ORDER) * len(NT_ORDER)}")
    print(f"available combinations: {len(rows)}")
    print(f"missing combinations: {len(missing_rows)}")
    print(f"qc failed combinations: {len(qc_rows)}")
    for reason, count in counter.items():
        print(f"{reason}: {count}")
    print(f"wrote {OUT_COVERAGE}")
    print(f"wrote {OUT_QC}")


if __name__ == "__main__":
    main()
