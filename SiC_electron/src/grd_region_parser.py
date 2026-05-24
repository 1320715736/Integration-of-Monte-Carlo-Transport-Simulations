#!/usr/bin/env python3
"""Helpers for mapping SiC PiN regions onto the global GRD vertex order.

For the current SiC workflow, the local vertex ordering used by each region in
the paired .dat file follows the GLOBAL vertex list order, filtered by depth.
The p/i/n layer boundaries therefore need to be consistent with the active
template mesh.  When a matching .dat file is available, this module infers the
depth boundaries dynamically from the template's PMIUserField0 vertex counts,
so different p-layer thicknesses can be used without editing Python code.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

# Fallback depth boundaries [µm] for the legacy SiC PiN template.
_DEFAULT_REGION_DEPTH_BOUNDS: dict[str, tuple[float, float]] = {
    "p_plus":  (0.0,   0.2),
    "i_layer": (0.2,   120.2),
    "n_plus":  (120.2, 120.7),
}

_PIN_REGION_NAMES = ("p_plus", "i_layer", "n_plus")


def parse_all_vertices(grd_path: Path) -> np.ndarray:
    """Return global vertex array, shape (N, 2): columns are (lateral_um, depth_um)."""
    verts: list[tuple[float, float]] = []
    inside = False
    with grd_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not inside:
                if s.startswith("Vertices (") and s.endswith("{"):
                    inside = True
                continue
            if s == "}":
                break
            tokens = s.split()
            for i in range(0, len(tokens) - 1, 2):
                verts.append((float(tokens[i]), float(tokens[i + 1])))
    if not verts:
        raise ValueError(f"No vertices found in {grd_path}.")
    return np.asarray(verts, dtype=float)


def parse_dataset_region_value_counts(
    dat_path: Path,
    dataset_name: str = "PMIUserField0",
) -> dict[str, int]:
    """Return {region_name: value_count} for one DAT dataset grouped by validity."""
    counts: dict[str, int] = {}
    inside_dataset = False
    current_region: str | None = None
    current_count: int | None = None

    with dat_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not inside_dataset:
                if stripped == f'Dataset ("{dataset_name}") {{':
                    inside_dataset = True
                    current_region = None
                    current_count = None
                continue

            if current_region is None and stripped.startswith("validity"):
                matches = re.findall(r'"([^"]+)"', stripped)
                if len(matches) != 1:
                    raise ValueError(
                        f"Expected exactly one validity region in {dat_path}, got: {stripped}"
                    )
                current_region = matches[0]
                continue

            if current_count is None and stripped.startswith("Values ("):
                current_count = int(stripped.split("(", 1)[1].split(")", 1)[0])
                continue

            if stripped == "}":
                if current_region is not None and current_count is not None:
                    if current_region in counts:
                        raise ValueError(
                            f"Duplicate {dataset_name} block for region {current_region!r} in {dat_path}."
                        )
                    counts[current_region] = current_count
                    inside_dataset = False
                    current_region = None
                    current_count = None

    if not counts:
        raise ValueError(f"No dataset {dataset_name!r} found in {dat_path}.")
    return counts


def infer_pin_layer_depth_bounds(
    grd_path: Path,
    dat_path: Path,
) -> dict[str, tuple[float, float]]:
    """Infer p/i/n depth boundaries from the active DAT template vertex counts.

    The DAT template already stores how many vertex values belong to each
    semiconductor region. Because those local region values follow the global
    GRD vertex ordering filtered by depth, the p/i and i/n interfaces can be
    recovered directly from the sorted GRD vertex depths.
    """
    region_counts = parse_dataset_region_value_counts(dat_path)
    missing = [name for name in _PIN_REGION_NAMES if name not in region_counts]
    if missing:
        raise KeyError(
            "Missing PiN region counts in DAT template: " + ", ".join(missing)
        )

    vertices = parse_all_vertices(grd_path)
    depth = np.asarray(vertices[:, 1], dtype=float)
    sorted_depth = np.sort(depth)

    p_count = int(region_counts["p_plus"])
    n_count = int(region_counts["n_plus"])
    if p_count <= 0 or n_count <= 0:
        raise ValueError("Template region counts must be positive.")
    if p_count > sorted_depth.size or n_count > sorted_depth.size:
        raise ValueError("Template region counts exceed GRD vertex count.")

    p_boundary = float(sorted_depth[p_count - 1])
    n_boundary = float(sorted_depth[sorted_depth.size - n_count])
    depth_min = float(sorted_depth[0])
    depth_max = float(sorted_depth[-1])

    inferred_bounds = {
        "p_plus": (depth_min, p_boundary),
        "i_layer": (p_boundary, n_boundary),
        "n_plus": (n_boundary, depth_max),
    }

    tolerance = 1e-12
    inferred_counts = {
        "p_plus": int(np.count_nonzero(depth <= p_boundary + tolerance)),
        "i_layer": int(
            np.count_nonzero(
                (depth >= p_boundary - tolerance)
                & (depth <= n_boundary + tolerance)
            )
        ),
        "n_plus": int(np.count_nonzero(depth >= n_boundary - tolerance)),
    }
    expected_counts = {name: int(region_counts[name]) for name in _PIN_REGION_NAMES}
    if inferred_counts != expected_counts:
        raise ValueError(
            "Inferred layer bounds do not reproduce DAT region counts: "
            f"expected {expected_counts}, got {inferred_counts}."
        )

    return inferred_bounds


def get_region_vertex_ids(
    grd_path: Path,
    region_names: list[str],
    *,
    dat_path: Path | None = None,
    region_depth_bounds: dict[str, tuple[float, float]] | None = None,
) -> dict[str, list[int]]:
    """Return {region_name: [global_vertex_id, ...]} for each named region.

    The returned lists are in GLOBAL vertex index order (ascending), which
    matches the local vertex ordering used in the .dat file for each region.

    Parameters
    ----------
    grd_path     : path to the DF-ISE .grd file
    region_names : subset of ["p_plus", "i_layer", "n_plus"]
    dat_path     : optional paired DAT template used to infer dynamic bounds
    region_depth_bounds : optional precomputed depth bounds per region

    Raises
    ------
    KeyError  : if a region_name is not in the known depth-bound table
    """
    vertices = parse_all_vertices(grd_path)  # (N, 2): (lateral, depth)
    depth = vertices[:, 1]

    if region_depth_bounds is None:
        if dat_path is not None:
            region_depth_bounds = infer_pin_layer_depth_bounds(grd_path, dat_path)
        else:
            region_depth_bounds = _DEFAULT_REGION_DEPTH_BOUNDS

    result: dict[str, list[int]] = {}
    tolerance = 1e-12
    for name in region_names:
        d_lo, d_hi = region_depth_bounds[name]
        ids = [
            int(i)
            for i in np.where(
                (depth >= d_lo - tolerance) & (depth <= d_hi + tolerance)
            )[0]
        ]
        result[name] = ids

    return result
