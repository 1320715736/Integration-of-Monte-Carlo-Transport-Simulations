#!/usr/bin/env python3
"""Batch pipeline: Step 4 → 5 → 6 for the CSV files listed below.

How to use
==========
1. Put all grid3d_*.csv input files under SiC_electron/raw_data/geant4_csv/.
2. Set CLEAR_OUTPUT_DIR_BEFORE_RUN below if you want to remove old outputs
    before each batch run.
3. Run from the SiC_electron/ directory:
       python src/run_all_energies.py

The script auto-discovers every grid3d_*.csv file in INPUT_CSV_DIR and runs
them all in one pass. Known energies are ordered by the preferred list below.

For each entry the pipeline performs:
  Step 4 – collapse the 3-D voxel grid → 2-D G(x, z) intermediate picture
  Step 5 – bilinear interpolation of G onto the TCAD mesh vertices
    Step 6 – write an OpticalGeneration .dat file for Sentaurus

The TCAD template files are auto-discovered from BASE_DIR by matching any
common prefix shared by one ``*.grd`` and one ``*.dat`` file.  If more than
one prefix pair is present, set ``TCAD_TEMPLATE_STEM`` below.

If two or more entries are processed an overlay G(z) comparison plot is
also generated under output/comparison_G_depth.png.

Outputs per label
-----------------
  output/<label>/step4_output/   intermediate_picture_*  +  step4_summary.json
  output/<label>/heatmap_G_<label>.png
  output/<label>/step5_output/   optical_generation_values.csv (+ diagnostics)
    output/<label>/step6_output/   <template_prefix><label>_optical_generation.dat
"""
from __future__ import annotations

import csv
import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

# ── allow sibling-module imports ──────────────────────────────────────────────
_SRC = Path(__file__).parent
sys.path.insert(0, str(_SRC))

# D: drive conda package fallback.  Reuse the tdx environment packages even
# when this script is launched by a different Python interpreter.
EXTRA_SITE_PACKAGES = [
    r"D:\conda_envs\tdx\Lib\site-packages",
]


def _prepend_existing_site_packages(paths: list[str]) -> None:
    for raw_path in reversed(paths):
        package_path = Path(raw_path)
        if package_path.is_dir():
            package_path_str = str(package_path)
            if package_path_str not in sys.path:
                sys.path.insert(0, package_path_str)


_prepend_existing_site_packages(EXTRA_SITE_PACKAGES)

np = importlib.import_module("numpy")
NDArray = Any

from step4_from_grid3d import (
    parse_grid3d_csv,
    build_2d_grid,
    save_outputs as save_step4_outputs,
)
from step5_interpolate_tcad import (
    parse_vertices_from_grd,
    interpolate_regular_grid_2d,
)
from grd_region_parser import get_region_vertex_ids, infer_pin_layer_depth_bounds

try:
    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")
    HAS_MPL = True
except ImportError:
    plt = None
    HAS_MPL = False

try:
    scipy_ndimage = importlib.import_module("scipy.ndimage")
    gaussian_filter = scipy_ndimage.gaussian_filter
    gaussian_filter1d = scipy_ndimage.gaussian_filter1d
    HAS_SCIPY = True
except ImportError:
    gaussian_filter = None
    gaussian_filter1d = None
    HAS_SCIPY = False


# ══════════════════════════════════════════════════════════════════════════════
#                              ✏️  EDIT BELOW  ✏️
# ══════════════════════════════════════════════════════════════════════════════
# Toggle this to remove the previous output/ contents before each batch run.
CLEAR_OUTPUT_DIR_BEFORE_RUN = True

# Project root – everything else is relative to this folder by default
BASE_DIR = Path(__file__).parent.parent          # SiC_electron/

# TCAD template prefix (without extension).
# None  -> auto-detect a unique *.grd / *.dat prefix pair in BASE_DIR
# "n4_" -> use n4_.grd + n4_.dat explicitly
TCAD_TEMPLATE_STEM: str | None = None

# Plot smoothing controls (visualisation only; never affect TCAD data values)
HEATMAP_GAUSSIAN_SIGMA: tuple[float, float] = (0.8, 1.2)
COMPARISON_GAUSSIAN_SIGMA: float = 1.0

OUTPUT_DIR = BASE_DIR / "output"
INPUT_CSV_DIR = (
    BASE_DIR / "raw_data" / "geant4_csv"
    if (BASE_DIR / "raw_data" / "geant4_csv").is_dir()
    else BASE_DIR / "csv"
)
PREFERRED_JOB_ORDER = [
    "10keV",
    "20keV",
    "30keV",
    "49keV",
    "100keV",
    "156p5keV",
    "c14",
]
# ══════════════════════════════════════════════════════════════════════════════
#                              ✏️  EDIT ABOVE  ✏️
# ══════════════════════════════════════════════════════════════════════════════

PLOT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
               "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

PIN_REGION_NAMES = ["p_plus", "i_layer", "n_plus"]
REGION_DISPLAY_NAMES = {
    "p_plus": "p+",
    "i_layer": "i",
    "n_plus": "n+",
}


def resolve_tcad_template(
    base_dir: Path,
    template_stem: str | None,
) -> tuple[Path, Path, str]:
    """Resolve the TCAD .grd/.dat template pair for any supported prefix.

    The supported naming convention is <prefix>.grd + <prefix>.dat, where the
    same stem appears in both files.  When template_stem is None, exactly one
    common stem must exist under BASE_DIR.
    """
    grd_files = {path.stem: path for path in base_dir.glob("*.grd")}
    dat_files = {path.stem: path for path in base_dir.glob("*.dat")}
    common_stems = sorted(set(grd_files) & set(dat_files))

    if template_stem is not None:
        stem = template_stem
        if stem not in grd_files or stem not in dat_files:
            raise FileNotFoundError(
                "Requested TCAD template stem "
                f"{stem!r} not found as both .grd and .dat under {base_dir}."
            )
        return grd_files[stem], dat_files[stem], stem

    if not common_stems:
        raise FileNotFoundError(
            f"No matching .grd/.dat template pair found under {base_dir}."
        )
    if len(common_stems) > 1:
        stems = ", ".join(common_stems)
        raise RuntimeError(
            "Multiple TCAD template prefixes found. "
            f"Set TCAD_TEMPLATE_STEM explicitly. Candidates: {stems}"
        )

    stem = common_stems[0]
    return grd_files[stem], dat_files[stem], stem


def build_output_dat_name(template_stem: str, label: str) -> str:
    separator = "" if template_stem.endswith("_") else "_"
    return f"{template_stem}{separator}{label}_optical_generation.dat"


def build_layer_markers(
    region_depth_bounds: dict[str, tuple[float, float]],
) -> list[tuple[float, str, str]]:
    """Return plot markers for the active PiN template boundaries."""
    markers: list[tuple[float, str, str]] = []
    boundary_specs = [
        ("p_plus", "i_layer", "--"),
        ("i_layer", "n_plus", ":"),
    ]
    for left_name, right_name, linestyle in boundary_specs:
        if left_name not in region_depth_bounds or right_name not in region_depth_bounds:
            continue
        boundary_depth = float(region_depth_bounds[left_name][1])
        label = (
            f"{REGION_DISPLAY_NAMES.get(left_name, left_name)}/"
            f"{REGION_DISPLAY_NAMES.get(right_name, right_name)}"
        )
        markers.append((boundary_depth, label, linestyle))
    return markers


def _build_plot_label_from_energy_token(energy_token: str) -> str:
    energy_value = float(energy_token.replace("p", "."))
    return f"{energy_value:g} keV"


def discover_jobs(input_csv_dir: Path) -> list[tuple[Path, str, str]]:
    if not input_csv_dir.is_dir():
        raise FileNotFoundError(f"Input CSV directory not found: {input_csv_dir}")

    preferred_order_index = {
        label.lower(): index for index, label in enumerate(PREFERRED_JOB_ORDER)
    }
    discovered: list[tuple[tuple[int, float, str], Path, str, str]] = []

    for csv_path in input_csv_dir.glob("grid3d_*.csv"):
        label = csv_path.stem[len("grid3d_"):]
        label_lower = label.lower()
        energy_sort = float("inf")

        if label_lower == "c14":
            plot_label = "C-14 β spectrum"
        elif label_lower.endswith("kev"):
            energy_token = label[:-3]
            plot_label = _build_plot_label_from_energy_token(energy_token)
            energy_sort = float(energy_token.replace("p", "."))
        else:
            plot_label = label

        preferred_index = preferred_order_index.get(label_lower)
        if preferred_index is None:
            sort_key = (1, energy_sort, label_lower)
        else:
            sort_key = (0, float(preferred_index), label_lower)

        discovered.append((sort_key, csv_path.resolve(), label, plot_label))

    if not discovered:
        raise FileNotFoundError(
            f"No grid3d_*.csv files found under {input_csv_dir}"
        )

    discovered.sort(key=lambda item: item[0])
    return [
        (csv_path, label, plot_label)
        for _sort_key, csv_path, label, plot_label in discovered
    ]


def clear_previous_outputs(output_dir: Path) -> None:
    if not output_dir.exists():
        return

    for child in output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()




# ─── Step-5 with correct axis mapping ────────────────────────────────────────

def _run_step5(
    label: str,
    x_centers: NDArray,
    z_centers: NDArray,
    G_2d: NDArray,
    vertices: NDArray,
    output_dir: Path,
) -> dict:
    """Interpolate G(x,z) onto TCAD mesh vertices.

        TCAD .grd axis convention
        -------------------------
      vertices[:, 0]  lateral x [µm], already centred at 0
      vertices[:, 1]  depth   z [µm], starts from 0
    """
    lateral_um = vertices[:, 0]   # coord-0 → lateral
    depth_um   = vertices[:, 1]   # coord-1 → depth

    query_points = np.column_stack([lateral_um, depth_um])
    interpolated = interpolate_regular_grid_2d(
        x_centers, z_centers, G_2d, query_points
    )

    summary = {
        "label": label,
        "vertex_count": int(vertices.shape[0]),
        "tcad_lateral_min_um": float(lateral_um.min()),
        "tcad_lateral_max_um": float(lateral_um.max()),
        "tcad_depth_min_um":   float(depth_um.min()),
        "tcad_depth_max_um":   float(depth_um.max()),
        "interpolated_min":    float(interpolated.min()),
        "interpolated_max":    float(interpolated.max()),
        "interpolated_nonzero_count": int(np.count_nonzero(interpolated)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Values-only CSV – fed directly to TCAD via Step 6
    values_path = output_dir / "optical_generation_values.csv"
    with values_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["optical_generation_cm^-3_s^-1"])
        for v in interpolated:
            writer.writerow([float(v)])

    # 2. Diagnostic vertex table
    vtable_path = output_dir / "tcad_vertices_interpolated.csv"
    with vtable_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "vertex_id",
            "tcad_lateral_um",
            "tcad_depth_um",
            "optical_generation_cm^-3_s^-1",
        ])
        for vid, (lat, dep, val) in enumerate(
            zip(lateral_um, depth_um, interpolated)
        ):
            writer.writerow([vid, float(lat), float(dep), float(val)])

    # 3. Compressed array for downstream use
    np.savez_compressed(
        output_dir / "step5_interpolated_vertices.npz",
        tcad_lateral_um=lateral_um,
        tcad_depth_um=depth_um,
        optical_generation_cm3_s=interpolated,
    )

    # 4. Summary JSON
    with (output_dir / "step5_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    return summary


# ─── Per-energy G heatmap ─────────────────────────────────────────────────────

def _make_heatmap(
    label: str,
    plot_label: str,
    x_centers: NDArray,
    z_centers: NDArray,
    G_2d: NDArray,
    output_path: Path,
    layer_markers: list[tuple[float, str, str]],
) -> None:
    """Save a 2-D heatmap of G(x, z) for one energy point.

    Style mirrors the Silicon_electron reference plot:
    - turbo colormap, linear scale
    - depth axis with origin='lower' (0 at bottom, increases upward)
    - cropped to the non-zero bounding box + small margin
    - dashed lines mark p+/i and i/n+ layer boundaries
    """
    if not HAS_MPL:
        return

    # ── Apply noise floor before bbox detection ──────────────────────────────
    # Voxels below 0.5 % of peak are MC statistical noise – treat as zero
    G_clean = G_2d.copy()
    noise_floor = 0.005 * float(G_clean.max())
    G_clean[G_clean < noise_floor] = 0.0

    # ── Find non-zero bounding box ──────────────────────────────────────────
    # G_2d shape: (n_x, n_z)
    nonzero = np.argwhere(G_clean > 0.0)
    if nonzero.size == 0:
        return
    xi0, xi1 = int(nonzero[:, 0].min()), int(nonzero[:, 0].max())
    zi0, zi1 = int(nonzero[:, 1].min()), int(nonzero[:, 1].max())

    margin_x = max(1, int(0.1 * (xi1 - xi0 + 1)))
    margin_z = max(2, int(0.05 * (zi1 - zi0 + 1)))
    xi0 = max(0, xi0 - margin_x)
    xi1 = min(G_clean.shape[0] - 1, xi1 + margin_x)
    zi0 = max(0, zi0 - margin_z)
    zi1 = min(G_clean.shape[1] - 1, zi1 + margin_z)

    G_crop = G_clean[xi0 : xi1 + 1, zi0 : zi1 + 1]   # (n_x_crop, n_z_crop)
    x_crop = x_centers[xi0 : xi1 + 1]
    z_crop = z_centers[zi0 : zi1 + 1]

    # ── Force the lateral extent to be symmetric around the beam axis ───────
    # The primary beam is centred at x = 0; any single-side bias in the data
    # is a Monte-Carlo sampling fluctuation.  We zero-pad the cropped array
    # on whichever side is shorter so the figure is centred on x = 0.
    if x_crop.size > 1:
        xy_bin = float(x_centers[1] - x_centers[0])
        x_extreme = max(abs(float(x_crop[0])), abs(float(x_crop[-1])))
        n_left_needed = int(round((x_extreme - (-x_crop[0])) / xy_bin)) \
            if -x_crop[0] < x_extreme else 0
        n_right_needed = int(round((x_extreme - x_crop[-1]) / xy_bin)) \
            if x_crop[-1] < x_extreme else 0
        if n_left_needed > 0 or n_right_needed > 0:
            pad_left = np.zeros((n_left_needed, G_crop.shape[1]))
            pad_right = np.zeros((n_right_needed, G_crop.shape[1]))
            G_crop = np.vstack([pad_left, G_crop, pad_right])
            new_x_left = x_crop[0] - n_left_needed * xy_bin
            new_x_right = x_crop[-1] + n_right_needed * xy_bin
            x_crop = np.linspace(new_x_left, new_x_right, G_crop.shape[0])

    # ── Light Gaussian smoothing for visualization only ─────────────────────
    # Removes the sharp voxel boundaries while preserving overall shape.
    # NOTE: This affects the figure only; TCAD .dat data are untouched.
    if HAS_SCIPY:
        G_smooth = gaussian_filter(
            G_crop,
            sigma=HEATMAP_GAUSSIAN_SIGMA,
            mode="nearest",
        )
    else:
        G_smooth = G_crop

    # ── imshow needs (n_z, n_x) with origin='lower' ─────────────────────────
    G_plot = G_smooth.T   # (n_z_crop, n_x_crop)

    dx = (x_centers[1] - x_centers[0]) / 2.0 if len(x_centers) > 1 else 5.0
    dz = (z_centers[1] - z_centers[0]) / 2.0 if len(z_centers) > 1 else 0.1
    extent = [
        float(x_crop[0]) - dx, float(x_crop[-1]) + dx,
        float(z_crop[0]) - dz, float(z_crop[-1]) + dz,
    ]

    # ── Figure size: match Silicon aspect (wider than tall) ─────────────────
    x_span = float(x_crop[-1] - x_crop[0]) + 2 * dx
    z_span = float(z_crop[-1] - z_crop[0]) + 2 * dz
    aspect_data = x_span / z_span
    fig_h = 5.2
    fig_w = max(7.0, min(14.0, fig_h * aspect_data * 1.3))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=180)

    im = ax.imshow(
        G_plot,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="turbo",
        interpolation="bilinear",
    )

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r"Optical Generation (cm$^{-3}$ s$^{-1}$)", fontsize=10)

    # ── Layer boundary lines ─────────────────────────────────────────────────
    for depth_um, text, linestyle in layer_markers:
        if float(z_crop[0]) - dz <= depth_um <= float(z_crop[-1]) + dz:
            ax.axhline(depth_um, color="white", linewidth=1.2,
                       linestyle=linestyle, alpha=0.9)
            ax.text(
                float(x_crop[0]) - dx + 0.5 * dx,
                depth_um + 0.5,
                text, color="white", fontsize=8, va="bottom",
            )

    ax.set_xlabel("Lateral position x (µm)", fontsize=11)
    ax.set_ylabel("Depth (µm)", fontsize=11)
    ax.set_title(
        f"Carrier Generation Rate G(x, z)  –  4H-SiC PiN  –  {plot_label}",
        fontsize=11,
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _make_comparison_plot(
    profiles: list[tuple[str, NDArray, NDArray]],
    output_path: Path,
    layer_markers: list[tuple[float, str, str]],
) -> None:
    """Plot G_mean(z) for all energies on one figure (log-y scale).

    The lateral mean G(z) is smoothed with a Gaussian kernel (sigma = 1 bin)
    *for visualisation only* – TCAD input data are never modified.
    Voxels below 0.1 % of each curve's peak are treated as MC noise floor
    and masked out.
    """
    if not HAS_MPL:
        print("  [skip] matplotlib not available – comparison plot not generated.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for (plot_label, z_centers, G_mean_z), color in zip(profiles, PLOT_COLORS):
        # Gaussian smoothing (sigma = 1 bin, visualisation only)
        if HAS_SCIPY:
            G_smooth = gaussian_filter1d(
                G_mean_z.astype(float),
                sigma=COMPARISON_GAUSSIAN_SIGMA,
            )
        else:
            G_smooth = G_mean_z.copy()

        # Mask noise floor: keep only values > 0.1 % of peak
        peak = G_smooth.max()
        noise_floor = peak * 1e-3
        mask = G_smooth > noise_floor
        if mask.any():
            ax.plot(
                z_centers[mask],
                G_smooth[mask],
                color=color,
                label=plot_label,
                linewidth=1.8,
            )

    for depth_um, text, ls in layer_markers:
        ax.axvline(depth_um, color="gray", linestyle=ls, linewidth=0.8, alpha=0.7)
        ax.text(
            depth_um + 0.3,
            ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1e10,
            text,
            fontsize=7,
            color="gray",
            va="top",
        )

    ax.set_xlabel("Depth (µm)", fontsize=12)
    ax.set_ylabel(r"$\bar{G}(z)$  [cm$^{-3}$ s$^{-1}$]", fontsize=12)
    ax.set_title(
        "Carrier generation rate vs. depth – 4H-SiC PiN\n"
        "(laterally averaged,  1 primary s⁻¹ normalisation)",
        fontsize=11,
    )
    ax.set_yscale("log")
    ax.set_xlim(left=0.0)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.6)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Comparison plot → {output_path}")


# ─── Per-energy processing ────────────────────────────────────────────────────

def process_one_energy(
    label: str,
    csv_path: Path,
    plot_label: str,
    vertices: NDArray,
    region_vertex_ids: dict[str, list[int]],
    layer_markers: list[tuple[float, str, str]],
    dat_template_path: Path,
    template_stem: str,
) -> tuple[NDArray, NDArray]:
    """Run Step 4 → 5 → 6 for a single CSV; return (z_centers, G_mean_z).

    The two returned arrays are used afterwards by the comparison plot.
    """
    print(f"\n── {label} ({csv_path.name}) ──")

    # Step 4 ──────────────────────────────────────────────────────────────────
    rows, n_events, xy_bin_mm, z_bin_um = parse_grid3d_csv(csv_path)
    x_centers, z_centers, G_2d, s4_summary = build_2d_grid(
        rows, n_events, xy_bin_mm, z_bin_um
    )
    s4_dir = OUTPUT_DIR / label / "step4_output"
    save_step4_outputs(s4_dir, x_centers, z_centers, G_2d, s4_summary)
    print(
        f"  Step 4: {s4_summary['x_bins']}x{s4_summary['z_bins']} grid  "
        f"x=[{s4_summary['x_min_um']:.1f},{s4_summary['x_max_um']:.1f}] um  "
        f"z=[{s4_summary['z_min_um']:.1f},{s4_summary['z_max_um']:.1f}] um  "
        f"G_max={G_2d.max():.3e} cm-3/s"
    )

    # Heatmap ──────────────────────────────────────────────────────────────────
    heatmap_path = OUTPUT_DIR / label / f"heatmap_G_{label}.png"
    _make_heatmap(
        label,
        plot_label,
        x_centers,
        z_centers,
        G_2d,
        heatmap_path,
        layer_markers,
    )
    if HAS_MPL:
        print(f"  Heatmap   -> {heatmap_path.relative_to(OUTPUT_DIR.parent)}")

    # Step 5 ──────────────────────────────────────────────────────────────────
    s5_dir = OUTPUT_DIR / label / "step5_output"
    s5_summary = _run_step5(label, x_centers, z_centers, G_2d, vertices, s5_dir)
    print(
        f"  Step 5: {s5_summary['interpolated_nonzero_count']} non-zero vertices  "
        f"G_max={s5_summary['interpolated_max']:.3e} cm-3/s"
    )

    # Step 6 ──────────────────────────────────────────────────────────────────
    # Write one OpticalGeneration Dataset block spanning all semiconductor
    # regions, using the global TCAD vertex order from Step 5.
    s6_dir = OUTPUT_DIR / label / "step6_output"
    s6_dir.mkdir(parents=True, exist_ok=True)
    output_dat = s6_dir / build_output_dat_name(template_stem, label)

    s5_npz = np.load(s5_dir / "step5_interpolated_vertices.npz")
    G_all = s5_npz["optical_generation_cm3_s"]

    dat_lines = dat_template_path.read_text(encoding="utf-8").splitlines(keepends=True)
    _SEM_REGIONS = PIN_REGION_NAMES
    updated = list(dat_lines)

    # Add "OpticalGeneration" to header datasets/functions lists once
    def _add_to_header(lines: list[str]) -> list[str]:
        result = list(lines)
        for idx, line in enumerate(result):
            if "datasets    = [" in line and '"OpticalGeneration"' not in line:
                result[idx] = line.replace(" ]", ' "OpticalGeneration" ]')
            if "functions   = [" in line and "OpticalGeneration" not in line:
                result[idx] = line.replace(" ]", " OpticalGeneration ]")
        return result

    updated = _add_to_header(updated)

    validity_str = " ".join(f'"{region}"' for region in _SEM_REGIONS)
    G_values = [float(value) for value in G_all]
    s6_info: dict = {}
    for region in _SEM_REGIONS:
        v_ids = region_vertex_ids[region]
        region_G = [float(G_all[vid]) for vid in v_ids]
        s6_info[region] = {
            "vertex_count": len(v_ids),
            "nonzero": int(sum(1 for g in region_G if g > 0)),
            "G_max": float(max(region_G)) if region_G else 0.0,
        }

    combined_block = (
        f'  Dataset ("OpticalGeneration") {{\n'
        f"    function  = OpticalGeneration\n"
        f"    type      = scalar\n"
        f"    dimension = 1\n"
        f"    location  = vertex\n"
        f"    validity  = [ {validity_str} ]\n"
        f"    Values ({len(G_values)}) {{\n"
    )
    for start in range(0, len(G_values), 10):
        chunk = G_values[start:start + 10]
        combined_block += " " + " ".join(f"{v:.15e}" for v in chunk) + "\n"
    combined_block += "    }\n  }\n\n"

    insert_idx = None
    for k in range(len(updated) - 1, -1, -1):
        if updated[k].strip() == "}":
            insert_idx = k
            break
    updated = updated[:insert_idx] + [combined_block] + updated[insert_idx:]
    output_dat.write_text("".join(updated), encoding="utf-8")

    with (s6_dir / "step6_summary.json").open("w", encoding="utf-8") as _f:
        json.dump(
            {
                "source_dat": str(dat_template_path),
                "output_dat": str(output_dat),
                "optical_generation_value_count": len(G_values),
                "validity": _SEM_REGIONS,
                "regions": s6_info,
            },
            _f, ensure_ascii=False, indent=2,
        )
    total_nz = sum(r["nonzero"] for r in s6_info.values())
    print(f"  Step 6: {output_dat.name}  non-zero vertices={total_nz}")

    # Laterally averaged depth profile (for comparison plot)
    G_mean_z = G_2d.mean(axis=0)
    return z_centers, G_mean_z


# ─── Argument parsing ─────────────────────────────────────────────────────────
# (none – job list is configured at the top of this file)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    # Discover input CSV files ────────────────────────────────────────────────
    resolved_jobs: list[tuple[str, Path, str]] = []
    try:
        discovered_jobs = discover_jobs(INPUT_CSV_DIR)
    except FileNotFoundError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    for csv_path, label, plot_label in discovered_jobs:
        resolved_jobs.append((label, csv_path, plot_label))

    if CLEAR_OUTPUT_DIR_BEFORE_RUN:
        print(f"Clearing previous outputs under: {OUTPUT_DIR}")
        clear_previous_outputs(OUTPUT_DIR)

    print(f"Discovered {len(resolved_jobs)} input CSV files in {INPUT_CSV_DIR}:")
    for label, csv_path, plot_label in resolved_jobs:
        print(f"  {label:>8}  <-  {csv_path.name}  ({plot_label})")

    # Common TCAD setup (loaded once) ─────────────────────────────────────────
    grd_file, dat_file, template_stem = resolve_tcad_template(
        BASE_DIR,
        TCAD_TEMPLATE_STEM,
    )
    print(f"Loading TCAD grid:  {grd_file}")
    print(f"Loading TCAD data:  {dat_file}")
    print(f"TCAD template stem: {template_stem}")

    vertices = parse_vertices_from_grd(grd_file)
    print(
        f"  Vertices: {vertices.shape[0]}  "
        f"lateral=[{vertices[:,0].min():.1f}, {vertices[:,0].max():.1f}] µm  "
        f"depth=[{vertices[:,1].min():.1f}, {vertices[:,1].max():.1f}] µm"
    )

    region_depth_bounds = infer_pin_layer_depth_bounds(grd_file, dat_file)
    layer_markers = build_layer_markers(region_depth_bounds)
    if layer_markers:
        print("  Inferred layer boundaries:")
        for depth_um, text, _ls in layer_markers:
            print(f"    {text:>6} @ {depth_um:.4f} µm")

    region_vertex_ids = get_region_vertex_ids(
        grd_file,
        PIN_REGION_NAMES,
        dat_path=dat_file,
        region_depth_bounds=region_depth_bounds,
    )
    print(
        f"  Region vertices – p_plus:{len(region_vertex_ids['p_plus'])}  "
        f"i_layer:{len(region_vertex_ids['i_layer'])}  "
        f"n_plus:{len(region_vertex_ids['n_plus'])}"
    )

    # Run each job ────────────────────────────────────────────────────────────
    profiles: list[tuple[str, NDArray, NDArray]] = []
    for label, csv_path, plot_label in resolved_jobs:
        z_centers, G_mean_z = process_one_energy(
            label,
            csv_path,
            plot_label,
            vertices,
            region_vertex_ids,
            layer_markers,
            dat_file,
            template_stem,
        )
        profiles.append((plot_label, z_centers, G_mean_z))

    # Comparison plot only meaningful for ≥ 2 jobs ────────────────────────────
    if len(profiles) >= 2:
        print("\n── Generating comparison plot ──")
        _make_comparison_plot(
            profiles,
            OUTPUT_DIR / "comparison_G_depth.png",
            layer_markers,
        )

    print("\nAll done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
