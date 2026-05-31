#!/usr/bin/env python3
# Build and sync manuscript figures; helper moved from fig_set to src.
"""Build MEJ figure PNG/PDF files from fig_set and sync them to the draft."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
FIG_SET_DIR = ROOT_DIR / "fig_set"
MEJ_FIG_DIR = ROOT_DIR / "latex" / "MEJ" / "figures"

FIGURES = [
    ("fig1", "plot_fig1_sic_pin_structure.py", "fig1_sic_pin_structure"),
    ("fig2", "plot_fig2_geant4_tcad_workflow.py", "fig2_geant4_tcad_workflow"),
    ("fig3", "plot_fig3_tcad_generation_distribution.py", "fig3_tcad_generation_distribution"),
    ("fig4", "plot_fig4_c14_spectrum.py", "fig4_c14_spectrum"),
    ("fig5", "plot_fig5_dedx_profiles.py", "fig5_dedx_profiles"),
    ("fig6", "plot_fig5_cv.py", "fig6_cv_1overc2_baseline"),
    ("fig7", "plot_fig6_c14_cce.py", "fig7_c14_cce_vs_thickness_by_Nt"),
    ("fig8", "plot_fig7_c14_cce_design_map.py", "fig8_c14_cce_design_map"),
    ("fig9", "plot_fig8_optimal_thickness.py", "fig9_optimal_thickness_vs_Nt"),
    ("fig10", "plot_fig9_mono_energy_bias.py", "fig10_design_bias_matrix"),
]

EXTRA_OUTPUTS: list[tuple[str, str]] = []


def run_script(script_path: Path) -> None:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run([sys.executable, str(script_path)], check=True, cwd=ROOT_DIR, env=env)


def main() -> int:
    MEJ_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for folder, script, base in FIGURES:
        script_path = FIG_SET_DIR / folder / script
        if script_path.exists():
            run_script(script_path)
        elif (SRC_DIR / script).exists():
            run_script(SRC_DIR / script)
        else:
            expected_outputs = [FIG_SET_DIR / folder / f"{base}.{extension}" for extension in ("png", "pdf")]
            if not all(path.exists() for path in expected_outputs):
                raise FileNotFoundError(f"Missing generator and output files for {base}: {script_path}")
            print(f"Skipped missing generator and reused existing outputs: {script_path}")
        for extension in ("png", "pdf"):
            source = FIG_SET_DIR / folder / f"{base}.{extension}"
            destination = MEJ_FIG_DIR / source.name
            shutil.copy2(source, destination)
    for folder, base in EXTRA_OUTPUTS:
        for extension in ("png", "pdf"):
            source = FIG_SET_DIR / folder / f"{base}.{extension}"
            if source.exists():
                destination = MEJ_FIG_DIR / source.name
                shutil.copy2(source, destination)
    print(f"Synced {len(FIGURES)} manuscript figures and {len(EXTRA_OUTPUTS)} preserved figure to {MEJ_FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
