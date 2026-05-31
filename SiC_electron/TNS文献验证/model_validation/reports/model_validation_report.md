# Model Validation Results

This folder implements the first two validation items in `markdown/验证方法.md`.

## 1. Geant4-ESTAR Low-Energy Electron Range Benchmark

NIST ESTAR was queried as a user-defined silicon carbide material with density 3.21 g/cm3 and formula `SiC 1`. ESTAR reports CSDA range as mass thickness; it was converted to projected length by dividing by density. The 49 keV and 156.5 keV entries are log-log interpolated from the default ESTAR energy grid because NIST only reports range for default energies in this text-table mode.

The Geant4 depths below are extracted directly from the one-dimensional depth-deposition profiles under `raw_data/geant4_csv/OutPut_fulldata/*/depth_profile_*_all.csv`, not from the Step-4 2D generation map.

| Energy | ESTAR R_CSDA (um) | Geant4 z50 (um) | Geant4 z90 (um) | Edep/Ein |
| --- | --- | --- | --- | --- |
| 20.000 | 3.439 | 0.985 | 1.945 | 0.918 |
| 49.000 | 16.352 | 4.933 | 9.565 | 0.924 |
| 100.000 | 55.358 | 16.874 | 32.606 | 0.929 |
| 156.500 | 115.937 | 35.092 | 67.230 | 0.934 |

Interpretation: `z50` and `z90` are deposited-energy cumulative depths, whereas ESTAR `R_CSDA` is a continuous-slowing-down path-length range. They should agree in order and monotonic trend, not point-by-point equality.

Figure: `../figures/estar_geant4_depth_benchmark.png`

## 2. Raw Geant4 3D Grid to Step-4 Conservation

This check verifies that the original sparse 3D Geant4 deposition files are carried into Step 4 without changing total deposited energy or event count.

| Source | Raw file | Raw events | Step4 events | Raw Edep total (eV) | Step4 Edep total (eV) | Rel. error (%) |
| --- | --- | --- | --- | --- | --- | --- |
| 20 keV | grid3d_20keV.csv | 1.000e+05 | 1.000e+05 | 1.835e+09 | 1.835e+09 | -1.299e-14 |
| 49 keV | grid3d_49keV.csv | 1.000e+05 | 1.000e+05 | 4.528e+09 | 4.528e+09 | -2.106e-14 |
| 100 keV | grid3d_100keV.csv | 1.000e+05 | 1.000e+05 | 9.289e+09 | 9.289e+09 | 2.053e-14 |
| 156.5 keV | grid3d_156p5keV.csv | 1.000e+05 | 1.000e+05 | 1.462e+10 | 1.462e+10 | 0.000 |
| C-14 spectrum | grid3d_c14.csv | 9.999e+04 | 9.999e+04 | 4.617e+09 | 4.617e+09 | 2.066e-14 |

Maximum absolute relative error: 2.106e-14%.

## 3. Step-4 Energy-to-Generation Conservation

The check compares `raw_edep_total_eV / (N_events * E_eh)` with `sum(G) * V_voxel * T_int` from Step 4. This verifies the normalization used before writing `OpticalGeneration` into the TCAD input.

| Source | Neh Geant4 | Neh generation grid | Rel. error (%) |
| --- | --- | --- | --- |
| 20 keV | 2353.072 | 2353.072 | -1.933e-14 |
| 49 keV | 5805.293 | 5805.293 | 3.133e-14 |
| 100 keV | 1.191e+04 | 1.191e+04 | 0.000 |
| 156.5 keV | 1.875e+04 | 1.875e+04 | -1.941e-14 |
| C-14 spectrum | 5919.355 | 5919.355 | 0.000 |

Maximum absolute relative error: 3.133e-14%.

## 4. CCE Limit Check

The existing C-14 data support the mapped-source no-trap and trapped cases at the 120 um full-depletion baseline. The ideal uniform-generation control case is not present in the current `raw_data/tcad_it` set and should be run separately if we want to close this validation item exactly as written in `验证方法.md`.

| Test case | Wi (um) | Bias (V) | Nt (cm^-3) | Expected CCE (%) | TCAD CCE (%) | Status |
| --- | --- | --- | --- | --- | --- | --- |
| Ideal uniform generation, no trap |  |  | 0.000 | ~100 |  | needs dedicated control simulation |
| Geant4 mapped C-14, no trap | 120.000 | 75.000 | 0.000 | ~97.8 | 97.712 | available |
| Geant4 mapped C-14, Nt=1e13 | 120.000 | 75.000 | 1.000e+13 | lower than no-trap case | 73.324 | available |

## Files

- `../data/nist_estar_sic_table.csv`
- `../data/estar_geant4_depth_metrics.csv`
- `../data/raw3d_to_step4_conservation_table.csv`
- `../data/mapping_conservation_table.csv`
- `../data/cce_limit_check.csv`
- `../figures/estar_geant4_depth_benchmark.png`
- `../figures/estar_geant4_depth_benchmark.svg`
- `moscatelli_benchmark_next_steps.md`
