# Trend 3 selected-source best CCE statistics

Selected sources: 20 keV, 49 keV, 100 keV, 156.5 keV, and C-14 spectrum.

Best CCE definition: within each source and Nt group, choose the available thickness with maximum `CCE_signed_percent`.
A `boundary` label means the maximum lies at the scanned thickness edge, so the true optimum may lie outside the current scan range.

## Best thickness and best CCE

| Source | Nt=0 | Nt=1e11 | Nt=1e12 | Nt=1e13 | Nt=5e13 |
| --- | --- | --- | --- | --- | --- |
| 20 keV | 110 boundary, CCE=74.25% | 110 boundary, CCE=74.25% | 110 boundary, CCE=74.20% | 10 boundary, CCE=73.83% | 10 boundary, CCE=73.63% |
| 49 keV | 110 boundary, CCE=97.75% | 110 boundary, CCE=97.74% | 110 boundary, CCE=97.68% | 20, CCE=97.53% | 20, CCE=96.61% |
| 100 keV | 80, CCE=99.39% | 80, CCE=99.39% | 80, CCE=99.34% | 60, CCE=97.41% | 30, CCE=67.04% |
| 156.5 keV | 100 boundary, CCE=99.02% | 100 boundary, CCE=98.98% | 100 boundary, CCE=98.56% | 70, CCE=70.88% | 90 boundary, CCE=30.24% |
| C-14 spectrum | 100 boundary, CCE=97.80% | 100 boundary, CCE=97.79% | 100 boundary, CCE=97.74% | 60, CCE=95.11% | 30, CCE=81.97% |

## Aggregate trend across selected sources

| Nt | median best thickness um | mean best thickness um | median shift from Nt=0 um | median best CCE % | median best-CCE drop pp | boundary/internal |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 100 | 100 | 0 | 97.797494398 | 0 | 4/1 |
| 1e11 | 100 | 100 | 0 | 97.793917353 | 0.005370516 | 4/1 |
| 1e12 | 100 | 100 | 0 | 97.738915173 | 0.058579225 | 4/1 |
| 1e13 | 60 | 44 | -40 | 95.109428275 | 1.985834895 | 1/4 |
| 5e13 | 30 | 36 | -70 | 73.630963154 | 15.83236621 | 2/3 |

## Interpretation

- The selected-source set keeps the cases most relevant to the manuscript argument and excludes 10 keV / 30 keV from the main Trend 3 statistics.
- In the current data window, 100 keV and C-14 show clear internal optimum shifts under high Nt.
- 20 keV and 49 keV move to the low-thickness edge under high Nt, which still supports trap-limited thinning but should be described as a boundary optimum.
- 156.5 keV still has boundary sensitivity at Nt=5e13 because the current high-Nt dataset ends at 90 um; this point should be treated cautiously until the missing thick points are completed.
