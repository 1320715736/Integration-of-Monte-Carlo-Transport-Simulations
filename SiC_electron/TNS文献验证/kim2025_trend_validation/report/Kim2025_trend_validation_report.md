# Kim 2025 文献趋势验证初步结果

本报告基于 `SiC_electron/raw_data/tcad_it/*.csv` 中当前已经完成的 TCAD i-t 数据生成。由于厚度扫描尚未全部完成，所有结论均只针对当前已有数据。

## 输出文件

- 指标总表：`../data/validation_metrics.csv`
- 覆盖矩阵：`../data/coverage_matrix.csv`
- 缺失组合：`../data/missing_combinations.csv`
- 趋势 1 统计：`../data/trend1_charge_vs_Nt_summary.csv`
- 趋势 2 统计：`../data/trend2_loss_vs_thickness_summary.csv`
- 趋势 3 统计：`../data/trend3_best_thickness_summary.csv`
- 图件目录：`../figures/`

## 数据覆盖

- 已解析曲线数：`364`
- 完整 `Nt=0,1e11,1e12,1e13,5e13` 组合数：`69`
- 不完整厚度组合数：`6`
- 初始点疑似异常曲线数：`41`

首点疑似异常曲线没有丢弃；主验证计算对这些曲线使用 `median_first_10_points_initial_outlier` 基线，其余曲线保持前 5 点均值基线。

未完成组合已写入 `missing_combinations.csv`。趋势图会保留缺失处的断点，不做插值。

## 趋势 1：trap density 增大时积分电荷变化

对照 Kim 2025 Fig.7：trap density 增大时，JSC/VOC/Pout_max 下降。本项目用瞬态积分得到的 `CCE = Qcol/Qgen` 对应这个趋势。`Qcol(Nt)/Qcol(Nt=0)` 仍保留在统计表中用于归一化检查。

| Source | Nt=0 | Nt=1e11 | Nt=1e12 | Nt=1e13 | Nt=5e13 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10 keV | 46.88 | 46.88 | 46.85 | 45.53 | 31.04 |
| 20 keV | 73.94 | 73.94 | 73.90 | 72.93 | 46.23 |
| 30 keV | 96.40 | 96.39 | 96.34 | 94.98 | 63.66 |
| 49 keV | 97.68 | 97.67 | 97.63 | 94.30 | 64.79 |
| 100 keV | 99.31 | 99.27 | 99.19 | 85.03 | 53.83 |
| 156.5 keV | 76.67 | 76.56 | 75.29 | 57.69 | 29.29 |
| C-14 spectrum | 97.17 | 97.16 | 97.04 | 87.13 | 58.08 |

## 趋势 2：厚器件更受 trap 影响

对照 Kim 2025 Fig.6：薄 i-layer 下 trap/no-trap 差别小，厚 i-layer 下差别扩大。这里比较 `<=30 um` 和 `>=80 um` 的中位 charge loss。

注意：overview 第二幅图只画 `thick loss - thin loss` 的中位差，把完整厚度曲线压缩成一个柱子，所以不同源项之间的视觉差异会被弱化。判断趋势 2 时应优先看各源项子图 `trend2_loss_vs_thickness/*.png`，而不是只看 overview 柱状图。

| Source | Nt | thin loss % | thick loss % | thick-thin pp |
| --- | ---: | ---: | ---: | ---: |
| 10 keV | 1e13 | 0.21 | 19.91 | 19.70 |
| 10 keV | 5e13 | 1.40 | 48.38 | 46.98 |
| 20 keV | 1e13 | 0.19 | 19.99 | 19.81 |
| 20 keV | 5e13 | 1.37 | 49.45 | 48.08 |
| 30 keV | 1e13 | 0.16 | 20.01 | 19.85 |
| 30 keV | 5e13 | 1.30 | 48.47 | 47.17 |
| 49 keV | 1e13 | 0.13 | 18.44 | 18.31 |
| 49 keV | 5e13 | 1.07 | 48.33 | 47.26 |
| 100 keV | 1e13 | 0.08 | 18.18 | 18.10 |
| 100 keV | 5e13 | 3.26 | 49.66 | 46.40 |
| 156.5 keV | 1e13 | 0.07 | 27.17 | 27.11 |
| 156.5 keV | 5e13 | 3.63 | 69.03 | 65.41 |
| C-14 spectrum | 1e13 | 0.11 | 18.55 | 18.43 |
| C-14 spectrum | 5e13 | 1.98 | 48.67 | 46.69 |

## 趋势 3：有缺陷后 CCE vs thickness 的最优厚度

对照 Kim 2025 Fig.6：无 trap 时性能随 i-layer 厚度增加并趋于饱和；有 trap 时存在最优厚度，厚端可能下降。下表列出当前已有数据中 CCE 最大的厚度。若 `boundary=True`，说明最大值落在当前扫描边界，不能断言已经看到 rollover。

| Source | Nt=0 | Nt=1e12 | Nt=1e13 | Nt=5e13 |
| --- | ---: | ---: | ---: | ---: |
| 10 keV | 100 um | 100 um | 50 um | 10 um boundary |
| 20 keV | 110 um boundary | 110 um boundary | 10 um boundary | 10 um boundary |
| 30 keV | 110 um boundary | 110 um boundary | 10 um boundary | 10 um boundary |
| 49 keV | 110 um boundary | 110 um boundary | 20 um | 20 um |
| 100 keV | 80 um | 80 um | 60 um | 30 um |
| 156.5 keV | 100 um boundary | 100 um boundary | 70 um | 90 um boundary |
| C-14 spectrum | 100 um boundary | 100 um boundary | 60 um | 30 um |

## 解释口径

本次验证使用 `Qcol` 和 `CCE`，不使用 `Imax` 作为主要退化指标。`Imax` 可能受电场重分布、脉冲变窄和感应电流形状影响，不能直接等同于总收集电荷。

建议在论文中表述为：虽然 Kim 2025 使用 betavoltaic J-V 提取，而本文使用瞬态电流积分，但积分收集电荷随 trap density 增加而下降、厚 i 区退化更明显、以及高缺陷条件下最优厚度前移的趋势，可作为一致的 trap-assisted recombination 证据链。
