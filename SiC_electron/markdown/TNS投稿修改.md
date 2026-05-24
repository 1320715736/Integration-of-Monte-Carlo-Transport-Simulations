# TNS 投稿论文 Outline 

> 修订说明：
> - 去除全部 t90 / FWHM 相关指标，仅保留 CCE 作为单一品质因子
> - 章节结构对标 IEEE TNS 标准格式（Introduction → Methodology → Results → Discussion → Conclusion）
> - 核心叙事改为：**i 区厚度 × 外延层缺陷密度 的二维设计空间，CCE 作为唯一目标量**
> - 物理逻辑：无缺陷时 CCE 随厚度单调饱和（无最优）；引入材料固有 Z₁/₂、EH₆/₇ 缺陷后，厚端受陷阱捕获限制，CCE 变为非单调函数，**最优厚度自然涌现**
> - 图表清单单独列出在文档末尾

---

## Suggested Title

**"Geant4–Sentaurus Coupled Simulation of ¹⁴C Beta-Spectrum Detection in 4H-SiC PIN Detectors: Trade-off Between Intrinsic-Layer Thickness and Epitaxial Defect Density"**

---

## Abstract（约 200–250 词）

TNS 标准格式：

- 1 句应用背景（¹⁴C 监测的需求与固态探测器替代液闪的动机）
- 1 句方法（Geant4–Sentaurus 耦合仿真框架）
- 1 句关键设计变量（i 区厚度 × Z₁/₂ 缺陷密度二维空间）
- 2–3 句核心发现（缺陷使 CCE 从单调函数转为非单调函数；最优厚度随材料质量收缩；单能近似导致的设计偏差被材料缺陷放大）
- 1 句工程价值（提供基于外延材料质量等级的设计速查表）

---

## I. Introduction（约 1.5 页 / 2 栏）

### A. Background and Motivation

- ¹⁴C 监测的应用场景：核电站退役、低放废物表征、生物医学示踪剂、环境本底监测
- 现有方案（液闪计数器、气流计数器）的局限：体积大、需要试剂、难以便携化
- 固态半导体探测器作为替代的吸引力

### B. Why 4H-SiC

- 宽禁带 3.26 eV → 室温下极低暗电流，无需制冷
- 高位移阈能（25 eV）→ 高辐照硬度，适合长期工作
- 高载流子饱和速度 → 适合高计数率应用（背景介绍，不作为本文优化目标）
- 引 Frontiers review (2022) 给出综述性背景

### C. Knowledge Gap

明确指出**三个未填补的空白**：

1. SiC 探测器文献几乎全部针对 MIP（如 ⁹⁰Sr）、α 粒子或 X 射线优化设计，**针对低能连续 β 谱（如 ¹⁴C，E_max = 156.5 keV）的系统设计研究空白**
2. 现有研究常用平均能量或最大能量的单能 β 代替连续谱进行设计，但 ¹⁴C β 的能量沉积深度跨越数微米至上百微米，单能近似引入的设计偏差未被量化
3. 商用 4H-SiC 外延层中固有的 Z₁/₂、EH₆/₇ 深能级缺陷会在厚 i 区中造成显著载流子捕获损失，**厚度与材料质量之间的 trade-off 未被定量研究**

### D. Contributions

本文给出以下三点贡献：

1. **方法学贡献**：首个针对 ¹⁴C 连续 β 谱的 Geant4–Sentaurus 耦合仿真框架，包含双线性插值的二维能量沉积映射
2. **物理发现**：引入外延层固有缺陷模型后，CCE 由单调饱和函数转变为非单调函数，**最优 i 区厚度自然涌现**且随缺陷密度系统性变化
3. **工程贡献**：量化"用单能近似代替真实 ¹⁴C 谱"造成的设计偏差，并证明此偏差在材料缺陷增加时被放大；给出基于外延材料质量等级的设计速查表

### Key References（保留并补充）

- Gagg et al., "TCAD modeling of radiation-induced defects in 4H-SiC diodes", NIM A (2024) — 陷阱模型参数
- Kleppinger et al., J. Crystal Growth (2022) — 50/150/250 μm 外延层缺陷与探测器性能关系
- Kimoto, "Bulk and epitaxial growth of SiC", Prog. Crystal Growth (2016) — Z₁/₂ 缺陷综述
- Moscatelli et al., IEEE TNS 53, 1557 (2006) — SiC PIN MIP 探测，文献对标基准
- Bertuccio et al. (TNS 系列) — SiC X 射线探测器能量分辨率优化
- Nava et al., "Silicon carbide and its use as a radiation detector material" (2008)
- Frontiers Physics review (2022) — SiC 探测器综述

---

## II. Simulation Framework（约 2 页）

### A. Device Structure

- 4H-SiC PIN 结构：p⁺ (0.2 μm) / i-region (10–130 μm 扫描) / n⁺ (0.5 μm)
- 横向尺寸：240 μm × 240 μm
- i 区净掺杂：N_D = 5.6 × 10¹² cm⁻³
- 工作偏压：每个厚度取对应全耗尽电压 V_dep(W_i)
- → **Fig. 1**（示意图以 W_i = 120 μm 为参考器件展示，正文 caption 注明 "i-region thickness varied from 10 to 130 μm in this study"）

### B. Geant4 Monte Carlo Simulation

- Geant4 版本与物理列表（推荐 Livermore 或 Penelope，理由：覆盖 keV 量级低能电子的精确模拟）
- 粒子数：10⁵ 量级，配合收敛性检验
- ¹⁴C β⁻ 源：基于 Fermi 函数的理论谱采样（E_max = 156.5 keV，E_mean = 49 keV）
- 对照单能源：20, 49 (mean), 100, 156.5 (max) keV
- 能量沉积 → 体积网格离散化 → 生成率密度 G(x, y, z)
- → **Fig. 2**, **Fig. 3**, **Fig. 4**

### C. Sentaurus TCAD Simulation

- 求解方程：Poisson + 漂移扩散
- 载流子输运模型：4H-SiC 各向异性迁移率
- 复合模型：SRH（用于引入缺陷效应）
- 边界条件、网格密度、收敛准则
- 输出：稳态 C-V（提取 V_dep）+ 瞬态阴极电流 i(t)（用于 CCE 积分）

### D. Defect Model（⭐ 核心新增节，必须写扎实）

**叙事重点**：这是材料**固有**缺陷，不是辐照诱导。回答工程问题——"我能买到的 SiC 外延材料质量参差不齐，应该如何选择器件厚度？"

参数来源说明：

> "The dominant electrically active defect in as-grown n-type 4H-SiC epitaxial layers is the Z₁/₂ center (E_c − 0.67 eV), identified as a carbon vacancy-related defect. Its concentration in commercial epitaxial layers typically ranges from 10¹¹ to 10¹³ cm⁻³, depending on growth conditions and post-growth processing..."

**Table A: SRH Trap Parameters**

| Defect | Type | Energy Level | σ_e (cm²) | σ_h (cm²) | N_t range (cm⁻³) |
|--------|------|--------------|-----------|-----------|------------------|
| Z₁/₂ | Acceptor | E_c − 0.67 eV | 2×10⁻¹⁴ | ~10⁻¹⁵ | 10¹¹ – 5×10¹³ |
| EH₆/₇ | Donor | E_c − 1.55 eV | 2×10⁻¹⁴ | ~10⁻¹⁵ | 与 Z₁/₂ 同量级 |

扫描策略：

- 固定 Z₁/₂ : EH₆/₇ = 1 : 1
- N_t 对数扫描：10¹¹（研究级）、10¹²（商用高端）、10¹³（商用标准）、5×10¹³（工业级）
- 对应少数载流子寿命：~10 μs → ~10 ns

### E. CCE Computation

唯一品质因子：

$$
\text{CCE} = \frac{\int_0^{T_{\text{int}}} i_{\text{cathode}}(t)\,dt}{Q_{\text{generated}}} \times 100\%
$$

其中 Q_generated = E_dep / E_eh × q（E_eh = 7.8 eV）。积分时间窗 T_int 取至 i(t) 衰减到峰值 0.1% 以下，确保电荷收集完全。

> **说明**：瞬态 i(t) 仅作为 CCE 积分的中间量，不再报告 t90 等时间指标。

---

## III. Results（约 4–5 页）

### A. Device Electrical Characteristics

TNS 标准 PIN 探测器论文的开篇结果图——展示 Sentaurus 仿真出来的器件基本电学特性，建立后续 CCE 分析的物理基础。

**Fig. 5：baseline 器件 1/C²-V 特性（inset：C-V）**

- 只展示 baseline 器件（W_i = 120 μm）的稳态 C-V 特性，不再画 I-V 和 E(z)。
- 主图采用 **1/C²-V** 表示，这是 PIN / p-n 结 C-V 分析中展示耗尽过程和提取全耗尽电压的常用形式。
- inset 放原始 C-V 曲线，用于直观展示反偏升高后电容逐渐接近全耗尽几何电容。
- 在图中标注解析全耗尽电压：

$$
V_{dep}=\frac{qN_DW_i^2}{2\varepsilon_{SiC}}
$$

- 对 W_i = 120 μm、N_D = 5.6×10¹² cm⁻³ 的 baseline 器件，V_dep 约为 75 V。
- 该图的目的不是比较不同厚度或不同缺陷密度，而是确认器件在后续瞬态仿真使用的工作偏压附近已经进入全耗尽工作区。

**关键叙事**：Fig. 5 只承担器件电学自洽性验证功能。主图 1/C²-V 和 inset C-V 共同说明 baseline PIN 器件在约 75 V 附近进入全耗尽工作区；后续 CCE 厚度优化的核心证据由 Fig. 6–8 给出。

- → **Fig. 5**：baseline 4H-SiC PIN 的 1/C²-V 曲线（inset：C-V），标注解析 V_dep。

### B. ¹⁴C Beta Spectrum and Energy Deposition Profile

- Geant4 生成的 ¹⁴C β⁻ 谱与理论 Fermi 谱对比 → 验证源项准确性
- 能量分段说明：< 30 keV (33.3%), 30–80 keV (47.3%), > 80 keV (19.2%)
- dE/dx 深度分布对比：单能（20/49/100/156.5 keV）vs ¹⁴C 连续谱
- **关键观察**：¹⁴C 谱的沉积分布跨越 < 5 μm 至 ~100 μm，无法被任何单能源近似复现
- → **Fig. 3, Fig. 4**

### C. Effect of Defects on CCE（⭐ 全文核心结果）

**核心图**：CCE vs 厚度，参数化以 N_t（一族曲线），针对 ¹⁴C 连续谱

观察到的现象：

- N_t = 0（理想）：单调饱和
- N_t = 10¹¹：与理想几乎重合（厚端微小下降）
- N_t = 10¹²：厚端 CCE 开始明显下降
- N_t = 10¹³：CCE 出现明确峰值（先升后降）
- N_t = 5×10¹³：峰值进一步向薄端移动且峰值降低

**物理诠释**：

- 薄端：CCE 受限于几何覆盖率（高能 β 未被完全吸收）→ 厚度增加有利
- 厚端：CCE 受限于深处载流子的陷阱捕获 → 厚度增加有害
- 两个机制竞争 → 存在最优厚度

→ **Fig. 6**（论文最核心的图，审稿人看到这张图即理解全部贡献）

### D. Optimal Thickness as a Function of Material Quality

两张图配合呈现：

1. **最优 i 区厚度 vs N_t**（log scale）：五条曲线（20/49/100/156.5 keV/¹⁴C）
2. **最优 CCE vs N_t**：五条曲线

**关键观察**：

- 高能源（156.5 keV）受缺陷影响最大（穿透深，载流子要走最远）
- 低能源（20 keV）受缺陷影响最小（沉积浅，载流子靠近收集端）
- **¹⁴C 谱的最优厚度曲线与 49 keV（mean energy）显著不同**——这是反对单能近似的直接证据

→ **Fig. 7, Fig. 8**

### E. Design Error from Mono-Energetic Approximation

**交叉评估矩阵**：

- 行：用某个能量 X 优化得到的器件（X ∈ {49 keV, 100 keV, 156.5 keV, ¹⁴C}）
- 列：实际入射的能量或谱 Y
- 单元格：CCE 值

对每个 N_t 水平画一张矩阵（或合并为热图）。

**关键观察**：

- 用 49 keV 优化的设计在 ¹⁴C 实际探测中 CCE 损失 ~Δ₁%
- 用 156.5 keV 优化（保守）的设计 CCE 损失 ~Δ₂%
- 用 ¹⁴C 谱直接优化的设计才能最大化 ¹⁴C 探测 CCE
- **这个偏差在高缺陷材料下被放大**：低质量材料中错误的厚度选择损失更严重

→ **Fig. 9**

---

## IV. Discussion（约 1 页）

### A. Material-Quality-Driven Design Recommendations

→ **Table I**：设计速查表

| 材料质量等级 | N_t (cm⁻³) | τ (ns) | 推荐 W_i (μm) | V_dep (V) | 预期 CCE (%) |
|--------------|------------|--------|---------------|-----------|---------------|
| Research-grade | ~10¹¹ | ~10⁴ | ~100 | ~52 | ~97.5 |
| Commercial high | ~10¹² | ~10³ | ~70 | ~26 | ~96 |
| Commercial standard | ~10¹³ | ~100 | ~40 | ~8.4 | ~85 |
| Industrial | ~5×10¹³ | ~20 | ~25 | ~3.3 | ~70 |

（数值待仿真填入）

### B. Indirect Validation Against Literature

- Kleppinger et al. (2022) 实测 50/150/250 μm 三组器件，250 μm 能量分辨率最差——本文仿真预测在合理 N_t 下厚器件 CCE 下降，趋势一致
- Moscatelli 2006 实测 55 μm SiC PIN 在 ⁹⁰Sr 下 CCE ≈ 100%——本文仿真在 N_t < 10¹² 下复现此结果
- NJU 2022 实测 100 μm 4H-SiC PIN 在 ⁹⁰Sr 下高 CCE——本文仿真验证此厚度在低 N_t 区间为合理设计

### C. Model Limitations

- 漂移扩散近似的适用性
- 表面复合与边缘场效应被简化
- 仅纳入 Z₁/₂ 和 EH₆/₇ 主导缺陷，未包含其他次要中心
- 未包含读出电子学噪声（FWHM 的电子学项）
- 仅讨论室温（300 K），未涵盖高温场景

### D. Connections and Implications

- 同一框架可推广到 ⁶³Ni (E_max = 66.7 keV)、³H (E_max = 18.6 keV) 等其他低能 β 源
- 缺陷模型形式上与辐照诱导缺陷等价，本文结论可外推至辐照退化场景
- betavoltaic cell 设计共享同一厚度-材料质量 trade-off，方法学可迁移

---

## V. Conclusion（约 0.5 页）

三段式结论：

1. **方法学贡献**：建立 Geant4–Sentaurus 耦合仿真框架，针对 ¹⁴C 连续 β 谱在 4H-SiC PIN 中的探测响应做系统模拟，并引入材料缺陷模型

2. **物理发现**：在理想材料中 CCE 是厚度的单调饱和函数；引入 Z₁/₂、EH₆/₇ 缺陷后 CCE 变为非单调函数，**最优 i 区厚度作为材料缺陷密度的函数自然涌现**；¹⁴C 连续谱的最优厚度与单能近似显著不同，且该差异在低质量材料中被放大

3. **工程价值**：为 ¹⁴C 探测应用提供了基于外延材料质量等级的器件厚度设计速查表

---

# 📊 图表清单（统一编号）

## ✅ 保留自原 outline 的图（4 张）

| 编号 | 内容 | 数据来源 | 主要展示信息 |
|------|------|---------|--------------|
| **Fig. 1** | 4H-SiC PIN 器件结构三维示意图 | 原 Fig. 1（Image 8） | p⁺/i/n⁺ 层厚、横向尺寸、入射方向 |
| **Fig. 2** | Geant4–Sentaurus 耦合 workflow（含 2D 生成率示例） | 原 Fig. 2 + 2b（Image 4 + Image 1） | 仿真链路 + 沉积映射示例 |
| **Fig. 3** | ¹⁴C 理论 β 谱 vs Geant4 模拟谱（含三段能量区间标注） | 原 Fig. 3（Image 11） | 源项验证 + 谱形特征 |
| **Fig. 4** | dE/dx vs 深度，单能源（4 种能量）vs ¹⁴C 连续谱 | 原 Fig. 4（Image 10） | 沉积分布跨越数量级 |

## 🆕 必须新增的核心图（5 张）

| 编号 | 内容 | 重要性 | 备注 |
|------|------|--------|------|
| **Fig. 5** | baseline 器件 1/C²-V 特性（W_i = 120 μm，inset 为 C-V，标注 V_dep） | ⭐ | 只用于确认 baseline PIN 器件全耗尽，不再画 I-V 和 E(z) |
| **Fig. 6** | CCE vs i 区厚度，**参数化以 N_t**（5 档），针对 ¹⁴C 连续谱（N_t = 0 曲线作为基线对照内嵌其中） | ⭐⭐⭐ | **全文核心图**，展示 CCE 非单调性 |
| **Fig. 7** | 最优 i 区厚度 vs N_t，五条曲线对应五个源（log x-axis） | ⭐⭐ | 设计指南图 |
| **Fig. 8** | 最优 CCE vs N_t，五条曲线 | ⭐ | 配合 Fig. 7，量化材料质量惩罚 |
| **Fig. 9** | 交叉评估矩阵（设计能量 × 实际能量 × CCE），热图形式 | ⭐⭐ | 量化单能近似设计偏差 |

## ❌ 删除的原有图

| 原编号 | 内容 | 删除原因 |
|--------|------|----------|
| 原 Fig. 5 | i-t 瞬态电流曲线（五条能量曲线） | t90 指标已删除；i(t) 仅作 CCE 积分中间量 |
| 原 Fig. 6b | t90 vs 厚度 | t90 指标已删除 |
| 原 Fig. 6c | CCE-t90 Pareto 前沿 | Pareto 分析已放弃 |
| 原 Fig. 7a | 推荐厚度柱状图 | 内容并入新 Fig. 7 + Table I |
| 原 Fig. 7b | CCE & t90 柱状图 | t90 指标已删除 |

## 📋 表格

| 编号 | 内容 | 位置 |
|------|------|------|
| **Table A** | 4H-SiC SRH 缺陷参数 | Section II-D |
| **Table I** | 材料质量等级 → 推荐器件设计速查表 | Section IV-A |

## 总计

**9 张图 + 2 张表**

- Fig.1–4：器件结构、workflow、源验证、能量沉积（保留原图）
- Fig.5：baseline 器件 1/C²-V（inset：C-V），全耗尽验证
- Fig.6：CCE vs 厚度 vs N_t 一族曲线（核心图，含 N_t=0 基线对照）
- Fig.7–8：最优厚度 / 最优 CCE 随材料质量变化
- Fig.9：单能 vs 连续谱交叉评估

对照 TNS regular paper 8–10 页篇幅，9 张图属于可接受的标准密度。

---

# 仿真工作量估算

| 任务 | 仿真量 | 时间 |
|------|--------|------|
| 缺陷模型在 Sentaurus 中调试与验证 | – | 1–2 天 |
| CCE 主扫描：5 个 N_t × 13 个厚度 × 5 个源 = **325 个仿真点** | 大头 | ~5–7 天 |
| 数据后处理（提取 CCE、画 Fig. 6–9） | – | 2–3 天 |
| 文献对比数据查找（Section IV-B） | – | 1 天 |
| **总计** | | **约 1.5–2 周** |

由于不再需要分析 t90，瞬态时间窗仅需积分到 CCE 收敛（典型 10–20 ns），单次仿真时间相比原方案减少 ~30–50%。

---

# 章节字数预算（TNS regular paper 8 页）

| 章节 | 字数 | 页数 |
|------|------|------|
| Abstract | 200–250 | 0.25 |
| I. Introduction | 1500 | 1.5 |
| II. Simulation Framework | 1800 | 2.0 |
| III. Results | 3500 | 4.0 |
| IV. Discussion | 800 | 1.0 |
| V. Conclusion | 400 | 0.5 |
| References | – | 0.75 |
| **总计** | ~8000 | **~10 页** |

---

# 审稿人可能的质疑与应对

**Q1: "为什么用缺陷密度而不是辐照注量作参数？"**

> 我们有意将分析框架与特定的辐照历史解耦。Z₁/₂ 缺陷既存在于 as-grown 材料中，也可由辐照引入。以缺陷密度（或等价的少子寿命）为参数的设计曲线更具通用性——设计者只需通过 μ-PCD 等技术测量获得的材料的少子寿命，即可从我们的设计图中读取最优几何。

**Q2: "没有实验验证如何保证可信度？"**

> 我们的 Geant4 β 谱经过了与理论 Fermi 谱的严格验证（Fig. 3）。TCAD 模型参数取自经过实验校准的文献（Gagg et al. 2024, Kleppinger et al. 2022）。无缺陷情况下的 CCE 趋势与 Moscatelli 2006、NJU 2022、SICAR 2025 等已发表的 SiC 探测器实验数据在量级和趋势上一致（Section IV-B）。

**Q3: "¹⁴C 的 β 平均射程只有 30–40 μm，为什么不直接用薄器件？"**

> 这正是我们的核心发现之一。对于高质量材料（N_t < 10¹²），由于连续 β 谱中约 20% 的粒子能量超过 80 keV、射程可达数十微米，较厚的器件确实能提供更高的 CCE。但对于普通商用材料（N_t > 10¹³），最优厚度会向薄端显著回缩。设计决策严格依赖于可获得的外延材料质量。

**Q4: "为什么 t90 也很重要，但论文里没有？"**

> 本文聚焦于 ¹⁴C 计数应用中最核心的 CCE 优化。时间响应在不同应用场景下重要性不同（高计数率 vs 谱学）。t90 等时间指标的优化是本文方法学框架的自然延伸，将在后续工作中针对具体应用场景展开。

---

# 与现有文献的差异化（reviewer 最关心）

| 对比项 | 已有工作 | 本文 |
|--------|----------|------|
| 射线源 | MIP（⁹⁰Sr）、α、单能 e⁻ | **¹⁴C 连续 β 谱（E_max = 156.5 keV）** |
| 仿真工具 | TCAD only 或 MC only | **Geant4 + Sentaurus 严格耦合** |
| 优化变量 | 通常固定厚度，研究辐照 | **i 区厚度 × 缺陷密度二维空间** |
| 缺陷来源 | 辐照诱导 | **材料固有（更通用）** |
| 设计输出 | 单器件性能报告 | **参数化设计曲线 + 材料等级速查表** |
| FoM | 多种（CCE、FWHM、σ_t 等） | **聚焦 CCE 单一目标，避免方法论分散** |
