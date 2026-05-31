# Optimization Design of 4H-SiC PIN Devices Based on Geant4-TCAD Simulation

## Abstract

4H-SiC is a promising material for C-14 beta-spectrum detection because of its wide bandgap, high critical breakdown field, low leakage current, and strong radiation tolerance. C-14 is a pure beta-emitting radionuclide with a maximum electron energy of 156.5 keV and an average energy of about 49 keV. Its emitted electrons follow a continuous energy spectrum. In beta-detector simulations, the real beta spectrum is often replaced by a monoenergetic electron source at the average or maximum energy, but the validity of this approximation for transient current response remains unclear. In this work, a coupled Geant4-TCAD simulation method is established to extract charge collection efficiency, peak current, and t90. The results show that monoenergetic electrons can give CCE values close to those of the real C-14 spectrum in some cases, but their transient responses are not equivalent. Based on this conclusion, full-depletion i-region thickness design is performed directly for the real C-14 spectrum, and the recommended detector parameters are obtained. Finally, the same method is extended to selected monoenergetic source terms, giving source-dependent full-depletion device parameters for SiC PIN detector design driven by the real C-14 spectrum.

**Keywords:** 4H-SiC; PIN detector; C-14; Geant4; TCAD; monoenergetic approximation; charge collection efficiency

## 1. Introduction

### 1.1 Background

Wide-bandgap semiconductor radiation detectors have clear advantages in high-temperature, high-radiation, and low-leakage applications. Compared with conventional silicon detectors, 4H-SiC has a bandgap of about 3.26 eV, which can significantly suppress room-temperature dark current. Its high critical breakdown field enables operation under relatively high reverse bias. In addition, SiC has a high displacement threshold energy and good chemical stability, making it suitable for long-term operation in nuclear radiation environments. Therefore, 4H-SiC PIN diodes are representative semiconductor devices for modeling C-14 beta-spectrum response and optimizing detector structures.

C-14 decays according to

$$
{}^{14}\mathrm{C}\rightarrow{}^{14}\mathrm{N}+e^-+\bar{\nu}_e
$$

Because the antineutrino carries a continuous amount of energy, the emitted C-14 electron is not monoenergetic. Instead, its energy is continuously distributed from near zero to a maximum energy of 156.5 keV, with an average energy of about 49 keV. For a 4H-SiC PIN detector, this continuous spectrum produces a spatial generation distribution composed of shallow deposition and a deeper tail. The real spectrum therefore affects not only the total generated charge, but also the carrier drift path, transient current peak, and charge collection time.

### 1.2 Scientific Problem

In existing Geant4-TCAD device-response simulations, the real continuous beta spectrum is often replaced by a monoenergetic electron at the average or maximum energy to reduce computational complexity. This approximation may give similar results for integral metrics such as total deposited energy or total charge collection efficiency. However, transient current response also depends on carrier generation depth, local electric field, drift path, and collection-time distribution. Therefore, whether monoenergetic approximation can be used to analyze the i-t waveform, peak current, and t90 of C-14 detectors must be re-evaluated under a unified device structure and simulation workflow.

This leads to two core questions. First, can a monoenergetic electron source replace the real C-14 spectrum in device-response simulation, and if so, what is the validity boundary of this approximation. Second, if the real C-14 spectrum cannot be reliably replaced for transient response, should the device i-region thickness and full-depletion bias be designed directly for the real spectrum, and how should the relevant parameters be selected.

### 1.3 Contributions

This work addresses the above questions and makes the following contributions.

1. A coupled Geant4-TCAD simulation workflow is established. Energy deposition from the real C-14 continuous beta spectrum and selected monoenergetic particle sources in 4H-SiC is converted into a spatial charge-generation distribution that can be imported into TCAD. CCE, peak current, and t90 are extracted under the same device structure and post-processing procedure.

2. The validity boundary for replacing the real C-14 source with monoenergetic particle sources is clarified. The results show that monoenergetic approximation can reproduce some CCE results, but cannot replace the real spectrum for transient i-t response, peak current, or charge collection time analysis.

3. A full-depletion i-region design is performed for a 4H-SiC PIN detector under the real C-14 spectrum. The i-region thickness is scanned from 10 to 130 um using the corresponding full-depletion bias, and the recommended structure is selected using CCE and t90 as joint metrics.

4. The design method derived from the real C-14 spectrum is extended to selected monoenergetic source terms. Recommended device parameters are obtained for 20, 49, 100, and 156.5 keV incident conditions, showing that the optimal i-region thickness depends on the source spectrum and deposition-depth distribution.

## 2. Device Structure and Geant4-TCAD Coupling Method

### 2.1 4H-SiC PIN Device Structure

The studied device is a vertically irradiated 4H-SiC PIN detector. It consists of a top p+ contact layer, a lightly doped i region, and a bottom n+ contact layer. Electrons enter from the p+ side. The baseline structure uses a 0.2 um p+ layer, a 120 um i region, and a 0.5 um n+ layer. The net doping concentration in the i region is $5.6\times10^{12}\ \mathrm{cm^{-3}}$, and the p+ and n+ contact regions are doped at $1\times10^{19}\ \mathrm{cm^{-3}}$.

![Fig.1 Schematic structure of the 4H-SiC PIN detector](figures/fig1_sic_pin_structure.png)

**Fig.1.** Schematic structure of the 4H-SiC PIN detector. C-14 beta electrons are incident from the p+ side, and the device operates under reverse bias.

**Table 1. Device structure and material parameters**

| Parameter | Value |
| --- | ---: |
| p+ thickness | 0.2 um |
| Baseline i-region thickness | 120 um |
| n+ thickness | 0.5 um |
| Net doping in i region | $5.6\times10^{12}\ \mathrm{cm^{-3}}$ |
| p+ / n+ doping | $1\times10^{19}\ \mathrm{cm^{-3}}$ |
| Full-depletion bias for 120 um | 75 V |

For different i-region thicknesses, the full-depletion bias is estimated using the one-dimensional depletion approximation:

$$
V_{\mathrm{dep}}(W_i)=\frac{qN_DW_i^2}{2\varepsilon_{\mathrm{SiC}}}.
$$

### 2.2 Coupling Workflow from Geant4 to TCAD

Geant4 is used to calculate the energy deposition of different source terms in the 4H-SiC PIN structure. The energy deposition distribution is then converted into a spatial charge-generation distribution that can be imported into TCAD. The overall workflow is shown in Fig.2.

![Fig.2 Geant4-TCAD coupled simulation workflow](figures/fig2_geant4_tcad_workflow.png)

**Fig.2.** Geant4-TCAD coupled simulation workflow. The Geant4 input is either the real C-14 spectrum or a monoenergetic electron source. Geant4 provides the depth-dependent energy deposition, which is then converted into an electron-hole pair generation distribution and imported into TCAD to extract transient current and response metrics.

Geant4 records the energy deposition generated by different particle sources entering SiC. For each source term, 100,000 incident particles are simulated to obtain the single-particle averaged deposition distribution. This distribution is then converted into a TCAD-readable spatial charge-generation distribution using an average electron-hole pair creation energy of $E_{eh}=7.8\ \mathrm{eV}$ in 4H-SiC.

Next, the Geant4-generated spatial charge-generation data are written into the TCAD input data file. This procedure establishes the simulation link from Geant4 to TCAD.

In this work, t90 is defined as the time at which the cumulative collected charge reaches 90% of the final collected charge:

$$
\int_0^{t90}i(t)\,dt=0.9Q_{\mathrm{collect}}
$$

## 3. Validity Boundary for Replacing the Real C-14 Spectrum with Monoenergetic Sources

### 3.1 C-14 Beta Spectrum

C-14 beta decay produces a continuous electron energy spectrum. Its probability density is related to the electron momentum, total energy, remaining decay energy, and Coulomb correction factor. Fig.3 shows the C-14 beta spectrum used in this work and the corresponding Geant4 sampling result.

![Fig.3 C-14 beta spectrum](figures/fig3_c14_spectrum.png)

**Fig.3.** C-14 beta spectrum. The spectrum is not monoenergetic but spans multiple energy bands. The average energy is about 49 keV, and the maximum energy is 156.5 keV.

This spectral shape means that electrons from different energy bands have different deposition depths in 4H-SiC. Low-energy electrons mainly deposit energy near the surface, while high-energy electrons contribute to the deeper deposition tail. Therefore, replacing the real spectrum with a single energy point may reproduce the total deposited energy, but cannot simultaneously reproduce the shallow peak and the deep tail.

### 3.2 Depth Distribution of Energy Deposition for Different Source Terms

Fig.4 compares the depth-dependent energy deposition distributions of the real C-14 spectrum and 10, 20, 30, 49, 100, and 156.5 keV monoenergetic electrons in the 4H-SiC PIN structure.

![Fig.4 Depth-dependent energy deposition distributions for different source terms in 4H-SiC](figures/fig4_dedx_profiles.png)

**Fig.4.** Depth-dependent energy deposition distributions for different source terms in 4H-SiC. The real C-14 spectrum contains both strong shallow deposition and a deep long tail, whereas any single monoenergetic electron can represent only part of the spectral behavior.

The 20 keV monoenergetic electron deposits energy almost entirely near the surface and is more sensitive to the p+ dead layer and shallow i region. The 49 keV monoenergetic electron is close to the average C-14 energy, but its deposition depth remains limited. The 100 keV and 156.5 keV monoenergetic electrons have stronger penetration capability and significantly enhance deep deposition. The real C-14 spectrum is a weighted result of contributions from different energy bands. It is neither equivalent to the average-energy monoenergetic electron nor to the maximum-energy monoenergetic electron.

### 3.3 Comparison of Transient i-t Response

To quantitatively evaluate the validity boundary of monoenergetic approximation, transient TCAD simulations are performed for five inputs under the same 4H-SiC PIN structure, reverse bias, and post-processing procedure: 20 keV, 49 keV, real C-14, 100 keV, and 156.5 keV. The corresponding current waveforms are shown in Fig.5.

![Fig.5 Transient i-t curves under 20 keV, 49 keV, C-14, 100 keV, and 156.5 keV inputs](figures/fig5_it_curves.png)

**Fig.5.** Transient cathode current responses under 20 keV, 49 keV, real C-14, 100 keV, and 156.5 keV inputs. CCE values from different source terms can be close, but peak current, waveform shape, and t90 are not equivalent.

**Table 3. Comparison of CCE, Ipeak, and t90 between monoenergetic source terms and the real C-14 spectrum**

| Input | CCE (%) | Ipeak (nA) | t90 (ns) |
| --- | ---: | ---: | ---: |
| 20 keV | 74.08 | 1.180 | 4.875 |
| 49 keV | 97.62 | 1.990 | 4.809 |
| C-14 spectrum | 97.66 | 2.170 | 4.763 |
| 100 keV | 99.28 | 3.190 | 4.671 |
| 156.5 keV | 99.56 | 4.490 | 5.532 |

Table 3 shows that if only CCE is considered, the 49 keV average-energy monoenergetic electron is very close to the real C-14 spectrum, with a difference of only about 0.04 percentage points. Although 100 keV and 156.5 keV overestimate CCE, their deviations remain within a few percentage points.

However, the transient response leads to a different conclusion. The peak current of the real C-14 spectrum is about 2.170 nA, whereas the 49 keV monoenergetic electron gives 1.990 nA, 100 keV gives 3.190 nA, and 156.5 keV gives 4.490 nA. Therefore, the average-energy monoenergetic electron underestimates the peak current, while high-energy monoenergetic source terms significantly overestimate it. t90 also differs, especially for the maximum-energy 156.5 keV case, where t90 increases to 5.532 ns.

This indicates that CCE is a time-integrated total metric and can hide differences in carrier generation position and drift-time distribution. For readout circuit design, time integration window setting, and pulse-shape analysis, monoenergetic electrons cannot reliably replace the real C-14 spectrum. The first core conclusion of this work is therefore that monoenergetic approximation can be used for rough estimation of some integral metrics, but it cannot reliably replace the real C-14 spectrum for transient i-t response analysis.

This section shows that the transient response of the real C-14 spectrum cannot be reliably replaced by monoenergetic source terms. Therefore, if the goal is to design a C-14 detector, the real C-14 spectrum should be used directly as the device-optimization input, rather than using only the average energy of 49 keV or the maximum energy of 156.5 keV as a representative source.

## 4. Full-Depletion i-Region Design for the Real C-14 Spectrum

Full-depletion simulations are performed for detectors with different i-region thicknesses. This avoids comparing devices with different electric-field conditions under the same external bias. Each thickness is compared under its own just-fully-depleted condition, so that the effects of thickness on deposition coverage and drift path are clearer.

### 4.1 Thickness-Scan Results under the C-14 Spectrum

Fig.6 gives the CCE and t90 values for different source terms under full-depletion bias over an i-region thickness range from 10 to 130 um. The C-14 curve is the primary focus here.

![Fig.6a CCE as a function of i-region thickness under full-depletion thickness scan](figures/fig6_energy_cce.png)

![Fig.6b t90 as a function of i-region thickness under full-depletion thickness scan](figures/fig6_energy_t90.png)

**Fig.6.** CCE and t90 for selected source terms at different i-region thicknesses and corresponding full-depletion biases. The C-14 curve shows that CCE enters a high-collection plateau near 60 um, while t90 generally increases with thickness.

**Table 4. Full-depletion metrics for different i-region thicknesses under the C-14 spectrum.**

| Wi (um) | Vdep (V) | CCE (%) | t90 (ns) | Ipeak (nA) |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 1 | 58.156 | 3.918 | 2.127 |
| 20 | 3 | 81.075 | 5.864 | 2.145 |
| 30 | 5 | 90.927 | 6.777 | 1.781 |
| 40 | 9 | 94.965 | 6.211 | 1.753 |
| 50 | 14 | 96.685 | 6.200 | 1.740 |
| 60 | 19 | 97.332 | 6.559 | 1.688 |
| 70 | 26 | 97.610 | 6.569 | 1.701 |
| 80 | 33 | 97.738 | 6.809 | 1.681 |
| 90 | 42 | 97.811 | 6.839 | 1.687 |
| 100 | 52 | 97.828 | 6.882 | 1.690 |
| 110 | 63 | 97.757 | 6.933 | 1.688 |
| 120 | 75 | 97.759 | 6.986 | 1.686 |
| 130 | 88 | 97.785 | 7.021 | 1.686 |

Table 4 shows that CCE under the C-14 spectrum increases rapidly with i-region thickness. The 10 um device covers only about 61.5% of the generation region, and its CCE is only 58.156%. The 20 um and 30 um devices still suffer from clear deposition truncation. When the thickness increases to 50 um, CCE reaches 96.685%. At 60 um, CCE further increases to 97.332%. With further increases to 70-130 um, CCE enters a plateau, with a maximum value of about 97.828%.

However, t90 does not continuously improve with increasing thickness. A thicker i region can cover the deeper deposition tail, but it also increases the carrier drift path and lengthens the charge collection time. For the C-14 spectrum, t90 is about 6.57-7.02 ns for thicknesses from 70 to 130 um, while the 60 um device has a t90 of 6.559 ns, already close to the shorter response times within the high-CCE plateau.

### 4.2 Recommended C-14 i-Region Parameter

The design rule used in this work is as follows. First, the maximum CCE under the given source term is identified. Then, all thicknesses with CCE within 0.5 percentage points of that maximum are retained as candidates. Finally, the candidate with the shortest t90 is selected. This rule avoids the inevitable bias toward the thinnest device when only t90 is considered, and also avoids pushing the thickness toward overly thick structures when only CCE is considered.

For the C-14 spectrum, the maximum CCE is about 97.828%, and the 0.5 percentage-point window gives a threshold of 97.328%. The 60 um device has a CCE of 97.332%, just entering the high-CCE candidate region, while its t90 is shorter than that of devices with thicknesses of 70 um and above. Therefore, under the current rule, the recommended structure is

$$
W_i=60\ \mu\mathrm{m},\quad V_{\mathrm{dep}}=19\ \mathrm{V}.
$$

The main metrics of this structure are

$$
\mathrm{CCE}=97.332\%,\quad t90=6.559\ \mathrm{ns},\quad I_{\mathrm{peak}}=1.688\ \mathrm{nA}.
$$

This result shows that once the real C-14 spectrum is confirmed to be non-replaceable for transient response, device structure optimization should be performed directly for the real C-14 spectrum. The 60 um structure is not the maximum-CCE point. Instead, it is a compromise point that provides a shorter t90 while maintaining a high CCE constraint.

---

## 5. Extension of the Design Method to Selected Monoenergetic Source Terms

### 5.1 Multi-Source Full-Depletion Thickness Scan

To examine the generality of the above design rule, the same thickness scan, full-depletion bias setting, and CCE-t90 selection rule are applied to four monoenergetic source terms:

$$
20,\ 49,\ 100,\ 156.5\ \mathrm{keV}
$$

The full set of curves in Fig.6 shows that the thickness dependence of CCE and t90 differs significantly among source terms. As the source energy increases, the deposition tail becomes deeper, and a larger i-region thickness is generally required to reach high CCE. Because the real C-14 spectrum contains both shallow deposition components and a high-energy deposition tail, its optimal thickness lies in an intermediate range.

It is worth noting that the recommended thickness for the C-14 spectrum is 60 um, whereas that for the 49 keV monoenergetic source is 20 um. Although 49 keV is close to the average C-14 energy, it cannot represent the contribution of the high-energy tail of the real spectrum to the thickness requirement. From the device-design perspective, this result again confirms that the real C-14 spectrum cannot be simply replaced by an average-energy monoenergetic electron.

### 5.2 Recommended Design Points

The following figures summarize the recommended thickness, recommended full-depletion bias, CCE, and t90 for each source term.

![Fig.7a Recommended i-region thickness and full-depletion bias for different source terms](figures/fig7a_recommended_thickness.png)

**Fig.7a.** Recommended i-region thickness and full-depletion bias for different source terms.

![Fig.7b CCE and t90 at the recommended design points for different source terms](figures/fig7b_cce_t90_recommendations.png)

**Fig.7b.** CCE and t90 at the recommended design points for different source terms. The selection rule is to choose, within the 0.5 percentage-point window below the maximum CCE of each source term, the structure with the shortest t90.

**Table 5. Recommended device parameters for different source terms.**

| Source | Wi (um) | Vdep (V) | CCE (%) | t90 (ns) | Ipeak (nA) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20 keV | 10 | 1 | 73.952 | 2.922 | 2.609 |
| 49 keV | 20 | 3 | 97.677 | 3.968 | 2.755 |
| 100 keV | 70 | 26 | 99.384 | 6.725 | 2.318 |
| 156.5 keV | 130 | 88 | 99.600 | 7.885 | 3.306 |
| C-14 spectrum | 60 | 19 | 97.332 | 6.559 | 1.688 |

Table 5 shows that the recommended thickness generally increases with source energy. The recommended thickness is 10 um for the 20 keV monoenergetic source and 20 um for the 49 keV source. The 100 keV and 156.5 keV sources select 70 um and 130 um, respectively, indicating that high-energy electrons require a thicker effective region to avoid loss from deep deposition.

The recommended thickness for the C-14 spectrum is 60 um, which lies between the 49 keV source and the higher-energy monoenergetic sources. This result is consistent with the physical picture of the continuous C-14 spectrum: many electrons contribute to shallow deposition, but the high-energy tail still requires a device thickness clearly larger than the optimum thickness for the average-energy monoenergetic source. Therefore, after the proposed method is extended to monoenergetic source terms, it not only provides design parameters for each energy, but also shows that real beta-spectrum design cannot be directly extrapolated from any single monoenergetic point.

---

## 6. Discussion

The results first show that the applicability of monoenergetic approximation must be evaluated separately for integral metrics and transient metrics. CCE is obtained by integrating the entire current waveform and does not contain information on the temporal order of charge arrival. Therefore, different source terms may give similar CCE values while producing completely different peak currents, pulse widths, and t90 values. For applications concerned only with final count rate or total collected charge, monoenergetic approximation may have engineering value. However, for front-end readout circuit design, time-gate setting, and pulse-shape analysis, the real energy spectrum is necessary.

Second, the C-14 structure optimization in this work is not a simple search for the thinnest device. If only t90 is considered, a thinner i region is usually faster, but it truncates part of the deep deposition from the real C-14 spectrum and causes a clear CCE reduction. If only CCE is considered, a thicker i region more easily approaches the plateau value, but the response time becomes longer. The rule used here, choosing the shortest t90 within a window close to the maximum CCE, is essentially an engineering trade-off between charge completeness and transient speed. The current 0.5 percentage-point window is suitable for demonstrating the method. In future work, it can be adjusted according to readout-circuit noise, timing-resolution requirements, and minimum signal-amplitude constraints.

Third, the recommended thickness is 60 um for the C-14 spectrum but 20 um for the 49 keV monoenergetic source. This point is important for the main argument of the paper. It shows that the average energy represents only a first-order statistic of the spectrum and cannot represent the deposition-depth distribution. The device thickness is actually determined by the spatial deposition distribution of the spectrum in the material, not by the average energy itself.

Finally, low-energy detection can be treated as a target for future research rather than the main focus of this work. Based on the Geant4-TCAD coupling and CCE-t90 post-processing workflow established here, future work can introduce thinner dead-layer structures, different window materials, front-end circuit noise, capacitance, leakage current, and system detection limits, thereby extending the method to device design specifically for low-energy beta particles or low-energy electrons.

---

## 7. Conclusion

This work establishes a Geant4-TCAD coupled simulation workflow for C-14 beta-spectrum response and systematically compares the energy deposition and transient response of the real C-14 spectrum with several monoenergetic source terms in a 4H-SiC PIN device.

First, the validity boundary for replacing the real C-14 source with monoenergetic source terms is clarified. A 49 keV monoenergetic electron can give a CCE close to that of the real C-14 spectrum, but it cannot reliably reproduce the transient i-t waveform, peak current, or t90. Therefore, monoenergetic approximation cannot replace the real C-14 spectrum for transient response analysis.

Second, a full-depletion i-region thickness scan is performed for the 4H-SiC PIN detector under the real C-14 spectrum. The results show that the CCE of C-14 enters a high-collection plateau after 60 um. When the shortest t90 is selected within a 0.5 percentage-point window below the maximum CCE, the recommended parameters are $W_i=60\ \mu\mathrm{m}$ and $V_{\mathrm{dep}}=19\ \mathrm{V}$, corresponding to CCE of 97.332% and t90 of 6.559 ns.

Third, the same design rule is extended to 20, 49, 100, and 156.5 keV monoenergetic source terms, yielding source-dependent recommended device parameters. The results show that the optimal i-region thickness varies significantly with incident energy and deposition-depth distribution. The recommended thickness for the real C-14 spectrum differs from that for the 49 keV average-energy monoenergetic source, further demonstrating the necessity of real-spectrum-driven design.

In summary, this work provides a complete design route from real beta-spectrum transport and TCAD transient response to full-depletion device parameter selection. It can provide simulation support for structural optimization and readout-circuit matching of C-14-spectrum-driven 4H-SiC PIN detectors.

## References

> References should be completed according to the target journal format. At minimum, references should cover reviews of 4H-SiC radiation detectors, C-14 nuclear data, Geant4 electromagnetic models, TCAD semiconductor detector simulation, and electron-hole pair creation energy and mobility models in SiC.

[1] T. Kimoto and J. A. Cooper, *Fundamentals of Silicon Carbide Technology*, Wiley, 2014.

[2] F. Nava et al., Silicon carbide and its use as a radiation detector material, *Measurement Science and Technology*, 2008.

[3] S. Agostinelli et al., Geant4: A simulation toolkit, *Nuclear Instruments and Methods in Physics Research A*, 2003.

[4] J. Allison et al., Recent developments in Geant4, *Nuclear Instruments and Methods in Physics Research A*, 2016.

[5] Synopsys, *TCAD Device User Guide*, Synopsys Inc.
