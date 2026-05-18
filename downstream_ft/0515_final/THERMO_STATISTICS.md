# ThermoGen — Per-Property Statistics

**Source:** `downstream_ft/0515_final/per_property/*.csv`
**Generated:** by `scripts/build_thermo_statistics.py`

**Unique molecules:** 21,269     **Properties:** 43     **Total cells:** 64,477

## 1. Confidence-star distribution (overall, all cells)

| ★ | Count | Share | Tier labels |
|---:|---:|---:|---|
| ★★★★★ | 34,272 | 53.2% | tier1, tier1+confirmed |
| ★★★★☆ | 11,222 | 17.4% | tier1+disputed, tier2, tier2+confirmed |
| ★★★☆☆ | 9,173 | 14.2% | tier2+disputed, secondary_tight |
| — | 9,810 | 15.2% | downstream (no upstream tier; ML-bench targets) |

## 2. Per-property tier composition

Columns: n = total rows, mol = unique molecules, 5★/4★/3★/2★/1★/dn = cell counts at each confidence level (`dn` = downstream/ML targets).

| Property | n | mol | 5★ | 4★ | 3★ | 2★ | 1★ | dn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `BP_K` | 5,572 | 5,572 | 1,166 | 4,406 |  |  |  |  |
| `CEP_PCE` | 875 | 875 |  |  |  |  |  | 875 |
| `Cp_gas_298K` | 862 | 862 | 862 |  |  |  |  |  |
| `Cp_liq_298K` | 1,061 | 1,061 | 915 | 60 | 86 |  |  |  |
| `ESOL_logS` | 1,115 | 1,115 |  |  |  |  |  | 1,115 |
| `Gf_gas_kJmol` | 1,217 | 1,217 | 1,217 |  |  |  |  |  |
| `H_combus_kJmol` | 1,267 | 1,267 | 1,267 |  |  |  |  |  |
| `Hf_gas_kJmol` | 3,086 | 3,086 | 1,166 | 1,891 | 29 |  |  |  |
| `Hf_liq_kJmol` | 1,613 | 1,613 |  | 1,563 | 50 |  |  |  |
| `Hfus_at_TF_kJmol` | 654 | 654 | 654 |  |  |  |  |  |
| `Hvap_at_TB_kJmol` | 1,414 | 1,414 | 1,414 |  |  |  |  |  |
| `LEL_volpct` | 1,047 | 1,047 | 1,047 |  |  |  |  |  |
| `Lipophilicity_logD` | 4,191 | 4,191 |  |  |  |  |  | 4,191 |
| `PPBR_pct` | 1,386 | 1,386 |  |  |  |  |  | 1,386 |
| `Pc_bar` | 1,235 | 1,235 | 1,182 | 53 |  |  |  |  |
| `Pvap_log10mmHg` | 2,770 | 2,770 | 747 | 2,023 |  |  |  |  |
| `Q_10ppmv_mgg` | 779 | 779 | 779 |  |  |  |  |  |
| `RI_298K` | 876 | 876 | 812 | 17 | 47 |  |  |  |
| `ST_298K_mNm` | 2,264 | 2,264 | 2,254 | 10 |  |  |  |  |
| `S_gas_JmolK` | 981 | 981 | 981 |  |  |  |  |  |
| `Sf_gas_JmolK` | 1,201 | 1,201 | 1,201 |  |  |  |  |  |
| `Tc_K` | 1,267 | 1,267 | 1,162 | 105 |  |  |  |  |
| `UEL_volpct` | 1,060 | 1,060 | 1,060 |  |  |  |  |  |
| `Vc_cm3mol` | 1,236 | 1,236 | 1,236 |  |  |  |  |  |
| `autoignition_K` | 497 | 497 | 497 |  |  |  |  |  |
| `density_liq_298K_gcm3` | 1,034 | 1,034 | 1,023 | 11 |  |  |  |  |
| `dielectric_298K` | 1,362 | 1,362 | 1,234 | 89 | 39 |  |  |  |
| `dipole_moment_D` | 753 | 753 | 749 | 3 | 1 |  |  |  |
| `expand_coeff_liq_K-1` | 989 | 989 | 989 |  |  |  |  |  |
| `flash_point_K` | 1,040 | 1,040 | 1,040 |  |  |  |  |  |
| `freesolv_dG_kcalmol` | 641 | 641 |  |  |  |  |  | 641 |
| `fusion_T_K` | 1,694 | 1,694 | 1,126 | 54 | 514 |  |  |  |
| `gyration_radius_A` | 990 | 990 | 990 |  |  |  |  |  |
| `k_gas_298K` | 353 | 353 | 353 |  |  |  |  |  |
| `k_liq_298K` | 993 | 993 | 880 | 102 | 11 |  |  |  |
| `log_Henry_atmmolfrac` | 558 | 558 | 558 |  |  |  |  |  |
| `log_Koc` | 715 | 715 |  | 715 |  |  |  |  |
| `log_solubility_water_molL` | 8,310 | 8,310 |  |  | 8,310 |  |  |  |
| `log_solubility_water_ppm` | 816 | 816 | 816 |  |  |  |  |  |
| `omega` | 1,264 | 1,264 | 1,264 |  |  |  |  |  |
| `visc_gas_298K_uPas` | 626 | 626 | 626 |  |  |  |  |  |
| `visc_liq_298K_cP` | 1,211 | 1,211 | 1,005 | 120 | 86 |  |  |  |
| `visc_liq_298K_cP_manual` | 1,602 | 1,602 |  |  |  |  |  | 1,602 |

## 3. Per-property value statistics

| Property | n | mean | std | min | p25 | median | p75 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `BP_K` | 5,572 | 464.2 | 87.21 | 81.7 | 407.1 | 464.6 | 519.7 | 832 |
| `CEP_PCE` | 875 | -5.225 | 0.7716 | -7.76 | -5.741 | -5.12 | -4.631 | -3.51 |
| `Cp_gas_298K` | 862 | 153.8 | 81.4 | 29.08 | 102.7 | 137.1 | 181.9 | 572.4 |
| `Cp_liq_298K` | 1,061 | 241.4 | 104.6 | 68.65 | 168.8 | 221.6 | 289.7 | 864.8 |
| `ESOL_logS` | 1,115 | -3.052 | 2.093 | -11.6 | -4.312 | -2.863 | -1.6 | 1.58 |
| `Gf_gas_kJmol` | 1,217 | -46.09 | 260.5 | -1970 | -146.1 | 22.4 | 115 | 564 |
| `H_combus_kJmol` | 1,267 | 4450 | 2780 | 51.6 | 2505 | 3840 | 6030 | 1.554e+04 |
| `Hf_gas_kJmol` | 3,086 | -180 | 339.7 | -2897 | -338.8 | -149.6 | 32.56 | 970.7 |
| `Hf_liq_kJmol` | 1,613 | -267.5 | 318.4 | -2931 | -443 | -214.3 | -62.4 | 503.6 |
| `Hfus_at_TF_kJmol` | 654 | 15.04 | 11.9 | 0.477 | 7.776 | 11.77 | 18.23 | 99.16 |
| `Hvap_at_TB_kJmol` | 1,414 | 41 | 13.77 | 5.57 | 32.05 | 38.07 | 48.71 | 133.1 |
| `LEL_volpct` | 1,047 | 1.542 | 1.592 | 0.3 | 0.8 | 1.1 | 1.7 | 18 |
| `Lipophilicity_logD` | 4,191 | 2.189 | 1.201 | -1.5 | 1.41 | 2.36 | 3.1 | 4.5 |
| `PPBR_pct` | 1,386 | 88.74 | 15.75 | 11.18 | 86.05 | 95.53 | 98.61 | 99.95 |
| `Pc_bar` | 1,235 | 34.94 | 13.3 | 10 | 25.4 | 33.5 | 42.55 | 88.1 |
| `Pvap_log10mmHg` | 2,770 | -1.917 | 3.507 | -11.82 | -4.63 | -1.157 | 0.932 | 5.668 |
| `Q_10ppmv_mgg` | 779 | 25.58 | 15.98 | 0.0001 | 13.34 | 26.63 | 35.32 | 146.1 |
| `RI_298K` | 876 | 1.432 | 0.07516 | 1 | 1.4 | 1.421 | 1.46 | 1.815 |
| `ST_298K_mNm` | 2,264 | 29.04 | 7.868 | 0.23 | 24.33 | 28.77 | 33.27 | 84.22 |
| `S_gas_JmolK` | 981 | 419.4 | 134.5 | 186.3 | 336.7 | 389.8 | 458.8 | 1149 |
| `Sf_gas_JmolK` | 1,201 | -583.3 | 404.4 | -1999 | -800.2 | -486.9 | -287.4 | 168.3 |
| `Tc_K` | 1,267 | 637 | 119.9 | 132.9 | 561 | 629 | 722.7 | 1113 |
| `UEL_volpct` | 1,060 | 9.904 | 8.103 | 1.5 | 5.3 | 7.7 | 11.1 | 80 |
| `Vc_cm3mol` | 1,236 | 446 | 217.7 | 16.5 | 300 | 404 | 524 | 1460 |
| `autoignition_K` | 497 | 658.8 | 128.4 | 349.8 | 555.4 | 664.3 | 738.1 | 1283 |
| `density_liq_298K_gcm3` | 1,034 | 0.926 | 0.2809 | 0.493 | 0.7545 | 0.851 | 0.994 | 3.306 |
| `dielectric_298K` | 1,362 | 11 | 17.51 | 1.423 | 3.292 | 5.642 | 10.57 | 187.9 |
| `dipole_moment_D` | 753 | 1.596 | 1.23 | 0 | 0.55 | 1.6 | 2.23 | 6.39 |
| `expand_coeff_liq_K-1` | 989 | 0.001165 | 0.0005748 | 0.000429 | 0.000896 | 0.001056 | 0.00122 | 0.009359 |
| `flash_point_K` | 1,040 | 329.9 | 71.83 | 137 | 284.3 | 316.8 | 374.8 | 658.1 |
| `freesolv_dG_kcalmol` | 641 | -3.796 | 3.843 | -25.47 | -5.72 | -3.52 | -1.21 | 3.43 |
| `fusion_T_K` | 1,694 | 274.7 | 95.72 | 68.15 | 200 | 264.9 | 334.8 | 710.8 |
| `gyration_radius_A` | 990 | 4.162 | 1.39 | 0.558 | 3.317 | 3.976 | 4.714 | 11.66 |
| `k_gas_298K` | 353 | 0.009451 | 0.003768 | 0.00244 | 0.00696 | 0.00892 | 0.0106 | 0.03557 |
| `k_liq_298K` | 993 | 0.129 | 0.03223 | 0.0447 | 0.1109 | 0.1291 | 0.1421 | 0.4115 |
| `log_Henry_atmmolfrac` | 558 | 3.309 | 1.799 | 0.0018 | 1.822 | 3.127 | 5.055 | 5.973 |
| `log_Koc` | 715 | 2.703 | 1.228 | 0 | 1.84 | 2.49 | 3.385 | 6.5 |
| `log_solubility_water_molL` | 8,310 | -3.004 | 2.311 | -13.17 | -4.37 | -2.701 | -1.403 | 2.138 |
| `log_solubility_water_ppm` | 816 | 2.103 | 2.329 | -6.699 | 0.0961 | 2.141 | 3.822 | 8 |
| `omega` | 1,264 | 0.4701 | 0.2409 | 0.011 | 0.2967 | 0.403 | 0.5952 | 1.578 |
| `visc_gas_298K_uPas` | 626 | 6.77 | 2.335 | 3.236 | 5.358 | 6.14 | 7.66 | 17.27 |
| `visc_liq_298K_cP` | 1,211 | 4.213 | 13.68 | 0.039 | 0.538 | 0.951 | 2.912 | 187.8 |
| `visc_liq_298K_cP_manual` | 1,602 | 6.021 | 81.84 | 0.05212 | 0.4521 | 0.7463 | 1.9 | 3018 |

## 4. Per-property top sources (rows by upstream provider)

Top-3 sources per property, plus total number of distinct sources.

| Property | n_sources | top sources (rows) |
|---|---:|---|
| `BP_K` | 0 | (no sources column — downstream) |
| `CEP_PCE` | 0 | (no sources column — downstream) |
| `Cp_gas_298K` | 0 | (no sources column — downstream) |
| `Cp_liq_298K` | 0 | (no sources column — downstream) |
| `ESOL_logS` | 0 | (no sources column — downstream) |
| `Gf_gas_kJmol` | 0 | (no sources column — downstream) |
| `H_combus_kJmol` | 0 | (no sources column — downstream) |
| `Hf_gas_kJmol` | 0 | (no sources column — downstream) |
| `Hf_liq_kJmol` | 0 | (no sources column — downstream) |
| `Hfus_at_TF_kJmol` | 0 | (no sources column — downstream) |
| `Hvap_at_TB_kJmol` | 0 | (no sources column — downstream) |
| `LEL_volpct` | 0 | (no sources column — downstream) |
| `Lipophilicity_logD` | 0 | (no sources column — downstream) |
| `PPBR_pct` | 0 | (no sources column — downstream) |
| `Pc_bar` | 0 | (no sources column — downstream) |
| `Pvap_log10mmHg` | 0 | (no sources column — downstream) |
| `Q_10ppmv_mgg` | 0 | (no sources column — downstream) |
| `RI_298K` | 0 | (no sources column — downstream) |
| `ST_298K_mNm` | 0 | (no sources column — downstream) |
| `S_gas_JmolK` | 0 | (no sources column — downstream) |
| `Sf_gas_JmolK` | 0 | (no sources column — downstream) |
| `Tc_K` | 0 | (no sources column — downstream) |
| `UEL_volpct` | 0 | (no sources column — downstream) |
| `Vc_cm3mol` | 0 | (no sources column — downstream) |
| `autoignition_K` | 0 | (no sources column — downstream) |
| `density_liq_298K_gcm3` | 0 | (no sources column — downstream) |
| `dielectric_298K` | 0 | (no sources column — downstream) |
| `dipole_moment_D` | 0 | (no sources column — downstream) |
| `expand_coeff_liq_K-1` | 0 | (no sources column — downstream) |
| `flash_point_K` | 0 | (no sources column — downstream) |
| `freesolv_dG_kcalmol` | 0 | (no sources column — downstream) |
| `fusion_T_K` | 0 | (no sources column — downstream) |
| `gyration_radius_A` | 0 | (no sources column — downstream) |
| `k_gas_298K` | 0 | (no sources column — downstream) |
| `k_liq_298K` | 0 | (no sources column — downstream) |
| `log_Henry_atmmolfrac` | 0 | (no sources column — downstream) |
| `log_Koc` | 0 | (no sources column — downstream) |
| `log_solubility_water_molL` | 0 | (no sources column — downstream) |
| `log_solubility_water_ppm` | 0 | (no sources column — downstream) |
| `omega` | 0 | (no sources column — downstream) |
| `visc_gas_298K_uPas` | 0 | (no sources column — downstream) |
| `visc_liq_298K_cP` | 0 | (no sources column — downstream) |
| `visc_liq_298K_cP_manual` | 0 | (no sources column — downstream) |

## 5. Split sizes (random + scaffold, 5-fold)

| Property | n_mol | n_scaffolds | random fold sizes | scaffold fold sizes |
|---|---:|---:|---|---|
| `BP_K` | 5572 | 459 | `1115|1115|1114|1114|1114` | `2772|1281|507|506|506` |
| `CEP_PCE` | 875 | 438 | `175|175|175|175|175` | `175|175|175|175|175` |
| `Cp_gas_298K` | 862 | 89 | `173|173|172|172|172` | `555|166|47|47|47` |
| `Cp_liq_298K` | 1061 | 71 | `213|212|212|212|212` | `767|129|55|55|55` |
| `ESOL_logS` | 1115 | 269 | `223|223|223|223|223` | `315|253|183|182|182` |
| `Gf_gas_kJmol` | 1217 | 109 | `244|244|243|243|243` | `808|196|71|71|71` |
| `H_combus_kJmol` | 1267 | 116 | `254|254|253|253|253` | `838|206|75|74|74` |
| `Hf_gas_kJmol` | 3086 | 533 | `618|617|617|617|617` | `1589|444|351|351|351` |
| `Hf_liq_kJmol` | 1613 | 201 | `323|323|323|322|322` | `898|221|165|165|164` |
| `Hfus_at_TF_kJmol` | 654 | 78 | `131|131|131|131|130` | `410|126|40|39|39` |
| `Hvap_at_TB_kJmol` | 1414 | 124 | `283|283|283|283|282` | `962|201|84|84|83` |
| `LEL_volpct` | 1047 | 103 | `210|210|209|209|209` | `666|186|65|65|65` |
| `Lipophilicity_logD` | 4191 | 2396 | `839|838|838|838|838` | `839|838|838|838|838` |
| `PPBR_pct` | 1386 | 878 | `278|277|277|277|277` | `278|277|277|277|277` |
| `Pc_bar` | 1235 | 117 | `247|247|247|247|247` | `808|206|74|74|73` |
| `Pvap_log10mmHg` | 2770 | 411 | `554|554|554|554|554` | `1292|538|314|313|313` |
| `Q_10ppmv_mgg` | 779 | 56 | `156|156|156|156|155` | `582|95|34|34|34` |
| `RI_298K` | 876 | 63 | `176|175|175|175|175` | `640|110|42|42|42` |
| `ST_298K_mNm` | 2264 | 118 | `453|453|453|453|452` | `1556|248|154|153|153` |
| `S_gas_JmolK` | 981 | 103 | `197|196|196|196|196` | `628|184|57|56|56` |
| `Sf_gas_JmolK` | 1201 | 109 | `241|240|240|240|240` | `798|195|70|69|69` |
| `Tc_K` | 1267 | 117 | `254|254|253|253|253` | `840|202|75|75|75` |
| `UEL_volpct` | 1060 | 103 | `212|212|212|212|212` | `671|193|66|65|65` |
| `Vc_cm3mol` | 1236 | 117 | `248|247|247|247|247` | `809|206|74|74|73` |
| `autoignition_K` | 497 | 46 | `100|100|99|99|99` | `333|86|26|26|26` |
| `density_liq_298K_gcm3` | 1034 | 71 | `207|207|207|207|206` | `756|127|51|50|50` |
| `dielectric_298K` | 1362 | 91 | `273|273|272|272|272` | `877|261|75|75|74` |
| `dipole_moment_D` | 753 | 88 | `151|151|151|150|150` | `466|144|48|48|47` |
| `expand_coeff_liq_K-1` | 989 | 69 | `198|198|198|198|197` | `714|128|49|49|49` |
| `flash_point_K` | 1040 | 106 | `208|208|208|208|208` | `680|187|58|58|57` |
| `freesolv_dG_kcalmol` | 641 | 61 | `129|128|128|128|128` | `320|152|57|56|56` |
| `fusion_T_K` | 1694 | 210 | `339|339|339|339|338` | `872|404|140|139|139` |
| `gyration_radius_A` | 990 | 106 | `198|198|198|198|198` | `629|185|59|59|58` |
| `k_gas_298K` | 353 | 22 | `71|71|71|70|70` | `281|20|18|17|17` |
| `k_liq_298K` | 993 | 72 | `199|199|199|198|198` | `716|125|51|51|50` |
| `log_Henry_atmmolfrac` | 558 | 21 | `112|112|112|111|111` | `447|68|15|14|14` |
| `log_Koc` | 715 | 181 | `143|143|143|143|143` | `288|107|107|107|106` |
| `log_solubility_water_molL` | 8310 | 1640 | `1662|1662|1662|1662|1662` | `2105|1592|1538|1538|1537` |
| `log_solubility_water_ppm` | 816 | 57 | `164|163|163|163|163` | `576|135|35|35|35` |
| `omega` | 1264 | 116 | `253|253|253|253|252` | `836|204|75|75|74` |
| `visc_gas_298K_uPas` | 626 | 44 | `126|125|125|125|125` | `471|66|30|30|29` |
| `visc_liq_298K_cP` | 1211 | 75 | `243|242|242|242|242` | `902|135|58|58|58` |
| `visc_liq_298K_cP_manual` | 1602 | 78 | `321|321|320|320|320` | `1255|138|70|70|69` |

## 6. Per-property value distributions

Composite (all 43 properties): [`distributions/_all_distributions.png`](distributions/_all_distributions.png)

![all](distributions/_all_distributions.png)

### Per-property histograms

Click any property name to open the standalone PNG.

| | | |
|---|---|---|
| **[BP_K](distributions/BP_K.png)**<br><img src="distributions/BP_K.png" width="280"> | **[CEP_PCE](distributions/CEP_PCE.png)**<br><img src="distributions/CEP_PCE.png" width="280"> | **[Cp_gas_298K](distributions/Cp_gas_298K.png)**<br><img src="distributions/Cp_gas_298K.png" width="280"> |
| **[Cp_liq_298K](distributions/Cp_liq_298K.png)**<br><img src="distributions/Cp_liq_298K.png" width="280"> | **[ESOL_logS](distributions/ESOL_logS.png)**<br><img src="distributions/ESOL_logS.png" width="280"> | **[Gf_gas_kJmol](distributions/Gf_gas_kJmol.png)**<br><img src="distributions/Gf_gas_kJmol.png" width="280"> |
| **[H_combus_kJmol](distributions/H_combus_kJmol.png)**<br><img src="distributions/H_combus_kJmol.png" width="280"> | **[Hf_gas_kJmol](distributions/Hf_gas_kJmol.png)**<br><img src="distributions/Hf_gas_kJmol.png" width="280"> | **[Hf_liq_kJmol](distributions/Hf_liq_kJmol.png)**<br><img src="distributions/Hf_liq_kJmol.png" width="280"> |
| **[Hfus_at_TF_kJmol](distributions/Hfus_at_TF_kJmol.png)**<br><img src="distributions/Hfus_at_TF_kJmol.png" width="280"> | **[Hvap_at_TB_kJmol](distributions/Hvap_at_TB_kJmol.png)**<br><img src="distributions/Hvap_at_TB_kJmol.png" width="280"> | **[LEL_volpct](distributions/LEL_volpct.png)**<br><img src="distributions/LEL_volpct.png" width="280"> |
| **[Lipophilicity_logD](distributions/Lipophilicity_logD.png)**<br><img src="distributions/Lipophilicity_logD.png" width="280"> | **[PPBR_pct](distributions/PPBR_pct.png)**<br><img src="distributions/PPBR_pct.png" width="280"> | **[Pc_bar](distributions/Pc_bar.png)**<br><img src="distributions/Pc_bar.png" width="280"> |
| **[Pvap_log10mmHg](distributions/Pvap_log10mmHg.png)**<br><img src="distributions/Pvap_log10mmHg.png" width="280"> | **[Q_10ppmv_mgg](distributions/Q_10ppmv_mgg.png)**<br><img src="distributions/Q_10ppmv_mgg.png" width="280"> | **[RI_298K](distributions/RI_298K.png)**<br><img src="distributions/RI_298K.png" width="280"> |
| **[ST_298K_mNm](distributions/ST_298K_mNm.png)**<br><img src="distributions/ST_298K_mNm.png" width="280"> | **[S_gas_JmolK](distributions/S_gas_JmolK.png)**<br><img src="distributions/S_gas_JmolK.png" width="280"> | **[Sf_gas_JmolK](distributions/Sf_gas_JmolK.png)**<br><img src="distributions/Sf_gas_JmolK.png" width="280"> |
| **[Tc_K](distributions/Tc_K.png)**<br><img src="distributions/Tc_K.png" width="280"> | **[UEL_volpct](distributions/UEL_volpct.png)**<br><img src="distributions/UEL_volpct.png" width="280"> | **[Vc_cm3mol](distributions/Vc_cm3mol.png)**<br><img src="distributions/Vc_cm3mol.png" width="280"> |
| **[autoignition_K](distributions/autoignition_K.png)**<br><img src="distributions/autoignition_K.png" width="280"> | **[density_liq_298K_gcm3](distributions/density_liq_298K_gcm3.png)**<br><img src="distributions/density_liq_298K_gcm3.png" width="280"> | **[dielectric_298K](distributions/dielectric_298K.png)**<br><img src="distributions/dielectric_298K.png" width="280"> |
| **[dipole_moment_D](distributions/dipole_moment_D.png)**<br><img src="distributions/dipole_moment_D.png" width="280"> | **[expand_coeff_liq_K-1](distributions/expand_coeff_liq_K-1.png)**<br><img src="distributions/expand_coeff_liq_K-1.png" width="280"> | **[flash_point_K](distributions/flash_point_K.png)**<br><img src="distributions/flash_point_K.png" width="280"> |
| **[freesolv_dG_kcalmol](distributions/freesolv_dG_kcalmol.png)**<br><img src="distributions/freesolv_dG_kcalmol.png" width="280"> | **[fusion_T_K](distributions/fusion_T_K.png)**<br><img src="distributions/fusion_T_K.png" width="280"> | **[gyration_radius_A](distributions/gyration_radius_A.png)**<br><img src="distributions/gyration_radius_A.png" width="280"> |
| **[k_gas_298K](distributions/k_gas_298K.png)**<br><img src="distributions/k_gas_298K.png" width="280"> | **[k_liq_298K](distributions/k_liq_298K.png)**<br><img src="distributions/k_liq_298K.png" width="280"> | **[log_Henry_atmmolfrac](distributions/log_Henry_atmmolfrac.png)**<br><img src="distributions/log_Henry_atmmolfrac.png" width="280"> |
| **[log_Koc](distributions/log_Koc.png)**<br><img src="distributions/log_Koc.png" width="280"> | **[log_solubility_water_molL](distributions/log_solubility_water_molL.png)**<br><img src="distributions/log_solubility_water_molL.png" width="280"> | **[log_solubility_water_ppm](distributions/log_solubility_water_ppm.png)**<br><img src="distributions/log_solubility_water_ppm.png" width="280"> |
| **[omega](distributions/omega.png)**<br><img src="distributions/omega.png" width="280"> | **[visc_gas_298K_uPas](distributions/visc_gas_298K_uPas.png)**<br><img src="distributions/visc_gas_298K_uPas.png" width="280"> | **[visc_liq_298K_cP](distributions/visc_liq_298K_cP.png)**<br><img src="distributions/visc_liq_298K_cP.png" width="280"> |
| **[visc_liq_298K_cP_manual](distributions/visc_liq_298K_cP_manual.png)**<br><img src="distributions/visc_liq_298K_cP_manual.png" width="280"> |  |  |

## 7. Top 20 upstream sources across all properties

| Source | Rows |
|---|---:|
