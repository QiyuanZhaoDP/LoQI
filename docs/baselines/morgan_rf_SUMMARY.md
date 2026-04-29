# Morgan FP + Random Forest 5-fold CV baseline

Reference 2D-only baseline. RF on Morgan FPs (no 3D info)
with small radius × n_estimators grid; reports the best
config per dataset (selected on mean val MAE).

5-fold CV, seed=42, sklearn RandomForestRegressor (max_features='sqrt', min_samples_leaf=1).

| dataset | n | target σ | best config | MAE | RMSE | R² |
|---|---:|---:|---|---:|---:|---:|
| Cp | 1,459 | 106.237 | r=3, n=500 | **43.967**±2.199 | 65.699 | 0.611 |
| V_cp | 813 | 4.230 | r=3, n=200 | **1.666**±0.162 | 3.116 | 0.436 |
| de | 778 | 12.080 | r=2, n=500 | **4.274**±0.466 | 8.722 | 0.472 |
| gas_Hf | 2,419 | 433.770 | r=2, n=500 | **111.644**±8.649 | 222.177 | 0.736 |
| k | 755 | 0.025 | r=2, n=500 | **0.009**±0.001 | 0.013 | 0.723 |
| liquid_Hf | 1,624 | 430.785 | r=2, n=500 | **100.855**±8.853 | 209.815 | 0.753 |
| delaney_s | 1,117 | 1.013 | r=2, n=500 | **0.427**±0.012 | 0.568 | 0.682 |
| freesolv_s | 641 | 3.843 | r=2, n=200 | **1.375**±0.114 | 2.141 | 0.685 |
| lipo_s | 4,199 | 1.203 | r=2, n=500 | **0.647**±0.013 | 0.849 | 0.501 |

MAE / RMSE are in target physical units. R² ≥ 0.7 is decent for a 2D-only baseline; gas_Hf and similar quantum-mechanically-demanding properties typically need 3D info to push R² higher.

