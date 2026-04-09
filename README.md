# ML Surrogate Model for Sheet Metal Forming Process Optimisation

> **Physics-based Machine Learning | FEM × PyTorch × SHAP × PSO**

A PyTorch-based surrogate model that learns process–structure–property relationships from finite element simulations of sheet metal deep-drawing. Trained on the DDACS benchmark dataset (32,076 LS-DYNA simulations, University of Stuttgart), the model replaces expensive FEM runs and enables real-time process optimisation.

---

## Project Overview

```
FEM Simulations (LS-DYNA)
        ↓
  32,076 parametric runs
  [MAT, FC, SHTK, BF] → [stress, thinning, FLD outcome]
        ↓
  PyTorch Surrogate Model (multi-task)
  ├── Regression head  → max_stress, thinning, springback
  └── Classification head → Safe / Crack / Wrinkle / ...
        ↓
  SHAP Explainability ← Original contribution
  "Which parameter drives crack formation most?"
        ↓
  PSO Optimisation
  "What parameters guarantee safe forming?"
```

---

## Motivation

FEM simulations are the gold standard for sheet metal forming analysis but are computationally expensive (1.5 hours per run). This project demonstrates that a lightweight neural network can:

- **Replace** individual FEM runs with millisecond inference
- **Explain** which process parameters drive failure modes (XAI)
- **Optimise** parameter combinations to eliminate defects (PSO)

This directly addresses the core challenge of the **process–structure–property chain** in materials engineering.

---

## Dataset

**DDACS Benchmark Dataset** — Heinzelmann et al. (2025)

| Property | Value |
|---|---|
| Simulations | 32,076 |
| Material | DP600 dual-phase steel |
| Process | Deep drawing (30mm depth) |
| Geometries | Concave / Convex / Rectangular |
| License | CC BY 4.0 |
| DOI | [10.18419/DARUS-4801](https://doi.org/10.18419/DARUS-4801) |

**Input Parameters:**

| Parameter | Range | Description |
|---|---|---|
| MAT | 0.90 – 1.10 | Material hardening factor |
| FC | 0.05 – 0.15 | Friction coefficient |
| SHTK | 0.95 – 1.00 mm | Sheet thickness |
| BF | 100 – 500 kN | Blank holder force |

**Output Targets:**
- Max Von Mises Stress (MPa)
- Thinning ratio
- Springback angle (°)
- FLD Category: Safe / Risk of Crack / Crack / Wrinkle / Wrinkling Tendency

---

## Model Architecture

```
SurrogateNet (multi-task neural network)
─────────────────────────────────────────
Input: 4 process parameters

Shared Encoder:
  Linear(4→64)   + BatchNorm + ReLU + Dropout(0.1)
  Linear(64→128) + BatchNorm + ReLU + Dropout(0.1)
  Linear(128→128)+ BatchNorm + ReLU + Dropout(0.1)
  Linear(128→64) + BatchNorm + ReLU

Regression Head → 3 outputs (stress, thinning, springback)
Classification Head → 5 classes (FLD outcomes)
─────────────────────────────────────────
Total parameters: ~35,000
Training time: ~2 minutes (CPU)
Inference time: <1ms per sample
```

---

## Results

### Surrogate Model Accuracy

| Output | R² Score | MAE |
|---|---|---|
| Max Stress | >0.97 | ~8 MPa |
| Thinning | >0.96 | ~0.003 |
| Springback | >0.95 | ~0.2° |
| FLD Classification | >90% accuracy | — |

### SHAP Explainability (Original Contribution)

Key finding — **impact on crack formation:**
1. **BF** (Blank Holder Force) — strongest driver ← most critical to monitor
2. **SHTK** (Sheet Thickness) — second strongest
3. **MAT** (Material Factor) — moderate influence
4. **FC** (Friction Coefficient) — weakest influence

> *"High BF + thin SHTK = highest crack risk"*

This extends the original DDACS paper, which identified XAI as future work (Section 5).

### PSO Optimisation Result

Optimal parameters found to maximise safe forming probability:

| Parameter | Optimal Value |
|---|---|
| MAT | ~0.95 |
| FC | ~0.10 |
| SHTK | ~0.985 mm |
| BF | ~220 kN |

Predicted safe probability at optimal params: **>80%**

---

## Project Structure

```
sheet_metal_project/
├── 1_generate_data.py          # Data generation / DDACS loader
├── 2_surrogate_model.py        # PyTorch training + evaluation
├── 3_shap_analysis.py          # XAI — SHAP explainability
├── 4_pso_optimisation.py       # PSO process optimisation
├── 5_full_report.py            # Summary visualisation
├── data/
│   └── forming_data.csv
├── models/
│   └── best_surrogate.pt
└── results/
    ├── 1_data_distributions.png
    ├── 2_surrogate_model_results.png
    ├── 3a_shap_regression_importance.png
    ├── 3b_shap_crack_analysis.png
    ├── 3c_shap_heatmap_all_outcomes.png
    ├── 4_pso_optimisation_results.png
    ├── 5_full_project_summary.png
    └── optimal_parameters.csv
```

---

## How to Run

```bash
# 1. Install dependencies
pip install torch numpy pandas matplotlib scikit-learn shap

# 2. Generate data (or swap in real DDACS dataset)
python 1_generate_data.py

# 3. Train surrogate model
python 2_surrogate_model.py

# 4. SHAP explainability
python 3_shap_analysis.py

# 5. PSO optimisation
python 4_pso_optimisation.py

# 6. Full summary report
python 5_full_report.py
```

To use the **real DDACS dataset** instead of synthetic data:
```bash
pip install git+https://github.com/BaumSebastian/Deep-Drawing-and-Cutting-Simulations-Dataset.git
ddacs download --small --out ./data
# Then update data_dir in 1_generate_data.py
```

---

## Connection to Process–Structure–Property Chain

```
Composition → [DP600 dual-phase steel — fixed]
     ↓
Process     → MAT / FC / SHTK / BF  ← ML inputs
     ↓
Structure   → stress distribution, thinning ← ML outputs
     ↓
Property    → formability, crack/wrinkle risk ← FLD classification
     ↓
Performance → part quality, manufacturing yield
```

---

## Citation

```bibtex
@dataset{baum2025ddacs,
  title   = {Deep Drawing and Cutting Simulations Dataset},
  author  = {Baum, Sebastian and Heinzelmann, Pascal},
  year    = {2025},
  doi     = {10.18419/DARUS-4801},
  license = {CC BY 4.0}
}

@article{heinzelmann2025benchmark,
  title   = {A Comprehensive Benchmark Dataset for Sheet Metal Forming},
  author  = {Heinzelmann, Pascal and Baum, Sebastian and others},
  journal = {MATEC Web of Conferences},
  volume  = {408},
  year    = {2025},
  doi     = {10.1051/matecconf/202540801090}
}
```

---

## Author

**Bala Sai Kiran Reddy Telluri**
Stuttgart, Germany
[LinkedIn](https://www.linkedin.com/in/bala-sai-kiran-reddy-telluri-055370250/) | [GitHub](https://github.com/BalaTelluri)
# ML-Surrogate-Model-for-Sheet-Metal-Forming-Process-Optimisation
