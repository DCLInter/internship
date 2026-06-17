# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PhD internship project: non-invasive blood pressure (BP) estimation from PPG signals using LightGBM, with SHAP-based feature importance and demographic stratification analysis. The dataset is PulseDB/VitalDB.

Targets: **SBP** (systolic), **DBP** (diastolic), **MAP** (mean arterial pressure).

---

## Environment Setup

Python 3.10 environment is in `.venv310/`.

Install dependencies:
```
pip install -r Scripts/bp_lgbm/Requirements.txt
```

The `NPEET` package (Kraskov MI) installs from GitHub — requires git access.

### `local_paths.py` — required, gitignored

`Scripts/bp_lgbm/local_paths.py` is **gitignored**. Before running any script, create it with absolute paths for your machine:

```python
from pathlib import Path

DATA_DIR = Path(r"<path to Scripts/bp_lgbm/data_features>")
LABELS_DIR = Path(r"<path to project root>")
PULSE_DB_SUP_DIR = Path(r"<path to PulseDB supplementary subsets>")

# Saving folders
EMBC_BA_RESULTS = Path(r"<...>")
EMBC_SHAP_RESULTS = Path(r"<...>")
EMBC_GS_CONFIGS = Path(r"<...>")
MI_RESULTS_PAPER = Path(r"<...>")
PERFORMANCE_RESULTS_PAPER = Path(r"<...>")
GS_RESULT_PAPER = Path(r"<...>")
SHAP_RESULTS_PAPER = Path(r"<...>")
DEMOG_RESULTS_PAPER = Path(r"<...>")
ABBLATION_RESULTS_PAPER = Path(r"<...>")
DATASET_EXPLORATION_RESULTS = Path(r"<...>")
```

### Data files — gitignored

All `.h5` files and output directories (`CV_results*`, `mi_results`, `shap_results`, `configs`, `Bland_Altman_results`) are gitignored and must exist locally.

---

## Running Scripts

All `Scripts/bp_lgbm/` scripts use **relative imports** (e.g., `import preprocessing`, `from data import ...`). Run them from inside `Scripts/bp_lgbm/`:

```bash
cd Scripts/bp_lgbm
python Experiment_GridSearch.py
python Experiment_SHAPAnalysis.py
python Experiment_eval.py
```

`main.py` is a prototype/exploratory entrypoint, not the primary runner.

The `Experiment_*.py` scripts are the runnable experiment entrypoints. Each has an `if __name__ == "__main__":` block with a documented experiment description at the top.

---

## Architecture

### Two-layer structure

**Layer 1 — Signal processing (`Scripts/` root):**  
PPG signal quality checking, feature extraction, and cleaning. Uses the `pyPPG` library with project-specific overrides in `Scripts/lib_changes/`. Key scripts: `new_checker.py`, `new_extraction.py`, `cleaning.py`, `new_comparator.py`. This layer operates on raw fiducial-point HDF5 files to produce cleaned feature HDF5 files consumed by Layer 2.

**Layer 2 — ML pipeline (`Scripts/bp_lgbm/`):**  
Clean, modular Python package for BP estimation. Data flows linearly through:

```
data.py → preprocessing.py → [config.py + models.py] → gs.py / cv.py → eval.py
                                                                        ↓
                                                              shap_analysis.py
```

### `bp_lgbm/` module roles

| File | Role |
|---|---|
| `data.py` | Loads HDF5 → pandas DataFrames; handles patient-grouped (multi-dataset) and flat (PulseDB supplementary) H5 formats |
| `preprocessing.py` | Patient-wise median imputation, signal-share greedy train/test split, BP NaN filling, X/Y merging |
| `config.py` | `ExperimentConfig` dataclass; all LightGBM hyperparameter grids; `save_config`/`load_config` |
| `models.py` | `build_lgbm(cfg)` — constructs `LGBMRegressor` from config |
| `gs.py` | `run_grid_search()` — wraps sklearn's `GridSearchCV`/`RandomizedSearchCV` with group or sample CV |
| `cv.py` | `patient_wise_cv()` (GroupKFold) and `sample_wise_cv()` (KFold) |
| `eval.py` | Evaluation metrics per Elgendi (2024): MAE, ME, SDE, RMSE, R², Bland–Altman; saves plots |
| `shap_analysis.py` | SHAP computation, rank stability across iterations, all SHAP visualization functions |
| `mutual_information.py` | Info-Fraction (MI/H) using Kraskov KNN estimator via NPEET |
| `demo_strata_utils.py` | Demographic stratification by threshold rules; per-stratum SHAP analysis |
| `plots.py` | Demographic distribution plots |
| `local_paths.py` | Machine-local absolute paths (gitignored, must be created per machine) |

### Data formats

- **Patient-grouped H5** (`features_cleaned.h5`, `features_original.h5`, `BP_values.h5`): groups keyed by patient ID (e.g., `p000001`); each group contains datasets (`mean`, `median`, `segments`) of shape `(features, samples)` — note: shape is transposed on load.
- **Flat PulseDB supplementary H5** (`Features_VitalDB_Train_Subset.h5`): top-level datasets per variable (`Age`, `SBP`, `PPG_Features` etc.); loaded by `load_PulseDB_sup_ds()`.

### Multi-output regression pattern

LightGBM is wrapped for three simultaneous BP targets:

```python
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", MultiOutputRegressor(LGBMRegressor(**params)))
])
```

Grid search parameter names must use the `model__estimator__` prefix (e.g., `model__estimator__num_leaves`). Single-target pipelines use just `model__`.

### CV strategies

- **Patient-wise** (`GroupKFold`): correct clinical approach — no patient appears in both train and val. Use `groups=X["Patient"]` with `cv_type="group"`.
- **Sample-wise** (`KFold`): ignores patient grouping; inflates performance metrics. Only used for ablation/comparison experiments explicitly studying this effect.

### The 28 PPG features

`IPR, Tsp, TWRRF25, TWRRF50, Tsw25, Tsw50, Tsw75, Tdw25, Tdw50, Tdw75, AUCpi, IPA, Av-Au_ratio, Ab-Aa_ratio, Ac-Aa_ratio, Ad-Aa_ratio, Ap2-Ap1_ratio, AGI, Kurtosis, Skewness, L-H_ratio, ShannonEntropy, Tpp, PRV, FullKurt, FullSkew, sdPRV, IQR_PRV`

The first 23 are morphological PPG biomarkers; the last 5 (PRV, FullKurt, FullSkew, sdPRV, IQR_PRV) are pulse-rate variability features added later.

### SHAP rank stability

`shap_analysis.shap_rank_stability()` runs `n_iter` subject-disjoint train splits (pre-allocated by `allocate_subjects_for_train()` using inverse-signal-count weighting to balance subject appearances), fits the pipeline fresh each iteration, and returns rank matrices + diff matrices showing how feature importance rankings change across resamples.

---

## Pre-submission TODOs

### Reproducibility (deferred — results production is priority)

Seeds are consistently `42` everywhere, which is sufficient for paper claims. However, before submission:

1. **Wire seeds through config** — `gs.py` lines 59 and 84, and `demo_strata_utils.py` lines 253/257 hardcode `random_state=42` instead of accepting it as a parameter. If the seed ever needs to change, all these must be hunted down manually.

2. **Fix `n_jobs` coupling** — `models.py` hardcodes `n_jobs=-1` and ignores `cfg.n_jobs`. LightGBM with `n_jobs=-1` is not guaranteed to produce bit-for-bit identical results across runs due to parallel floating-point aggregation. The practical effect is <0.001 mmHg — invisible at 2 decimal places — so results are valid. Add this sentence to the paper methods: *"All experiments used random seed 42; minor numerical differences (<0.001 mmHg) may arise from floating-point non-determinism in parallel LightGBM execution."*

3. **Add global RNG initialization** to the top of each `Experiment_*.py`:
   ```python
   import numpy as np
   np.random.seed(42)
   ```

### Model persistence (deferred)

No trained models are currently saved to disk. This is not a problem for producing results (retraining is fast and deterministic), but before submission:

- Save the final trained pipeline (post grid search, fit on full train set) using `joblib.dump(pipeline, path)`. LightGBM models are typically a few MB.
- This allows post-hoc analyses a reviewer might request (new SHAP plots, error breakdowns, subgroup evaluations) without retraining from scratch.
- It also guards against any future change to the data or preprocessing code making exact reproduction harder.
