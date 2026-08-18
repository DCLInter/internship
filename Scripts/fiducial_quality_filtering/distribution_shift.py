"""
Unified distributional-shift analysis: how much does fiducial quality
filtering change feature/demographic/label distributions?

This replaces two scripts that had drifted apart:
  - Scripts/comparator.py (+ its drivers comparator_example.py,
    new_comparator.py): raw (unnormalized) Wasserstein distance, KS test,
    %-change stats, IQR/MAD outlier detection - but only ever run over the
    fiducial-derived PPG features, never demographics/labels. Also had
    per-patient / per-patient-median / pooled granularities.
  - Scripts/bp_lgbm/Dataset_exploration.py: IQR-normalized Wasserstein,
    but only pooled granularity, and demographics/labels only got
    describe()-based summaries - no actual distributional-distance metric
    for them, despite being equally affected by segment-dropping.

Per the project owner's decision: this keeps only the POOLED (whole-
column, all-samples) granularity - not per-patient or per-patient-median -
and extends the Wasserstein/KS treatment to demographics/labels too, not
just the 28 PPG features, since dropping a segment affects both equally.

Cross-package import note: this pulls preprocessing.median_impute_patientwise
from Scripts/bp_lgbm/ via the sys.path insert below. This is a deliberate,
temporary shortcut - cross-package imports and relative paths across the
whole repo get cleaned up in one pass before publication, not per-script.

Run from inside this folder:
    ..\..\.venv310\Scripts\python.exe distribution_shift.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bp_lgbm"))
from preprocessing import median_impute_patientwise  # noqa: E402

from run_pipeline import SUBSETS, DATA_DIR, load_features

# The 28 PPG features, in the order they appear in PPG_Features (see
# CLAUDE.md "The 28 PPG features" - same order/naming used throughout
# Scripts/bp_lgbm/).
PPG_FEATURE_NAMES = [
    "IPR", "Tsp", "TWRRF25", "TWRRF50", "Tsw25", "Tsw50", "Tsw75",
    "Tdw25", "Tdw50", "Tdw75", "AUCpi", "IPA", "Av-Au_ratio", "Ab-Aa_ratio",
    "Ac-Aa_ratio", "Ad-Aa_ratio", "Ap2-Ap1_ratio", "AGI", "Kurtosis",
    "Skewness", "L-H_ratio", "ShannonEntropy", "Tpp", "PRV", "FullKurt",
    "FullSkew", "sdPRV", "IQR_PRV",
]

# Demographic/label columns: NOT fiducial-derived, but equally affected by
# segment-dropping, so they get the same shift treatment as the 28 features.
DEMOGRAPHIC_LABEL_COLUMNS = ["Age", "BMI", "Height", "Weight", "SBP", "DBP", "MAP"]

# Gender (categorical) and SF (constant) are loaded but excluded from shift
# stats - Wasserstein/KS aren't meaningful for a binary category or a
# constant.

OUTPUT_ROOT = Path(
    r"C:\Users\Felipe Saldarriaga\OneDrive - City, University of London"
    r"\3.PhD\9. Experiments\2.LightGBM_SHAP\Distribution_Analysis"
)

THRESHOLDS_TO_ANALYZE = [90, 80]


def features_dict_to_dataframe(features: dict) -> pd.DataFrame:
    """
    Converts run_pipeline.load_features()'s dict into one DataFrame:
    Subject (decoded), Gender/SF (kept, not analyzed), the 7 demographic/
    label columns, and the 28 named PPG features.
    """
    df = pd.DataFrame({
        "Subject": [s.decode() if isinstance(s, bytes) else s for s in features["Subject"]],
        "Gender": [g.decode() if isinstance(g, bytes) else g for g in features["Gender"]],
        "SF": features["SF"],
    })
    for col in DEMOGRAPHIC_LABEL_COLUMNS:
        df[col] = features[col]
    ppg = features["PPG_Features"].copy()
    ppg.columns = PPG_FEATURE_NAMES
    return pd.concat([df, ppg], axis=1)


def normalized_wasserstein_and_ks(reference: pd.Series, comparison: pd.Series) -> dict:
    """
    Pooled (whole-sample) distributional shift for one column:
      - reference_q1 / reference_q3 / reference_iqr: the reference
        distribution's own IQR, kept alongside the result so the
        normalization below is auditable without re-deriving it.
      - wasserstein: raw Wasserstein (Earth Mover's) distance.
      - normalized_wasserstein: wasserstein / reference_iqr (NaN if that
        IQR is 0) - makes the distance comparable across features/labels
        with very different scales.
      - ks_statistic_D / ks_pvalue: two-sample Kolmogorov-Smirnov test.

    Both sides are expected to already be imputed (see run_shift_analysis)
    - the dropna() here is just a safety net for any leftover NaNs.
    """
    ref = reference.dropna().to_numpy(dtype=float)
    comp = comparison.dropna().to_numpy(dtype=float)

    q1, q3 = np.percentile(ref, 25), np.percentile(ref, 75)
    iqr = q3 - q1

    w = wasserstein_distance(ref, comp)
    normalized_w = w / iqr if iqr != 0 else np.nan

    ks_stat, ks_pvalue = ks_2samp(ref, comp)

    return {
        "reference_q1": q1,
        "reference_q3": q3,
        "reference_iqr": iqr,
        "wasserstein": w,
        "normalized_wasserstein": normalized_w,
        "ks_statistic_D": ks_stat,
        "ks_pvalue": ks_pvalue,
    }


def compare_distributions(reference_df: pd.DataFrame, comparison_df: pd.DataFrame,
                           columns: list) -> pd.DataFrame:
    """Runs normalized_wasserstein_and_ks() for every column in `columns`."""
    rows = {col: normalized_wasserstein_and_ks(reference_df[col], comparison_df[col])
            for col in columns}
    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = "column"
    return result


def run_shift_analysis(subset_name: str, threshold: int, output_dir: Path):
    original_path = SUBSETS[subset_name]["features"]
    clean_path = DATA_DIR / f"Clean_Features_{subset_name}_{threshold}.h5"

    print(f"\n=== {subset_name} @ threshold {threshold} ===")
    if not clean_path.exists():
        print(f"  Skipping: {clean_path} not found.")
        return None

    original = features_dict_to_dataframe(load_features(original_path))
    clean = features_dict_to_dataframe(load_features(clean_path))
    print(f"  original: {len(original)} signals, clean: {len(clean)} signals")

    original_imputed = median_impute_patientwise(original, patient_col="Subject")
    clean_imputed = median_impute_patientwise(clean, patient_col="Subject")

    columns = PPG_FEATURE_NAMES + DEMOGRAPHIC_LABEL_COLUMNS
    shift = compare_distributions(original_imputed, clean_imputed, columns)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"shift_pooled_{subset_name}_{threshold}.csv"
    shift.to_csv(out_path)
    print("  Saved:", out_path)
    return shift


if __name__ == "__main__":
    for subset_name in SUBSETS:
        for threshold in THRESHOLDS_TO_ANALYZE:
            run_shift_analysis(subset_name, threshold, OUTPUT_ROOT)
