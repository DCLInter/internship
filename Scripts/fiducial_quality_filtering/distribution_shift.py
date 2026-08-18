"""
Unified distributional-shift analysis: how much does fiducial quality
filtering change feature/demographic/label distributions?

This replaces two scripts that had drifted apart:
  - Scripts/comparator.py (+ its drivers comparator_example.py,
    new_comparator.py): raw (unnormalized) Wasserstein distance, KS test,
    %-change stats, IQR/MAD outlier detection + Jaccard overlap - but only
    ever run over the fiducial-derived PPG features, never demographics/
    labels. Also had per-patient / per-patient-median / pooled
    granularities.
  - Scripts/bp_lgbm/Dataset_exploration.py: IQR-normalized Wasserstein,
    but only pooled granularity, and demographics/labels only got
    describe()-based summaries - no actual distributional-distance metric
    for them, despite being equally affected by segment-dropping.

Per the project owner's decision: this keeps only the POOLED (whole-
column, all-samples) granularity - not per-patient or per-patient-median -
but keeps EVERYTHING comparator.py computed at that granularity (%-change
of mean/median/Q1/Q3/IQR/MAD, IQR/MAD outlier detection, Jaccard overlap
of outlier sets), adds the IQR-normalized Wasserstein from
Dataset_exploration.py, and extends ALL of it to demographics/labels too,
not just the 28 PPG features, since dropping a segment affects both
equally.

One consistency fix vs. comparator.py: every %-change calc here returns
NaN (not a silent 0) when its denominator is 0 - the original mixed the
two (e.g. IQR/MAD %-change guarded with `else 0`, Q1/Q3 %-change had no
guard at all and could raise). Jaccard overlap is computed the same way
comparator.py did it: over the outlier subsets' own positional indices
within the pooled "before"/"after" arrays (not a shared physical-signal
identity - "before" and "after" have different lengths/signals, so this
measures index-position overlap of the outlier subsets, exactly matching
what byFeature_all() in comparator.py did).

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
from scipy.stats import iqr as scipy_iqr
from scipy.stats import ks_2samp, median_abs_deviation, wasserstein_distance

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


def _pct_change(before: float, after: float) -> float:
    """100 * (after - before) / before, NaN if before is 0 (see module docstring)."""
    return 100 * (after - before) / before if before != 0 else np.nan


def pct_change_stats(before: np.ndarray, after: np.ndarray) -> dict:
    """
    Percent change (before -> after) of mean, median, Q1, Q3, IQR, MAD -
    ports comparator.py's cleaningAnalysis() stats_changes dict.
    """
    return {
        "pct_change_mean": _pct_change(np.nanmean(before), np.nanmean(after)),
        "pct_change_median": _pct_change(np.nanmedian(before), np.nanmedian(after)),
        "pct_change_q1": _pct_change(np.nanquantile(before, 0.25), np.nanquantile(after, 0.25)),
        "pct_change_q3": _pct_change(np.nanquantile(before, 0.75), np.nanquantile(after, 0.75)),
        "pct_change_iqr": _pct_change(scipy_iqr(before, nan_policy="omit"),
                                       scipy_iqr(after, nan_policy="omit")),
        "pct_change_mad": _pct_change(median_abs_deviation(before, nan_policy="omit"),
                                       median_abs_deviation(after, nan_policy="omit")),
    }


def _iqr_mad_outlier_mask(array: np.ndarray) -> tuple:
    """
    Ports comparator.py's outliers_IQRandMAD(): returns (iqr_mask, mad_mask),
    boolean arrays flagging IQR-fence (1.5x) and modified-z-score
    (MAD, threshold 3) outliers.
    """
    median = np.nanmedian(array)
    q1, q3 = np.nanquantile(array, [0.25, 0.75])
    iqr_val = q3 - q1
    low, high = q1 - 1.5 * iqr_val, q3 + 1.5 * iqr_val
    iqr_mask = (array < low) | (array > high)

    mad = median_abs_deviation(array, nan_policy="omit")
    modified_z = np.abs((array - median) / mad) * 0.6745 if mad != 0 else np.zeros_like(array)
    mad_mask = modified_z > 3

    return iqr_mask, mad_mask


def outlier_overlap_stats(before: np.ndarray, after: np.ndarray) -> dict:
    """
    IQR/MAD outlier counts before/after (raw + % of that array's own size)
    and the Jaccard overlap of the outlier subsets' positional indices -
    ports comparator.py's byFeature_all() outlier/overlap block exactly
    (see module docstring for the positional-index caveat).
    """
    before_iqr_mask, before_mad_mask = _iqr_mad_outlier_mask(before)
    after_iqr_mask, after_mad_mask = _iqr_mad_outlier_mask(after)

    def jaccard(mask_a, mask_b):
        idx_a, idx_b = set(np.where(mask_a)[0]), set(np.where(mask_b)[0])
        union = idx_a | idx_b
        return 100 * len(idx_a & idx_b) / len(union) if union else np.nan

    return {
        "n_outliers_iqr_before": int(before_iqr_mask.sum()),
        "n_outliers_iqr_after": int(after_iqr_mask.sum()),
        "pct_outliers_iqr_before": 100 * before_iqr_mask.sum() / len(before),
        "pct_outliers_iqr_after": 100 * after_iqr_mask.sum() / len(after),
        "jaccard_overlap_iqr_pct": jaccard(before_iqr_mask, after_iqr_mask),
        "n_outliers_mad_before": int(before_mad_mask.sum()),
        "n_outliers_mad_after": int(after_mad_mask.sum()),
        "pct_outliers_mad_before": 100 * before_mad_mask.sum() / len(before),
        "pct_outliers_mad_after": 100 * after_mad_mask.sum() / len(after),
        "jaccard_overlap_mad_pct": jaccard(before_mad_mask, after_mad_mask),
    }


def compare_distributions(reference_df: pd.DataFrame, comparison_df: pd.DataFrame,
                           columns: list) -> pd.DataFrame:
    """
    Runs the full pooled-granularity comparison for every column in
    `columns`: normalized Wasserstein + KS, %-change stats, and IQR/MAD
    outlier + Jaccard overlap.
    """
    rows = {}
    for col in columns:
        ref, comp = reference_df[col], comparison_df[col]
        ref_arr = ref.dropna().to_numpy(dtype=float)
        comp_arr = comp.dropna().to_numpy(dtype=float)
        rows[col] = {
            **normalized_wasserstein_and_ks(ref, comp),
            **pct_change_stats(ref_arr, comp_arr),
            **outlier_overlap_stats(ref_arr, comp_arr),
        }
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
