############ DATASET EXPLORATION ##############################
#                                                             #
# Here, we want to extract some statistics from the dataset   #
#                                                             #
###############################################################
import preprocessing
import pandas as pd
import numpy as np
from pathlib import Path
from local_paths import PULSE_DB_SUP_DIR, DATASET_EXPLORATION_RESULTS
from data import load_PulseDB_sup_ds
from scipy.stats import wasserstein_distance


# Functions
# ============================================================
# 1️⃣ Basic Summaries
# ============================================================

def summarize_dataset(df: pd.DataFrame, name: str, subj_col: str = "subject_id") -> pd.DataFrame:
    """Summarize basic dataset statistics."""
    summary = {
        "dataset": name,
        "n_signals": len(df),
        "n_subjects": df[subj_col].nunique(),
    }
    return pd.DataFrame([summary])


# ============================================================
# 2️⃣ Cleaning Impact Between Two Datasets
# ============================================================

def compare_datasets(
    df_base: pd.DataFrame,
    df_new: pd.DataFrame,
    subj_col: str = "subject_id"
) -> pd.DataFrame:
    """
    Compare datasets and compute how many signals and subjects were removed.
    """
    base_subj = set(df_base[subj_col])
    new_subj = set(df_new[subj_col])
    
    lost_subjects = base_subj - new_subj
    subject_overlap = len(base_subj & new_subj) / len(base_subj)
    
    summary = {
        "comparison": f"{df_base.name if hasattr(df_base, 'name') else 'base'} → {df_new.name if hasattr(df_new, 'name') else 'new'}",
        "lost_signals": len(df_base) - len(df_new),
        "lost_subjects": len(lost_subjects),
        "subject_overlap_ratio": subject_overlap
    }
    return pd.DataFrame([summary])


# ============================================================
# 3️⃣ Demographic Summaries
# ============================================================

def demographic_summary(df: pd.DataFrame, demo_cols: list) -> pd.DataFrame:
    """
    Compute mean, std, min, max, and quartiles for demographic variables.
    """
    return df[demo_cols].describe().T


# ============================================================
# 4️⃣ Demographics of Completely Removed Subjects
# ============================================================

def demographics_removed_subjects(
    df_base: pd.DataFrame,
    df_new: pd.DataFrame,
    subj_col: str,
    demo_cols: list
) -> pd.DataFrame:
    """
    Identify subjects completely removed after cleaning and summarize their demographics.
    """
    base_subj = set(df_base[subj_col])
    new_subj = set(df_new[subj_col])
    lost_subj = base_subj - new_subj
    
    if not lost_subj:
        return pd.DataFrame(columns=demo_cols)
    
    lost_df = df_base[df_base[subj_col].isin(lost_subj)]
    return lost_df[demo_cols].describe().T


# ============================================================
# 5️⃣ Wasserstein Distance Normalized by Original IQR
# ============================================================

def compute_normalized_wasserstein(
    df_ref: pd.DataFrame,
    df_comp: pd.DataFrame,
    feature_cols: list
) -> pd.DataFrame:
    """
    Compute Wasserstein distances between datasets for each feature,
    normalized by the IQR of the reference (original) dataset.
    """
    iqr = df_ref[feature_cols].quantile(0.75) - df_ref[feature_cols].quantile(0.25)
    records = []
    
    for f in feature_cols:
        w = wasserstein_distance(df_ref[f].dropna(), df_comp[f].dropna())
        normalized_w = w / iqr[f] if iqr[f] != 0 else np.nan
        records.append({"feature": f, "wasserstein": w, "normalized_wasserstein": normalized_w})
    
    return pd.DataFrame(records)


# ============================================================
# 6️⃣ Pipeline Function to Run Everything
# ============================================================

def analyze_datasets(
    df_original: pd.DataFrame,
    df_90: pd.DataFrame,
    df_80: pd.DataFrame,
    subj_col: str,
    demo_cols: list,
    feature_cols: list,
    save_dir: str = None
) -> dict:
    """
    Run all comparisons and optionally save outputs as CSV.
    """
    # Attach names for labeling
    df_original.name, df_90.name, df_80.name = "original", "90", "80"
    
    # --- Summaries ---
    summaries = pd.concat([
        summarize_dataset(df_original, "original", subj_col),
        summarize_dataset(df_90, "90", subj_col),
        summarize_dataset(df_80, "80", subj_col)
    ])
    
    cleaning_impact = pd.concat([
        compare_datasets(df_original, df_90, subj_col),
        compare_datasets(df_original, df_80, subj_col)
    ])
    
    demo_original = demographic_summary(df_original, demo_cols)
    demo_90 = demographic_summary(df_90, demo_cols)
    demo_80 = demographic_summary(df_80, demo_cols)
    
    demo_removed_90 = demographics_removed_subjects(df_original, df_90, subj_col, demo_cols)
    demo_removed_80 = demographics_removed_subjects(df_original, df_80, subj_col, demo_cols)
    
    wdist_90 = compute_normalized_wasserstein(df_original, df_90, feature_cols)
    wdist_80 = compute_normalized_wasserstein(df_original, df_80, feature_cols)
    
    results = {
        "summary": summaries,
        "cleaning_impact": cleaning_impact,
        "demographics_original": demo_original,
        "demographics_90": demo_90,
        "demographics_80": demo_80,
        "demographics_removed_90": demo_removed_90,
        "demographics_removed_80": demo_removed_80,
        "wasserstein_90": wdist_90,
        "wasserstein_80": wdist_80,
    }

    # --- Optional saving ---
    if save_dir is not None:
        for key, df in results.items():
            df.to_csv(f"{save_dir}/{key}.csv", index=True)
    
    return results

if __name__ == "__main__":

    # =========================================================
    # Paths
    #==========================================================
    train_original_path = PULSE_DB_SUP_DIR / "Features_VitalDB_Train_Subset.h5"
    train_clean_path_90 = PULSE_DB_SUP_DIR / "Clean_Features_VitalDB_Train_Subset_90.h5"
    train_clean_path_80 = PULSE_DB_SUP_DIR / "Clean_Features_VitalDB_Train_Subset_80.h5"

    test_clean_path_90 = PULSE_DB_SUP_DIR / "Clean_Features_VitalDB_CalFree_Test_Subset_90.h5"
    test_clean_path_80 = PULSE_DB_SUP_DIR/ "Clean_Features_VitalDB_CalFree_Test_Subset_80.h5"
    test_original_path = PULSE_DB_SUP_DIR / "Features_VitalDB_CalFree_Test_Subset.h5"

    # =========================================================
    # Load Data
    #==========================================================
    feature_names = ["IPR", "Tsp", "TWRRF25", "TWRRF50", "Tsw25", 
                 "Tsw50", "Tsw75", "Tdw25", "Tdw50", "Tdw75", 
                 "AUCpi", "IPA",  "Av-Au_ratio", "Ab-Aa_ratio", "Ac-Aa_ratio", 
                 "Ad-Aa_ratio", "Ap2-Ap1_ratio", "AGI", "Kurtosis", "Skewness", 
                 "L-H_ratio", "ShannonEntropy", "Tpp", "PRV", "FullKurt", 
                 "FullSkew", "sdPRV", "IQR_PRV"]
    df_train = load_PulseDB_sup_ds(train_original_path, feature_names=feature_names)
    df_train_90 = load_PulseDB_sup_ds(train_clean_path_90, feature_names=feature_names)
    df_train_80 = load_PulseDB_sup_ds(train_clean_path_80, feature_names=feature_names)
    df_test = load_PulseDB_sup_ds(test_original_path, feature_names=feature_names)
    df_test_90 = load_PulseDB_sup_ds(test_clean_path_90, feature_names=feature_names)
    df_test_80 = load_PulseDB_sup_ds(test_clean_path_80, feature_names=feature_names)

    # =========================================================
    # Check NaNs and fill them
    #==========================================================
    df_train = preprocessing.median_impute_patientwise(df_train, patient_col= "Subject")
    df_test = preprocessing.median_impute_patientwise(df_test, patient_col= "Subject")

    # =========================================================
    # Perform the statistical analysis
    #==========================================================
    save_dir_train = DATASET_EXPLORATION_RESULTS/"Training"
    results_train = analyze_datasets(
    df_train,
    df_train_90,
    df_train_80,
    subj_col="Subject",
    demo_cols=["Age", "Gender", "BMI", "Height", "Weight"],
    feature_cols=feature_names,
    save_dir=save_dir_train
    )

    save_dir_test = DATASET_EXPLORATION_RESULTS/"Testing"
    results_test = analyze_datasets(
    df_test,
    df_test_90,
    df_test_80,
    subj_col="Subject",
    demo_cols=["Age", "Gender", "BMI", "Height", "Weight"],
    feature_cols=feature_names,
    save_dir=save_dir_test
    )