############ EXPERIMENT MUTUAL INFORMATION ####################
#                                                             #
# Here, the mutual information experiment will be performed   #
#                                                             #
###############################################################
import numpy as np
import local_paths
import preprocessing
import cv
import gs
import mutual_information as mi
import shap_analysis as sa
from data import load_patient_dataset, load_group_attributes
from models import build_lgbm
from config import ExperimentConfig, lightGBM_best_guess_1
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error as mse
from pathlib import Path
from h5_inspector import inspect_file


if __name__ == "__main__":

    # =========================================================
    # Paths
    #==========================================================
    data_path_cleaned = local_paths.DATA_DIR /"features_patients_clean.h5"
    labels_path = local_paths.LABELS_DIR / "BP_values.h5"
    data_messy_path = local_paths.DATA_DIR /"features_patients.h5"

    # =========================================================
    # Load Data
    #==========================================================
    """
    inspect_file(data_messy_path, show_attrs=True)
    inspect_file(labels_path, show_attrs=True)
    """

    df_feat_orig_mean = load_patient_dataset(data_messy_path, dataset_type="mean")
    labels_original_df = load_patient_dataset(file_path=labels_path, column_names=["SBP", "DBP", "MAP", "segment_ID"])
    metadata_original_df = load_group_attributes(data_messy_path)
    df_feat_cleaned_mean = load_patient_dataset(data_path_cleaned, dataset_type="mean")
    # =========================================================
    # Check NaNs and fill them
    #==========================================================
    """
    print("************************************************************")
    print("Check NaNs")
    print("************************************************************")
    
    # Check if any NaN at all
    print("Is there any NaN in: df_features_mean?")
    print(df_feat_orig_mean.isna().any().any())
    # Count total number of NaNs
    print(df_feat_orig_mean.isna().sum())
    """

    # Fill X NaNs
    X_df_original_feat = preprocessing.median_impute_patientwise(df_feat_orig_mean)
    print("Checking NaN after inputation in X:", X_df_original_feat.isna().any().any())

    # Fill Y NaNs
    Y_df_original_feat = preprocessing.fill_missing_bp(labels_original_df)

    # =========================================================
    # Solving Macheetazos
    #==========================================================
    uncommon = preprocessing.find_uncommon_rows(X_df_original_feat, df_feat_cleaned_mean)

    # =========================================================
    # X and Y dfs merging
    #==========================================================
    """ Checking duplicates in both datasets"""
    """
    keys = ["Patient", "segment_ID"]  # or whatever you use for `on`
    print( X_df_original_feat[keys].duplicated().sum())
    print( Y_df_original_feat[keys].duplicated().sum())

    # In X
    dupes_X = X_df_original_feat[X_df_original_feat.duplicated(keys, keep=False)].sort_values(keys)
    print(dupes_X)

    # In Y
    dupes_Y = Y_df_original_feat[Y_df_original_feat.duplicated(keys, keep=False)].sort_values(keys)
    print(dupes_Y)

    set_X = set(map(tuple, X_df_original_feat[keys].to_numpy()))
    set_Y = set(map(tuple, Y_df_original_feat[keys].to_numpy()))

    missing_in_Y = set_X - set_Y
    missing_in_X = set_Y - set_X

    print("Keys in X but not in Y:", missing_in_Y)
    print("Keys in Y but not in X:", missing_in_X)
    """

    XY_df_original = preprocessing.merge_XY(X_df_original_feat, Y_df_original_feat) # When possible, add the parmeter "on: Segements_ID"
    XY_df_original = preprocessing.reset_segment_ids(XY_df_original, patient_col="Patient", segment_col="segment_ID")

     #~~~~~~~~~~ Testing pipeline only ~~~~~~~~~~~~~~~~~~~~~~~~
    """
    # Count signals per patient
    counts = XY_df.groupby("Patient").size()

    # Get top 5 patients
    top5_patients = counts.nlargest(5).index   # patient IDs of top 5

    # If you want to extract all their rows:
    top5_XY_df = XY_df[XY_df["Patient"].isin(top5_patients)]
    print(top5_XY_df.info())
    """

    

    # =========================================================
    # Splitting into X and Y
    #==========================================================
    X_df_t_orig, Y_df_t_orig = preprocessing.split_XY(XY_df_original,
                                                    target_cols= ["SBP", "DBP", "MAP"],
                                                    id_cols= ["Patient", "segment_ID"]
                                                    )
    # =========================================================
    # Solving Machetazos 2.0
    #==========================================================
    X_df_t_clean = preprocessing.align_segment_ids(X_df_t_orig, df_feat_cleaned_mean, seg_col="segment_ID")
    uncommon = preprocessing.align_segment_ids(X_df_t_orig, uncommon, seg_col="segment_ID")

    # Dropping the labels
    Y_df_t_clean = preprocessing.drop_rows_by_keys(Y_df_t_orig, uncommon, keys = ["Patient", "segment_ID"])

    # =========================================================
    # Mutual Information Analysis
    #==========================================================
    X_df_t_clean = X_df_t_clean.drop(columns=["Patient", "segment_ID"])
    Y_df_t_clean = Y_df_t_clean.drop(columns=["Patient", "segment_ID"])
    
    
    targets = Y_df_t_clean.columns
    for target in targets:
        print(f"Analysing - {target} - target")
        mi_results = mi.compute_mi_summary(X_df_t_clean, Y_df_t_clean, X_df_t_clean.columns, target, k = 5, save_path=Path(f"mi_results/cleaned/{target}.csv"))
        

    