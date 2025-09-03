############ EXPERIMENT SHAPLEY VALUES ########################
#                                                             #
# Here, the mutual information experiment will be performed   #
#                                                             #
###############################################################
"""
Note: The implementation of SHAP is done using the shap library.

For the iterative process for the feature ranking is borrowed from:

"""

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

    df_feat_cleaned_mean = load_patient_dataset(data_path_cleaned, dataset_type="mean")
    labels_cleaned_df = load_patient_dataset(file_path=labels_path, column_names=["SBP", "DBP", "MAP", "segment_ID"])
    print(df_feat_cleaned_mean.head())
    print(labels_cleaned_df.head())
    # =========================================================
    # Check NaNs and fill them
    #==========================================================
    """
    print("************************************************************")
    print("Check NaNs")
    print("************************************************************")
    
    # Check if any NaN at all
    print("Is there any NaN in: df_features_mean?")
    print(df_feat_cleaned_mean.isna().any().any())
    # Count total number of NaNs
    print(df_feat_cleaned_mean.isna().sum())
    """

    # Note: ONLY THE ORGINAL FEATURE DATASET HAS NaNs
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
    XY_df_original = preprocessing.merge_XY(X_df_original_feat, Y_df_original_feat) # When possible, add the parmeter "on: Segements_ID"
    XY_df_original = preprocessing.reset_segment_ids(XY_df_original, patient_col="Patient", segment_col="segment_ID")

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
    print(uncommon.head())

    # Dropping the labels
    Y_df_t_clean = preprocessing.drop_rows_by_keys(Y_df_t_orig, uncommon, keys = ["Patient", "segment_ID"])

    """
    # Check the length of the datasets
    print(X_df_t_clean.info())
    print(Y_df_t_clean.info())
    XY_df_clean = preprocessing.merge_XY(X_df_t_clean, Y_df_t_clean, on=["Patient", "segment_ID"])
    print(XY_df_clean.info())
    """
    
    # =========================================================
    # Shapley Analysis
    #==========================================================
    """
    # I need to refit a model (this time over the full train dataset)
    final_lgbm = pipeline.fit(X_train, Y_train)
    avg_rank, rank_matrix, avg_abs_shap, abs_shap_matrix, rank_diff_matrix, feature_names = sa.shap_rank_stability(final_lgbm, X_train, Y_train, n_iter=50)
    """