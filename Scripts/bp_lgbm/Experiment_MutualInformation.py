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
    data_path = local_paths.DATA_DIR /"features_cleaned.h5"
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

    XY_df = preprocessing.merge_XY(X_df_original_feat, Y_df_original_feat) # When possible, add the parmeter "on: Segements_ID"
    print(XY_df.head())
    print(XY_df.info())

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
    X_df_train, Y_df_train = preprocessing.split_XY(XY_df,
                                                    target_cols= ["SBP", "DBP", "MAP"],
                                                    id_cols= ["Patient"]
                                                    )
    
    # =========================================================
    # Initialize model
    #==========================================================
    # drop ID columns before training
    id_cols = ["Patient", "segment_ID"]
    targets = ["SBP", "DBP", "MAP"]
    groups = X_df_train["Patient"]
    X_train = X_df_train.drop(columns=id_cols)
    Y_train = Y_df_train[targets]   # or whatever target you want

    # build the model
    cfg = ExperimentConfig(n_splits=5,
                           random_state=42, 
                           experiment_name="Shapley_test",
                           verbose = -1,
                           model_params=lightGBM_best_guess_1)
    lgbm = build_lgbm(cfg)

    # build the pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", lgbm)
    ])

    # =========================================================
    # Mutual Information Analysis
    #==========================================================
    for target in Y_train.columns:
        print(f"Analysing - {target} - target")
        mi_results = mi.compute_mi_summary(X_train, Y_train, X_train.columns, target, k = 5, save_path=Path(f"mi_results/{target}.csv"))

    