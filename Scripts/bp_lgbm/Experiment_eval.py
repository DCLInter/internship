############ EXPERIMENT SHAPLEY VALUES ########################
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
import eval
from data import load_patient_dataset, load_group_attributes
from models import build_lgbm
from config import ExperimentConfig, lightGBM_best_guess_1
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from pathlib import Path

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
    df_feat_orig_mean = load_patient_dataset(data_messy_path, dataset_type="mean")
    labels_original_df = load_patient_dataset(file_path=labels_path, column_names=["SBP", "DBP", "MAP", "segment_ID"])
    metadata_original_df = load_group_attributes(data_messy_path)

    # =========================================================
    # Check NaNs and fill them
    #==========================================================
    # Note: ONLY THE ORGINAL FEATURE DATASET HAS NaNs
    # Fill X NaNs
    X_df_original_feat = preprocessing.median_impute_patientwise(df_feat_orig_mean)
    print("Checking NaN after inputation in X:", X_df_original_feat.isna().any().any())

    # Fill Y NaNs
    Y_df_original_feat = preprocessing.fill_missing_bp(labels_original_df)

    # =========================================================
    # X and Y dfs merging
    #==========================================================
    XY_df_original = preprocessing.merge_XY(X_df_original_feat, Y_df_original_feat) # When possible, add the parmeter "on: Segements_ID"
    XY_df_original = preprocessing.reset_segment_ids(XY_df_original, patient_col="Patient", segment_col="segment_ID")

    # =========================================================
    # Splitting process of the dataset IDs - patyient wise
    #==========================================================
    split_results = preprocessing.split_patients_by_signal_share(meta_df= metadata_original_df,
                                                 threshold= 0.8,
                                                 margin = 0.02,
                                                 patient_col="Patient",
                                                 signals_col= "Total_signals",
                                                 prefer="asc"
                                                 )
    print("Training subjects:", len(split_results["selected_ids"])) # OJOOO the logic here states that for me the "selected" set is for training
    print("Testing subjects:", len(split_results["remaining_ids"]))

    df_train, df_test = preprocessing.split_train_test(XY=XY_df_original, split_results=split_results, patient_col="Patient")

    # =========================================================
    # Splitting into X and Y
    #==========================================================
    X_orig_test, Y_orig_test = preprocessing.split_XY(df_test,
                                                    target_cols= ["SBP", "DBP", "MAP"],
                                                    id_cols= ["Patient", "segment_ID"]
                                                    )
    X_orig_train, Y_orig_train = preprocessing.split_XY(df_train,
                                                    target_cols= ["SBP", "DBP", "MAP"],
                                                    id_cols= ["Patient", "segment_ID"]
                                                    )
    X_train, X_val, Y_train, Y_val = train_test_split(X_orig_train, Y_orig_train, test_size=0.2, random_state= 42, shuffle=True)

    # =========================================================
    # Initialize model
    #==========================================================
    # build the model
    cfg = ExperimentConfig(n_splits=5, # it is not used in shap
                           random_state=42, 
                           experiment_name="ShapAnalysis",
                           verbose = -1,
                           model_params=lightGBM_best_guess_1)
    lgbm = build_lgbm(cfg)

    # build the pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", lgbm)
    ])

    # =========================================================
    # Train model
    #==========================================================
    # Dropping strg columns
    id_cols = ["Patient", "segment_ID"]
    targets = ["SBP", "DBP", "MAP"]
    X_train = X_train.drop(columns=id_cols)
    X_val = X_val.drop(columns=id_cols)
    X_test = X_orig_test.drop(columns=id_cols)
    Y_train = Y_train[targets[0]]
    Y_val = Y_val[targets[0]]
    Y_test = Y_orig_test[targets[0]]

    pipeline.fit(X_train,Y_train)

    # Predictions
    pred_val = pipeline.predict(X_val)
    pred_test = pipeline.predict(X_test)

    metrics_val = eval.evaluate(Y_val, pred_val, path=r"Bland_Altman_results\val_sbp.png")
    print("=========== This is Val metrics ==========")
    print(metrics_val)
    
    metrics_test = eval.evaluate(Y_test, pred_test, path=r"Bland_Altman_results\test_sbp.png")
    print("=========== This is Test metrics ==========")
    print(metrics_test)