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
import preprocessing
import shap_analysis as sa
from local_paths import PULSE_DB_SUP_DIR
from data import load_PulseDB_sup_ds
from models import build_lgbm
from config import ExperimentConfig, lightGBM_best_guess_1
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pathlib import Path

if __name__ == "__main__":

    # =========================================================
    # Paths
    #==========================================================
    train_original_path = PULSE_DB_SUP_DIR / "Features_VitalDB_Train_Subset.h5"
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
    df_test = load_PulseDB_sup_ds(test_original_path, feature_names=feature_names)
    
    # =========================================================
    # Check NaNs and fill them
    #==========================================================
    """
    print("************************************************************")
    print("Check NaNs")
    print("************************************************************")
    
    # Check if any NaN at all
    print("Is there any NaN in: df_features_mean?")
    print(df_test.isna().any().any())
    # Count total number of NaNs
    print(df_test.isna().sum())
    """
    # Fill X NaNs
    df_train = preprocessing.median_impute_patientwise(df_train, patient_col= "Subject")
    df_test = preprocessing.median_impute_patientwise(df_test, patient_col= "Subject")
    """
    print(df_test.isna().any().any())
    """

    # =========================================================
    # Initialize model
    #==========================================================
    print(df_train.info())
    df_train = preprocessing.downsample_per_patient(df_train, patient_col="Subject", proportion = 0.1)
    print(df_train.info())
    
    """
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
    # Shapley Analysis
    #==========================================================
    #~~~~~~~~~~ Testing pipeline only ~~~~~~~~~~~~~~~~~~~~~~~~
    
    # Count signals per patient
    counts = XY_df_original.groupby("Patient").size()

    # Get top 5 patients
    top5_patients = counts.nlargest(5).index   # patient IDs of top 5

    # If you want to extract all their rows:
    top5_XY_df = XY_df_original[XY_df_original["Patient"].isin(top5_patients)]
    print(top5_XY_df.info())
    X_train, Y_train = preprocessing.split_XY(top5_XY_df,
                                                    target_cols= ["SBP", "DBP", "MAP"],
                                                    id_cols= ["Patient", "segment_ID"]
                                                    )
    #Dropping strg columns
    id_cols = ["Patient", "segment_ID"]
    groups = X_df_t_orig["Patient"]
    targets = ["SBP", "DBP", "MAP"]
    X_train = X_df_t_orig.drop(columns=id_cols)
    Y_train = Y_df_t_orig["SBP"]   # or whatever target you want

    # I need to refit a model (this time over the full train dataset)
    avg_rank, rank_matrix, avg_abs_shap, abs_shap_matrix, rank_diff_matrix, feature_names = sa.shap_rank_stability(pipeline, 
                                                                                                                   X_train, 
                                                                                                                   Y_train,
                                                                                                                   groups = groups, 
                                                                                                                   n_iter=50, 
                                                                                                                   save_path=Path(f"shap_results/test/{targets[0]}"))
    print("end")
    """
    