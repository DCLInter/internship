############ EXPERIMENT DEMOGRAPHIC STRATIFICATION ############
#                                                             #
# Here, THE ds WILL BE SPLITTED ACCORDING TO SOME STRATA      #
#                                                             #
###############################################################

import demo_strata_utils as ds
import preprocessing
from pathlib import Path
from local_paths import PULSE_DB_SUP_DIR, GS_RESULT_PAPER
from data import load_PulseDB_sup_ds, load_config

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
    
    print(df_train.head())
    print(df_test.head())

    # =========================================================
    # Check NaNs and fill them
    #==========================================================
    """
    print("************************************************************")
    print("Check NaNs")
    print("************************************************************")
    
    # Check if any NaN at all
    print("Is there any NaN in: df_features_mean?")
    print(df_train.isna().any().any())
    # Count total number of NaNs
    print(df_train.isna().sum())
    """
    # Fill X NaNs
    df_train = preprocessing.median_impute_patientwise(df_train, patient_col= "Subject")
    df_test = preprocessing.median_impute_patientwise(df_test, patient_col= "Subject")

    # =========================================================
    # Segmentation by demo thresholds
    #==========================================================
    thresholds = {
        "Age": [40, 60],
        "BMI": [25],
        "Gender": ["M", "F"]
    }
    dfs_dict_train = ds.segment_thresholds(df_train, rules=thresholds, subject_col_name="Subject", verbose=True)
    dfs_dict_test = ds.segment_thresholds(df_test, rules=thresholds, subject_col_name="Subject", verbose=True)

    # =========================================================
    # Initialize model
    #==========================================================
    # build the model
    grid_path = GS_RESULT_PAPER / "Full_Grid_Randomized_search_3targets.json"
    cfg = load_config(grid_path)
    lgbm = build_lgbm(cfg)

    # build the pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", lgbm)
    ])
