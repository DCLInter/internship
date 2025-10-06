############ EXPERIMENT MUTUAL INFORMATION ####################
#                                                             #
# Here, the mutual information experiment will be performed   #
#                                                             #
###############################################################
import preprocessing
import mutual_information as mi
from local_paths import PULSE_DB_SUP_DIR, MI_RESULTS_PAPER
from data import load_PulseDB_sup_ds

if __name__ == "__main__":

    #if __name__ == "__main__":

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
    print(df_train.info())
    
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
    """
    print(df_test.isna().any().any())
    """

    # =========================================================
    # Mutual Information Analysis
    #==========================================================
    id_cols = ["Subject", "Age", "Gender", "Height", "Weight", "BMI", "SF"]
    targets = ["SBP", "DBP", "MAP"]
    id_cols.extend(targets)
    X_df = df_train.drop(columns=id_cols)
    Y_df = df_train[targets]

    for target in targets:
        print(f"Analysing - {target} - target")
        mi_path = MI_RESULTS_PAPER / f"Original_DS/{target}.csv"
        mi_results = mi.compute_mi_summary(X_df, Y_df, X_df.columns, target, k = 5, save_path= mi_path)
        

    