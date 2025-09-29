############ DEMOGRAPHIC ANALYSIS #############################
#                                                             #
# Here, all the distributions and balances                    #
# are gonna be checked                                        #
#                                                             #
###############################################################

import preprocessing
from pathlib import Path
from local_paths import PULSE_DB_SUP_DIR
from data import load_PulseDB_sup_ds
import plots

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
    """
    # =========================================================
    # Check NaNs and fill them
    #==========================================================
    # Fill X NaNs
    df_train = preprocessing.median_impute_patientwise(df_train, patient_col= "Subject")
    df_test = preprocessing.median_impute_patientwise(df_test, patient_col= "Subject")

    # =========================================================
    # Distribution Plots
    #==========================================================
    num_features = ["Age", "BMI", "Height", "Weight", "SBP", "DBP"]
    plots.plot_numeric_distributions(df_train, num_features, "Train")
    plots.plot_numeric_distributions(df_test, num_features, "Test")
    """
    plots.plot_gender_distribution(df_train, "Train")
    plots.plot_gender_distribution(df_test, "Test")
    
    thresholds = {
        "Age": [40, 60],
        "BMI": [18.5, 25],
        "SBP": [90, 120, 140], # dont pay aattention to the subject number
        "DBP": [40, 90]
    }

    plots.plot_threshold_proportions(df_train, thresholds, dataset_name="Train")
    plots.plot_threshold_proportions(df_test, thresholds, dataset_name="Test")