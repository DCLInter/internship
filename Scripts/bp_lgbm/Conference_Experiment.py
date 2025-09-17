############ EXPERIMENT CONFERENCE ############################
#                                                             #
# Here, the experiments for the EMBC wll be conducted. It is  #
#   single training + evaluation + SHAP claculation (1 iter). #
#                                                             #
###############################################################

import preprocessing
import eval
from pathlib import Path
from local_paths import PULSE_DB_SUP_DIR
from data import load_PulseDB_sup_ds
from config import ExperimentConfig, load_config
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from models import build_lgbm

if __name__ == "__main__":

    # =========================================================
    # Paths
    #==========================================================
    train_original_path = PULSE_DB_SUP_DIR / "Features_complete_VitalDB_Train_Subset.h5"
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
    
    print(df_train["MAP"].head())
    print(df_test["MAP"].head())
    
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
    # drop ID columns before training
    id_cols = ["Subject", "Age", "Gender", "Height", "Weight", "BMI", "SF"]
    targets = ["SBP", "DBP", "MAP"]
    id_cols.extend(targets)
    groups = df_train["Subject"]
    X_train = df_train.drop(columns=id_cols)
    X_test = df_test.drop(columns=id_cols)
    print(X_train.head())
    # build the model
    """
    cfg = ExperimentConfig(n_splits=5,
                           random_state=42, 
                           experiment_name="Conference_Experiment_Test_Shap",
                           verbose = -1,
                           model_params=lightGBM_best_guess_1)
    """
    grid_path = Path(r"C:\Users\addp972\OneDrive - City, University of London\3.PhD\9. Experiments\2.LightGBM_SHAP\ICASSP_Submission\GridSearchResults") / "Full_Grid_Randomized_search_3targets.json"
    cfg = load_config(grid_path)
    lgbm = build_lgbm(cfg)

    # build the pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", lgbm)
    ])

    # =========================================================
    # Training and Eval
    #==========================================================
    
    Y_train = df_train[targets[2]]
    Y_test = df_test[targets[2]]

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state= 42, shuffle=True)

    # Train
    pipeline.fit(X_train, Y_train)
    # Pred
    Y_val_pred = pipeline.predict(X_val)
    Y_pred = pipeline.predict(X_test)
    # Eval
    Bland_Altman_Path = Path(r"C:\Users\addp972\OneDrive - City, University of London\3.PhD\9. Experiments\2.LightGBM_SHAP\ICASSP_Submission\Bland_Altman")
    metrics_val_original = eval.evaluate(Y_val, Y_val_pred, Bland_Altman_Path / f"val_original_{targets[2]}.png")
    metrics_test_original = eval.evaluate(Y_test, Y_pred, Bland_Altman_Path / f"test_original_{targets[2]}.png")

    print(f"=== Results for {targets[2]}, in VAL ===")
    print(metrics_val_original)

    print(f"=== Results for {targets[2]}, in TEST ===")
    print(metrics_test_original)
