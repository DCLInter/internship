############ EXPERIMENT SHAPLEY VALUES ########################
#                                                             #
# Here, the mutual information experiment will be performed   #
#                                                             #
###############################################################
import preprocessing
import eval
import shap_analysis
from pathlib import Path
from local_paths import PULSE_DB_SUP_DIR, PERFORMANCE_RESULTS_PAPER, GS_RESULT_PAPER
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
    #train_original_path = PULSE_DB_SUP_DIR / "Features_VitalDB_Train_Subset.h5"
    test_original_path = PULSE_DB_SUP_DIR / "Features_VitalDB_CalFree_Test_Subset.h5"
    
    train_clean_path = PULSE_DB_SUP_DIR / "Clean_Features_VitalDB_Train_Subset.h5"
    """
    test_clean_path = PULSE_DB_SUP_DIR / "Clean_Features_VitalDB_CalFree_Test_Subset.h5"
    """
    # =========================================================
    # Load Data
    #==========================================================
    feature_names = ["IPR", "Tsp", "TWRRF25", "TWRRF50", "Tsw25", 
                 "Tsw50", "Tsw75", "Tdw25", "Tdw50", "Tdw75", 
                 "AUCpi", "IPA",  "Av-Au_ratio", "Ab-Aa_ratio", "Ac-Aa_ratio", 
                 "Ad-Aa_ratio", "Ap2-Ap1_ratio", "AGI", "Kurtosis", "Skewness", 
                 "L-H_ratio", "ShannonEntropy", "Tpp", "PRV", "FullKurt", 
                 "FullSkew", "sdPRV", "IQR_PRV"]
    df_train = load_PulseDB_sup_ds(train_clean_path, feature_names=feature_names)
    df_test = load_PulseDB_sup_ds(test_original_path, feature_names=feature_names)
    
    print(len(df_train["Subject"].unique()))
    # =========================================================
    # Check NaNs and fill them
    #==========================================================
    
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
    
    print(df_test.isna().any().any())
    """

    # =========================================================
    # Initialize model
    #==========================================================
    print(df_train.info())
    """
    df_train = preprocessing.downsample_per_patient(df_train, patient_col="Subject", proportion = 0.1)
    print(df_train.info())
    """
    

    # drop ID columns before training
    id_cols = ["Age", "Gender", "Height", "Weight", "BMI", "SF"]
    targets = ["SBP", "DBP", "MAP"]
    id_cols.extend(targets)
    groups = df_train["Subject"]
    X_train = df_train.drop(columns=id_cols)
    X_test = df_test.drop(columns=id_cols)

    Y_train = df_train[targets]
    Y_test = df_test[targets]

    # Splittin sample wise
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state= 42, shuffle=True)#, stratify=X_train["Subject"]) # Cant stratify in the cleaned as dataset is quite imbalanced
    
    counts = X_val["Subject"].value_counts()
    print(counts)
    

    X_train = X_train.drop(columns=["Subject"])
    X_val = X_val.drop(columns=["Subject"])
    X_test = X_test.drop(columns=["Subject"])

    # build the model
    grid_path = GS_RESULT_PAPER / "Full_Grid_Randomized_search_3targets.json"
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
    for target in targets:
        Y_tr = Y_train[target]
        Y_v = Y_val[target]
        Y_t = Y_test[target]

        # Train
        pipeline.fit(X_train, Y_tr)                               

        # Pred
        Y_val_pred = pipeline.predict(X_val)
        Y_pred = pipeline.predict(X_test)
        # Eval
        Bland_Altman_Path = PERFORMANCE_RESULTS_PAPER / r"Ideal_Case/BA"
        R2_path = PERFORMANCE_RESULTS_PAPER / r"Ideal_Case/R2"
        res_path = PERFORMANCE_RESULTS_PAPER / r"Ideal_Case/Performance"
        metrics_val_original = eval.evaluate(Y_v, Y_val_pred, 
                                             BA_path=Bland_Altman_Path / f"val_clean_{target}.png",
                                             R2_path= R2_path / f"val_clean_{target}.png",
                                             save_results=res_path/ f"val_clean_{target}.csv")
        metrics_test_original = eval.evaluate(Y_t, Y_pred, 
                                              BA_path=Bland_Altman_Path / f"test_clean_{target}.png",
                                              R2_path= R2_path / f"test_original_{target}.png",
                                              save_results=res_path/ f"test_clean_{target}.csv")

        print(f"=== Results for {target}, in VAL ===")
        print(metrics_val_original)
        
        print(f"=== Results for {target}, in TEST ===")
        print(metrics_test_original)
        
        