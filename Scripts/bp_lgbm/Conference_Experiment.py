############ EXPERIMENT CONFERENCE ############################
#                                                             #
# Here, the experiments for the EMBC wll be conducted. It is  #
#   single training + evaluation + SHAP claculation (1 iter). #
#                                                             #
###############################################################

import preprocessing
import eval
import shap_analysis
from pathlib import Path
from local_paths import PULSE_DB_SUP_DIR, EMBC_BA_RESULTS, EMBC_SHAP_RESULTS, EMBC_GS_CONFIGS
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
    """
    df_train = preprocessing.downsample_per_patient(df_train, patient_col="Subject", proportion = 0.2)
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
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state= 42, shuffle=True, stratify=X_train["Subject"])
    
    """
    counts = X_val["Subject"].value_counts()
    print(counts)
    """

    X_train = X_train.drop(columns=["Subject"])
    X_val = X_val.drop(columns=["Subject"])
    X_test = X_test.drop(columns=["Subject"])

    # build the model
    grid_path = EMBC_GS_CONFIGS / "SAMPLE_Full_Grid_Randomized_search_3targets.json"
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
    
        #~~~~~~ Shap Calculation ~~~~~~
        Shap_path = EMBC_SHAP_RESULTS / "Original_DS_Sample_wise_GS"
        shap_values, _, features = shap_analysis.compute_shap_values(model=pipeline["model"], X=X_train, feature_names=X_train.columns.to_list())
        shap_analysis.plot_shap_beeswarm_summary(shap_values=shap_values,
                                                 X = X_train,
                                                 feature_names= features,
                                                 target_name= target,
                                                 save_path=Shap_path / f"SHAP_original_{target}.png",
                                                 show=False,
                                                 max_display=15
                                                )
                                                

        # Pred
        Y_val_pred = pipeline.predict(X_val)
        Y_pred = pipeline.predict(X_test)
        # Eval
        Bland_Altman_Path = EMBC_BA_RESULTS / "Garbage"
        metrics_val_original = eval.evaluate(Y_v, Y_val_pred, Bland_Altman_Path / f"val_original_{target}.png")
        metrics_test_original = eval.evaluate(Y_t, Y_pred, Bland_Altman_Path / f"test_original_{target}.png")

        print(f"=== Results for {target}, in VAL ===")
        print(metrics_val_original)
        
        print(f"=== Results for {target}, in TEST ===")
        print(metrics_test_original)
        
        