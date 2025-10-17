############ EXPERIMENT SHAPLEY VALUES ########################
#                                                             #
# Here, the mutual information experiment will be performed   #
#                                                             #
###############################################################
import preprocessing
import eval
from pathlib import Path
from local_paths import PULSE_DB_SUP_DIR, PERFORMANCE_RESULTS_PAPER, GS_RESULT_PAPER
from data import load_PulseDB_sup_ds
from config import ExperimentConfig, load_config, lightGBM_default_params
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
    print(df_train.isna().any().any())
    # Count total number of NaNs
    print(df_train.isna().sum())
    """
    # Fill X NaNs
    df_train = preprocessing.median_impute_patientwise(df_train, patient_col= "Subject")
    df_test = preprocessing.median_impute_patientwise(df_test, patient_col= "Subject")
    """
    # =========================================================
    # Gender as categorical
    # =========================================================
    df_train["Gender"] = df_train["Gender"].astype("category")
    df_test["Gender"] = df_test["Gender"].astype("category")
    """
    # =========================================================
    # Downsampling of subjects and samples
    #==========================================================
    chosen_subjects = preprocessing.select_subjects_by_target_distribution(
        df_train,
        subject_col='Subject',
        target_col='SBP',   # or 'SBP'
        n_subjects=100,
        n_bins=10,
        random_state=42
    )

    # Filter the main dataset
    df_train_subset = df_train[df_train['Subject'].isin(chosen_subjects)].copy()

    # Downsample by subject
    df_train_subset = preprocessing.downsample_per_patient(df_train_subset, 
                                                           patient_col="Subject",
                                                           proportion=0.025,
                                                           random_state=42)

    print(f"Selected {len(chosen_subjects)} subjects, {len(df_train_subset)} total samples.")

    # ----- Same for testing --------
    chosen_subjects_test = preprocessing.select_subjects_by_target_distribution(
        df_test,
        subject_col='Subject',
        target_col='SBP',   # or 'SBP'
        n_subjects=85,
        n_bins=10,
        random_state=42
    )

    # Filter the main dataset
    df_test_subset = df_test[df_test['Subject'].isin(chosen_subjects_test)].copy()

    # Downsample by subject
    df_test_subset = preprocessing.downsample_per_patient(df_test_subset, 
                                                           patient_col="Subject",
                                                           proportion=0.01,
                                                           random_state=42)

    print(f"Selected {len(chosen_subjects_test)} subjects, {len(df_test_subset)} total samples.")

    # =========================================================
    # Initialize model
    #==========================================================
    
    # drop ID columns before training
    id_cols = ["Age","Gender", "Height", "Weight", "BMI", "SF"]
    targets = ["SBP", "DBP", "MAP"]
    id_cols.extend(targets)
    groups = df_train["Subject"]
    X_train = df_train.drop(columns=id_cols)
    X_test = df_test.drop(columns=id_cols)

    Y_train = df_train[targets]
    Y_test = df_test[targets]
    
    # Splittin sample wise
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state= 42, shuffle=True, stratify=X_train["Subject"])

    #-----------------------------------------
    # Just for diagnosing the model
    #-----------------------------------------
    _, X_sub_train, _, Y_sub_train = train_test_split(X_train, Y_train, test_size=0.1, random_state= 42, shuffle=True, stratify=X_train["Subject"])

    X_train = X_train.drop(columns=["Subject"])
    X_val = X_val.drop(columns=["Subject"])
    X_test = X_test.drop(columns=["Subject"])
    X_sub_train = X_sub_train.drop(columns=["Subject"])

    # build the model
    grid_path = GS_RESULT_PAPER / "Full_Grid_Randomized_search_3targets.json"
    cfg = load_config(grid_path)
    """
    cfg = ExperimentConfig(n_splits=5,
                           random_state=42, 
                           experiment_name="Overperformance",
                           verbose = -1,
                           model_params=lightGBM_default_params)
                           """
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
        Y_tr_sub = Y_sub_train[target]

        # Train
        pipeline.fit(X_train, Y_tr)                               

        # Pred
        Y_val_pred = pipeline.predict(X_val)
        Y_pred = pipeline.predict(X_test)
        Y_tr_sub_pred = pipeline.predict(X_sub_train)

        # Eval
        Path_res = PERFORMANCE_RESULTS_PAPER / r"Overperformance_Trial"
        metrics_train = eval.evaluate(Y_tr_sub, Y_tr_sub_pred, 
                                             BA_path=Path_res / f"BA_train_subset_{target}.png",
                                             R2_path= Path_res / f"R2_train_subset_{target}.png")
        metrics_val = eval.evaluate(Y_v, Y_val_pred, 
                                             BA_path=Path_res / f"BA_val_{target}.png",
                                             R2_path= Path_res / f"R2_val_{target}.png")
        metrics_test_original = eval.evaluate(Y_t, Y_pred, 
                                              BA_path=Path_res / f"BA_test_original_{target}.png",
                                              R2_path= Path_res / f"R2_test_original_{target}.png")
        # Create dict for saving results
        results = {
            "Train": metrics_train,
            "Val": metrics_val,
            "Test_original": metrics_test_original,
        }
        eval.save_results_dict(results, Path_res/ f"Results_{target}.csv", "Data_Subset")

        # Print results
        print(f"=== Results for {target}, in TRAIN ===")
        print(metrics_train, "\n")
        print(f"=== Results for {target}, in VAL ===")
        print(metrics_val, "\n")
        print(f"=== Results for {target}, in TEST ===")
        print(metrics_test_original, "\n")
        