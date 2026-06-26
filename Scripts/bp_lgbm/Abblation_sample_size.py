############ EXPERIMENT ABBLATION STUDIES #####################
#                                                             #
# Here, sample size ablation study will be performed          #
#                                                             #
###############################################################
import preprocessing
import eval
from pathlib import Path
from local_paths import PULSE_DB_SUP_DIR, ABBLATION_RESULTS_PAPER, GS_RESULT_PAPER
from data import load_PulseDB_sup_ds
from config import load_config, ExperimentConfig, stress_test_params
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from models import build_lgbm
import pandas as pd

def run_single_ablation(df_train, df_test, proportion, targets, lgbm, base_path):
    # Downsample
    df_train_sub = preprocessing.downsample_per_patient(df_train, patient_col="Subject", proportion=proportion)
    
    # Prepare data
    id_cols = ["Age","Gender","Height","Weight","BMI","SF"]
    id_cols.extend(targets)
    groups = df_train_sub["Subject"]

    X_train = df_train_sub.drop(columns=id_cols)
    X_test = df_test.drop(columns=id_cols)
    Y_train = df_train_sub[targets]
    Y_test = df_test[targets]

    # Split (sample-wise)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, test_size=0.1, random_state=42, shuffle=True, stratify=X_train["Subject"]
    )

    # Diagnostic subset
    """
    _, X_sub_train, _, Y_sub_train = train_test_split(
        X_train, Y_train, test_size=0.1, random_state=42, shuffle=True, stratify=X_train["Subject"]
    )
    """
    # Drop Subject col
    for df_ in [X_train, X_val, X_test,]:# X_sub_train]:
        df_.drop(columns=["Subject"], inplace=True)

    # Run for each target
    all_metrics = []
    for target in targets:
        Y_tr, Y_v, Y_t = Y_train[target], Y_val[target], Y_test[target]
        #Y_tr_sub = Y_sub_train[target]

        # ---- Integrate model + pipeline
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", lgbm)
        ])
        pipeline.fit(X_train, Y_tr)

        Y_val_pred = pipeline.predict(X_val)
        Y_pred = pipeline.predict(X_test)
        Y_tr_pred = pipeline.predict(X_train)
        #Y_tr_sub_pred = pipeline.predict(X_sub_train)

        Path_res = base_path / f"Sample_Size_{proportion:.2f}"
        Path_res.mkdir(parents=True, exist_ok=True)

        metrics_train = eval.evaluate(Y_tr, Y_tr_pred,
                                      BA_path=Path_res / f"BA_train_subset_{target}.png",
                                      R2_path=Path_res / f"R2_train_subset_{target}.png")
        metrics_val = eval.evaluate(Y_v, Y_val_pred,
                                    BA_path=Path_res / f"BA_val_{target}.png",
                                    R2_path=Path_res / f"R2_val_{target}.png")
        metrics_test = eval.evaluate(Y_t, Y_pred,
                                     BA_path=Path_res / f"BA_test_original_{target}.png",
                                     R2_path=Path_res / f"R2_test_original_{target}.png")

        # Append to global results
        for subset_name, m in zip(["Train", "Val", "Test"], [metrics_train, metrics_val, metrics_test]):
            m["target"] = target
            m["proportion"] = proportion
            m["subset"] = subset_name
            all_metrics.append(m)
    
    return pd.DataFrame(all_metrics)


if __name__ == "__main__":
    # Load data once
    feature_names = ["IPR","Tsp","TWRRF25","TWRRF50","Tsw25","Tsw50","Tsw75","Tdw25","Tdw50","Tdw75",
                     "AUCpi","IPA","Av-Au_ratio","Ab-Aa_ratio","Ac-Aa_ratio","Ad-Aa_ratio","Ap2-Ap1_ratio",
                     "AGI","Kurtosis","Skewness","L-H_ratio","ShannonEntropy","Tpp","PRV","FullKurt",
                     "FullSkew","sdPRV","IQR_PRV"]

    df_train = load_PulseDB_sup_ds(PULSE_DB_SUP_DIR / "Features_VitalDB_Train_Subset.h5", feature_names=feature_names)
    df_test = load_PulseDB_sup_ds(PULSE_DB_SUP_DIR / "Features_VitalDB_CalFree_Test_Subset.h5", feature_names=feature_names)

    df_train = preprocessing.median_impute_patientwise(df_train, "Subject")
    df_test = preprocessing.median_impute_patientwise(df_test, "Subject")
    print(df_train.info())

    proportions = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    targets = ["SBP","DBP","MAP"]
    """
    grid_path = GS_RESULT_PAPER / "Full_Grid_Randomized_search_3targets.json"
    cfg = load_config(grid_path)
    """
    cfg = ExperimentConfig(n_splits=5,
                           random_state=42, 
                           experiment_name="Overperformance",
                           verbose = -1,
                           model_params=stress_test_params)
    

    base_path = ABBLATION_RESULTS_PAPER / "Sample_Size_Stress_Test"
    base_path.mkdir(parents=True, exist_ok=True)

    all_results = []
    for p in proportions:
        lgbm = build_lgbm(cfg)
        df_res = run_single_ablation(df_train, df_test, p, targets, lgbm, base_path)
        all_results.append(df_res)

    # Aggregate and save unified CSV
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(base_path / "Aggregated_Results.csv", index=False)
    print("✅ Ablation study complete. Results saved to:", base_path / "Aggregated_Results.csv")
    
