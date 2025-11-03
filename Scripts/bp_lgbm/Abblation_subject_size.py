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

def run_subject_ablation(df_train, df_test, n_subjects, targets, lgbm, base_path):
    # --------------------------------------------------------
    # Select subjects preserving MAP distribution
    # --------------------------------------------------------
    chosen_subjects = preprocessing.select_subjects_by_target_distribution(
        df_train,
        subject_col='Subject',
        target_col='MAP',  # preserve MAP distribution
        n_subjects=n_subjects,
        n_bins=12,
        random_state=42
    )

    df_train_sub = df_train[df_train['Subject'].isin(chosen_subjects)].copy()

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------
    id_cols = ["Age","Gender","Height","Weight","BMI","SF"]
    id_cols.extend(targets)
    groups = df_train_sub["Subject"]

    X_train = df_train_sub.drop(columns=id_cols)
    X_test = df_test.drop(columns=id_cols)
    Y_train = df_train_sub[targets]
    Y_test = df_test[targets]

    # Stratified split (same as your previous logic)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train,
        test_size=0.1,
        random_state=42,
        shuffle=True,
        stratify=X_train["Subject"]
    )

    # Diagnostic subset (same as before)
    _, X_sub_train, _, Y_sub_train = train_test_split(
        X_train, Y_train,
        test_size=0.1,
        random_state=42,
        shuffle=True,
        stratify=X_train["Subject"]
    )

    # Drop subject before model input
    for df_ in [X_train, X_val, X_test, X_sub_train]:
        df_.drop(columns=["Subject"], inplace=True, errors='ignore')

    # --------------------------------------------------------
    # Run per target
    # --------------------------------------------------------
    all_metrics = []
    for target in targets:
        Y_tr = Y_train[target]
        Y_v = Y_val[target]
        Y_t = Y_test[target]
        Y_tr_sub = Y_sub_train[target]

    # ---- Integrate model + pipeline
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", lgbm)
        ])
        # Train model
        pipeline.fit(X_train, Y_tr)

        # Predict
        Y_val_pred = pipeline.predict(X_val)
        Y_pred = pipeline.predict(X_test)
        Y_tr_sub_pred = pipeline.predict(X_sub_train)

        # Evaluate
        Path_res = base_path / f"Subject_Size_{n_subjects}"
        Path_res.mkdir(parents=True, exist_ok=True)

        metrics_train = eval.evaluate(
            Y_tr_sub, Y_tr_sub_pred,
            BA_path=Path_res / f"BA_train_subset_{target}.png",
            R2_path=Path_res / f"R2_train_subset_{target}.png"
        )
        metrics_val = eval.evaluate(
            Y_v, Y_val_pred,
            BA_path=Path_res / f"BA_val_{target}.png",
            R2_path=Path_res / f"R2_val_{target}.png"
        )
        metrics_test = eval.evaluate(
            Y_t, Y_pred,
            BA_path=Path_res / f"BA_test_{target}.png",
            R2_path=Path_res / f"R2_test_{target}.png"
        )

        # Collect metrics
        for subset_name, m in zip(
            ["Train", "Val", "Test"],
            [metrics_train, metrics_val, metrics_test]
        ):
            m["target"] = target
            m["n_subjects"] = n_subjects
            m["subset"] = subset_name
            all_metrics.append(m)

    return pd.DataFrame(all_metrics)


if __name__ == "__main__":
    # --------------------------------------------------------
    # Load and preprocess data
    # --------------------------------------------------------
    feature_names = [
        "IPR","Tsp","TWRRF25","TWRRF50","Tsw25","Tsw50","Tsw75",
        "Tdw25","Tdw50","Tdw75","AUCpi","IPA","Av-Au_ratio",
        "Ab-Aa_ratio","Ac-Aa_ratio","Ad-Aa_ratio","Ap2-Ap1_ratio",
        "AGI","Kurtosis","Skewness","L-H_ratio","ShannonEntropy",
        "Tpp","PRV","FullKurt","FullSkew","sdPRV","IQR_PRV"
    ]

    df_train = load_PulseDB_sup_ds(PULSE_DB_SUP_DIR / "Features_VitalDB_Train_Subset.h5", feature_names=feature_names)
    df_test = load_PulseDB_sup_ds(PULSE_DB_SUP_DIR / "Features_VitalDB_CalFree_Test_Subset.h5", feature_names=feature_names)

    df_train = preprocessing.median_impute_patientwise(df_train, "Subject")
    df_test = preprocessing.median_impute_patientwise(df_test, "Subject")

    # --------------------------------------------------------
    # Experiment setup
    # --------------------------------------------------------
    targets = ["SBP","DBP","MAP"]
    subject_counts = [25, 50, 100, 200, 400, 800, 1293]  # adjust to your dataset
    """
    grid_path = GS_RESULT_PAPER / "Full_Grid_Randomized_search_3targets.json"
    cfg = load_config(grid_path)
    """
    cfg = ExperimentConfig(n_splits=5,
                           random_state=42, 
                           experiment_name="Overperformance",
                           verbose = -1,
                           model_params=stress_test_params)

    base_path = ABBLATION_RESULTS_PAPER / "Subject_Size_Stress_Test"
    base_path.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Run ablation
    # --------------------------------------------------------
    all_results = []
    for n in subject_counts:
        lgbm = build_lgbm(cfg)
        df_res = run_subject_ablation(df_train, df_test, n, targets, lgbm, base_path)
        all_results.append(df_res)

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(base_path / "Aggregated_Subject_Size_Results.csv", index=False)
    print("✅ Subject-size ablation complete. Results saved to:", base_path / "Aggregated_Subject_Size_Results.csv")
