############ BOOTSTRAP CI — AGE-STRATIFIED ############################
#                                                                      #
# Trains a separate PPG-only LightGBM per age stratum — each model is  #
# fit only on the age-filtered training subset — then runs bootstrap  #
# CI evaluation on the matching age stratum of the test set:           #
#   age_lt60  : Age < 60   (mirrors demo_strata_utils label "Age<60") #
#   age_gte60 : Age >= 60  (mirrors demo_strata_utils label "Age>=60") #
#                                                                      #
# Segmentation uses the exact boolean operations from                  #
# demo_strata_utils.segment_thresholds (single numeric threshold):     #
#   below  : df["Age"] < threshold                                     #
#   above  : df["Age"] >= threshold                                    #
#                                                                      #
# Saves per stratum per target (under Bootstrap/age_stratified/):      #
#   Distribution_{target}.csv   — 1000 rows × metrics                 #
#   PointEstimates_{target}.csv — metrics on the full stratum          #
#   CI_{target}.csv             — 95% CI summary                      #
########################################################################

import pandas as pd
from tqdm import tqdm

import preprocessing
import eval
from local_paths import PULSE_DB_SUP_DIR, PERFORMANCE_RESULTS_PAPER, GS_RESULT_PAPER
from data import load_PulseDB_sup_ds
from config import load_config
from models import build_lgbm
from sklearn.model_selection import train_test_split

N_RESAMPLES   = 1000
RANDOM_STATE  = 42
AGE_THRESHOLD = 60
TARGETS       = ["SBP", "DBP", "MAP"]

FEATURE_NAMES = [
    "IPR", "Tsp", "TWRRF25", "TWRRF50", "Tsw25",
    "Tsw50", "Tsw75", "Tdw25", "Tdw50", "Tdw75",
    "AUCpi", "IPA", "Av-Au_ratio", "Ab-Aa_ratio", "Ac-Aa_ratio",
    "Ad-Aa_ratio", "Ap2-Ap1_ratio", "AGI", "Kurtosis", "Skewness",
    "L-H_ratio", "ShannonEntropy", "Tpp", "PRV", "FullKurt",
    "FullSkew", "sdPRV", "IQR_PRV",
]

if __name__ == "__main__":

    # =========================================================
    # Paths
    # =========================================================
    train_path = PULSE_DB_SUP_DIR / "Features_VitalDB_Train_Subset.h5"
    test_path  = PULSE_DB_SUP_DIR / "Features_VitalDB_CalFree_Test_Subset.h5"
    res_root   = PERFORMANCE_RESULTS_PAPER / "Bootstrap" / "age_stratified"

    # =========================================================
    # Load
    # =========================================================
    df_train = load_PulseDB_sup_ds(train_path, feature_names=FEATURE_NAMES)
    df_test  = load_PulseDB_sup_ds(test_path,  feature_names=FEATURE_NAMES)

    # =========================================================
    # Impute
    # =========================================================
    df_train = preprocessing.median_impute_patientwise(df_train, patient_col="Subject")
    df_test  = preprocessing.median_impute_patientwise(df_test,  patient_col="Subject")

    grid_path = GS_RESULT_PAPER / "Full_Grid_Randomized_search_3targets.json"
    cfg = load_config(grid_path)

    # =========================================================
    # Age segmentation — applied to BOTH train and test
    # Exact boolean from demo_strata_utils.segment_thresholds
    # =========================================================
    strata = {
        "age_lt60":  (df_train[df_train["Age"] <  AGE_THRESHOLD].copy(),
                      df_test[df_test["Age"]   <  AGE_THRESHOLD].copy()),
        "age_gte60": (df_train[df_train["Age"] >= AGE_THRESHOLD].copy(),
                      df_test[df_test["Age"]   >= AGE_THRESHOLD].copy()),
    }

    for stratum_name, (df_train_s, df_stratum) in strata.items():
        print(f"\n--- Stratum: {stratum_name} ---")
        print(f"  Train: {df_train_s['Subject'].nunique()} subjects, {len(df_train_s)} signals")
        print(f"  Test:  {df_stratum['Subject'].nunique()} subjects, {len(df_stratum)} signals")

        res_dir = res_root / stratum_name
        res_dir.mkdir(parents=True, exist_ok=True)

        # =========================================================
        # Train on stratum-specific training subset
        # =========================================================
        X_train_full = df_train_s.drop(columns=["SF"] + TARGETS)
        Y_train_full = df_train_s[TARGETS]

        X_train, _, Y_train, _ = train_test_split(
            X_train_full, Y_train_full,
            test_size=0.1, random_state=RANDOM_STATE, shuffle=True,
            stratify=X_train_full["Subject"],
        )
        X_train = X_train.drop(columns=["Subject"])[FEATURE_NAMES]

        lgbm_models = {}
        for target in TARGETS:
            lgbm = build_lgbm(cfg)
            lgbm.fit(X_train, Y_train[target])
            lgbm_models[target] = lgbm
            print(f"  Trained {target}")

        # =========================================================
        # Point estimates on full stratum
        # =========================================================
        for target in TARGETS:
            y_pred = lgbm_models[target].predict(df_stratum[FEATURE_NAMES])
            pt = eval.compute_metrics(df_stratum[target].values, y_pred)
            pt_df = pd.DataFrame([{"metric": k, "value": v} for k, v in pt.items()])
            pt_df.to_csv(res_dir / f"PointEstimates_{target}.csv", index=False)

        # =========================================================
        # Bootstrap pre-allocation (per stratum, same seed)
        # =========================================================
        resamples     = eval.allocate_bootstrap_resamples(
            df_stratum["Subject"], n_resamples=N_RESAMPLES, random_state=RANDOM_STATE
        )
        subject_index = eval.build_subject_index(df_stratum, subject_col="Subject")

        # =========================================================
        # Bootstrap loop
        # =========================================================
        boot_metrics = {t: [] for t in TARGETS}

        for resample in tqdm(resamples, total=N_RESAMPLES, desc=stratum_name):
            sample = eval.build_bootstrap_sample(df_stratum, resample, subject_index)
            for target in TARGETS:
                y_true = sample[target].values
                y_pred = lgbm_models[target].predict(sample[FEATURE_NAMES])
                boot_metrics[target].append(eval.compute_metrics(y_true, y_pred))

        # =========================================================
        # Save distributions + CI
        # =========================================================
        for target in TARGETS:
            dist_df = pd.DataFrame(boot_metrics[target])
            dist_df.to_csv(res_dir / f"Distribution_{target}.csv", index=False)

            pt_df       = pd.read_csv(res_dir / f"PointEstimates_{target}.csv")
            point_metrics = dict(zip(pt_df["metric"], pt_df["value"]))

            ci_df = eval.compute_bootstrap_ci(dist_df, alpha=0.05)
            eval.save_bootstrap_ci(
                ci_df,
                path=res_dir / f"CI_{target}.csv",
                point_metrics=point_metrics,
            )

    print("\nDone.")
