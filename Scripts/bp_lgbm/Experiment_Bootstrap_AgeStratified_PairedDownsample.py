############ BOOTSTRAP CI — AGE-STRATIFIED, SUBJECT-COUNT-MATCHED, PAIRED ####
#                                                                             #
# Branch of Experiment_Bootstrap_AgeStratified.py. The original Age>=60 vs   #
# Full contrast bootstraps two UNEQUAL-sized, INDEPENDENT test pools (144    #
# full-test subjects vs 76 Age>=60 subjects) — see                           #
# Experiment_MDE_Equivalence_AgeStratified.py, which has to fall back to a   #
# Wald/quadrature approximation because of this. This script instead:        #
#                                                                             #
#   1. Downsamples the full test set to N_TARGET subjects (= however many    #
#      Age>=60 test subjects exist, computed at runtime) using               #
#      preprocessing.select_subjects_by_target_distribution to preserve the  #
#      MAP distribution — the same technique Abblation_subject_size.py uses  #
#      for training-subject downsampling, applied here to the TEST set.      #
#   2. Trains two fresh PPG-only models: "full_downsampled" (trained on ALL  #
#      ages, evaluated on the N_TARGET downsampled test subjects) and        #
#      "age_gte60" (trained + evaluated on the Age>=60 stratum, unchanged in #
#      size from the original experiment).                                  #
#   3. Bootstraps BOTH arms using one shared array of resample POSITIONS     #
#      (not subject IDs) applied independently to each arm's own subject     #
#      list, so row i of both arms' Distribution_{target}.csv comes from the #
#      same underlying random draw -> genuinely paired, unlike the original  #
#      independent-bootstrap age contrast.                                  #
#                                                                             #
# Does NOT read or write anything under Bootstrap/ppg or                    #
# Bootstrap/age_stratified/ — fully self-contained, new output root:         #
#   Bootstrap_AgeStratified_PairedDownsample/{full_downsampled,age_gte60}/   #
#     Distribution_{target}.csv, PointEstimates_{target}.csv, CI_{target}.csv#
#   Bootstrap_AgeStratified_PairedDownsample/Downsampled_Test_Subjects.csv   #
###############################################################################

import numpy as np
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


def allocate_paired_positions(n_subjects, n_resamples=N_RESAMPLES, random_state=RANDOM_STATE):
    """
    Pre-allocate a shared (n_resamples, n_subjects) array of integer positions
    in [0, n_subjects). Applying the SAME position array to two different
    fixed-order subject-ID arrays of the same length yields two independent
    valid with-replacement bootstrap draws whose rows are paired by
    construction (row i of both draws comes from the same underlying random
    draw), without depending on which subjects are in either pool.
    """
    rng = np.random.default_rng(random_state)
    return rng.integers(0, n_subjects, size=(n_resamples, n_subjects))


def train_ppg_models(df_train_stratum, cfg, random_state=RANDOM_STATE):
    """Train one PPG-only LightGBM per target on a (possibly age-filtered)
    training stratum. Mirrors the per-stratum training block in
    Experiment_Bootstrap_AgeStratified.py exactly."""
    X_train_full = df_train_stratum.drop(columns=["SF"] + TARGETS)
    Y_train_full = df_train_stratum[TARGETS]

    X_train, _, Y_train, _ = train_test_split(
        X_train_full, Y_train_full,
        test_size=0.1, random_state=random_state, shuffle=True,
        stratify=X_train_full["Subject"],
    )
    X_train = X_train.drop(columns=["Subject"])[FEATURE_NAMES]

    models = {}
    for target in TARGETS:
        lgbm = build_lgbm(cfg)
        lgbm.fit(X_train, Y_train[target])
        models[target] = lgbm
    return models


if __name__ == "__main__":

    # =========================================================
    # Paths
    # =========================================================
    train_path = PULSE_DB_SUP_DIR / "Features_VitalDB_Train_Subset.h5"
    test_path  = PULSE_DB_SUP_DIR / "Features_VitalDB_CalFree_Test_Subset.h5"
    res_root   = PERFORMANCE_RESULTS_PAPER / "Bootstrap_AgeStratified_PairedDownsample"
    res_root.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # Load + impute
    # =========================================================
    df_train = load_PulseDB_sup_ds(train_path, feature_names=FEATURE_NAMES)
    df_test  = load_PulseDB_sup_ds(test_path,  feature_names=FEATURE_NAMES)

    df_train = preprocessing.median_impute_patientwise(df_train, patient_col="Subject")
    df_test  = preprocessing.median_impute_patientwise(df_test,  patient_col="Subject")

    grid_path = GS_RESULT_PAPER / "Full_Grid_Randomized_search_3targets.json"
    cfg = load_config(grid_path)

    # =========================================================
    # Age>=60 test stratum (unchanged size) + dynamic target N
    # =========================================================
    df_test_gte60 = df_test[df_test["Age"] >= AGE_THRESHOLD].copy()
    N_TARGET = df_test_gte60["Subject"].nunique()
    print(f"Age>={AGE_THRESHOLD} test stratum: {N_TARGET} subjects, {len(df_test_gte60)} signals")

    # =========================================================
    # Downsample full test set to N_TARGET subjects, preserving MAP dist.
    # Same call signature as Abblation_subject_size.py's subject selection.
    # =========================================================
    chosen_subjects = preprocessing.select_subjects_by_target_distribution(
        df_test,
        subject_col="Subject",
        target_col="MAP",
        n_subjects=N_TARGET,
        n_bins=12,
        random_state=RANDOM_STATE,
    )
    df_test_full_ds = df_test[df_test["Subject"].isin(chosen_subjects)].copy()

    n_full_ds = df_test_full_ds["Subject"].nunique()
    print(f"Full test set downsampled: {n_full_ds} subjects, {len(df_test_full_ds)} signals")
    assert n_full_ds == N_TARGET, (
        f"Downsampled full test set has {n_full_ds} subjects, expected {N_TARGET}"
    )

    pd.DataFrame({"Subject": sorted(df_test_full_ds["Subject"].unique())}).to_csv(
        res_root / "Downsampled_Test_Subjects.csv", index=False
    )

    strata = {
        "full_downsampled": (df_train.copy(),      df_test_full_ds),
        "age_gte60":        (df_train[df_train["Age"] >= AGE_THRESHOLD].copy(), df_test_gte60),
    }

    # =========================================================
    # Train both arms
    # =========================================================
    arm_models        = {}
    arm_test          = {}
    arm_subjects_arr  = {}
    arm_subject_index = {}

    for arm_name, (df_train_s, df_test_s) in strata.items():
        print(f"\n--- Arm: {arm_name} ---")
        print(f"  Train: {df_train_s['Subject'].nunique()} subjects, {len(df_train_s)} signals")
        print(f"  Test:  {df_test_s['Subject'].nunique()} subjects, {len(df_test_s)} signals")

        res_dir = res_root / arm_name
        res_dir.mkdir(parents=True, exist_ok=True)

        arm_models[arm_name] = train_ppg_models(df_train_s, cfg, random_state=RANDOM_STATE)
        arm_test[arm_name]   = df_test_s

        for target in TARGETS:
            y_pred = arm_models[arm_name][target].predict(df_test_s[FEATURE_NAMES])
            pt = eval.compute_metrics(df_test_s[target].values, y_pred)
            pt_df = pd.DataFrame([{"metric": k, "value": v} for k, v in pt.items()])
            pt_df.to_csv(res_dir / f"PointEstimates_{target}.csv", index=False)

        arm_subjects_arr[arm_name]  = np.array(sorted(df_test_s["Subject"].unique()))
        arm_subject_index[arm_name] = eval.build_subject_index(df_test_s, subject_col="Subject")

        assert len(arm_subjects_arr[arm_name]) == N_TARGET, (
            f"Arm {arm_name}: {len(arm_subjects_arr[arm_name])} unique test subjects, "
            f"expected {N_TARGET}"
        )

    # =========================================================
    # Shared paired bootstrap positions (drives BOTH arms)
    # =========================================================
    paired_positions = allocate_paired_positions(N_TARGET, N_RESAMPLES, RANDOM_STATE)
    assert paired_positions.shape == (N_RESAMPLES, N_TARGET)

    # =========================================================
    # Bootstrap loop
    # =========================================================
    boot_metrics = {arm: {t: [] for t in TARGETS} for arm in strata}

    for positions in tqdm(paired_positions, total=N_RESAMPLES, desc="paired bootstrap"):
        for arm_name in strata:
            resample_subjects = arm_subjects_arr[arm_name][positions]
            sample = eval.build_bootstrap_sample(
                arm_test[arm_name], resample_subjects, arm_subject_index[arm_name]
            )
            for target in TARGETS:
                y_true = sample[target].values
                y_pred = arm_models[arm_name][target].predict(sample[FEATURE_NAMES])
                boot_metrics[arm_name][target].append(eval.compute_metrics(y_true, y_pred))

    # =========================================================
    # Save distributions + CI
    # =========================================================
    for arm_name in strata:
        res_dir = res_root / arm_name
        for target in TARGETS:
            dist_df = pd.DataFrame(boot_metrics[arm_name][target])
            assert len(dist_df) == N_RESAMPLES
            dist_df.to_csv(res_dir / f"Distribution_{target}.csv", index=False)

            pt_df         = pd.read_csv(res_dir / f"PointEstimates_{target}.csv")
            point_metrics = dict(zip(pt_df["metric"], pt_df["value"]))

            ci_df = eval.compute_bootstrap_ci(dist_df, alpha=0.05)
            eval.save_bootstrap_ci(
                ci_df,
                path=res_dir / f"CI_{target}.csv",
                point_metrics=point_metrics,
            )

    print(f"\nDone. N_TARGET={N_TARGET}. Results under: {res_root}")
