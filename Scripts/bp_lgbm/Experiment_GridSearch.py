############ EXPERIMENT GRID SEARCH ###########################
#                                                             #
# This script is for executing the grid search only.          #
#                                                             #
###############################################################
"""
EXPERIMENT DESCRIPTION:

- Grid Search using Hyperion.
- Original dataset (without cleaning)
- Settings: 5 folds CV - group patient wise splitting, full grid search, optimization for the three targets (SBP, DBP, MAP).
- Preprocessing details: Inputation strategy is meadian based on patient entries! Scaler is fitted only on training set.
- Imputation for labels:
•	It appears that MAP is full, so I am going to fill those NaNs with the formula for calculating MAP based on DBP and SBP.
•	If by any chance, both SBP and DBP are missing, I am going to impute one of them based on the mean of the adjacent 5 values 
    for SBP and use the method above to calculate the DBP.

"""
import preprocessing
import gs
from pathlib import Path
from data import load_patient_dataset, load_group_attributes
from config import ExperimentConfig, lightGBM_default_params, save_config, lightGBM_baseline_grid_3target
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from models import build_lgbm


if __name__ == "__main__":

    # =========================================================
    # Paths
    #==========================================================
    folder_path = Path("data_features")
    labels_path = folder_path/ "BP_values.h5"
    data_messy_path = folder_path /"features_original.h5"

    # =========================================================
    # Load Data
    #==========================================================
    df_feat_orig_mean = load_patient_dataset(data_messy_path, dataset_type="mean")
    labels_original_df = load_patient_dataset(labels_path, column_names=["SBP", "DBP", "MAP"])
    metadata_original_df = load_group_attributes(data_messy_path)

    # =========================================================
    # Solucionar machetazos
    #==========================================================
    df_feat_orig_mean = preprocessing.drop_abp_features(df_feat_orig_mean, patient_col="Patient")
    metadata_original_df["Total_signals"] = metadata_original_df["Total_signals"]//2

    # =========================================================
    # Check NaNs and fill them
    #==========================================================

    # Fill X NaNs
    X_df_original_feat = preprocessing.median_impute_patientwise(df_feat_orig_mean, patient_col= "Patient")
    print("Checking NaN after inputation in X:", X_df_original_feat.isna().any().any())

    # Fill Y NaNs
    Y_df_original_feat = preprocessing.fill_missing_bp(labels_original_df)
    print(Y_df_original_feat.info())

    # =========================================================
    # X and Y dfs merging
    #==========================================================
    XY_df = preprocessing.merge_XY(X_df_original_feat, Y_df_original_feat) # When possible, add the parmeter "on: Segements_ID"

    # =========================================================
    # Splitting process of the dataset IDs - patyient wise
    #==========================================================
    split_results = preprocessing.split_patients_by_signal_share(meta_df= metadata_original_df,
                                                 threshold= 0.8,
                                                 margin = 0.02,
                                                 patient_col="Patient",
                                                 signals_col= "Total_signals",
                                                 prefer="asc"
                                                 )
    print("Training subjects:", len(split_results["selected_ids"])) # OJOOO the logic here states that for me the "selected" set is for training
    print("Testing subjects:", len(split_results["remaining_ids"]))

    df_train, df_test = preprocessing.split_train_test(XY=XY_df, split_results=split_results, patient_col="Patient")

    # =========================================================
    # Splitting into X and Y
    #==========================================================
    X_df_train, Y_df_train = preprocessing.split_XY(df_train,
                                                    target_cols= ["SBP", "DBP", "MAP"],
                                                    id_cols= ["Patient"]
                                                    )

    X_df_test, Y_df_test = preprocessing.split_XY(df_test,
                                                    target_cols= ["SBP", "DBP", "MAP"],
                                                    id_cols= ["Patient"]
                                                    )
    
    # =========================================================
    # Initialize model
    #==========================================================
    # drop ID columns before training
    id_cols = ["Patient"]
    targets = ["SBP", "DBP", "MAP"]
    groups = X_df_train["Patient"]
    X_train = X_df_train.drop(columns=id_cols)
    Y_train = Y_df_train[targets]   # or whatever target you want

    # build the model
    cfg = ExperimentConfig(n_splits=5,
                           random_state=42, 
                           experiment_name="Grid_Search_3Targets",
                           verbose = -1,
                           model_params=lightGBM_default_params)
    lgbm = build_lgbm(cfg)
    model = MultiOutputRegressor(lgbm)

    # build the pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    # =========================================================
    # Grid Search, standarization and CV happens inside.
    #==========================================================
    print("**********************************************************************it is running until here")
    search, results_df = gs.run_grid_search(pipeline=pipeline,
                                            X=X_train,
                                            y=Y_train,
                                            groups=groups,
                                            param_grid=lightGBM_baseline_grid_3target,
                                            n_splits=cfg.n_splits,
                                            cv_type="group",
                                            save_results=True,
                                            verbose=0,
                                            search_mode="grid",
                                            n_iter=100
                                            )
    
    print("Best parameters:", search.best_params_)
    print("Best CV score (MSE):", -search.best_score_)
    print("Best CV score (RMSE):", (-search.best_score_)**0.5)

    best_params_clean = {k.replace("model__", ""): v 
                     for k, v in search.best_params_.items()}
    print(best_params_clean)

    # Update config with best hyperparameters
    cfg.model_params.update(best_params_clean)
    save_config(cfg, Path(f"configs/{cfg.experiment_name}.json"))