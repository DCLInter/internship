############ MAIN #############################################
#                                                             #
# Here, all of the functions are going to be called           #
#                                                             #
###############################################################
import numpy as np
import local_paths
import preprocessing
import cv
import gs
from data import load_patient_dataset, load_group_attributes
from models import build_lgbm
from config import ExperimentConfig, lightGBM_default_params, lightGBM_tiny_grid_3target, save_config
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from pathlib import Path


if __name__ == "__main__":

    # =========================================================
    # Paths
    #==========================================================
    data_path = local_paths.DATA_DIR /"features_cleaned.h5"
    labels_path = local_paths.LABELS_DIR / "BP_values.h5"
    data_messy_path = local_paths.DATA_DIR /"features_original.h5"

    # =========================================================
    # Load Data
    #==========================================================
    df_features_mean = load_patient_dataset(data_path, dataset_type="mean")
    df_feat_orig_mean = load_patient_dataset(data_messy_path, dataset_type="mean")
    labels_original_df = load_patient_dataset(labels_path, column_names=["SBP", "DBP", "MAP"])
    metadata_clean_df = load_group_attributes(data_path)
    metadata_original_df = load_group_attributes(data_messy_path)

    # =========================================================
    # Solucionar machetazos
    #==========================================================
    #X_df_original, Y_df_original = preprocessing.add_segment_id(X_df_original, labels_original_df, data_messy_path)
    df_feat_orig_mean = preprocessing.drop_abp_features(df_feat_orig_mean, patient_col="Patient")
    metadata_original_df["Total_signals"] = metadata_original_df["Total_signals"]//2
    """
    print(df_feat_orig_mean.shape)      
    print(df_feat_orig_mean.head()) # show first rows with patient column
    print(df_feat_orig_mean.info())
    print("************************************************************")
    print(metadata_original_df.head())
    print(metadata_original_df.info())
    print("************************************************************")
    print(labels_original_df.info())
    """
    # =========================================================
    # Check NaNs and fill them
    #==========================================================
    """
    print("************************************************************")
    print("Check NaNs")
    print("************************************************************")
    
    # Check if any NaN at all
    print("Is there any NaN in: df_features_mean?")
    print(df_feat_orig_mean.isna().any().any())
    # Count total number of NaNs
    print(df_feat_orig_mean.isna().sum())
    """

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
    print(XY_df.head())
    print(XY_df.info())

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
    # Grid Search, standarization and CV happens inside.
    #==========================================================
    # drop ID columns before training
    id_cols = ["Patient"]
    targets = ["SBP", "DBP", "MAP"]
    groups = X_df_train["Patient"]
    X_train = X_df_train.drop(columns=id_cols)
    Y_train = Y_df_train[targets]   # or whatever target you want

    # build the model
    cfg = ExperimentConfig(n_splits=10,
                           random_state=42, 
                           experiment_name="Grid_Search_3Targets",
                           model_params=lightGBM_default_params)
    lgbm = build_lgbm(cfg)
    model = MultiOutputRegressor(lgbm)

    # build the pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])
    
    search, results_df = gs.run_grid_search(pipeline=pipeline,
                                            X=X_train,
                                            y=Y_train,
                                            groups=groups,
                                            param_grid=lightGBM_tiny_grid_3target,
                                            n_splits=cfg.n_splits,
                                            cv_type="group",
                                            save_results=True,
                                            verbose=2
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
    # =========================================================
    # Eval. - with Testing set
    #==========================================================
    