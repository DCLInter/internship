############ MAIN #############################################
#                                                             #
# Here, all of the functions are going to be called           #
#                                                             #
###############################################################

import local_paths
from data import load_patient_dataset, load_group_attributes
import preprocessing
import lightgbm as lgb
import numpy as np

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
    # Standardize the dataset
    #==========================================================
    print("************************************************************")
    print("Standarization")
    print("************************************************************")
    #preprocessing.check_standarization(df=df_features_mean)
    X_df_train_scaled, scaler = preprocessing.standardize_z_score(X_df_train, exclude_cols=["Patient"])
    preprocessing.check_standarization(df=X_df_train_scaled)

    """
    # Note: This scaler is used for new data and for inverse scalling if needed.
    """

    # Scalling the testing data as well.
    numeric_cols = X_df_test.select_dtypes(include="number").columns
    X_df_test_scaled = X_df_test.copy()
    X_df_test_scaled[numeric_cols] = scaler.transform(X_df_test[numeric_cols])

    # =========================================================
    # Training the model
    #==========================================================
    # drop ID columns before training
    id_cols = ["Patient"]
    X_train = X_df_train_scaled.drop(columns=id_cols)
    Y_train = Y_df_train["MAP"]   # or whatever target you want

    # train LightGBM
    model = lgb.LGBMRegressor()
    model.fit(X_train, Y_train)

    # Make predictions on train set (or use validation/test if you have it)
    y_pred = model.predict(X_train)

    # Absolute errors
    abs_errors = np.abs(Y_train.values - y_pred)

    # Metrics
    mae = abs_errors.mean()
    mae_sd = abs_errors.std()

    print(f"MAE: {mae:.3f} ± {mae_sd:.3f}")