import numpy as np
import local_paths
import matplotlib.pyplot as plt
import preprocessing
from data import load_patient_dataset, load_group_attributes

if __name__ == "__main__":

    # =========================================================
    # Paths
    #==========================================================
    data_path_cleaned = local_paths.DATA_DIR /"features_patients_clean.h5"
    labels_path = local_paths.LABELS_DIR / "BP_values.h5"
    data_messy_path = local_paths.DATA_DIR /"features_patients.h5"

    # =========================================================
    # Load Data
    #==========================================================
    """
    inspect_file(data_messy_path, show_attrs=True)
    inspect_file(labels_path, show_attrs=True)
    """

    df_feat_orig_mean = load_patient_dataset(data_messy_path, dataset_type="mean")
    labels_original_df = load_patient_dataset(file_path=labels_path, column_names=["SBP", "DBP", "MAP", "segment_ID"])
    metadata_original_df = load_group_attributes(data_messy_path)
    df_feat_cleaned_mean = load_patient_dataset(data_path_cleaned, dataset_type="mean")
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
    X_df_original_feat = preprocessing.median_impute_patientwise(df_feat_orig_mean)
    print("Checking NaN after inputation in X:", X_df_original_feat.isna().any().any())

    # Fill Y NaNs
    Y_df_original_feat = preprocessing.fill_missing_bp(labels_original_df)

    plt.figure()