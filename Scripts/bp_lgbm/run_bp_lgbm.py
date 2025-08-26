############ MAIN #############################################
#                                                             #
# Here, all of the functions are going to be called           #
#                                                             #
###############################################################

import local_paths
from data import load_patient_dataset, load_group_attributes
import preprocessing

if __name__ == "__main__":
    data_path = local_paths.DATA_DIR /"features_cleaned.h5"
    labels_path = local_paths.LABELS_DIR / "BP_values.h5"
    bp_values_file = ("BP_values.h5")

    df_features_mean = load_patient_dataset(data_path, dataset_type="mean")
    labels_df = load_patient_dataset(labels_path, column_names=["SBP", "DBP", "MAP"])
    metadata_df = load_group_attributes(data_path)
    print(df_features_mean.shape)      
    #print(df_mean.head()) # show first rows with patient column
    print(df_features_mean.info())

    print(metadata_df.head())

    split_results = preprocessing.split_patients_by_signal_share(meta_df= metadata_df,
                                                 threshold= 0.8,
                                                 margin = 0.02,
                                                 patient_col="Patient",
                                                 signals_col= "Total_signals",
                                                 prefer="desc"
                                                 )
    print(split_results)

