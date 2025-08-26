######## DATA LOADER ################################################################
#                                                                                   #
# This scrip is meant for loading the data from the feature extraction made by Juan #
#                                                                                   #
#####################################################################################

import local_paths
from pathlib import Path
import h5py
from h5_inspector import inspect_file
import numpy as np
import pandas as pd
from typing import List, Optional, Union

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FUNCTIONS FOR LOADING           ~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def load_patient_dataset(
    file_path: str | Path,
    dataset_type: str | None = None,  # optional now
    column_names: Optional[List[str]] = None     # custom names for dataset columns
) -> pd.DataFrame:
    """
    Load one dataset type across all patients into a single DataFrame.

    Args:
        file_path: path to the .h5 file
        dataset_type: one of {"mean", "median", "segments"} or None.
                      If None and each patient group has only one dataset,
                      that dataset is used automatically.
        column_names: optional list of names for dataset columns.
                      Length must match number of features in the dataset.
    

    Returns:
        pd.DataFrame with all rows concatenated, patient id included
    """
    file_path = Path(file_path)
    all_data = []

    with h5py.File(file_path, "r") as f:
        for patient_id in f.keys():
            group = f[patient_id]

            # --- dataset selection logic ---
            if dataset_type is None:
                keys = list(group.keys())
                if len(keys) != 1:
                    raise ValueError(
                        f"Patient {patient_id} has {len(keys)} datasets. "
                        f"Please specify dataset_type explicitly."
                    )
                ds_name = keys[0]
            elif dataset_type == "segments":
                ds_name = "segments"
            else:
                ds_name = f"{dataset_type}_{patient_id}"

            if ds_name not in group:
                print(f"⚠️ Skipping {patient_id}, dataset {ds_name} not found")
                continue

            ds = group[ds_name][()]
            arr = np.asarray(ds).T  # transpose: features x samples → samples x features

            # --- column names handling ---
            if column_names is not None:
                if len(column_names) != arr.shape[1]:
                    raise ValueError(
                        f"Length of column_names ({len(column_names)}) does not "
                        f"match number of features ({arr.shape[1]}) for patient {patient_id}"
                    )
                cols = column_names
            else:
                cols = list(range(arr.shape[1]))  # default numeric names

            # wrap into dataframe
            df = pd.DataFrame(arr, columns=cols)
            df.insert(0, "Patient", patient_id)  # patient as first col
            all_data.append(df)

    if not all_data:
        raise RuntimeError(
            f"No data loaded from {file_path} with dataset_type={dataset_type}."
        )

    return pd.concat(all_data, ignore_index=True)

def load_group_attributes(
        file_path: str | Path,
        dataset_for_count: str = "mean"   # e.g. "segments", "mean", "median"
        ) -> pd.DataFrame:
    """
    Collect attributes from each patient group in an HDF5 file and
    return them as a DataFrame (one row per patient).
    Attribute names become column names.
    Adds an extra column 'Total_signals' based on dataset_for_count.
    Adds extra column named BMI

    Args:
        file_path: Path to HDF5 file where each patient is a group
        dataset_for_count: which dataset to use to compute total signals

    Returns:
        pd.DataFrame with patient attributes + total_signals
    """
    file_path = Path(file_path)
    records = []

    with h5py.File(file_path, "r") as f:
        for patient_id in f.keys():
            group = f[patient_id]

            # --- collect attributes ---
            attrs = {k: _stringify(v) for k, v in group.attrs.items()}
            attrs["Patient"] = patient_id

            # --- count signals ---
            if dataset_for_count == "segments":
                ds_name = "segments"
            else:
                ds_name = f"{dataset_for_count}_{patient_id}"

            if ds_name in group:
                ds = group[ds_name]
                # ds.shape = (features, samples) → signals = samples
                total_signals = ds.shape[1]
            else:
                total_signals = np.nan

            attrs["Total_signals"] = total_signals
            records.append(attrs)

    if not records:
        raise RuntimeError(f"No group attributes found in {file_path}")

    df = pd.DataFrame(records)
    # Create the BMI column
    df['Height'] = df['Height']/100
    df['BMI'] = (df['Weight']/df['Height']**2).round(2)
    # Ensure patient column is first
    cols = ["Patient"] + [c for c in df.columns if c != "Patient"]
    return df[cols]

def _stringify(val):
    """Convert HDF5 attribute to a JSON/pandas-friendly value."""
    if isinstance(val, (bytes, bytearray)):
        return val.decode("utf-8", errors="replace")
    if isinstance(val, np.generic):  # numpy scalar
        return val.item()
    if isinstance(val, (list, tuple, np.ndarray)):
        arr = np.array(val)
        if arr.size == 1:            # unwrap single values
            return arr.item()
        return arr.tolist()
    return val


#====================================================================
# TESTING THE CODE
data_path = local_paths.DATA_DIR /"features_cleaned.h5"
labels_path = local_paths.LABELS_DIR / "BP_values.h5"
bp_values_file = ("BP_values.h5")

#inspect_file(data_path, show_attrs=True)
#inspect_file(bp_values_file, show_attrs=True)

df_features_mean = load_patient_dataset(data_path, dataset_type="mean")
labels_df = load_patient_dataset(labels_path, column_names=["SBP", "DBP", "MAP"])
metadata_df = load_group_attributes(data_path)
print(df_features_mean.shape)      
#print(df_mean.head()) # show first rows with patient column
print(df_features_mean.info())

print(metadata_df.head())

