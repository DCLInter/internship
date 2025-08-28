######## PREPROCESSING SCRIPT ################################################################
#                                                                                            #
# This script handles the splitting, standarization and things like that.                    #
#                                                                                            #
#                                                                                            #
##############################################################################################

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, List
import h5py
from typing import List, Tuple


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FUNCTIONS FOR SPLITTING         ~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def split_patients_by_signal_share(
    meta_df: pd.DataFrame,
    threshold: float = 0.80,             # target fraction (0–1)
    margin: float = 0.02,                # allowed tolerance
    patient_col: str = "Patient",        # your DF’s column name
    signals_col: str = "Total_signals",  # your DF’s column name
    prefer: str = "desc",                # "desc" = biggest patients first, "asc" = smallest first
) -> Dict[str, object]:
    """
    Greedy splitter: selects patients until cumulative share of signals
    falls within [threshold - margin, threshold + margin].

    If adding a patient would overshoot threshold + margin, that patient is skipped.
    """

    # total number of signals across all patients
    total_signals = int(meta_df[signals_col].sum())
    if total_signals == 0:
        return {
            "selected_ids": [],
            "remaining_ids": meta_df[patient_col].tolist(),
            "selected_share": 0.0,
            "note": "No signals in dataset."
        }

    # sort patients
    df = meta_df[[patient_col, signals_col]].copy()
    df = df.sort_values(signals_col, ascending=(prefer == "asc"))

    # target band
    low = (threshold - margin) * total_signals
    high = (threshold + margin) * total_signals

    selected: List[str] = []
    selected_signals = 0

    for _, row in df.iterrows():
        pid, sigs = row[patient_col], int(row[signals_col])
        tentative = selected_signals + sigs

        if tentative > high:  # overshoot → skip this patient
            continue

        selected.append(pid)
        selected_signals = tentative

        if low <= selected_signals <= high:  # stop once inside band
            break

    selected_share = selected_signals / total_signals
    remaining = [pid for pid in df[patient_col].tolist() if pid not in selected]

    return {
        "selected_ids": selected,
        "remaining_ids": remaining,
        "selected_share": selected_share,
        "selected_signals": selected_signals,
        "remaining_signals": total_signals - selected_signals,
        "total_signals": total_signals,
        "note": (
            "Within margin."
            if low <= selected_signals <= high
            else "Could not hit band exactly; stopped at closest under upper bound."
        ),
    }

def split_train_test(XY: pd.DataFrame, split_results: dict, patient_col: str = "Patient"):
    """
    Split XY dataframe into train/test sets based on split_results dict (from split_patients_by_signal_share).
    
    Args:
        XY: Combined dataframe with a patient column.
        split_results: Dict containing keys:
            - "selected_ids": patient IDs for train/test split
            - "remaining_ids": patient IDs for the other partition
            - "selected_share": share of signals assigned
            - "selected_signals": expected number of signals in selected partition
            - "remaining_signals": expected number of signals in remaining partition
            - "total_signals": total signals across dataset
        patient_col: column name for patient IDs (default="Patient")
    
    Returns:
        XY_selected, XY_remaining (two DataFrames)
    """
    selected_ids = set(split_results["selected_ids"])
    remaining_ids = set(split_results["remaining_ids"])
    
    # --- Split ---
    XY_selected = XY[XY[patient_col].isin(selected_ids)].reset_index(drop=True)
    XY_remaining = XY[XY[patient_col].isin(remaining_ids)].reset_index(drop=True)
    
    # --- Safety checks ---
    # 1) Signal count check
    n_sel, n_rem = len(XY_selected), len(XY_remaining)
    if n_sel != split_results["selected_signals"]:
        raise ValueError(f"❌ Selected signals mismatch: expected {split_results['selected_signals']}, got {n_sel}")
    if n_rem != split_results["remaining_signals"]:
        raise ValueError(f"❌ Remaining signals mismatch: expected {split_results['remaining_signals']}, got {n_rem}")
    
    # 2) Unique patient count check
    n_sel_ids = XY_selected[patient_col].nunique()
    n_rem_ids = XY_remaining[patient_col].nunique()
    if n_sel_ids != len(selected_ids):
        raise ValueError(f"❌ Unique patients mismatch in selected: expected {len(selected_ids)}, got {n_sel_ids}")
    if n_rem_ids != len(remaining_ids):
        raise ValueError(f"❌ Unique patients mismatch in remaining: expected {len(remaining_ids)}, got {n_rem_ids}")
    
    # --- Print summary ---
    print("✅ Safety check passed")
    print(f"Selected partition: {n_sel} signals, {n_sel_ids} patients "
          f"({split_results['selected_share']:.2%} share)")
    print(f"Remaining partition: {n_rem} signals, {n_rem_ids} patients")
    print(f"Total signals: {split_results['total_signals']}")
    print(f"Note: {split_results['note']}")
    
    return XY_selected, XY_remaining

def split_XY(
    XY: pd.DataFrame,
    target_cols: List[str],
    id_cols: List[str] = ["Patient", "Segment_ID"]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split XY dataframe into X (features) and Y (targets).
    
    Args:
        XY: Combined dataframe.
        target_cols: List of target column names (Y).
        id_cols: Columns to keep in both X and Y (default: ["Patient", "Segment_ID"]).
    
    Returns:
        X (DataFrame), Y (DataFrame)
    """
    # --- Validate ---
    missing_targets = [c for c in target_cols if c not in XY.columns]
    if missing_targets:
        raise KeyError(f"Missing target columns: {missing_targets}")

    # Keep IDs if present in DataFrame
    ids_in_df = [c for c in id_cols if c in XY.columns]

    # Y = targets + IDs
    Y = XY[ids_in_df + target_cols].copy()

    # X = all other columns except targets, but keep IDs
    X_cols = [c for c in XY.columns if c not in target_cols]
    X = XY[X_cols].copy()

    return X, Y

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FUNCTIONS FOR STANDARDIZING     ~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def standardize_z_score(df: pd.DataFrame, exclude_cols=None) -> pd.DataFrame:
    """
    Standardize DataFrame features using sklearn's StandardScaler (z-score).
    
    Args:
        df: Input DataFrame.
        exclude_cols: List of columns to exclude from scaling (e.g., 'Patient').
    
    Returns:
        DataFrame with scaled features, same column order.
        Fitted scaler object (to allow inverse transform later).
    """
    if exclude_cols is None:
        exclude_cols = []

    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c not in exclude_cols]
    
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df_scaled, scaler


def check_standarization(df: pd.DataFrame):

    features_df = df.drop(columns=["Patient"])
    # Column-wise mean and std
    print("Means:")
    print(features_df.mean())

    print("\nStandard deviations:")
    print(features_df.std())

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FUNCTIONS FOR DATA IMPUTATION   ~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def fill_missing_bp(Y: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing SBP/DBP values based on MAP column.
    
    Rules:
    - If MAP has any NaN -> raise ValueError immediately.
    - If SBP or DBP is NaN but the other is present:
        Use MAP = DBP + 1/3*(SBP - DBP) to fill the missing one.
    - If both SBP and DBP are NaN:
        Estimate SBP as the mean of the 5 nearest segments
        from the same patient (non-NaN SBP values).
        Then compute DBP with formula: DBP = 1.5*MAP - 0.5*SBP.
    
    Args:
        Y: DataFrame with columns ["Patient", "MAP", "SBP", "DBP"].
    
    Returns:
        DataFrame with SBP and DBP filled in.
    """
    Y_filled = Y.copy()

    # --- Check MAP first ---
    if Y_filled["MAP"].isna().any():
        raise ValueError("❌ MAP column contains NaN values. Cannot proceed.")

    # --- Filter only rows where SBP or DBP are NaN ---
    mask = Y_filled["SBP"].isna() | Y_filled["DBP"].isna()
    missing_rows = Y_filled[mask]

    for idx, row in missing_rows.iterrows():
        patient_id = row["Patient"]
        sbp, dbp, map_val = row["SBP"], row["DBP"], row["MAP"]

        # Case 1: only one missing
        if pd.isna(sbp) and pd.notna(dbp):
            # MAP = DBP + (SBP - DBP)/3  -> SBP = 3*MAP - 2*DBP
            sbp_est = 3 * map_val - 2 * dbp
            Y_filled.at[idx, "SBP"] = sbp_est

        elif pd.isna(dbp) and pd.notna(sbp):
            # MAP = DBP + (SBP - DBP)/3  -> DBP = 1.5*MAP - 0.5*SBP
            dbp_est = 1.5 * map_val - 0.5 * sbp
            Y_filled.at[idx, "DBP"] = dbp_est

        # Case 2: both missing
        elif pd.isna(sbp) and pd.isna(dbp):
            # get same-patient rows (excluding current one)
            patient_rows = Y_filled[Y_filled["Patient"] == patient_id].drop(idx)

            # find 5 closest by index
            diffs = np.abs(patient_rows.index - idx)
            nearest_idx = diffs.nsmallest(5).index

            sbp_est = patient_rows.loc[nearest_idx, "SBP"].dropna().mean()
            if pd.isna(sbp_est):
                raise ValueError(f"❌ Not enough SBP data to estimate for patient {patient_id} at row {idx}")

            Y_filled.at[idx, "SBP"] = sbp_est
            dbp_est = 1.5 * map_val - 0.5 * sbp_est
            Y_filled.at[idx, "DBP"] = dbp_est

    return Y_filled

def median_impute_patientwise(df: pd.DataFrame, patient_col: str = "Patient") -> pd.DataFrame:
    """
    Fill NaN values with the median of the same feature within each patient.
    
    Args:
        df: Input DataFrame.
        patient_col: Column name identifying patients (default="Patient").
    
    Returns:
        DataFrame with NaNs imputed patient-wise.
    """
    df_imputed = df.copy() # This is for safety! This avoids modifications on the original DF in memory.
    # group by patient and apply median imputation
    df_imputed = df_imputed.groupby(patient_col, group_keys=False).apply(
        lambda g: g.fillna(g.median(numeric_only=True)) # The fillna function is native from pandas
    )

    # Note: When doing groupby and then apply a lmbda function, the function is only used over the samll DF that is generated by grouping
    return df_imputed.reset_index(drop=True)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FUNCTIONS FOR ARREGLAR MACHETAZOS ~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def add_segment_id(X: pd.DataFrame,
                   Y: pd.DataFrame,
                   file_path: str,
                   dataset_type: str = "segments") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add Segment_ID column to X and Y DataFrames based on the first row length
    of the HDF5 'segments' dataset for each patient.

    Args:
        X, Y: pandas DataFrames with a "Patient" column.
        file_path: path to the .h5 file.
        dataset_type: which dataset to use inside each group (default="segments").
    
    Returns:
        (X_new, Y_new): DataFrames with Segment_ID column added.

    Raises:
        ValueError: if number of rows in X/Y for a patient does not match
                    the length of the first row in the HDF5 dataset.
    """
    X_new, Y_new = X.copy(), Y.copy()

    with h5py.File(file_path, "r") as f:
        for patient_id in f.keys():
            group = f[patient_id]
            if dataset_type not in group:
                raise KeyError(f"Dataset '{dataset_type}' not found in group '{patient_id}'")

            ds = group[dataset_type]
            first_row = ds[0]                  # shape = (L,)
            n_expected = len(first_row)        # number of elements in that row

            # rows in X and Y for this patient
            idx_X = X_new.index[X_new["Patient"] == patient_id]
            idx_Y = Y_new.index[Y_new["Patient"] == patient_id]

            if len(idx_X) != n_expected or len(idx_Y) != n_expected:
                raise ValueError(
                    f"Mismatch for patient {patient_id}: "
                    f"HDF5 first row length = {n_expected}, "
                    f"X rows = {len(idx_X)}, Y rows = {len(idx_Y)}"
                )

            # assign Segment_IDs
            segment_ids = [f"{patient_id}_seg{i}" for i in range(n_expected)]
            X_new.loc[idx_X, "Segment_ID"] = segment_ids
            Y_new.loc[idx_Y, "Segment_ID"] = segment_ids

            print(f"✅ Added Segment_ID for {patient_id} ({n_expected} segments)")

    return X_new, Y_new

import pandas as pd

def drop_abp_features(df: pd.DataFrame, patient_col: str = "Patient") -> pd.DataFrame:
    """
    Drop the last half of rows for each patient in the DataFrame.

    Args:
        df: Input DataFrame with a patient identifier column.
        patient_col: Name of the column that contains patient codes (default="Patient").

    Returns:
        A new DataFrame with only the first half of rows kept for each patient.
    """
    keep_rows = []
    for patient_id, group in df.groupby(patient_col):
        n = len(group)
        half = n // 2  # floor division, keeps first half if odd number
        keep_rows.extend(group.index[:half])  # take the first half

    return df.loc[keep_rows].reset_index(drop=True)

def merge_XY(X: pd.DataFrame, Y: pd.DataFrame, on: str = None) -> pd.DataFrame:
    """
    Merge X and Y DataFrames into one.

    Args:
        X: Features DataFrame.
        Y: Targets DataFrame.
        on: Column name to merge on. If None:
            - If 'Segment_ID' exists in both, use it.
            - Otherwise, merge assuming correct order patient-wise.

    Returns:
        Merged DataFrame.
    """
    # Case 1: explicit or Segment_ID present
    if on is not None or ("Segment_ID" in X.columns and "Segment_ID" in Y.columns):
        key = on if on is not None else "Segment_ID"
        merged = pd.merge(X, Y, on=key, suffixes=("_X", "_Y"))

    else:
        # Case 2: assume order matches within each patient
        if not all(X["Patient"].values == Y["Patient"].values):
            raise ValueError("Patient order mismatch between X and Y. Cannot merge safely.")
        
        merged = pd.concat([X.reset_index(drop=True), 
                            Y.drop(columns=["Patient"]).reset_index(drop=True)], axis=1)

    return merged
