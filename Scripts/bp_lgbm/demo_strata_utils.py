############# DEMOGRAPHIC STRATIFICATION UTILS ################
#                                                             #
# Here, the utils for the satrata will be placed              #
#                                                             #
###############################################################

import pandas as pd
import numbers

def segment_thresholds(df: pd.DataFrame, rules: dict, subject_col_name: str = "Subject", verbose: bool = True):
    """
    Segment a DataFrame by numeric thresholds or categorical values and summarize each subset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing the columns to segment and a 'Subject' identifier.
    rules : dict
        Segmentation rules in the form:
            {"Age": [30, 45, 60], "BMI": [18.5, 25, 30], "Gender": ["M", "F"]}
    subject_col : str, default="Subject"
        Column name containing subject IDs to count unique subjects per subset.
    verbose : bool, default=True
        If True, prints a summary of segment sizes.

    Returns
    -------
    dict
        Nested dictionary with:
        {
            "Age": [
                ("Age<30", subset_df, n_rows, n_subjects),
                ("Age_30-45", subset_df, n_rows, n_subjects),
                ...
            ],
            "Gender": [...]
        }
    """
    results = {}

    for col, vals in rules.items():
        if col not in df.columns:
            print(f"⚠️ Column '{col}' not found — skipping.")
            continue

        # Handle numeric thresholds
        if all(isinstance(v, numbers.Number) for v in vals):
            vals = sorted(vals)
            segments = []

            # Below first threshold
            mask = df[col] < vals[0]
            sub = df.loc[mask].copy()
            label = f"{col}<{vals[0]}"
            segments.append((label, sub, len(sub), sub[subject_col_name].nunique()))

            # Between thresholds
            for i in range(len(vals) - 1):
                lo, hi = vals[i], vals[i + 1]
                mask = (df[col] >= lo) & (df[col] < hi)
                sub = df.loc[mask].copy()
                label = f"{col}_{lo}-{hi}"
                segments.append((label, sub, len(sub), sub[subject_col_name].nunique()))

            # Above last threshold
            mask = df[col] >= vals[-1]
            sub = df.loc[mask].copy()
            label = f"{col}>={vals[-1]}"
            segments.append((label, sub, len(sub), sub[subject_col_name].nunique()))

            results[col] = segments

        else:
            # Handle categorical splits
            segments = []
            for val in vals:
                mask = df[col] == val
                sub = df.loc[mask].copy()
                label = f"{col}={val}"
                segments.append((label, sub, len(sub), sub[subject_col_name].nunique()))
            results[col] = segments

    # Optional summary
    if verbose:
        print("📊 Stratification summary:\n")
        for col, segs in results.items():
            print(f"Variable: {col}")
            for i, (label, _, n_rows, n_subj) in enumerate(segs):
                print(f"  {i}. {label:<20} -> {n_rows:>6} samples | {n_subj:>5} unique subjects")
            print("-" * 60)

    return results