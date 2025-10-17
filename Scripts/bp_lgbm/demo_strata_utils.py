############# DEMOGRAPHIC STRATIFICATION UTILS ################
#                                                             #
# Here, the utils for the satrata will be placed              #
#                                                             #
###############################################################
import re
import pandas as pd
import numbers
from preprocessing import split_XY
from pathlib import Path
from typing import List
from sklearn.model_selection import train_test_split

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

def run_analysis_for_target(
    dfs_dict_train,
    dfs_dict_test,
    target: str,
    model_fn,
    val_split_size,
    train_subset_size,
    evaluate_fn,
    base_results_dir,
    drop_features: List[str] = None,
):
    """
    Train and evaluate a model for one BP target across demographic subsets.

    Parameters
    ----------
    dfs_dict_train, dfs_dict_test : dict
        Nested dictionaries with demographic splits (same structure).
    target : str
        Target variable name (e.g. 'SBP', 'DBP', 'MAP').
    model_fn : callable
        Function that returns a model instance.
    val_split_size : float
        Proportion for the size of the splitting for validation.
    copy_portion_fn : callable
        Proportion for the size of the splitting for the training subset for evaluating training.
    evaluate_fn : callable
        Evaluation function: evaluate(y_true, y_pred, R2_path, BA_path) → dict
    base_results_dir : str
        Base directory for saving all results.
    drop_features : list of str, optional
        Columns to drop from X (e.g. demographics, IDs, biometrics).
    """

    drop_features = drop_features or []
    results = []

    print(f"\n🚀 Running analysis for target: {target}")
    base_dir = Path(base_results_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    for variable, train_segments in dfs_dict_train.items():
        test_segments = dfs_dict_test[variable]
        print(f"\n📊 Demographic variable: {variable}")

        for (label, df_train, _, _), (_, df_test, _, _) in zip(train_segments, test_segments):
            print(f"  ▶️ {variable} - {label}")

            # --- Create output folder for this demographic group ---
            subset_dir = base_dir / variable / sanitize_label(label)
            subset_dir.mkdir(parents=True, exist_ok=True)

            # --- Split into X/Y for training ---
            X_train, Y_train = split_XY(df_train, [target], id_cols=["Subject"], drop_cols=drop_features)
            X_test,  Y_test  = split_XY(df_test, [target], id_cols=["Subject"], drop_cols=drop_features)

            # --- Split into training and validation sets ---
            X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=val_split_size, random_state= 42, shuffle=True, stratify=X_train["Subject"])

            # --- Leak portion for training evaluation ---
            _, X_leak,_, Y_leak = train_test_split(X_train, Y_train, test_size=train_subset_size, random_state= 42, shuffle=True, stratify=X_train["Subject"])

            X_train = X_train.drop(columns=["Subject"])
            Y_train = Y_train.drop(columns=["Subject"])
            X_val = X_val.drop(columns=["Subject"])
            Y_val = Y_val.drop(columns=["Subject"])
            X_leak = X_leak.drop(columns=["Subject"])
            Y_leak = Y_leak.drop(columns=["Subject"])
            X_test = X_test.drop(columns=["Subject"])
            Y_test = Y_test.drop(columns=["Subject"])

            # --- Train model ---
            model = model_fn()
            model.fit(X_train, Y_train)

            # --- Evaluate on leak, val, and test ---
            datasets = {
                "train": (X_leak, Y_leak),
                "val": (X_val, Y_val),
                "test": (X_test, Y_test),
            }

            for dataset_name, (X, y_true) in datasets.items():
                """
                print(f"\n🧩 DEBUG — {variable} | {label} | {dataset_name}")
                print(f"X type: {type(X)}, shape: {getattr(X, 'shape', 'N/A')}")
                """
                # ensure y_true is a 1-D array
                if isinstance(y_true, pd.DataFrame):
                    y_true = y_true.squeeze().values   # (n,) array
                elif isinstance(y_true, pd.Series):
                    y_true = y_true.values

                y_pred = model.predict(X)

                # Define paths for R² and BA plots
                R2_path = subset_dir / f"{target}_{dataset_name}_R2.png"
                BA_path = subset_dir / f"{target}_{dataset_name}_BA.png"

                metrics = evaluate_fn(y_true, y_pred, R2_path, BA_path)
                metrics.update({
                    "target": target,
                    "variable": variable,
                    "segment": label,
                    "dataset": dataset_name,
                })
                results.append(metrics)

            # --- Save subgroup CSV incrementally ---
            subgroup_df = pd.DataFrame([r for r in results if r["variable"] == variable and r["segment"] == label])
            subgroup_csv = subset_dir / f"stratified_results_{target}.csv"
            subgroup_df.to_csv(subgroup_csv, index=False)

    # --- Aggregate results for this target ---
    df_all = pd.DataFrame(results)
    agg_csv = base_dir / f"aggregated_{target}.csv"
    df_all.to_csv(agg_csv, index=False)
    print(f"\n✅ Aggregated results for {target} saved to {agg_csv}")

    return df_all

# -------------------------------------------------------
# HELPER
# -------------------------------------------------------

def sanitize_label(label: str) -> str:
    """
    Replace invalid filename characters for cross-platform compatibility.
    Example: 'Age<40' -> 'Age_lt_40'
    """
    replacements = {
        "<": "_lt_",
        ">": "_gt_",
        "=": "_eq_",
        " ": "_",
        "/": "_",
        "\\": "_"
    }
    for k, v in replacements.items():
        label = label.replace(k, v)
    # Remove any remaining forbidden chars just in case
    label = re.sub(r'[<>:"/\\|?*]', "_", label)
    return label