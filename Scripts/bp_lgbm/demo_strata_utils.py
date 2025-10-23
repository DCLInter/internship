############# DEMOGRAPHIC STRATIFICATION UTILS ################
#                                                             #
# Here, the utils for the satrata will be placed              #
#                                                             #
###############################################################
import re
import pandas as pd
import numbers
import itertools
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


def segment_multilabel_thresholds(
    df: pd.DataFrame,
    rules: dict,
    subject_col_name: str = "Subject",
    verbose: bool = True,
    skip_empty: bool = True
) -> dict:
    """
    Segment a DataFrame into multilabel demographic strata
    based on multiple thresholding rules.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing all segmentation columns.
    rules : dict
        Segmentation rules in the form:
            {
                "Age": [30, 45, 60],
                "BMI": [18.5, 25, 30],
                "Gender": ["M", "F"]
            }
    subject_col_name : str, default="Subject"
        Column name with subject IDs.
    verbose : bool, default=True
        Print summary of each multilabel subset.

    Returns
    -------
    dict
        Dictionary mapping multilabel segment names to DataFrames:
        {
            "Age<30_BMI<18.5_Gender=M": df_subset,
            "Age_30-45_BMI_25-30_Gender=F": df_subset,
            ...
        }
    """

    # --- Build segment masks for each variable (like your single-var version) ---
    var_segments = {}

    for col, vals in rules.items():
        if col not in df.columns:
            print(f"⚠️ Column '{col}' not found — skipping.")
            continue

        segs = []

        # numeric thresholds
        if all(isinstance(v, numbers.Number) for v in vals):
            vals = sorted(vals)
            segs.append((f"{col}<{vals[0]}", df[col] < vals[0]))
            for lo, hi in zip(vals[:-1], vals[1:]):
                segs.append((f"{col}_{lo}-{hi}", (df[col] >= lo) & (df[col] < hi)))
            segs.append((f"{col}>={vals[-1]}", df[col] >= vals[-1]))

        # categorical values
        else:
            for v in vals:
                segs.append((f"{col}={v}", df[col] == v))

        var_segments[col] = segs

    # Cartesian product of all variable segments
    keys = list(var_segments.keys())
    combos = list(itertools.product(*[var_segments[k] for k in keys]))

    multi_segments = {}
    if verbose:
        print("📊 Multilabel stratification summary:\n")

    for combo in combos:
        labels, masks = zip(*combo)
        label = "_".join(labels)
        combined_mask = pd.Series(True, index=df.index)
        for m in masks:
            combined_mask &= m

        subset = df.loc[combined_mask].copy()
        n_rows = len(subset)
        n_subj = subset[subject_col_name].nunique() if n_rows > 0 else 0

        if not skip_empty or n_rows > 0:
            multi_segments[label] = subset

        if verbose:
            print(f"{label:<50} -> {n_rows:>6} samples | {n_subj:>5} subjects")

    return multi_segments

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
    multilabel_mode: bool = False,
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
    multilable_mode: bool.
        Changes the iteration logic to adjust for any incomming dictionary.
        If multilabel_mode=False → expects nested dict structure:
            {"Age": [(label, df, n_rows, n_subj), ...], "Gender": [...], ...]}
        If multilabel_mode=True → expects flat dict:
            {"Age<40_BMI<25_Gender=M": df, ...}
    """

    drop_features = drop_features or []
    results = []

    print(f"\n🚀 Running analysis for target: {target}")
    base_dir = Path(base_results_dir)
    base_dir.mkdir(parents=True, exist_ok=True)


    # --- unified loop ---
    for variable, label, df_train, df_test in iter_segments(dfs_dict_train, dfs_dict_test, multilabel_mode):
        print(f"▶️ {variable} - {label}")

        subset_dir = base_dir / variable / sanitize_label(label)
        subset_dir.mkdir(parents=True, exist_ok=True)

        # --- Split into X/Y ---
        X_train, Y_train = split_XY(df_train, [target], id_cols=["Subject"], drop_cols=drop_features)
        X_test,  Y_test  = split_XY(df_test, [target], id_cols=["Subject"], drop_cols=drop_features)

        # --- Train/val/leak splits ---
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_train, Y_train, test_size=val_split_size, random_state=42,
            shuffle=True, stratify=X_train["Subject"]
        )
        _, X_leak, _, Y_leak = train_test_split(
            X_train, Y_train, test_size=train_subset_size, random_state=42,
            shuffle=True, stratify=X_train["Subject"]
        )

        # --- Drop ID columns ---
        for X in (X_train, X_val, X_leak, X_test):
            if "Subject" in X.columns:
                X.drop(columns=["Subject"], inplace=True)
        for Y in (Y_train, Y_val, Y_leak, Y_test):
            if "Subject" in Y.columns:
                Y.drop(columns=["Subject"], inplace=True)

        # --- Train model ---
        model = model_fn()
        model.fit(X_train, Y_train)

        # --- Evaluate ---
        datasets = {"train": (X_leak, Y_leak), "val": (X_val, Y_val), "test": (X_test, Y_test)}
        for dataset_name, (X, y_true) in datasets.items():
            if isinstance(y_true, pd.DataFrame):
                y_true = y_true.squeeze().values
            elif isinstance(y_true, pd.Series):
                y_true = y_true.values

            y_pred = model.predict(X)
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

        # --- Save subgroup CSV ---
        subgroup_df = pd.DataFrame([r for r in results if r["variable"] == variable and r["segment"] == label])
        subgroup_df.to_csv(subset_dir / f"stratified_results_{target}.csv", index=False)

    # --- Aggregate ---
    df_all = pd.DataFrame(results)
    if multilabel_mode:
        agg_csv = base_dir / f"aggregated_multilabel_{target}.csv"
    else:
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

def iter_segments(dfs_train, dfs_test, multilabel_mode=False):
    """
    Unified iterator yielding (variable, label, df_train, df_test).
    This is quite an important concept.
    Here, qwe have a function that standardize the way we iterate over the dictionaries that I am creating and returns a sequence of tuples for each iteration.

    CONCEPT OF "yield":
    Turns a function into a generator --> a function that pauses and resumes.
    My understanding: allows a function to becaome iterable. Look at this:
        def count_to_three():
    for i in [1, 2, 3]:
        yield i

    for x in count_to_three():
        print(x)

    The difference is that the iterator only holds 1 element at a time in memory --> way more efficient than returing the whole array.
    """
    if multilabel_mode:
        # flat dict: {"Age<40_BMI<25_Gender=M": df, ...}
        """ This one is mimicking the previous syntax by introducing "Multilabel". But the rest is pretty much the same"""
        for label, df_tr in dfs_train.items():
            df_te = dfs_test.get(label)
            if df_te is None:
                print(f"⚠️ Missing test segment for {label}, skipping.")
                continue
            yield ("Multilabel", label, df_tr, df_te)
    else:
        # nested dict: {"Age": [(label, df, n, nsubj), ...], "Gender": [...], ...}
        """ The same logic as before """
        for variable, train_segments in dfs_train.items():
            test_segments = dfs_test.get(variable, [])
            for (label, df_tr, *_), (_, df_te, *_) in zip(train_segments, test_segments):
                yield (variable, label, df_tr, df_te)