######## SHAP SCRIPT #################################################################
#                                                                                    #
# This script handles all the experimentation of shapley analysis.                   #
#                                                                                    #
######################################################################################

import shap
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit
from scipy import stats

# ----------------------------
# Computation
# ----------------------------
def compute_shap_values(model, X, target_idx: int = None, feature_names=None, verbose = True):
    """
    Compute SHAP values for a given target in a trained LightGBM model.

    OJO: PASS A MODEL TYPE OBJECT, not a pipeline.

    Parameters
    ----------
    model : MultiOutputRegressor or LGBMRegressor
        Trained model.
    
    X : array-like of shape (n_samples, n_features)
        Dataset on which to compute SHAP values.
    
    target_idx : int, optional (default=0)
        Index of the target if using MultiOutputRegressor.
        - 0 = SBP
        - 1 = DBP
        - 2 = MAP
    
    feature_names : list of str, optional
        Names of features (for plotting/interpretation). If None,
        will be inferred if X is a DataFrame.

    verbose : bool, optional (default=True)
        If True, prints runtime and summary of results.

    Returns
    -------
    shap_values : np.ndarray
        SHAP values array of shape (n_samples, n_features).
    
    explainer : shap.TreeExplainer
        The SHAP explainer object (can be reused for plots).
    """
    # Select correct model
    if hasattr(model, "estimators_"):  # MultiOutputRegressor case
        if target_idx is None:
            raise ValueError(
            "MultiOutputRegressor detected. You must provide 'target_idx' "
            "to specify which target to explain (0=SBP, 1=DBP, 2=MAP)."
        )
        base_model = model.estimators_[target_idx]
    else:
        base_model = model

    # Build SHAP explainer
    explainer = shap.TreeExplainer(base_model)

    # Convert DataFrame to numpy if needed
    if hasattr(X, "values"):
        X_array = X.values
        if feature_names is None:
            feature_names = list(X.columns)
    else:
        X_array = np.asarray(X)
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(X_array.shape[1])]

    # Compute SHAP values
    start = time.time()
    shap_values = explainer.shap_values(X_array)
    runtime = time.time() - start

    if verbose:
        print(f"[SHAP Analysis] Runtime: {runtime:.2f} s "
              f"| n_samples={X_array.shape[0]}, n_features={X_array.shape[1]}")

    return shap_values, explainer, feature_names

def rank_features_from_shap(shap_values, feature_names):
    """
    Compute mean(|SHAP|) across samples and rank features.
    """
    import numpy as np
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    ranking = np.argsort(-mean_abs_shap)  # descending order
    return ranking, mean_abs_shap

def shap_rank_stability(
    pipeline, 
    X, 
    y, 
    groups,                 # subject IDs
    n_iter: int = 50, 
    target_idx: int = None, 
    random_state: int = 42,
    verbose: bool = True,
    save_path: str = None
):
    """
    Evaluate SHAP feature ranking stability using *preallocated subject splits*.
    Splits are generated once with allocate_subjects_for_train, ensuring that
    subject appearances in training are balanced based on inverse signal contribution.

    Parameters
    ----------
    pipeline : sklearn.Pipeline
        Pipeline ending with an estimator compatible with shap.TreeExplainer.
    
    X : pd.DataFrame
        Feature matrix (with column names).
    
    y : array-like
        Target vector.
    
    groups : array-like
        Subject IDs aligned with X/y.
    
    n_iter : int, default=50
        Number of iterations/splits to allocate and run.
    
    target_idx : int, optional
        If using MultiOutputRegressor, specify which target to explain.
    
    random_state : int, default=42
        Seed for reproducibility in subject allocation.
    
    verbose : bool, default=True
        Print progress info.
    
    save_path : str, optional
        If provided, saves matrices/summary to disk.

    Returns
    -------
    avg_rank : np.ndarray
        Average rank of each feature across iterations.
    rank_matrix : np.ndarray
        Rank of each feature at each iteration.
    avg_abs_shap : np.ndarray
        Average mean(|SHAP|) across iterations.
    abs_shap_matrix : np.ndarray
        Mean(|SHAP|) per feature per iteration.
    rank_diff_matrix : np.ndarray
        Rank changes between consecutive iterations.
    feature_names : list
        List of feature names.
    """

    # === Preallocate splits ===
    splits = allocate_subjects_for_train(groups, n_iter=n_iter, train_size=0.8, random_state=random_state)

    feature_names = list(X.columns)
    n_features = X.shape[1]

    rank_matrix = np.zeros((n_iter, n_features))
    abs_shap_matrix = np.zeros((n_iter, n_features))
    rank_diff_matrix = np.zeros((n_iter - 1, n_features))

    prev_ranks = None

    for i, split in enumerate(splits):
        if verbose:
            print(f"[Iteration {i+1}/{n_iter}]")
        iter_start = time.time()

        # === Build masks ===
        mask_train = np.isin(groups, split["train_subjects"]) # It is accessing the dicrtionary in the train set
        X_train, y_train = X[mask_train], np.array(y)[mask_train]

        # === Fit pipeline ===
        pipe = clone(pipeline)
        pipe.fit(X_train, y_train)
        estimator = pipe.named_steps["model"]  # explicit access to estimator

        # === Compute SHAP on train set ===
        shap_values, _, feature_names = compute_shap_values(
            estimator, X_train, target_idx=target_idx, feature_names=feature_names, verbose=False
        )

        # === Rank features ===
        ranks, mean_abs_shap = rank_features_from_shap(shap_values, feature_names)
        for pos, feat_idx in enumerate(ranks):
            rank_matrix[i, feat_idx] = pos + 1
        abs_shap_matrix[i, :] = mean_abs_shap

        # === Rank differences ===
        if prev_ranks is not None:
            rank_diff_matrix[i - 1, :] = np.abs(rank_matrix[i] - prev_ranks)

        prev_ranks = rank_matrix[i].copy()

        if verbose:
            elapsed = time.time() - iter_start
            print(f"  Iteration time: {elapsed:.2f} seconds")

    # === Averages ===
    avg_rank = rank_matrix.mean(axis=0)
    avg_abs_shap = abs_shap_matrix.mean(axis=0)

    # === Optional save ===
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        pd.DataFrame({
            "feature": feature_names,
            "avg_rank": avg_rank,
            "avg_abs_shap": avg_abs_shap
        }).to_csv(f"{save_path}_summary.csv", index=False)

        pd.DataFrame(rank_matrix, columns=feature_names).to_csv(f"{save_path}_rank_matrix.csv", index=False)
        pd.DataFrame(abs_shap_matrix, columns=feature_names).to_csv(f"{save_path}_abs_shap_matrix.csv", index=False)
        pd.DataFrame(rank_diff_matrix, columns=feature_names).to_csv(f"{save_path}_rank_diff_matrix.csv", index=False)

        if verbose:
            print(f"Results saved to: {os.path.dirname(save_path)}")

    return avg_rank, rank_matrix, avg_abs_shap, abs_shap_matrix, rank_diff_matrix, feature_names

# ----------------------------
# Utilities
# ----------------------------
def get_train_test_masks(X, y, groups, mask_train=None, mask_test=None, seed=42, iter_idx=0): # Not really used.
    """
    Create subject-wise train/test masks for iteration.

    Iteration 0:
        - 80% train / 20% test (subject-wise).
    Iteration >=1:
        - Resplit only the previous train into 75% train / 25% new test.
        - Merge: new_train = subtrain ∪ old_test, new_test = subtest.
    """
    if mask_train is None and mask_test is None:
        # First split: 80/20 subject-wise
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, test_idx = next(gss.split(X, y, groups))

        mask_train = np.zeros(len(X), dtype=bool)
        mask_test = np.zeros(len(X), dtype=bool)
        mask_train[train_idx] = True
        mask_test[test_idx] = True
        return mask_train, mask_test

    else:
        # Iteration >= 1: resplit within previous train (75/25 → global ~80/20)
        gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed+iter_idx)
        subtrain_idx, subtest_idx = next(gss_inner.split(X[mask_train], y[mask_train], groups[mask_train]))

        # Build new masks (relative to original dataset indices)
        mask_subtrain = np.zeros(len(X), dtype=bool)
        mask_subtest = np.zeros(len(X), dtype=bool)

        mask_subtrain[np.where(mask_train)[0][subtrain_idx]] = True
        mask_subtest[np.where(mask_train)[0][subtest_idx]] = True

        # Rotate: old test returns to train, new test comes from old train
        new_mask_train = mask_subtrain | mask_test
        new_mask_test = mask_subtest

        return new_mask_train, new_mask_test

def allocate_subjects_for_train(groups, n_iter=50, train_size=0.8, random_state=42):
    """
    Allocate subjects into n_iter training sets, balancing appearances
    based on the inverse of their dataset contribution (signal counts).
    
    Parameters
    ----------
    groups : array-like of shape (n_samples,)
        Subject IDs for each row of the dataset.
    n_iter : int, default=50
        Number of splits to allocate.
    train_size : float, default=0.8
        Proportion of subjects to assign to train set each split.
    random_state : int, default=42
        Random seed for reproducibility when shuffling assignments.

    Returns
    -------
    splits : list of dict
        List of length n_iter. Each entry is a dict with:
            - "train_subjects": list of subject IDs
            - "test_subjects": list of subject IDs
    """
    rng = np.random.RandomState(random_state)
    unique_subjects, counts = np.unique(groups, return_counts=True)
    N = len(unique_subjects)

    # Contribution proportion per subject
    contributions = counts / counts.sum()

    # Inverse weights
    inv_weights = 1.0 / contributions
    probs = inv_weights / inv_weights.sum()

    # Expected number of training slots across all iterations
    total_train_slots = int(n_iter * train_size * N)
    expected_train = probs * total_train_slots

    # Round to integers
    train_counts = np.floor(expected_train).astype(int)
    remainder = total_train_slots - train_counts.sum()

    # Distribute leftover slots (highest fractional parts get one extra slot)
    frac_parts = expected_train - train_counts
    extra_indices = np.argsort(-frac_parts)[:remainder]
    train_counts[extra_indices] += 1

    # === Allocate across iterations ===
    # Build a "pool" of subject IDs repeated by their quota
    pool = []
    for subj, count in zip(unique_subjects, train_counts):
        pool.extend([subj] * count)

    rng.shuffle(pool)

    splits = []
    slot_size = int(train_size * N)  # train subjects per iteration
    for i in range(n_iter):
        start = i * slot_size
        end = (i + 1) * slot_size
        train_subjects = pool[start:end]

        # Ensure uniqueness per iteration
        train_subjects = list(set(train_subjects))
        test_subjects = [s for s in unique_subjects if s not in train_subjects]

        splits.append({
            "train_subjects": train_subjects,
            "test_subjects": test_subjects
        })

    return splits

def summarize_rank_matrix(rank_matrix, feature_names, save_path=None):
    """
    Compute summary statistics for feature ranks across iterations.
    
    Parameters
    ----------
    rank_matrix : np.ndarray, shape (n_iter, n_features)
        Rank of each feature at each iteration.
    
    feature_names : list of str
        Names of the features.
    
    save_path : str, optional
        If provided, saves the summary as CSV.
    
    Returns
    -------
    summary_df : pd.DataFrame
        DataFrame with summary statistics for each feature.
    """
    # Convert to DataFrame
    df = pd.DataFrame(rank_matrix, columns=feature_names)

    summary = pd.DataFrame({
        "feature": feature_names,
        "mean": df.mean().values,
        "median": df.median().values,
        "std": df.std().values,
        "min": df.min().values,
        "max": df.max().values,
        "iqr": (df.quantile(0.75) - df.quantile(0.25)).values,
        "mode": [stats.mode(df[col], keepdims=True).mode[0] for col in df.columns]
    })

    if save_path is not None:
        summary.to_csv(save_path, index=False)

    return summary
# ----------------------------
# Visualization
# ----------------------------
def plot_shap_beeswarm_summary(shap_values, X, feature_names=None, target_name="target", save_path=None, show=True, max_display=20):
    """
    Plot a SHAP summary (beeswarm) plot for global feature importance.
    Features sorted by importance (mean |SHAP|).
    Colors show the feature value (blue=low, red=high).

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values array of shape (n_samples, n_features).
    
    X : array-like or DataFrame
        Input dataset used to compute SHAP values.
    
    feature_names : list of str, optional
        Names of features. If None and X is a DataFrame, uses X.columns.
    
    target_name : str, optional (default="target")
        Name of the target variable (e.g., SBP, DBP, MAP).
    
    save_path : str or Path, optional
        If provided, saves the plot as a PNG file at this path.
    
    show : bool, optional (default=True)
        If True, displays the plot interactively.

     max_display : int, optional (default=20)
        Maximum number of top features to display in the plot.
    """
    # Handle feature names
    if feature_names is None and hasattr(X, "columns"):
        feature_names = list(X.columns)

    # Define title
    title = f"SHAP Summary - {target_name}"

    # Make the plot
    plt.figure()
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        show=False,
        max_display=max_display
    )
    plt.title(title)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    elif show:
        plt.show()

def plot_rank_diff_trajectories(
    rank_diff_matrix, 
    feature_names, 
    top_k=None, 
    figsize=(12,6),
    show=True,
    save_path=None
):
    """
    Plot trajectories of rank differences across iterations.

    Parameters
    ----------
    rank_diff_matrix : np.ndarray, shape (n_iter-1, n_features)
        Rank differences between consecutive iterations.
    feature_names : list of str
        Feature names.
    top_k : int, optional
        If given, plot only top_k features with highest average variability.
    figsize : tuple
        Figure size.
    show : bool, default=True
        If True, display the plot interactively.
    save_path : str or Path, optional
        If given, save the plot as PNG at this path.
    """
    n_iter = rank_diff_matrix.shape[0] + 1

    # Compute average variability
    avg_var = rank_diff_matrix.mean(axis=0)
    order = np.argsort(-avg_var)
    if top_k is not None:
        selected = order[:top_k]
    else:
        selected = range(len(feature_names))

    plt.figure(figsize=figsize)
    for idx in selected:
        plt.plot(range(2, n_iter+1), rank_diff_matrix[:, idx], label=feature_names[idx], alpha=0.7)

    plt.xlabel("Iteration")
    plt.ylabel("Rank difference vs previous iteration")
    plt.title("Feature rank difference trajectories")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    elif show:
        plt.show()

def plot_rank_boxplot(
    rank_matrix, 
    feature_names, 
    top_k=10, 
    figsize=(10,6),
    show=True,
    save_path=None
):
    """
    Plot horizontal boxplots of feature ranks across iterations.

    Parameters
    ----------
    rank_matrix : np.ndarray, shape (n_iter, n_features)
        Rank of each feature at each iteration.
    feature_names : list of str
        Feature names.
    top_k : int, default=10
        Number of top features (by average rank) to display.
    figsize : tuple
        Figure size.
    show : bool, default=True
        If True, display the plot interactively.
    save_path : str or Path, optional
        If given, save the plot as PNG at this path.
    """
    df = pd.DataFrame(rank_matrix, columns=feature_names)
    avg_rank = df.mean().sort_values()  # lower = more important
    top_features = avg_rank.head(top_k).index.tolist()

    plt.figure(figsize=figsize)
    df[top_features].boxplot(vert=False)
    plt.title(f"Top {top_k} features: rank distributions across iterations")
    plt.xlabel("Rank (lower = more important)")
    plt.ylabel("Feature")
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    elif show:
        plt.show()