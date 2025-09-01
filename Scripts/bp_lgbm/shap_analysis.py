######## SHAP SCRIPT #################################################################
#                                                                                    #
# This script handles all the experimentation of shapley analysis.                   #
#                                                                                    #
######################################################################################

import shap
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.base import clone

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
    target_idx: int = None, 
    n_iter: int = 50, 
    tol: float = 1.0, 
    random_state: int = 42,
    verbose: bool = True
):
    """
    Evaluate the stability of SHAP feature importance rankings and absolute values 
    across multiple stochastic runs of a model (e.g., LightGBM inside a pipeline).

    Tracks *per-feature* rank changes between iterations, so feature trajectories 
    can be visualized directly.

    Parameters
    ----------
    pipeline : sklearn.Pipeline
        Pipeline ending with an estimator compatible with TreeExplainer 
        (e.g., LightGBM). The estimator must accept `random_state`.
    
    X : pandas.DataFrame
        Feature matrix. Must have column names.
    
    y : array-like
        Target vector.
    
    target_idx : int, optional (default=None)
        If pipeline's final estimator is a MultiOutputRegressor, specify which 
        target to explain (0=SBP, 1=DBP, 2=MAP).
    
    n_iter : int, optional (default=50)
        Maximum number of iterations.
    
    tol : float, optional (default=1.0)
        Convergence threshold for the average rank shift between iterations.
    
    random_state : int, optional (default=42)
        Base random seed. Each iteration increments this by +i for variability.
    
    verbose : bool, optional (default=True)
        Print progress and stability information.

    Returns
    -------
    avg_rank : np.ndarray of shape (n_features,)
        Average rank of each feature across iterations.
    
    rank_matrix : np.ndarray of shape (n_iter_eff, n_features)
        Rank of each feature at each iteration.
    
    avg_abs_shap : np.ndarray of shape (n_features,)
        Average mean(|SHAP|) across iterations for each feature.
    
    abs_shap_matrix : np.ndarray of shape (n_iter_eff, n_features)
        Per-iteration mean(|SHAP|) for each feature.
    
    rank_diff_matrix : np.ndarray of shape (n_iter_eff-1, n_features)
        Per-feature rank changes between consecutive iterations.
    
    feature_names : list of str
        Feature names corresponding to columns of X.
    """
    import numpy as np
    from sklearn.base import clone

    feature_names = list(X.columns)
    n_features = X.shape[1]

    rank_matrix = np.zeros((n_iter, n_features))
    abs_shap_matrix = np.zeros((n_iter, n_features))
    rank_diff_matrix = np.zeros((n_iter - 1, n_features))

    prev_ranks = None
    effective_iters = 0

    for i in range(n_iter):
        if verbose:
            print(f"[Iteration {i+1}/{n_iter}]")

        # Clone and fit pipeline
        pipe = clone(pipeline)
        pipe["model"].set_params(**{"random_state": random_state + i})
        pipe.fit(X, y)

        # Extract final estimator
        estimator = pipe[-1]

        # Compute SHAP values
        shap_values, _, feature_names = compute_shap_values(
            estimator, X, target_idx=target_idx, feature_names=feature_names, verbose=False
        )

        # Rank features
        ranks, mean_abs_shap = rank_features_from_shap(shap_values, feature_names)

        # Store rank positions and mean(|SHAP|)
        for pos, feat_idx in enumerate(ranks):
            rank_matrix[i, feat_idx] = pos + 1
        abs_shap_matrix[i, :] = mean_abs_shap

        # Per-feature rank changes
        if prev_ranks is not None:
            rank_diff = np.abs(rank_matrix[i] - prev_ranks)
            rank_diff_matrix[i - 1, :] = rank_diff

            avg_shift = rank_diff.mean()
            if verbose:
                print(f"  Avg. rank shift = {avg_shift:.3f}")

            if avg_shift < tol:
                if verbose:
                    print("  Converged — stopping early.")
                effective_iters = i + 1
                break

        prev_ranks = rank_matrix[i].copy()
        effective_iters = i + 1

    # Truncate matrices to effective iterations
    rank_matrix = rank_matrix[:effective_iters]
    abs_shap_matrix = abs_shap_matrix[:effective_iters]
    rank_diff_matrix = rank_diff_matrix[:effective_iters - 1]

    # Averages
    avg_rank = rank_matrix.mean(axis=0)
    avg_abs_shap = abs_shap_matrix.mean(axis=0)

    return avg_rank, rank_matrix, avg_abs_shap, abs_shap_matrix, rank_diff_matrix, feature_names

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