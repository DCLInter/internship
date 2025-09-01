######## MI SCRIPT ###################################################################
#                                                                                    #
# This script calls the implamentation of Kraskov Mutual Information and             #
# calculates the index of Info-Fraction proposed in Examining the challenges         #
# of blood pressure estimation via photoplethysmogram (2024)                         #
#                                                                                    #
######################################################################################
"""
For Info-Fraction and concept:
@article{Mehta2024,
  author  = {S. Mehta and N. Kwatra and M. Jain and D. McDuff},
  title   = {Examining the challenges of blood pressure estimation via photoplethysmogram},
  journal = {Scientific Reports},
  volume  = {14},
  number  = {18318},
  year    = {2024},
  doi     = {10.1038/s41598-024-68862-1}
}

For actual implementation of Mutual Information algorithm from Kraskov:
@misc{VerSteeg2020,
  author       = {G. Ver Steeg and A. Galstyan},
  title        = {NPEET: Non-parametric Entropy Estimation Toolbox},
  howpublished = {GitHub repository},
  year         = {2020},
  url          = {https://github.com/gregversteeg/NPEET}
}
"""

import numpy as np
import pandas as pd
import time
from npeet import entropy_estimators as ee
from pathlib import Path

def compute_mi_info_fraction(X, y, k=5, verbose=True):
    """
    Compute Mutual Information (MI), entropy of the target (H),
    and Info-Fraction (MI/H) using Kraskov's KNN estimator.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input features (NumPy array or DataFrame).
    
    y : array-like of shape (n_samples,)
        Continuous target variable (NumPy array or Series).
    
    k : int, optional (default=5)
        Number of nearest neighbors for the Kraskov MI/entropy estimator.
    
    verbose : bool, optional (default=True)
        If True, prints runtime and results.

    Returns
    -------
    results : dict
        Dictionary with:
        - "MI": float, mutual information between X and y
        - "H": float, entropy of y
        - "Info-Fraction": float, normalized MI (MI/H)
        - "Runtime": float, time in seconds for the computation

    Notes
    -----
    - MI is invariant under smooth invertible transformations of X and y.
    - Info-Fraction ranges from 0 to 1, indicating the fraction of the 
      target's uncertainty explained by the features.
    - For very high-dimensional X, consider dimensionality reduction 
      (PCA or autoencoder bottleneck) before calling this function.
    """
    # Ensure numpy arrays
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1, 1)

    start = time.time()

    # Compute MI and entropy
    mi = ee.mi(X, y, k=k)
    h_y = ee.entropy(y, k=k)
    info_fraction = mi / h_y if h_y > 0 else np.nan

    runtime = time.time() - start

    results = {
        "MI": mi,
        "H": h_y,
        "Info-Fraction": info_fraction,
        "Runtime": runtime
    }

    if verbose:
        print(f"[MI Analysis] Runtime: {runtime:.2f} s | "
              f"MI={mi:.4f}, H={h_y:.4f}, Info-Fraction={info_fraction:.4f}")

    return results

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def compute_mi_summary(X, y, compute_func, feature_names=None,
                       target_name="target", k=5, save_path=None):
    """
    Wrapper that computes MI, entropy, Info-Fraction, and runtime for each feature
    individually and for the full feature set, using an existing
    compute_mi_info_fraction() function.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input features (NumPy array or DataFrame).
    
    y : array-like of shape (n_samples,)
        Continuous target variable (NumPy array or Series).
    
    compute_func : callable
        Function with signature compute_func(X, y, k=...) that returns
        {"MI": float, "H": float, "Info-Fraction": float, "Runtime": float}.
    
    feature_names : list of str, optional
        Names of the features. If None, features are indexed numerically.
    
    target_name : str, optional (default="target")
        Name of the target variable.
    
    k : int, optional (default=5)
        Number of nearest neighbors for the MI estimator.
    
    save_path : str or Path, optional
        If provided, saves the results DataFrame as a CSV file at this path.

    Returns
    -------
    results_df : pandas.DataFrame
        DataFrame with MI, entropy, Info-Fraction, and Runtime for each feature
        and for the collective feature set.
    """
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]

    records = []

    # Per-feature MI
    for i, fname in enumerate(feature_names):
        res = compute_func(X[:, i].reshape(-1, 1), y, k=k, verbose=True)
        records.append({
            "Target": target_name,
            "Feature": fname,
            "MI": res["MI"],
            "H": res["H"],
            "Info-Fraction": res["Info-Fraction"],
            "Runtime": res["Runtime"]
        })

    # Collective MI
    res_all = compute_func(X, y, k=k, verbose=True)
    records.append({
        "Target": target_name,
        "Feature": "ALL_FEATURES",
        "MI": res_all["MI"],
        "H": res_all["H"],
        "Info-Fraction": res_all["Info-Fraction"],
        "Runtime": res_all["Runtime"]
    })

    results_df = pd.DataFrame(records)

    # Optional save
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(save_path, index=False)

    return results_df