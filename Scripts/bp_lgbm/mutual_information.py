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
def compute_mi_summary(
    X: pd.DataFrame,
    y: pd.DataFrame | pd.Series,
    feature_names: list[str] | None = None,
    target_name: str = "target",
    k: int = 5,
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Compute MI, entropy, Info-Fraction, and runtime for each feature
    individually and for the full feature set, using an existing
    compute_mi_info_fraction() function.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (samples x features).
    
    y : pd.Series or pd.DataFrame
        Target variable(s). If DataFrame, `target_name` must be a column.
    
    feature_names : list of str, optional
        Names of the features. If None, use DataFrame columns.
    
    target_name : str, optional (default="target")
        Name of the target variable to use (if y is DataFrame).
    
    k : int, optional (default=5)
        Number of nearest neighbors for the MI estimator.
    
    save_path : str or Path, optional
        If provided, saves the results DataFrame as a CSV file at this path.

    Returns
    -------
    results_df : pd.DataFrame
    """
    # --- validate input
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")

    if isinstance(y, pd.DataFrame):
        if target_name not in y.columns:
            raise ValueError(f"Target column '{target_name}' not found in y DataFrame")
        y_vec = y[target_name].to_numpy().ravel()
    else:
        y_vec = y.to_numpy().ravel()

    # --- feature names
    if feature_names is None:
        feature_names = X.columns.tolist()

    records = []

    # --- per-feature MI
    for i, fname in enumerate(feature_names):
        Xi = X.iloc[:, i].to_numpy().reshape(-1, 1)
        if len(Xi) != len(y_vec):
            raise ValueError(
                f"Length mismatch for feature '{fname}': X has {len(Xi)}, y has {len(y_vec)}"
            )
        res = compute_mi_info_fraction(Xi, y_vec, k=k, verbose=True)
        records.append({
            "Target": target_name,
            "Feature": fname,
            "MI": res["MI"],
            "H": res["H"],
            "Info-Fraction": res["Info-Fraction"],
            "Runtime": res["Runtime"]
        })

    # --- collective MI
    res_all = compute_mi_info_fraction(X.to_numpy(), y_vec, k=k, verbose=True)
    records.append({
        "Target": target_name,
        "Feature": "ALL_FEATURES",
        "MI": res_all["MI"],
        "H": res_all["H"],
        "Info-Fraction": res_all["Info-Fraction"],
        "Runtime": res_all["Runtime"]
    })

    results_df = pd.DataFrame(records)

    # --- optional save
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(save_path, index=False)

    return results_df
