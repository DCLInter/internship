######## EVAL SCRIPT #################################################################
#                                                                                    #
# This script handles the model eval with all of the apropriate metrics              #
# for BP estimation evaluation according to Elgendi (2024                            #
#                                                                                    #
######################################################################################
"""
Citation:

Elgendi, M., Haugg, F., Fletcher, R.R. et al. 
Recommendations for evaluating photoplethysmography-based algorithms for blood pressure assessment. 
Commun Med 4, 140 (2024). https://doi.org/10.1038/s43856-024-00555-2
"""

# Metrics: MAE (+SD), ME (+SD), RMSE, MSE, R2, absolute errors, Bland–Altman plot (7 metrics)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Tuple

# ----------------------------
# Evaluation
# ----------------------------
def bland_altman_plot(y_true: np.ndarray, y_pred: np.ndarray, path: str = "bland_altman.png") -> Dict[str, float]:
    """
    Bland–Altman: difference vs mean, with bias and 95% LoA.
    Returns bias and LoA stats; saves plot to file.
    """
    diffs = y_pred - y_true
    means = (y_pred + y_true) / 2.0

    bias = np.mean(diffs)
    sd = np.std(diffs, ddof=1)
    loa_low = bias - 1.96 * sd
    loa_high = bias + 1.96 * sd

    plt.figure(figsize=(6, 5), dpi=140)
    plt.scatter(means, diffs, alpha=0.5, s=12)
    plt.axhline(bias, linestyle="--", linewidth=1.5, label=f"Bias = {bias:.2f}")
    plt.axhline(loa_low, linestyle=":", linewidth=1.5, label=f"LoA low = {loa_low:.2f}")
    plt.axhline(loa_high, linestyle=":", linewidth=1.5, label=f"LoA high = {loa_high:.2f}")
    plt.xlabel("Mean of prediction and reference")
    plt.ylabel("Prediction − Reference")
    plt.title("Bland–Altman Plot")
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    return {
        "bias_ME": float(bias),
        "sd_diff": float(sd),
        "loa_low": float(loa_low),
        "loa_high": float(loa_high),
        "plot_path": path,
    }

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute requested metrics:
    - MAE (+SD of |error|)
    - ME (+SD of signed error)
    - RMSE, MSE
    - R2
    - Absolute errors (returned as array for further analysis)
    - Bland–Altman stats (also saves a plot)
    """
    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    mae = mean_absolute_error(y_true, y_pred)
    mae_sd = float(np.std(abs_errors, ddof=1))

    me = float(np.mean(errors))
    me_sd = float(np.std(errors, ddof=1))

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))

    r2 = r2_score(y_true, y_pred)

    ba_stats = bland_altman_plot(y_true, y_pred, path="bland_altman.png")

    metrics = {
        "MAE": float(mae),
        "MAE_SD": float(mae_sd),
        "ME": me,
        "ME_SD": me_sd,
        "MSE": float(mse),
        "RMSE": rmse,
        "R2": float(r2),
        "AbsError_mean": float(np.mean(abs_errors)),
        "AbsError_std": float(np.std(abs_errors, ddof=1)),
        "AbsError_min": float(np.min(abs_errors)),
        "AbsError_max": float(np.max(abs_errors)),
        # Bland–Altman
        "BA_bias_ME": ba_stats["bias_ME"],
        "BA_sd_diff": ba_stats["sd_diff"],
        "BA_loa_low": ba_stats["loa_low"],
        "BA_loa_high": ba_stats["loa_high"],
        "BA_plot_path": ba_stats["plot_path"],
    }
    return metrics