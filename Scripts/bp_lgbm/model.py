######## MODEL SCRIPT ################################################################
#                                                                                    #
# This script handles the trainingm, the evaluation, the grid search,                #
# and eventually the cross validation                                                #
#                                                                                    #
######################################################################################

# lgbm_pipeline.py
# Simple preprocessing + LightGBM training + evaluation on dummy data
# Metrics: MAE (+SD), ME (+SD), RMSE, MSE, R2, absolute errors, Bland–Altman plot

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from lightgbm import LGBMRegressor
except ImportError:
    raise SystemExit(
        "LightGBM is not installed. Install it with:\n"
        "  pip install lightgbm"
    )

# ----------------------------
# Config
# ----------------------------
@dataclass # Decorator only to create a class that holds data. (this creates the builder and so on automatically)
class Config:
    random_state: int = 42
    n_samples: int = 2000
    n_features: int = 20
    test_size: float = 0.2
    val_size: float = 0.1  # of the remaining train split
    scale_features: bool = True
    # LightGBM hyperparameters (tweak as needed)
    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 500
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0

# ----------------------------
# Data Generation (Dummy)
# ----------------------------
def make_dummy_regression(cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a dummy regression dataset.
    """
    X, y = make_regression(
        n_samples=cfg.n_samples,
        n_features=cfg.n_features,
        n_informative=12,
        noise=12.0,
        random_state=cfg.random_state,
    )
    # Optionally shift/scale target to resemble a physiological range (e.g., SBP 80–180)
    y = (y - y.min()) / (y.max() - y.min())  # [0,1]
    y = 80 + y * 100  # ~[80, 180]
    return X.astype(np.float32), y.astype(np.float32)

# ----------------------------
# Model
# ----------------------------
def build_lgbm(cfg: Config) -> LGBMRegressor:
    return LGBMRegressor(
        num_leaves=cfg.num_leaves,
        learning_rate=cfg.learning_rate,
        n_estimators=cfg.n_estimators,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        random_state=cfg.random_state,
        n_jobs=-1,
    )

# ----------------------------
# Training
# ----------------------------
def train_model(model: LGBMRegressor, X_train, y_train, X_val, y_val) -> LGBMRegressor:
    model.fit(
        X_train,
        y_train,
        eval_set=[("val", X_val, y_val)],
        eval_metric="l2",  # MSE
        verbose=False,
    )
    return model

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


# ----------------------------
# Main
# ----------------------------
def main():
    cfg = Config()

    # 1) Dummy data
    X, y = make_dummy_regression(cfg)

    # 2) Preprocess (split + optional scale)
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = preprocess_split(X, y, cfg)

    # 3) Model
    model = build_lgbm(cfg)

    # 4) Train
    model = train_model(model, X_train, y_train, X_val, y_val)

    # 5) Evaluate on test
    y_pred = model.predict(X_test)
    metrics = evaluate(y_test, y_pred)

    # Print neat summary
    print("\n=== Evaluation (Test Set) ===")
    for k in [
        "MAE", "MAE_SD", "ME", "ME_SD", "MSE", "RMSE", "R2",
        "AbsError_mean", "AbsError_std", "AbsError_min", "AbsError_max",
        "BA_bias_ME", "BA_sd_diff", "BA_loa_low", "BA_loa_high",
        "BA_plot_path",
    ]:
        print(f"{k:>14}: {metrics[k]:.4f}" if isinstance(metrics[k], float) else f"{k:>14}: {metrics[k]}")

    # If you also want the absolute error vector for downstream analysis:
    # (Recompute here to avoid storing a huge array in 'metrics' by default)
    abs_errors = np.abs(y_pred - y_test)
    # Example: show first 10 absolute errors
    print("\nFirst 10 absolute errors:", np.round(abs_errors[:10], 3))

if __name__ == "__main__":
    main()