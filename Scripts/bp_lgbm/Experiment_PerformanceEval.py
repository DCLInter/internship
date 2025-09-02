############ PERFORMANCE EVALUATION SCRIPT ####################
#                                                             #
# Here, the evaluation of the lgbm model will be done.        #
#                                                             #
###############################################################
import numpy as np
import local_paths
import preprocessing
import cv
import gs
import mutual_information as mi
import shap_analysis as sa
from data import load_patient_dataset, load_group_attributes
from models import build_lgbm
from config import ExperimentConfig, lightGBM_default_params, lightGBM_baseline_grid_3target, save_config, lightGBM_best_guess_1, lightGBM_small_grid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error as mse
from pathlib import Path
from h5_inspector import inspect_file


if __name__ == "__main__":

    # =========================================================
    # Paths
    #==========================================================
    data_path = local_paths.DATA_DIR /"features_cleaned.h5"
    labels_path = local_paths.LABELS_DIR / "BP_values.h5"
    data_messy_path = local_paths.DATA_DIR /"features_patients.h5"

    # =========================================================
    # Load Data
    #==========================================================
    inspect_file(data_messy_path, show_attrs=True)
    df_feat_orig_mean = load_patient_dataset(data_messy_path, dataset_type="mean")
    print(df_feat_orig_mean.head())