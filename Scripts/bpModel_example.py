import h5py
import numpy as np
import pandas as pd
import lightgbm as lgb
from bp_estimation import BPModel_LightGBM

# This script demonstrates how to use the BPModel_LightGBM class for blood pressure estimation.
data_path = 'C:/Users/adhn565/Documents/Data/clean_6_8_2025.h5'
data_path_target = 'BP_values.h5'
data = {}
data_target = {}
target_label = ["SBP","DBP","MAP"]

with h5py.File(data_path, 'r') as f:
    for group_name in f:
        group = f[group_name]
        data[group_name] = {}
        for dtset_name in group:
            data[group_name][dtset_name] = group[dtset_name][()]
            data[group_name][dtset_name] = data[group_name][dtset_name]
    
    fiducial = f[group_name]["segments"].attrs['fiducial_order']
    features = f[group_name]["mean"].attrs['features']
    fiducial = [f.decode() if isinstance(f, bytes) else f for f in fiducial]
    features = [f.decode() if isinstance(f, bytes) else f for f in features]

with h5py.File(data_path_target, 'r') as f:
    for group_name in f:
        group = f[group_name]
        data_target[group_name] = {}
        for dtset_name in group:
            data_target[group_name][dtset_name] = group[dtset_name][()]
            data_target[group_name][dtset_name] = data_target[group_name][dtset_name]

# Initialize the BPModel_LightGBM with the data and target
# Note: The default_model parameter is set to True to use the default model setup and perform initial predictions.
bp = BPModel_LightGBM(data,data_target,target_label= target_label,features_list=features,default_model=True)
errors_test, errors_valid = bp.error_test, bp.error_valid

""" You can do your own splitting of the data.
You can also use the functions to change the model however you want after calling the class
and update the model with the function model_setup(), use prediction() and results() to test the model
There is a function for feature ranking using the SHAP values
"""
# bp.split(test_size=0.2, n_split=1, valid_set=True)

# # Example of how to use the grid search for hyperparameter tuning
# # Note: The function grid_searchCV will return the best model, best parameters and the grid search object.
# # The grid_searchCV function will automatically update the model with the best parameters so you can use prediction directly after it.
# param_grid = {
#         'estimator__learning_rate': [0.05, 0.1],
#         'estimator__n_estimators': [400, 800],
#         'estimator__max_depth': [-1, 15],
#         'estimator__num_leaves': [50]
#     }
# bp.grid_searchCV(param_grid=param_grid,n_splits=2,score="neg_mean_squared_error")
# errors_test, errors_valid = bp.prediction(valid_set=True)

# # Example of how to use the model for own parameters and prediction
# # Note: The parameters should be a dictionary with the same keys as the default parameters.
# parameters = {
#         "random_state": 42,
#         "n_estimators": 800,
#         "learning_rate": 0.05,
#         "max_depth": -1,
#         "num_leaves": 50
#         }
# bp.model_setup(parameters=parameters)
# errors_test, errors_valid = bp.prediction(valid_set=True)