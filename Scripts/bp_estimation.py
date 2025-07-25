import h5py
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.model_selection import KFold, GridSearchCV, GroupShuffleSplit, train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

data_path = 'C:/Users/adhn565/Documents/Data/completo_conAttrs_16_7_25.h5'
data_path_target = 'BP_values.h5'
data = {}
data_target = {}
segment_ids = {}

def evaluate(true, pred, label):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    
    print(f"{label} — MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")

def multioutput_mae(y_true, y_pred):
    return np.mean([
        mean_absolute_error(y_true[:, i], y_pred[:, i])
        for i in range(y_true.shape[1])
    ])

def SHAPvalues(target_labels: list, model):
    multi_model = model
    for i in range(len(target_labels)):
        # Extract the i-th model from the MultiOutputRegressor
        model_i = multi_model.estimators_[i]
        # Use TreeExplainer (fast & efficient for LightGBM)
        explainer = shap.Explainer(model_i)
        shap_values = explainer(dataframe_X[:10000])
        # Summary plot
        shap.summary_plot(shap_values, dataframe_X[:10000], feature_names=dataframe_X.columns)

with h5py.File(data_path, 'r') as f:
    for group_name in f:
        group = f[group_name]
        data[group_name] = {}
        for dtset_name in group:
            data[group_name][dtset_name] = group[dtset_name][()]
            data[group_name][dtset_name] = data[group_name][dtset_name]
        segment_ids[group_name] = group["segments"][0]
    
    fiducial = f[group_name]["segments"].attrs['fiducial_order']
    features = f[group_name][f"mean_{group_name}"].attrs['features']
    fiducial = [f.decode() if isinstance(f, bytes) else f for f in fiducial]
    features = [f.decode() if isinstance(f, bytes) else f for f in features]

with h5py.File(data_path_target, 'r') as f:
    for group_name in f:
        group = f[group_name]
        data_target[group_name] = {}
        for dtset_name in group:
            data_target[group_name][dtset_name] = group[dtset_name][()]
            data_target[group_name][dtset_name] = data_target[group_name][dtset_name]

dataframe_X = pd.DataFrame(columns=features)
for p in data.keys():
    p_array = data[p][f"mean_{p}"].T
    p_array = p_array[:len(p_array)//2]
    col_p = np.full(len(p_array), p, dtype=object)
    p_df = pd.DataFrame(p_array,columns=features)
    p_df["patient"] = col_p
    dataframe_X = pd.concat([dataframe_X,p_df],ignore_index=True)
groups = dataframe_X["patient"].values
dataframe_X = dataframe_X.drop(columns="patient")

target_label = ["SBP","DBP","MAP"]
df_target = pd.DataFrame(columns= target_label)
for p in data.keys():
    p_array = data_target[p]["Bp_values"].T
    p_df = pd.DataFrame(p_array,columns= target_label)
    df_target = pd.concat([df_target,p_df],ignore_index=True)

dataframe_X = dataframe_X[:10000]
groups = groups[:10000]
df_target = df_target[:10000]
print(dataframe_X.shape,groups.shape,df_target.shape)

# Splitting the data patient wise to avoid data leakage
gss = GroupShuffleSplit(n_splits=1, train_size=0.8, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(dataframe_X, df_target, groups=groups))
X_train, X_test = dataframe_X.iloc[train_idx], dataframe_X.iloc[test_idx]
y_train, y_test = df_target.iloc[train_idx], df_target.iloc[test_idx]

# Impute the data since I have a couple of nan values, doing it after the splitting to avoid data leakage
imputer_X = SimpleImputer(strategy='median')
X_train = imputer_X.fit_transform(X_train)
X_test = imputer_X.transform(X_test)

imputer_y = SimpleImputer(strategy='median')
y_train = imputer_y.fit_transform(y_train)
y_test = imputer_y.transform(y_test) 

# Model for multiple targets
model = lgb.LGBMRegressor(random_state= 42)
multi_model = MultiOutputRegressor(model)

# Grid Search (tuning fot the best hyperparameters for my LGBM model)
# Note: Since MultiOutputRegressor wraps the estimator, prefix parameters with estimator__.
param_grid = {
    'estimator__learning_rate': [0.05, 0.1],
    'estimator__n_estimators': [500, 1000],
    'estimator__max_depth': [10, 15]
}

# Run cross-validation
mae_scorer = make_scorer(multioutput_mae, greater_is_better=False) # makes our own score function that averages the MAE for the 3 targets
cv = KFold(n_splits=5, shuffle=True, random_state=42)
# scores = cross_val_score(multi_model, X_train, y_train, scoring=mae_scorer, cv=cv)
grid_search = GridSearchCV(
    estimator=multi_model,
    param_grid=param_grid,
    scoring=mae_scorer,  # or 'r2', 'neg_mean_squared_error'
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train, groups=groups[train_idx])

print("Best parameters:", grid_search.best_params_)
print("Best MAE:", -grid_search.best_score_)

# best_model = grid_search.best_estimator_
# y_pred = best_model.predict(X_test)

# Report results
# print("Cross-validation MAE scores (negated):", -scores)
# print("Mean MAE:", -np.mean(scores))
# print("Standard deviation:", np.std(scores))

# multi_model.fit(X_train, y_train)

# predictions = multi_model.predict(X_test)
# sbp_pred = predictions[:, 0]
# dbp_pred = predictions[:, 1]
# map_pred = predictions[:, 2]

# evaluate(y_test[:, 0], sbp_pred, "SBP")
# evaluate(y_test[:, 1], dbp_pred, "DBP")
# evaluate(y_test[:, 2], map_pred, "MAP")

# SHAPvalues(target_label,multi_model)

# for i in range(len(target_label)):
#     plt.figure()
#     plt.scatter(y_test[:, i], predictions[:, i], alpha=0.5)
#     plt.plot([y_test[:, i].min(), y_test[:, i].max()],
#              [y_test[:, i].min(), y_test[:, i].max()],
#              'r--')
#     plt.xlabel("True")
#     plt.ylabel("Predicted")
#     plt.title(f"{target_label[i]}: Predicted vs True")
#     plt.grid(True)
#     plt.show()

# param_grid = {
#     'estimator__num_leaves': [31, 50],
#     'estimator__learning_rate': [0.01, 0.05],
#     'estimator__n_estimators': [100, 200]
# }
# grid = GridSearchCV(
#     MultiOutputRegressor(LGBMRegressor(random_state=42)),
#     param_grid,
#     cv=3,
#     scoring='neg_mean_absolute_error',
#     verbose=1,
#     n_jobs=-1
# )



