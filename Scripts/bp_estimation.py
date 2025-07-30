import h5py
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from scipy.stats import pearsonr
from sklearn.model_selection import KFold, GridSearchCV, GroupShuffleSplit, train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from other_functions import bland_altman_plot

def evaluate(true, pred, label):
    me = np.mean(true - pred)
    stdme = np.std(true - pred)
    mae = mean_absolute_error(true, pred)
    stdmae = np.std(abs(true - pred))
    mse = mean_squared_error(true, pred)
    stdmse = np.std((true - pred)**2)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r,_ = pearsonr(true, pred)
    abs_error = abs(true - pred)
    df = pd.DataFrame([me,stdme,mae,stdmae,mse,stdmse,rmse,r],index=["ME","stdME","MAE","stdMAE","MSE","stdMSE","RMSE","R Pearson"])

    # bland_altman_plot(true, pred, label)
    print(f"{label} \nME: {me:.2f} ± {stdme} \nMAE: {mae:.2f} ± {stdmae} \nMSE: {mse:.3f} ± {stdmse} \nRMSE: {rmse:.2f} \nPearson: {r:.2f}")

    return df

def results(predict, real, targets):
    df_pred_error = pd.DataFrame(columns=targets, index=["ME","stdME","MAE","stdMAE","MSE","stdMSE","RMSE","R Pearson"])
    for t in np.arange(len(targets)):
        pred = predict[:, t]
        d = evaluate(real[:, t], pred, targets[t])
        df_pred_error[targets[t]] = d
    
    return df_pred_error


def SHAPvalues(target_labels: list, model):
    multi_model = model
    for i in range(len(target_labels)):
        # Extract the i-th model from the MultiOutputRegressor
        model_i = multi_model.estimators_[i]
        # Use TreeExplainer (fast & efficient for LightGBM)
        explainer = shap.Explainer(model_i)
        shap_values = explainer(dataframe_X[:1000])
        # Summary plot
        shap.summary_plot(shap_values, dataframe_X[:1000], feature_names=dataframe_X.columns, show = False)
        plt.title(f"SHAP Plot for {target_labels[i]}", fontsize=14)
        plt.savefig(f"{target_labels[i]}_shap_summary_plot.png", dpi=300, bbox_inches='tight')
        plt.close()

def grid_searchCV(model, param_grid, n_splits, X_train, y_train, score):
    # Grid Search (tuning fot the best hyperparameters for LGBM model)
    # Note: Since MultiOutputRegressor wraps the estimator, prefix parameters with estimator__.
    cv = KFold(n_splits= n_splits, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator= model,
        param_grid= param_grid,
        scoring= score,  # or 'r2', 'neg_mean_squared_error'
        cv= cv,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train, groups=groups[train_idx])

    best_idx = grid_search.best_index_
    std = grid_search.cv_results_['std_test_score'][best_idx]
    print("Best parameters:", grid_search.best_params_)
    print(f"Best {score}:", -grid_search.best_score_,"+",std)
    best_param = grid_search.best_params_
    best_model = grid_search.best_estimator_

    return best_model, best_param, grid_search 



data_path = 'C:/Users/adhn565/Documents/Data/data_clean_features.h5'
data_path_target = 'BP_values.h5'
data = {}
data_target = {}
segment_ids = {}

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
a, X_valid, b, y_valid = train_test_split(X_train,y_train,test_size=0.2, random_state=42)

# Impute the data since I have a couple of nan values, doing it after the splitting to avoid data leakage
imputer_X = SimpleImputer(strategy='median')
X_train = imputer_X.fit_transform(X_train)
X_test = imputer_X.transform(X_test)
X_valid = imputer_X.fit_transform(X_valid)

imputer_y = SimpleImputer(strategy='median')
y_train = imputer_y.fit_transform(y_train)
y_test = imputer_y.transform(y_test) 
y_valid = imputer_y.fit_transform(y_valid)

# Model for multiple targets
model = lgb.LGBMRegressor(random_state= 42, num_leaves= 50, learning_rate= 0.05, max_depth= -1, n_estimators= 800)
multi_model = MultiOutputRegressor(model)

param_grid = {
        'estimator__learning_rate': [0.05, 0.1],
        'estimator__n_estimators': [400, 800],
        # 'estimator__max_depth': [-1, 15],
        # 'estimator__num_leaves': [50]
    }

# multi_model, best_parameters, grid_search = grid_searchCV(multi_model, param_grid, 5, X_train, y_train, "neg_mean_squared_error")

# Run cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
score = "neg_mean_absolute_error"
scores = cross_val_score(multi_model, X_train, y_train, scoring= score, cv=cv)
print(f"Cross-validation {score} scores (negated):", -scores, np.std(scores))

# Fit the model
multi_model.fit(X_train, y_train)
# Testing
predictions = multi_model.predict(X_test)
# Validation
predictions_valid = multi_model.predict(X_valid)

# Report results
# Testing
df_pred_error = results(predictions, y_test, target_label)

# Validation
df_valid_error = results(predictions_valid, y_valid, target_label)

print(df_pred_error, df_valid_error)
# SHAPvalues(target_label,multi_model)