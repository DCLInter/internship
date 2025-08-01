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

class BPModel_LightGBM:
    def __init__(self, data: dict, data_target: dict, target_label: list, limit_data: int = 10000):

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

        df_target = pd.DataFrame(columns= target_label)
        for p in data.keys():
            p_array = data_target[p]["Bp_values"].T
            p_df = pd.DataFrame(p_array,columns= target_label)
            df_target = pd.concat([df_target,p_df],ignore_index=True)

        dataframe_X = dataframe_X[:limit_data]
        groups = groups[:limit_data]
        df_target = df_target[:limit_data]

        self.X = dataframe_X
        self.target = df_target
        self.groups = groups
        self.target_label = target_label

        # Splitting the data patient wise to avoid data leakage
    def split(self, test_size: int = 0.2, n_split: int = 1, valid_set: bool = False):
        dataframe_X = self.X
        df_target = self.target
        groups = self.groups

        gss = GroupShuffleSplit(n_splits=n_split, test_size=test_size, random_state=42)
        train_idx, test_idx = next(gss.split(dataframe_X, df_target, groups=groups))
        X_train, X_test = dataframe_X.iloc[train_idx], dataframe_X.iloc[test_idx]
        y_train, y_test = df_target.iloc[train_idx], df_target.iloc[test_idx]

        self.train_idx = train_idx
        self.test_idx = test_idx
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        if valid_set:
            a, X_valid, b, y_valid = train_test_split(X_train,y_train,test_size=0.2, random_state=42)
            self.X_valid = X_valid
            self.y_valid = y_valid
            return X_train, y_train, X_test, y_test, X_valid, y_valid

        return X_train, y_train, X_test, y_test
    
    def model_setup(self, parameters: dict = None, valid_set: bool = False):
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test

        # Impute the data since I have a couple of nan values, doing it after the splitting to avoid data leakage
        imputer_X = SimpleImputer(strategy='median')
        X_train = imputer_X.fit_transform(X_train)
        X_test = imputer_X.transform(X_test)

        imputer_y = SimpleImputer(strategy='median')
        y_train = imputer_y.fit_transform(y_train)
        y_test = imputer_y.transform(y_test) 
        
        if valid_set:
            X_valid = self.X_valid
            y_valid = self.y_valid
            X_valid = imputer_X.fit_transform(X_valid)
            y_valid = imputer_y.fit_transform(y_valid)
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_valid = X_valid
        self.y_valid = y_valid

        default_params = {
        "random_state": 42,
        "n_estimators": 800,
        "learning_rate": 0.05,
        "max_depth": -1,
        "num_leaves": 50
        }

        if parameters:
            default_params.update(parameters)

        # Model for multiple targets
        model = lgb.LGBMRegressor(**default_params)
        multi_model = MultiOutputRegressor(model)
        self.model = multi_model
        
        return multi_model
    
    def prediction_default(self, valid_set: bool = True):

        self.split(valid_set=valid_set)
        multi_model = self.model_setup(valid_set=valid_set)
        X_train, y_train, X_test, y_test, X_valid, y_valid = self.X_train, self.y_train, self.X_test, self.y_test, self.X_valid, self.y_valid

        # Fit the model
        multi_model.fit(X_train, y_train)
        # Testing
        predictions = multi_model.predict(X_test)
        # Validation
        predictions_valid = multi_model.predict(X_valid)

        # Report results
        # Testing
        df_pred_error = self.results(predictions, y_test, target_label)
        # Validation
        df_valid_error = self.results(predictions_valid, y_valid, target_label)

        self.SHAPvalues(self.target_label)

        return df_pred_error, df_valid_error
        
    
    def grid_searchCV(self, param_grid, n_splits: int = 5, score: str = "neg_mean_squared_error"):
        groups = self.groups
        model = self.model
        X_train = self.X_train
        y_train = self.y_train
        train_idx = self.train_idx

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

    def evaluate(self, true, pred, label):

        me = np.mean(true - pred)
        stdme = np.sqrt(( 1 / (len(true)-1) ) * np.sum( ((pred - true) - me)**2 ))
        mae = mean_absolute_error(true, pred)
        stdmae = np.sqrt(( 1 / (len(true)-1) ) * np.sum( (abs(pred - true) - mae)**2 ))
        mse = mean_squared_error(true, pred)
        stdmse = np.sqrt(( 1 / (len(true)-1) ) * np.sum( ((pred - true)**2 - mse)**2 ))
        rmse = np.sqrt(mean_squared_error(true, pred))
        r,_ = pearsonr(true, pred)
        df = pd.DataFrame([me,stdme,mae,stdmae,mse,stdmse,rmse,r],index=["ME","stdME","MAE","stdMAE","MSE","stdMSE","RMSE","R Pearson"])
        
        print(f"Errors of {label} \nME: {me:.3f} ± {stdme:.3f} \nMAE: {mae:.3f} ± {stdmae:.3f} \nMSE: {mse:.3f} ± {stdmse:.3f} \nRMSE: {rmse:.3f} \nPearson: {r:.3f}")
        bland_altman_plot(true, pred, label)

        return df

    def results(self, predict, real, targets):
        df_pred_error = pd.DataFrame(columns=targets, index=["ME","stdME","MAE","stdMAE","MSE","stdMSE","RMSE","R Pearson"])
        for t in np.arange(len(targets)):
            pred = predict[:, t]
            d = self.evaluate(real[:, t], pred, targets[t])
            df_pred_error[targets[t]] = d
        
        return df_pred_error

    def SHAPvalues(self, target_labels: list):
        X = self.X
        multi_model = self.model
        for i in range(len(target_labels)):
            # Extract the i-th model from the MultiOutputRegressor
            model_i = multi_model.estimators_[i]
            # Use TreeExplainer (fast & efficient for LightGBM)
            explainer = shap.Explainer(model_i)
            shap_values = explainer(X[:1000])
            # Summary plot
            shap.summary_plot(shap_values, X[:1000], feature_names=X.columns, show = False)
            plt.title(f"SHAP Plot for {target_labels[i]}", fontsize=14)
            plt.savefig(f"{target_labels[i]}_shap_summary_plot.png", dpi=300, bbox_inches='tight')
            plt.close()


data_path = 'C:/Users/adhn565/Documents/Data/data_clean_features.h5'
data_path_target = 'BP_values.h5'
data = {}
data_target = {}
segment_ids = {}
target_label = ["SBP","DBP","MAP"]

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

bp = BPModel_LightGBM(data,data_target,target_label= target_label)
param_grid = {
        'estimator__learning_rate': [0.05, 0.1],
        'estimator__n_estimators': [400, 800],
        # 'estimator__max_depth': [-1, 15],
        # 'estimator__num_leaves': [50]
    }
bp.grid_searchCV(param_grid=param_grid,n_splits=2)
# d1, d2 = bp.prediction()

# dataframe_X = pd.DataFrame(columns=features)
# for p in data.keys():
#     p_array = data[p][f"mean_{p}"].T
#     p_array = p_array[:len(p_array)//2]
#     col_p = np.full(len(p_array), p, dtype=object)
#     p_df = pd.DataFrame(p_array,columns=features)
#     p_df["patient"] = col_p
#     dataframe_X = pd.concat([dataframe_X,p_df],ignore_index=True)
# groups = dataframe_X["patient"].values
# dataframe_X = dataframe_X.drop(columns="patient")

# target_label = ["SBP","DBP","MAP"]
# df_target = pd.DataFrame(columns= target_label)
# for p in data.keys():
#     p_array = data_target[p]["Bp_values"].T
#     p_df = pd.DataFrame(p_array,columns= target_label)
#     df_target = pd.concat([df_target,p_df],ignore_index=True)

# dataframe_X = dataframe_X[:10000]
# groups = groups[:10000]
# df_target = df_target[:10000]
# print(dataframe_X.shape,groups.shape,df_target.shape)

# # Splitting the data patient wise to avoid data leakage
# gss = GroupShuffleSplit(n_splits=1, train_size=0.8, test_size=0.2, random_state=42)
# train_idx, test_idx = next(gss.split(dataframe_X, df_target, groups=groups))
# X_train, X_test = dataframe_X.iloc[train_idx], dataframe_X.iloc[test_idx]
# y_train, y_test = df_target.iloc[train_idx], df_target.iloc[test_idx]
# a, X_valid, b, y_valid = train_test_split(X_train,y_train,test_size=0.2, random_state=42)

# # Impute the data since I have a couple of nan values, doing it after the splitting to avoid data leakage
# imputer_X = SimpleImputer(strategy='median')
# X_train = imputer_X.fit_transform(X_train)
# X_test = imputer_X.transform(X_test)
# X_valid = imputer_X.fit_transform(X_valid)

# imputer_y = SimpleImputer(strategy='median')
# y_train = imputer_y.fit_transform(y_train)
# y_test = imputer_y.transform(y_test) 
# y_valid = imputer_y.fit_transform(y_valid)

# # Model for multiple targets
# model = lgb.LGBMRegressor(random_state= 42, num_leaves= 50, learning_rate= 0.05, max_depth= -1, n_estimators= 800)
# multi_model = MultiOutputRegressor(model)

# param_grid = {
#         'estimator__learning_rate': [0.05, 0.1],
#         'estimator__n_estimators': [400, 800],
#         # 'estimator__max_depth': [-1, 15],
#         # 'estimator__num_leaves': [50]
#     }

# # multi_model, best_parameters, grid_search = grid_searchCV(multi_model, param_grid, 5, X_train, y_train, "neg_mean_squared_error")

# # Run cross-validation
# # cv = KFold(n_splits=5, shuffle=True, random_state=42)
# # score = "neg_mean_absolute_error"
# # scores = cross_val_score(multi_model, X_train, y_train, scoring= score, cv=cv)
# # print(f"Cross-validation {score} scores (negated):", -scores, np.std(scores))

# # Fit the model
# multi_model.fit(X_train, y_train)
# # Testing
# predictions = multi_model.predict(X_test)
# # Validation
# predictions_valid = multi_model.predict(X_valid)

# # Report results
# # Testing
# df_pred_error = results(predictions, y_test, target_label)

# # Validation
# df_valid_error = results(predictions_valid, y_valid, target_label)

# # print(df_pred_error, df_valid_error)
# # SHAPvalues(target_label,multi_model)