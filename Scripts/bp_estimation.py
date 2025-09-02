import h5py
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from scipy.stats import pearsonr
from sklearn.model_selection import KFold, GridSearchCV, GroupShuffleSplit, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from other_functions import bland_altman_plot

class BPModel_LightGBM:
    def __init__(self, data: dict, data_target: dict, target_label: list, features_list: list, default_model: bool = True, limit_data: int = 10000):

        dataframe_X = pd.DataFrame(columns=features_list)
        for p in data.keys():
            p_array = data[p]["mean"].T
            p_array = p_array[:len(p_array)//2]
            col_p = np.full(len(p_array), p, dtype=object)
            p_df = pd.DataFrame(p_array,columns=features_list)
            p_df["patient"] = col_p
            dataframe_X = pd.concat([dataframe_X,p_df],ignore_index=True)
        groups = dataframe_X["patient"].values
        dataframe_X = dataframe_X.drop(columns="patient")

        df_target = pd.DataFrame(columns= target_label)
        for p in data.keys():
            p_array = data_target[p]["Bp_values"].T
            target_label = target_label.insert(-1,"id")
            p_df = pd.DataFrame(p_array,columns = target_label)
            df_target = pd.concat([df_target,p_df],ignore_index=True)
        df_target = df_target.drop(columns="id")

        dataframe_X = dataframe_X[:limit_data]
        groups = groups[:limit_data]
        df_target = df_target[:limit_data]

        self.X = dataframe_X
        self.target = df_target
        self.groups = groups
        self.target_label = target_label
        self.model_setup()
        self.split(valid_set=True)
        if default_model:
            self.error_test, self.error_valid = self.prediction(valid_set=True, shap=True)

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
        # Impute the data since I have a couple of nan values, doing it after the splitting to avoid data leakage
        imputer_X = SimpleImputer(strategy='median')
        X_train = imputer_X.fit_transform(X_train)
        X_test = imputer_X.transform(X_test)

        imputer_y = SimpleImputer(strategy='median')
        y_train = imputer_y.fit_transform(y_train)
        y_test = imputer_y.transform(y_test) 
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        if valid_set:
            a, X_valid, b, y_valid = train_test_split(X_train,y_train,test_size=0.2, random_state=42)
            X_valid = imputer_X.fit_transform(X_valid)
            y_valid = imputer_y.fit_transform(y_valid)
            self.X_valid = X_valid
            self.y_valid = y_valid
            return X_train, y_train, X_test, y_test, X_valid, y_valid

        return X_train, y_train, X_test, y_test
    
    def model_setup(self, parameters: dict = None):

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
    
    def prediction(self, valid_set: bool = False, shap: bool = False):

        multi_model = self.model
        X_train, y_train, X_test, y_test, X_valid, y_valid = self.X_train, self.y_train, self.X_test, self.y_test, self.X_valid, self.y_valid

        # Fit the model
        multi_model.fit(X_train, y_train)
        # Testing
        predictions = multi_model.predict(X_test)
        df_pred_error = self.results(predictions, y_test, self.target_label)

        if shap:
            self.SHAPvalues(self.target_label)
        
        # Validation
        if valid_set:
            predictions_valid = multi_model.predict(X_valid)
            df_valid_error = self.results(predictions_valid, y_valid, self.target_label)
            return df_pred_error, df_valid_error

        return df_pred_error
        
    
    def grid_searchCV(self, param_grid, n_splits: int = 5, score = "neg_mean_squared_error"):

        groups = self.groups
        model = self.model
        X_train = self.X_train
        y_train = self.y_train
        train_idx = self.train_idx

        # Grid Search (tuning fot the best hyperparameters for LGBM model)
        # Note: Since MultiOutputRegressor wraps the estimator, prefix parameters with estimator__ in PARAM_GRID.
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

        self.model = best_model

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
        r2 = r2_score(true,pred)
        df = pd.DataFrame([me,stdme,mae,stdmae,mse,stdmse,rmse,r,r2],index=["ME","stdME","MAE","stdMAE","MSE","stdMSE","RMSE","R Pearson","R^2"])
        
        print(f"Errors of {label} \nME: {me:.3f} ± {stdme:.3f} \nMAE: {mae:.3f} ± {stdmae:.3f} \nMSE: {mse:.3f} ± {stdmse:.3f} \nRMSE: {rmse:.3f} \nPearson: {r:.3f} \nR^2: {r2:.3f}")
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
        print("Processing SHAP values:")
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