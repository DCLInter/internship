import h5py
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
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

with h5py.File(data_path, 'r') as f:
    for group_name in f:
        group = f[group_name]
        data[group_name] = {}
        for dtset_name in group:
            data[group_name][dtset_name] = group[dtset_name][()]
            data[group_name][dtset_name] = data[group_name][dtset_name]
        segment_ids[group_name] = group["segments"][0]
    
    fiducial = f[group_name]["segments"].attrs['fiducial_order']
    features = f["p000001"]["mean_p000001"].attrs['features']
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
    p_df = pd.DataFrame(p_array,columns=features)
    dataframe_X = pd.concat([dataframe_X,p_df],ignore_index=True)

target_label = ["SBP","DBP","MAP"]
df_target = pd.DataFrame(columns= target_label)
for p in data.keys():
    p_array = data_target[p]["Bp_values"].T
    p_df = pd.DataFrame(p_array,columns= target_label)
    df_target = pd.concat([df_target,p_df],ignore_index=True)

imputer_X = SimpleImputer(strategy='median')
X = imputer_X.fit_transform(dataframe_X)
imputer_y = SimpleImputer(strategy='median')
y = imputer_y.fit_transform(df_target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = lgb.LGBMRegressor(random_state= 42, max_depth= 15, learning_rate= 0.05, n_estimators= 1000)
multi_model = MultiOutputRegressor(model)
multi_model.fit(X_train, y_train)

predictions = multi_model.predict(X_test)
sbp_pred = predictions[:, 0]
dbp_pred = predictions[:, 1]
map_pred = predictions[:, 2]

evaluate(y_test[:, 0], sbp_pred, "SBP")
evaluate(y_test[:, 1], dbp_pred, "DBP")
evaluate(y_test[:, 2], map_pred, "MAP")

for i in range(len(target_label)):
    # Extract the i-th model from the MultiOutputRegressor
    model_i = multi_model.estimators_[i]
    # Use TreeExplainer (fast & efficient for LightGBM)
    explainer = shap.Explainer(model_i)
    shap_values = explainer(X[:10000])
    # Summary plot
    shap.summary_plot(shap_values, X[:10000], feature_names=dataframe_X.columns)

for i in range(len(target_label)):
    plt.figure()
    plt.scatter(y_test[:, i], predictions[:, i], alpha=0.5)
    plt.plot([y_test[:, i].min(), y_test[:, i].max()],
             [y_test[:, i].min(), y_test[:, i].max()],
             'r--')
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"{target_label[i]}: Predicted vs True")
    plt.grid(True)
    plt.show()

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



