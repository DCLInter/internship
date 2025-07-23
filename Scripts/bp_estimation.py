import h5py
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputRegressor

data_path = 'C:/Users/adhn565/Documents/Data/completo_conAttrs_16_7_25.h5'
data_target = 'BP_values.h5'
data = {}
segment_ids = {}

# with h5py.File(data_path, 'r') as f:
#     for group_name in f:
#         group = f[group_name]
#         data[group_name] = {}
#         for dtset_name in group:
#             data[group_name][dtset_name] = group[dtset_name][()]
#             data[group_name][dtset_name] = data[group_name][dtset_name]
#         segment_ids[group_name] = group["segments"][0]
    
#     fiducial = f["p000001"]["segments"].attrs['fiducial_order']
#     features = f["p000001"]["mean_p000001"].attrs['features']
#     fiducial = [f.decode() if isinstance(f, bytes) else f for f in fiducial]
#     features = [f.decode() if isinstance(f, bytes) else f for f in features]

datasource = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(datasource.data, datasource.target, test_size=0.2)
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
