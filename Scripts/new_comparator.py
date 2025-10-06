import h5py
import numpy as np
import pandas as pd
from comparator import Comparator

path_features = "D:/U/Practicas_City_University_of_London/Data/Features_VitalDB_Train_Subset.h5"
path_cleaned = "D:/U/Practicas_City_University_of_London/Data/Clean_Features_VitalDB_Train_Subset.h5"
filepath_results = "D:/U/Practicas_City_University_of_London/Data"
data = {}
with h5py.File(path_features, 'r') as f:
    for group_name in f:
        #print(group_name)
        obj = f[group_name]
        if isinstance(obj, h5py.Dataset):
            data[group_name] = f[group_name][:]
        elif isinstance(obj, h5py.Group):
            group = f[group_name]
            data[group_name] = {}
            for dst in group:
                data[group_name][dst] = group[dst][:]

data_clean = {}
with h5py.File(path_cleaned, 'r') as f:
    for group_name in f:
        obj = f[group_name]
        #print(group_name)
        if isinstance(obj, h5py.Dataset):
            data_clean[group_name] = f[group_name][:]
        elif isinstance(obj, h5py.Group):
            group = f[group_name]
            data_clean[group_name] = {}
            for dst in group:
                data_clean[group_name][dst] = group[dst][:]

data_ext = {}
data_ext["Full_set"] = data["PPG_features"]["Features"]
#print(data_ext["Full_set"][0].shape)
data_clean_ext = {}
#print(data_clean)
data_clean_ext["Full_set"] = data_clean["PPG_Features"]
#print(data_clean_ext["Full_set"])

features_names = ["IPR", "Tsp", "TWRRF25", "TWRRF50", "Tsw25", "Tsw50", "Tsw75", "Tdw25", "Tdw50", "Tdw75", "AUCpi", "IPA",  "Av-Au ratio", "Ab-Aa ratio", "Ac-Aa ratio", "Ad-Aa ratio", "Ap2-Ap1 ratio", "AGI", "Kurtosis", "Skewness", "L-H ratio", "ShannonEntropy", "Tpp"]

comp = Comparator(data_ext, data_clean_ext, feat_names=features_names, filepath_results=filepath_results)
comp.extractResults()