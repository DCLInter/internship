from checker import Checker
from cleaning import Cleaner
import h5py
import numpy as np
import pandas as pd

path_fiducials = "D:/U/Practicas_City_University_of_London/Data/Fiducial_Points_VitalDB_Train_Subset.h5"
path_originalData = "D:/U/Practicas_City_University_of_London/Data/Features_VitalDB_Train_Subset.h5"
filename_report = "D:/U/Practicas_City_University_of_London/Data/metrics_VitalDB_Train_Subset.h5"
filename_cleanData = "D:/U/Practicas_City_University_of_London/Data/Clean_Features_VitalDB_Train_Subset.h5"
filename_csvReport = "general_report.csv"

thresholds = {
            "sp_limit":2,
            "bmin":50,
            "bmax":180,
            "w_consistency":0.25,
            "w_alignment":0.75,
            "thresFiducials":90,
            "thresScores":90
            }

data = {}
with h5py.File(path_fiducials, 'r') as f:
    for group_name in f:
        obj = f[group_name]
        if isinstance(obj, h5py.Dataset):
            data[group_name] = f[group_name][()]
        elif isinstance(obj, h5py.Group):
            group = f[group_name]
            data[group_name] = {}
            for dst in group:
                data[group_name][dst] = group[dst][()]

data_ext = {}
for group in data.keys():
    data_ext[group] = {}
    data_ext[group]["segments"] = data["PPG_fiducial_points"]["Fiducials"]

print(data_ext[group]["segments"].shape)
ck = Checker(thresholds, data_ext=data_ext)

segment_ids = {}
demo_info = {}
Nsamples = {}
for group in data_ext.keys():
    segment_ids[group] = np.arange(1,data["PPG_fiducial_points"]["Fiducials"].shape[1]+1)
    demo_info[group] = {"SamplingFrequency": 125}
    Nsamples[group] = 1250
ck.ids = segment_ids
ck.demo_info = demo_info
ck.Nsamples = Nsamples
ck.fiducial_order = ['on','sp','dn','dp','off','u','v','w','a','b','c','d','e','f','p1','p2']
dictScore = ck.metrics()
dictResults = ck.results()
ck.report()
dictReport = ck.df_results
print(dictResults.keys())
ck.h5format(filename_report)

################## CLEANING ##################
data2 = {}
segment_ids = {}
with h5py.File(path_originalData, 'r') as f:
    for group_name in f:
        obj = f[group_name]
        if isinstance(obj, h5py.Dataset):
            data2[group_name] = f[group_name][()].T
            segment_ids[group_name] = np.arange(1,data2[group_name].shape[0]+1)
            data2[group_name] = pd.DataFrame(data2[group_name],index=segment_ids[group_name])
        elif isinstance(obj, h5py.Group):
            group = f[group_name]
            data2[group_name] = {}
            for dst in group:
                data2[group_name][dst] = group[dst][()].T
                segment_ids[group_name] = np.arange(1, data2[group_name][dst].shape[0]+1)
                data2[group_name][dst] = pd.DataFrame(data2[group_name][dst],index=segment_ids[group_name])

data_ext = {}
for group in data2.keys():
    data_ext[group] = {}
    data_ext[group] = data2[group]
    if group == "PPG_features":
        data_ext["PPG_fiducial_points"] = {}
        data_ext["PPG_fiducial_points"]["segments"] = data2["PPG_features"]["Features"]
print(data_ext.keys())
print(data_ext["PPG_fiducial_points"]["segments"].shape)

c = Cleaner(filename_report)
dictFlags = c.detect()
clean_data = c.clean(data_ext=data_ext)

print(clean_data.keys())
print(clean_data["PPG_fiducial_points"]["segments"].shape)

new_data = {}
for group, obj in clean_data.items():
    new_data[group] = {}
    if isinstance(obj, dict):
        for dst, df in obj.items():
            new_data[group][dst] = df
    else:
        new_data[group] = clean_data[group]
print(new_data.keys())
print(new_data["PPG_fiducial_points"]["segments"].shape)

with h5py.File(filename_cleanData, 'w') as f:
    for g in new_data.keys():
        if g == "PPG_features" or g == "PPG_fiducial_points":
            continue
        f.create_dataset(g, data=new_data[g].T)
    group = f.create_group("PPG_features")
    group.create_dataset("Features", data=new_data["PPG_fiducial_points"]["segments"].T.to_numpy(dtype=np.float64, na_value=np.nan))

