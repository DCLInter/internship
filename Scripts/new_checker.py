from checker import Checker
from cleaning import Cleaner
import h5py
import numpy as np
import pandas as pd

path_fiducials = "D:/U/Practicas_City_University_of_London/Data/Fiducial_Points_VitalDB_CalFree_Test_Subset 1.h5"
path_originalData = "C:/Users/adhn565/Documents/Data/Features_VitalDB_CalFree_Test_Subset 1.h5"
filename_report = "D:/U/Practicas_City_University_of_London/Data/metrics_465k.h5"
filename_cleanData = "D:/U/Practicas_City_University_of_London/Data/features_patients_clean.h5"
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
data_ext["P1"] = {}
data_ext["P1"]["segments"] = data["PPG_fiducial_points"]["Fiducials"]

print(data_ext["P1"]["segments"].shape)
ck = Checker(thresholds, data_ext=data_ext)

segment_ids = {}
segment_ids["P1"] = np.arange(1,data["PPG_fiducial_points"]["Fiducials"].shape[1]+1)
ck.ids = segment_ids
ck.demo_info = {"P1": {"SamplingFrequency": 125} }
ck.Nsamples = {"P1": 1250 }
ck.fiducial_order = ['on','sp','dn','dp','off','u','v','w','a','b','c','d','e','f','p1','p2']

dictScore = ck.metrics()
dictResults = ck.results()
ck.report()
dictReport = ck.df_results
print(dictResults.keys())
ck.h5format(filename_report)

# c = Cleaner(filename_report)
# dictFlags = c.detect()
# clean_data = c.clean(path_fiducials)
# c.csvReport(filename_csvReport)
# c.saveh5(filename_cleanData)