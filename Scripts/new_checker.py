from checker_copy import Checker
from cleaning import Cleaner
import h5py
import numpy as np
import pandas as pd

path_fiducials = "D:/U/Practicas_City_University_of_London/Data/Fiducial_Points_VitalDB_Train_Subset.h5"
path_originalData = "D:/U/Practicas_City_University_of_London/Data/Features_VitalDB_Train_Subset.h5"
filename_report = "D:/U/Practicas_City_University_of_London/Data/Thresholds_80/metrics_VitalDB_Train_Subset_80.h5"
filename_cleanData = "D:/U/Practicas_City_University_of_London/Data/Thresholds_80/Clean_Features_VitalDB_Train_Subset_80.h5"
filename_csvReport = "general_report.csv"

################## CHECKER ##################
thresholds = {
            "sp_limit":2,
            "bmin":50,
            "bmax":180,
            "w_consistency":0.25,
            "w_alignment":0.75,
            "thresFiducials":80,
            "thresScores":80
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
segment_ids = {}
#splits = [0,data["PPG_fiducial_points"]["Fiducials"].shape[1]]
splits = [0,50000,100000,150000,200000,250000,300000,350000,400000,data["PPG_fiducial_points"]["Fiducials"].shape[1]]
#splits = [0,20,40,60,80,100]
for i in range(0, len(splits)-1):
    data_ext[f"P{splits[i]}"] = {}
    data_ext[f"P{splits[i]}"]["segments"] = data["PPG_fiducial_points"]["Fiducials"][:,splits[i]:splits[i+1]]
    segment_ids[f"P{splits[i]}"] = np.arange(splits[i],splits[i+1])
    
print(data_ext.keys())
print(data_ext["P0"]["segments"].shape)
features_names = ["IPR", "Tsp", "TWRRF25", "TWRRF50", "Tsw25", "Tsw50", "Tsw75", "Tdw25", "Tdw50", "Tdw75", "AUCpi", "IPA",  "Av-Au ratio", "Ab-Aa ratio", "Ac-Aa ratio", "Ad-Aa ratio", "Ap2-Ap1 ratio", "AGI", "Kurtosis", "Skewness", "L-H ratio", "ShannonEntropy", "Tpp"]
demo_info = {}
Nsamples = {}
for group in data_ext.keys():
    demo_info[group] = {"SamplingFrequency": 125}
    Nsamples[group] = 1250

ck = Checker(thresholds, data_ext=data_ext, features_names=features_names, demo_info=demo_info, samples=Nsamples, ids=segment_ids)

Count_fiducials_problems = {}
for group in data_ext.keys():
    dictScore = ck.metrics(patient=group)
    dictResults = ck.results(patient=group)
    
    # Extra data to report some stats on the problems found with the fiducials
    fiducials_problematic = dictScore["percentageProblematicFiducials"]
    dataframe_storage = pd.DataFrame(columns=["Ratio (%)"],index=fiducials_problematic.keys())
    for fp in fiducials_problematic.keys():
        dataframe_storage.loc[fp,"Ratio (%)"] = np.mean(fiducials_problematic[fp])
    Count_fiducials_problems[group] = dataframe_storage

Problems_full_set = pd.DataFrame(index=["Ratio (%)"],columns=ck.fiducial_order)
y = pd.DataFrame()
for group in Count_fiducials_problems.keys():
    x = Count_fiducials_problems[group]
    x.columns = [group]
    y = pd.concat([y,x],axis=1)
y = y.T
for fidu in ck.fiducial_order:
    Problems_full_set.loc["Ratio (%)",fidu] = np.mean(y[fidu])
print(Problems_full_set)
with pd.ExcelWriter("D:/U/Practicas_City_University_of_London/Data/Thresholds_80/Problems_Fiducials.xlsx") as writer:
    Problems_full_set.to_excel(writer,sheet_name="Full_set")

#print(Full_metrics[group]["checkOrderFiducials"])

# print(ck.df_results.keys(), ck.resultsMetrics.keys())
# ck.report()
# results = ck.df_results
# complete_df = pd.DataFrame()
# for k in results:
#     complete_df = pd.concat([complete_df,results[k]])
# print(complete_df)
# complete_results = {"Full_set": complete_df}
# ck.df_results = complete_results
# ck.h5format(filename_report)

################## CLEANING ##################
# data2 = {}
# segment_ids = {}

# with h5py.File(path_originalData, 'r') as f:
#     for group_name in f:
#         obj = f[group_name]
#         if isinstance(obj, h5py.Dataset):
#             d = f[group_name][()]
#             segment_ids[group_name] = np.arange(0,d.shape[1])
#             data2[group_name] = pd.DataFrame(d.T,index=segment_ids[group_name])
            
#         elif isinstance(obj, h5py.Group):
#             group = f[group_name]
#             data2[group_name] = {}
#             for dst in group:
#                 d = group[dst][()]
#                 segment_ids[group_name] = np.arange(0, d.shape[1])
#                 data2[group_name][dst] = pd.DataFrame(d.T,index=segment_ids[group_name])
#                 #print(data2[group_name][dst])

# data_ext = {}
# for k in data2.keys():
#     if k == "PPG_features":
#         data_ext["Full_set"] = {}
#         data_ext["Full_set"]["Features"] = data2[k]["Features"]
#     else:
#         data_ext[k] = data2[k]
#     print(len(data2[k]), k)

# c = Cleaner(filename_report)
# dictFlags = c.detect()
# remove = c.remove
# clean_data = c.clean(data_ext=data_ext)
# for k in data_ext:
#     if k not in remove.keys():
#         clean_data[k] = c.clean_dataset(dataset=data_ext[k],ids_remove=remove[list(remove.keys())[0]])
#         print(k)
#     else:
#         pass
# print(clean_data.keys())

# # with h5py.File(filename_cleanData, 'w') as f:
# #     for g in clean_data.keys():
# #         if g == "Full_set":
# #             continue
# #         else:
# #             f.create_dataset(g, data=clean_data[g].T.to_numpy())
# #     f.create_dataset("PPG_Features", data=clean_data["Full_set"]["Features"].T.to_numpy(dtype=np.float64, na_value=np.nan))

# info = ["Age","Height","Weight","Total signals","Removed signals","Percentage removed (%)"]
# demo_info = pd.DataFrame(columns=info)

# g = "Age"
# dataset = data_ext[g]
# real_ids = np.array(data_ext["Subject"][0])
# dataset.index = [v.decode() if isinstance(v, bytes) else v for v in real_ids]
# unique_id = np.unique(dataset.index)

# dataset_clean = clean_data[g]
# real_ids_clean = np.array(clean_data["Subject"][0])
# dataset_clean.index = [v.decode() if isinstance(v, bytes) else v for v in real_ids_clean]
# unique_id_clean = np.unique(dataset_clean.index)

# for id in unique_id:
#     print(id)
#     for l in ["Age","Height","Weight"]:
#         a =  data_ext[l]
#         a.index = dataset.index
#         demo_info.loc[id,l] = a.loc[dataset.index == id][0][0]
#     total = len(dataset.loc[dataset.index == id])
#     total_clean = len(dataset_clean.loc[dataset_clean.index == id])
#     demo_info.loc[id,"Total signals"] = total
#     rem = total - total_clean
#     demo_info.loc[id,"Removed signals"] = rem
#     demo_info.loc[id,"Percentage removed (%)"] = (rem/total)*100
                
# # print(demo_info)
# # demo_info.to_excel("D:/U/Practicas_City_University_of_London/Data/Thresholds_80/Demographic_Info_VitalDB_Train_Subset_80.xlsx")
