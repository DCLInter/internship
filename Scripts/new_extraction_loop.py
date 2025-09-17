from fearture_extraction import Feature_Extraction
import numpy as np
import pandas as pd
import h5py
from bp_lgbm.local_paths import PULSE_DB_SUP_DIR

data_path = PULSE_DB_SUP_DIR / "VitalDB_Train_Subset.h5"
#data_path = 'D:/U/Practicas_City_University_of_London/Data/VitalDB_Train_Subset.h5'
data = {}
with h5py.File(data_path, 'r') as f:
    for group_name in f:
        group = f[group_name]
        data[group_name] = group[()].T        

print(data["PPG"].shape)
data_ext = {}

splits = [0, 100000, 200000, 300000, 400000, len(data["PPG"])]

features_list = list()
fiducials_list = list()

for i in range (4, len(splits)-1):

    data_ext = data["PPG"][splits[i]:splits[i+1]]

    ftext = Feature_Extraction(data_ext=data_ext)

    demo_info = {}
    demo_info = {}
    demo_info["SamplingFrequency"] = data["SF"][0][0] # 125 Hz
    print(demo_info["SamplingFrequency"])
    ftext.demo_info = demo_info

    segment_ids = {}
    segment_ids = np.arange(1,len(data_ext)+1)
    ftext.segment_ids = segment_ids

    features_means, features_medians, failed, fiducial_points = ftext.feature_extraction(save=False)
    df = features_means.T
    df.drop(columns=["segment_ID"], inplace=True)
    df = df.T

    # Convert into dictionaries and then append it
    dict_feat = df.to_dict(orient="list")
    features_list.append(dict_feat)

    df_fidu = fiducial_points.T
    df_fidu.drop(columns=["segment_ID"], inplace=True)
    df_fidu = df_fidu.T

    # Convert into dictionaries and then append it
    dict_fid = df_fidu.to_dict(orient="list")
    fiducials_list.append(dict_fid)

# later: rebuild DataFrames and concat
dfs_feat = [pd.DataFrame(d) for d in dict_feat]
final_feat_df = pd.concat(dfs_feat, ignore_index=True)
print(final_feat_df.head())

# later: rebuild DataFrames and concat
dfs_fid = [pd.DataFrame(d) for d in dict_feat]
final_fid_df = pd.concat(dfs_feat, ignore_index=True)
print(final_fid_df.head())

with h5py.File('D:/U/Practicas_City_University_of_London/Data/Features_VitalDB_Train_Subset.h5', 'w') as f:
    for g in data.keys():
        if g == "PPG" or g == "ABP":
            continue
        f.create_dataset(g, data=data[g].T)
    group = f.create_group("PPG_features")
    group.create_dataset("Features", data= final_feat_df.to_numpy(dtype=np.float64, na_value=np.nan))
    
with h5py.File('D:/U/Practicas_City_University_of_London/Data/Fiducial_Points_VitalDB_Train_Subset.h5', 'w') as f:
    group = f.create_group("PPG_fiducial_points")
    group.create_dataset("Fiducials", data= final_fid_df.to_numpy(dtype=np.float64, na_value=np.nan))