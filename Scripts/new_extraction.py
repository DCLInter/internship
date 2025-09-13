from fearture_extraction import Feature_Extraction
import numpy as np
import pandas as pd
import h5py

data_path = 'D:/U/Practicas_City_University_of_London/Data/VitalDB_Train_Subset.h5'
data = {}
with h5py.File(data_path, 'r') as f:
    for group_name in f:
        group = f[group_name]
        data[group_name] = group[()].T        

print(data["PPG"].shape)
data_ext = {}
data_ext["P1"] = data["PPG"][:200000]

ftext = Feature_Extraction(data_ext=data_ext)

demo_info = {}
demo_info["P1"] = {}
demo_info["P1"]["SamplingFrequency"] = data["SF"][0][0] # 125 Hz
print(demo_info["P1"]["SamplingFrequency"])
ftext.demo_info = demo_info

segment_ids = {}
segment_ids["P1"] = np.arange(1,len(data["PPG"])+1)
ftext.segment_ids = segment_ids

features_means, features_medians, failed, fiducial_points = ftext.feature_extraction(save=False)
df = features_means["P1"].T
df.drop(columns=["segment_ID"], inplace=True)
df = df.T

with h5py.File('D:/U/Practicas_City_University_of_London/Data/Features_VitalDB_Train_Subset.h5', 'w') as f:
    for g in data.keys():
        if g == "PPG" or g == "ABP":
            continue
        f.create_dataset(g, data=data[g])
    f.create_dataset("PPG_features", data=df.to_numpy(dtype = float, na_value = np.nan))
