from fearture_extraction import Feature_Extraction
import numpy as np
import pandas as pd
import h5py

data_path = 'D:/U/Practicas_City_University_of_London/Data/VitalDB_CalFree_Test_Subset.h5'
data = {}
with h5py.File(data_path, 'r') as f:
    for group_name in f:
        group = f[group_name]
        data[group_name] = group[()].T

print(data["PPG"].shape)
data_ext = {}
data_ext["P1"] = data["PPG"][:]

ftext = Feature_Extraction(data_ext=data_ext)

demo_info = {}
demo_info["P1"] = {}
demo_info["P1"]["SamplingFrequency"] = data["SF"][0][0] # 125 Hz
print(demo_info["P1"]["SamplingFrequency"])
ftext.demo_info = demo_info

segment_ids = {}
segment_ids["P1"] = np.arange(1,len(data_ext["P1"])+1)
ftext.segment_ids = segment_ids

fiducial_points = ftext.only_fiducials()
df_fidu = fiducial_points["P1"].T
df_fidu.drop(columns=["segment_ID"], inplace=True)
df_fidu = df_fidu.T
df_fidu = df_fidu.replace({pd.NA: np.nan})

with h5py.File('D:/U/Practicas_City_University_of_London/Data/Fiducial_Points_VitalDB_CalFree_Test_Subset.h5', 'w') as f:
    for k in data.keys():
        f.create_dataset(k, data=data[k].T)
    f.create_dataset("PPG_fiducial_points", data= df_fidu.to_numpy(dtype=np.float64, na_value=np.nan))


# data_path = 'D:/U/Practicas_City_University_of_London/Data/Fiducial_Points_VitalDB_Train_Subset.h5'
# data = {}
# with h5py.File(data_path, 'r') as f:
#     for group_name in f:
#         obj = f[group_name]
#         if isinstance(obj, h5py.Dataset):
#             data[group_name] = f[group_name][()].T
#         elif isinstance(obj, h5py.Group):
#             group = f[group_name]
#             data[group_name] = {}
#             for dst in group:
#                 data[group_name][dst] = group[dst][()].T
# print(data.keys())

# data_path = 'D:/U/Practicas_City_University_of_London/Data/Fiducial_Points_extra_VitalDB_Train_Subset.h5'
# data_extra = {}
# with h5py.File(data_path, 'r') as f:
#     for group_name in f:
#         obj = f[group_name]
#         if isinstance(obj, h5py.Dataset):
#             data_extra[group_name] = f[group_name][()].T
#         elif isinstance(obj, h5py.Group):
#             group = f[group_name]
#             data_extra[group_name] = {}
#             for dst in group:
#                 data_extra[group_name][dst] = group[dst][()].T

# df = pd.DataFrame()
# df = pd.concat([df, pd.DataFrame(data["PPG_fiducial_points"]["First_100k"])], axis=0)
# df = pd.concat([df, pd.DataFrame(data["PPG_fiducial_points"]["Second_100k"])], axis=0)
# df = pd.concat([df, pd.DataFrame(data["PPG_fiducial_points"]["Third_100k"])], axis=0)
# df = pd.concat([df, pd.DataFrame(data_extra["PPG_fiducial_points"]["Fourth_100k"])], axis=0)
# df = pd.concat([df, pd.DataFrame(data["PPG_fiducial_points"]["Last_data"])], axis=0)
# print(df.shape)

# with h5py.File('D:/U/Practicas_City_University_of_London/Data/Fiducial_points_full_VitalDB_Train_Subset.h5', 'w') as f:
#     group = f.create_group("PPG_fiducial_points")
#     group.create_dataset("First_100k", data= data["PPG_fiducial_points"]["First_100k"].T)
#     group.create_dataset("Second_100k", data= data["PPG_fiducial_points"]["Second_100k"].T)
#     group.create_dataset("Third_100k", data= data["PPG_fiducial_points"]["Third_100k"].T)
#     group.create_dataset("Fourth_100k", data= data_extra["PPG_fiducial_points"]["Fourth_100k"].T)
#     group.create_dataset("Last_data", data= data["PPG_fiducial_points"]["Last_data"].T)



# data_path = 'D:/U/Practicas_City_University_of_London/Data/VitalDB_Train_Subset.h5'
# data = {}
# with h5py.File(data_path, 'r') as f:
#     for group_name in f:
#         group = f[group_name]
#         data[group_name] = group[()].T  

# new_data = {}
# for k in data.keys():
#     new_data[k] = data[k]
# new_data["PPG_fiducial_points"] = df
# print(new_data.keys())
# with h5py.File('D:/U/Practicas_City_University_of_London/Data/Fiducials_complete_VitalDB_Train_Subset.h5', 'w') as f:
#     for g in new_data.keys():
#         f.create_dataset(g, data=new_data[g].T)   

# for sigs in data["PPG_fiducial_points"]:
#     if sigs == "Fourth_100k":
#         df = pd.concat([df, pd.DataFrame(data_extra["PPG_fiducial_points"][sigs])], axis=0)
#     else:
#         df = pd.concat([df, pd.DataFrame(data["PPG_fiducial_points"][sigs])], axis=0)
#     print(sigs, data["PPG_fiducial_points"][sigs].shape)
# print(df.shape)

# new_data = {}
# for k in data.keys():
#     if k != "PPG_features":
#         new_data[k] = data[k]
#     else:
#         new_data["PPG_features"] = df

# with h5py.File('D:/U/Practicas_City_University_of_London/Data/Features_complete_VitalDB_Train_Subset.h5', 'w') as f:
#     for g in new_data.keys():
#         f.create_dataset(g, data=new_data[g].T)



# data_path = 'D:/U/Practicas_City_University_of_London/Data/VitalDB_Train_Subset.h5'
# data = {}
# with h5py.File(data_path, 'r') as f:
#     for group_name in f:
#         group = f[group_name]
#         data[group_name] = group[()].T        

# print(data["PPG"].shape)
# data_ext = {}
# data_ext["P1"] = data["PPG"][:100000]

# ftext = Feature_Extraction(data_ext=data_ext)

# demo_info = {}
# demo_info["P1"] = {}
# demo_info["P1"]["SamplingFrequency"] = data["SF"][0][0] # 125 Hz
# print(demo_info["P1"]["SamplingFrequency"])
# ftext.demo_info = demo_info

# segment_ids = {}
# segment_ids["P1"] = np.arange(1,len(data_ext["P1"])+1)
# ftext.segment_ids = segment_ids

# features_means, features_medians, failed, fiducial_points = ftext.feature_extraction(save=False)

# df_fidu = fiducial_points["P1"].T
# df_fidu.drop(columns=["segment_ID"], inplace=True)
# df_fidu = df_fidu.T

# # data_ext["P1"] = data["PPG"][100000:200000]
# # ftext = Feature_Extraction(data_ext=data_ext)

# # demo_info["P1"]["SamplingFrequency"] = data["SF"][0][0] # 125 Hz
# # print(demo_info["P1"]["SamplingFrequency"])
# # ftext.demo_info = demo_info

# # segment_ids["P1"] = np.arange(1,len(data_ext["P1"])+1)
# # ftext.segment_ids = segment_ids
# # features_means, features_medians, failed, fiducial_points = ftext.feature_extraction(save=False)

# # df2_fidu = fiducial_points["P1"].T
# # df2_fidu.drop(columns=["segment_ID"], inplace=True)
# # df2_fidu = df2_fidu.T

# # data_ext["P1"] = data["PPG"][200000:300000]
# # ftext = Feature_Extraction(data_ext=data_ext)

# # demo_info["P1"]["SamplingFrequency"] = data["SF"][0][0] # 125 Hz
# # print(demo_info["P1"]["SamplingFrequency"])
# # ftext.demo_info = demo_info

# # segment_ids["P1"] = np.arange(1,len(data_ext["P1"])+1)
# # ftext.segment_ids = segment_ids
# # features_means, features_medians, failed, fiducial_points = ftext.feature_extraction(save=False)

# # df3_fidu = fiducial_points["P1"].T
# # df3_fidu.drop(columns=["segment_ID"], inplace=True)
# # df3_fidu = df3_fidu.T

# # data_ext["P1"] = data["PPG"][300000:400000]
# # ftext = Feature_Extraction(data_ext=data_ext)

# # demo_info["P1"]["SamplingFrequency"] = data["SF"][0][0] # 125 Hz
# # print(demo_info["P1"]["SamplingFrequency"])
# # ftext.demo_info = demo_info

# # segment_ids["P1"] = np.arange(1,len(data_ext["P1"])+1)
# # ftext.segment_ids = segment_ids
# # features_means, features_medians, failed, fiducial_points = ftext.feature_extraction(save=False)

# # df4_fidu = fiducial_points["P1"].T
# # df4_fidu.drop(columns=["segment_ID"], inplace=True)
# # df4_fidu = df4_fidu.T

# # data_ext["P1"] = data["PPG"][400000:]
# # ftext = Feature_Extraction(data_ext=data_ext)

# # demo_info["P1"]["SamplingFrequency"] = data["SF"][0][0] # 125 Hz
# # print(demo_info["P1"]["SamplingFrequency"])
# # ftext.demo_info = demo_info

# # segment_ids["P1"] = np.arange(1,len(data_ext["P1"])+1)
# # ftext.segment_ids = segment_ids
# # features_means, features_medians, failed, fiducial_points = ftext.feature_extraction(save=False)

# # df5_fidu = fiducial_points["P1"].T
# # df5_fidu.drop(columns=["segment_ID"], inplace=True)
# # df5_fidu = df5_fidu.T

# # df_fidu = df_fidu.replace({pd.NA: np.nan})
# # df2_fidu = df2_fidu.replace({pd.NA: np.nan})
# # df3_fidu = df3_fidu.replace({pd.NA: np.nan})
# # df4_fidu = df4_fidu.replace({pd.NA: np.nan})
# # df5_fidu = df5_fidu.replace({pd.NA: np.nan})

# with h5py.File('D:/U/Practicas_City_University_of_London/Data/Fiducial_Points_extra_VitalDB_Train_Subset.h5', 'w') as f:
#     group = f.create_group("PPG_fiducial_points")
#     group.create_dataset("Fourth_100k", data= df_fidu.to_numpy(dtype=np.float64, na_value=np.nan))
#     # group.create_dataset("Second_100k", data= df2_fidu.to_numpy(dtype=np.float64, na_value=np.nan))
#     # group.create_dataset("Third_100k", data= df3_fidu.to_numpy(dtype=np.float64, na_value=np.nan))
#     # group.create_dataset("Fourth_100k", data= df4_fidu.to_numpy(dtype=np.float64, na_value=np.nan))
#     # group.create_dataset("Last_data", data= df5_fidu.to_numpy(dtype=np.float64, na_value=np.nan))

