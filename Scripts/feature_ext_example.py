from fearture_extraction import Feature_Extraction
import numpy as np
import pandas as pd
import h5py

''' This will create a Feature_Extraction object with the data path and the names of the files to save
The class will read a h5 file and extract the features from the PPG signals
Its recommended to input your own data directly in the parameter "data_ext" if you are not using the same file as ours
since the reading of the h5 file is very specific to our data structure.
If you want you to input the data directly with the parameter "data_ext", it needs the following format:
'''
# data = {
#           "patient_id_1 or whatever you want to call it": np.array( [signal1, signal2, ...] )
#           }

############ Example of external data input ############
data = {}
with h5py.File("file.h5", 'r') as f:
    for group_name in f:
        group = f[group_name]
        data[group_name] = group[()].T        
print(data["PPG"].shape)
### data["PPG"] contains the PPG signals in rows, each row is a different signal ###
data_ext = {}
### Here the sigansl are added to a dictionary with the key "P1" to fit the format of the class ###
data_ext["P1"] = data["PPG"]

ftext = Feature_Extraction(data_ext=data_ext)
demo_info = {}
### For the class to properly work we need the sampling frequency of the signals ###
demo_info["P1"] = {}
demo_info["P1"]["SamplingFrequency"] = 125 # 125 Hz
ftext.demo_info = demo_info

segment_ids = {}
segment_ids["P1"] = [1,2,3,4] # List of signal ids
ftext.segment_ids = segment_ids

features_means, features_medians, failed, fiducial_points = ftext.feature_extraction(save=False)
########################################################

############ Feature extraction ############
data_path = 'patient_data.h5'
filename_save = "features_patients.h5"
filename_csv = "missing.csv"

ftext = Feature_Extraction(filename_save,filename_csv,data_path=data_path)

######### You can access to the signals with: #########
# signals = ftext.data["name of the group in .h5 file"]

'''Proceed with the feature extraction, it will generate a .h5file
The first column of the datasets will contain the signal_ids
it will save the mean and median of the features for each signal.
'''
features_means, features_medians, failed, fiducial_points = ftext.feature_extraction(save=True)

