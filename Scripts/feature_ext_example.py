from tkinter import filedialog
from fearture_extraction import Feature_Extraction
import numpy as np
import pandas as pd

#### Feature extraction ####
data_path = 'D:/U/Practicas_City_University_of_London/One_drive/Data/patient_data.h5'
filename_save = "features_patients.h5"
filename_csv = "missing.csv"

if data_path=="":
    data_path = filedialog.askopenfilename(title='Select signals file', filetypes=[("Input Files", ".h5")])
else:
    pass

''' This will create a Feature_Extraction object with the data path and the names of the files to save
The class will read a h5 file and extract the features from the PPG signals
Its recommended to input your own data directly in the parameter "data_ext" if you are not using the same file as ours
since the reading of the h5 file is very specific to our data structure.
If you want you to input the data directly with the parameter "data_ext", it needs the following format:
'''
# data = {
#           "patient_id_1 or whatever you want to call it": np.array( [signal1, signal2, ...] )
#           }

ftext = Feature_Extraction(filename_save,filename_csv,data_path=data_path)
######### You can access to the signals with: #########
# signals = ftext.data["name of the group in .h5 file"]

'''Proceed with the feature extraction, it will generate a .h5file
The first column of the datasets will contain the signal_ids
it will save the mean and median of the features for each signal.
'''
features_means, features_medians, failed, fiducial_points = ftext.feature_extraction()