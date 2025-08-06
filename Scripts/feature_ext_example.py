from tkinter import filedialog
from Scripts.fearture_extraction import Feature_Extraction
import numpy as np
import pandas as pd

#### Feature extraction ####
data_path = 'patient_data.h5'
filename_save = "a.h5"
filename_csv = "a.csv"

if data_path=="":
    data_path = filedialog.askopenfilename(title='Select signals file', filetypes=[("Input Files", ".h5")])
else:
    pass

''' This will create a Feature_Extraction object with the data path and the names of the files to save
The class will read h5 file and extract the features from the signals
Be sure that the h5 file has groups as patients and their dataset is the signals
and that the first 4 columns of the dataset can be removed (they are not needed for the features)
If you want you can input the data directly with the parameter "data", in a dictionary with this format:
'''
# data = {
#           "patient_id or whatever you want to call it": np.array( [signal1, signal2, ...] )
#           }

ftext = Feature_Extraction(data_path,filename_save,filename_csv)

######### You can access to the signals with: #########
# signals = ftext.data["name of the group in .h5 file"]

'''Proceed with the feature extraction, it will generate a .h5file
The first column of the segments dataset will contain the signal_id
it will save the mean and median of the features for each signal in the segments dataset
'''
features_means, features_medians, failed, fiducial_points = ftext.feature_extraction()