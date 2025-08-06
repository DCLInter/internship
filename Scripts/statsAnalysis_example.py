import os
import pandas as pd
import numpy as np
from statsAnalysis import STATS
import h5py
from tkinter import filedialog

data = {}
segment_ids = {}
data_path = 'C:/Users/adhn565/Documents/Data/completo_conAttrs_16_7_25.h5'

if data_path=="":
    data_path = filedialog.askopenfilename(title='Select signals file', filetypes=[("Input Files", ".h5")])
else:
    data_path=data_path

# Opens the archive read mode only with h5py
with h5py.File(data_path, 'r') as f:
    for group_name in f:
        group = f[group_name]
        data[group_name] = {}
        for dtset_name in group:
            data[group_name][dtset_name] = group[dtset_name][()]
        segment_ids[group_name] = group["segments"][0]
    ### The name and amount of fiducial points and features are the same for all patients
    fiducial = f["p000001"]["segments"].attrs['fiducial_order']
    features = f["p000001"]["mean_p000001"].attrs['features']
    fiducial = [f.decode() if isinstance(f, bytes) else f for f in fiducial]
    features = [f.decode() if isinstance(f, bytes) else f for f in features]

######################### Stadistical analysis #########################
'''Create an instance of the STATS class with the data, features, and save folder
You can change the savefolder to your desired path
Just by calling the class it will create the folders and save the results but you can also call the methods individually
Data is a dictionary with patients as keys and datasets as values
the datasets are dictionaries with the dataset name as keys and the data as values
be sure that the first 2 datasets are the ones you want to analyze
If you want to change the datasets to analyze, you can change the code in the class
Please enssure that the data is in the correct format and structure
'''
savefolder = "C:/Users/adhn565/Documents/Stats"
stats = STATS(data=data, features=features, savefolder=savefolder, segment_ids=segment_ids)

##### Perform the analysis on features and save the results #####
'''If you want to use the function directly you can do it like this:
you need to pass the data to analize, it can be a list, and array or dataframe (1D)'''
# data_analysis = []
# stats_data = pd.DataFrame(index=["Kurtosis","Skewness","IQR","STD","Mean","Median","Distribution","pvalue","Samples"])
# outliers_data = pd.DataFrame(index=["IQR","MAD"])
'''If you want to save the boxplots you can specify the folder'''
# saveBP = os.path.join(savefolder, "BoxPlots")
# os.makedirs(saveBP, exist_ok=True)
# stats, outliers = stats.analysis_feat(data_analysis=data_analysis, stats_data=stats_data, outliers_data=outliers_data, save_BP=saveBP)