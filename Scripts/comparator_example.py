import h5py
import numpy as np
import pandas as pd
from cleaning_comparison import Comparator

data = {}
data_clean = {}
segment_ids = {}
data_path = 'C:/Users/adhn565/Documents/Data/completo_conAttrs_16_7_25.h5'
data_path_clean = 'C:/Users/adhn565/Documents/Data/data_clean_features.h5'

with h5py.File(data_path, 'r') as f:
    for group_name in f:
        group = f[group_name]
        data[group_name] = {}
        for dtset_name in group:
            data[group_name][dtset_name] = group[dtset_name][()]
        segment_ids[group_name] = group["segments"][0]
    
    fiducial = f["p000001"]["segments"].attrs['fiducial_order']
    features = f["p000001"]["mean_p000001"].attrs['features']
    fiducial = [f.decode() if isinstance(f, bytes) else f for f in fiducial]
    features = [f.decode() if isinstance(f, bytes) else f for f in features]

with h5py.File(data_path_clean, 'r') as f:
    for group_name in f:
        group = f[group_name]
        data_clean[group_name] = {}
        for dtset_name in group:
            data_clean[group_name][dtset_name] = group[dtset_name][()]

c = Comparator(data, data_clean, features, filepath_results="")
c.extractResults()
c.extractResults_byPatient()