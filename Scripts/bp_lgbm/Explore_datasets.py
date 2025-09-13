############ EXPLORE DATASETS #################################
#                                                             #
# Here, new data functions for loading the datasets           #
#  wll be tested                                              #
#                                                             #
###############################################################

from local_paths import PULSE_DB_SUP_DIR
from h5_inspector import inspect_file
from pathlib import Path
from data import load_PulseDB_sup_ds

test_original_path = PULSE_DB_SUP_DIR / "Features_VitalDB_CalFree_Test_Subset.h5"
"""
inspect_file(test_original_path, show_attrs=True)
"""
feature_names = ["IPR", "Tsp", "TWRRF25", "TWRRF50", "Tsw25", 
                 "Tsw50", "Tsw75", "Tdw25", "Tdw50", "Tdw75", 
                 "AUCpi", "IPA",  "Av-Au_ratio", "Ab-Aa_ratio", "Ac-Aa_ratio", 
                 "Ad-Aa_ratio", "Ap2-Ap1_ratio", "AGI", "Kurtosis", "Skewness", 
                 "L-H_ratio", "ShannonEntropy", "Tpp", "PRV", "FullKurt", 
                 "FullSkew", "sdPRV", "IQR_PRV"]
df = load_PulseDB_sup_ds(test_original_path, feature_names=feature_names)
print(df.info())
print(df.head())
