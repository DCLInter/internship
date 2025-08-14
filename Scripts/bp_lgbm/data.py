######## DATA LOADER ################################################################
#                                                                                   #
# This scrip is meant for loading the data from the feature extraction made by Juan #
#                                                                                   #
#####################################################################################

import local_paths
from pathlib import Path
import h5py
from h5_inspector import inspect_file

data_path = local_paths.DATA_DIR / "patient_data.h5"
inspect_file(data_path)