import numpy as np
import pandas as pd
import h5py

from fearture_extraction import Feature_Extraction
from checker import Checker
from cleaning import Cleaner

''' The feature extraction and statistical analysis process its the most time consuming 
    if you have a large dataset of signals know that its going to take a lot of time '''

def main():
    #### Feature Extraction ####
    data_path = "patient_data.h5"
    filename_save = "extracted_features.h5"
    filename_csv = "null_detection.csv"

    ftext = Feature_Extraction(data_path,filename_save,filename_csv)

    #### Checking and generating report the fiducial points ####
    path_fiducials = "extracted_features.h5"
    path_originalData = "patient_data.h5"
    filename_report = "metrics.h5"
    filename_cleanData = "clean_data.h5"
    filename_csvReport = "report.csv"

    thresholds = {
                "sp_limit":2,
                "bmin":50,
                "bmax":180,
                "w_consistency":0.25,
                "w_alignment":0.75,
                "thresFiducials":80,
                "thresScores":80
                }
    
    ### Checking the fiducial points
    ck = Checker(path_fiducials,thresholds)
    dictScore = ck.metrics()
    df_results = ck.results()
    ck.report()
    ck.h5format(filename_report)

    ### Cleaning  the original dataset based on the report of the Checker
    c = Cleaner(filename_report)
    dictFlags = c.detect()
    # You can clean the data contaning the features or the original data with just the signals
    clean_data = c.clean(path_fiducials) # path_originalData
    c.csvReport(filename_csvReport)
    c.saveh5(filename_cleanData)

if __name__ == "__main__":
    main()