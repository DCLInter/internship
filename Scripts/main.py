from checker import Checker
from cleaning import Cleaner

def main():
    path_metrics = "C:/Users/adhn565/Documents/Python_3.10/completo_conAttrs_16_7_25.h5"
    path_originalData = "C:/Users/adhn565/Documents/Data/patient_data.h5"
    filename_report = "C:/Users/adhn565/Documents/Data/final_metrics_2.h5"
    filename_cleanData = "C:/Users/adhn565/Documents/Data/clean_patient_data_2.h5"
    filename_csvReport = "C:/Users/adhn565/Documents/Data/general_report3.csv"
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
    ck = Checker(path_metrics,thresholds)
    dictScore = ck.metrics()
    df_results = ck.results()
    ck.report()
    ck.h5format(filename_report)

    ### Cleaning  the original dataset based on the report of the Checker
    c = Cleaner(filename_report)
    dictFlags = c.detect()
    clean_data = c.clean(path_originalData)
    c.csvReport(filename_csvReport)
    c.saveh5(filename_cleanData)

if __name__ == "__main__":
    main()