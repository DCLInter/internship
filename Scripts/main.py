from checker import Checker
from cleaning import Cleaner

def main():
    path_fiducials = "C:/Users/adhn565/Documents/Data/completo_conAttrs_16_7_25.h5"
    path_originalData = "C:/Users/adhn565/Documents/Data/patient_data.h5"
    filename_report = "C:/Users/adhn565/Documents/Data/mtrics_28_7_2025.h5"
    filename_cleanData = "C:/Users/adhn565/Documents/Data/clean_29_7_2025.h5"
    filename_csvReport = "C:/Users/adhn565/Documents/Data/report_28_7_2025.csv"
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
    clean_data = c.clean(path_fiducials)
    c.csvReport(filename_csvReport)
    c.saveh5(filename_cleanData)

if __name__ == "__main__":
    main()