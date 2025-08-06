from checker import Checker
from cleaning import Cleaner

#### Checking and generating a report for the fiducial points ####
'''Please note that this needs to have a certain format in the .h5 file where the signals are rows in the dataset of the groups
- the name of datasets should be "segments" since this was done to recieve the data from the Feature Extraction, this is important for the Checker
- the first column of the datasets will be used as "signal ids"
If you dont have any you can just add at the beggining of your signals a random number as long as they dont repeat, this is actually important for the checker
- if you dont have any "signal ids" and you arent using the data from Feature Extraction please add them

For the cleaning alone note that if you dont have "signal ids" it doesn't matter that much, it will eliminate the corresponding signals without problems
-  just know that the "signal ids" will be the first value of the signal 
'''

path_fiducials = "C:/Users/adhn565/Documents/Data/completo_conAttrs_16_7_25.h5"
path_originalData = "C:/Users/adhn565/Documents/Data/patient_data.h5"
filename_report = "C:/Users/adhn565/Documents/Data/final_metrics_2.h5"
filename_cleanData = "C:/Users/adhn565/Documents/Data/clean_29_7_2025.h5"
filename_csvReport = "C:/Users/adhn565/Documents/Data/report_28_7_2025.csv"

# The thresholds are for the metrics implemented and follows this format and, you can change it
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
''' The checker will generate a report based on stablished metrics with the objective to evaluate the detected fiducials of the signals
You can check all the metrics generated as additional and more specific information
or check the report which uses 4 specific metrics to give a confidence score of the signal
This means that the signal most likely didnt have too many or any problem with the fiducial points and probably
the features extracted from those signals arent noise or are biased for badly locations of the fiducials
Note that the dictResults for each key has a dataframe, that is the results of the metrics for each group of the .h5 file
the report will just add a column to this dataframe with a flag (0 or 1) where 1 means we recommend discarding the signal or manually review the fiducials location
'''
ck = Checker(path_fiducials,thresholds)
dictScore = ck.metrics()
dictResults = ck.results()
ck.report()
dictReport = ck.df_results
ck.h5format(filename_report)

### Cleaning  the original dataset based on the report of the Checker
''' The cleanear wil read the report generate by the checker and will only use the "report" column to eliminate the signals an create a new .h5 file
with the remaning signals, it wont modify the original data, as an extra the "csvReport" function will generate a resume for each group of the 4 metrics
and the number of signals that were eliminated
'''
c = Cleaner(filename_report)
dictFlags = c.detect()
# You can clean the data contaning the features or the original data with just the signals
clean_data = c.clean(path_fiducials) # path_originalData
c.csvReport(filename_csvReport)
c.saveh5(filename_cleanData)