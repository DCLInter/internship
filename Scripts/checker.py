import h5py
import pandas as pd
import numpy as np
from metrics_functions import Metrics
class Checker:
    def __init__(self, datapath: str, threshold: dict = {}):

        self.path = datapath
        self.data = {}
        self.ids = {}
        self.demo_info = {}
        self.Nsamples = {}
        self.df_results = {}
        self.threshold = threshold
        
        with h5py.File(self.path, 'r') as f:
            for group_name in f:
                group = f[group_name]
                self.data[group_name] = {}
                if "N-samples" in group["segments"].attrs:
                    self.Nsamples[group_name] = f[group_name]["segments"].attrs["N-samples"]
                self.ids[group_name] = list(group["segments"][0])
                self.ids[group_name] = self.ids[group_name][:len(self.ids[group_name])//2]
                for dtset_name in group:
                    self.data[group_name][dtset_name] = group[dtset_name][()]
                    if dtset_name == "segments":
                        self.data[group_name][dtset_name] = group[dtset_name][1:]

                if group_name not in self.demo_info:
                    self.demo_info[group_name] = {}
                for attr_name, attr_value in group.attrs.items():
                    self.demo_info[group_name][attr_name] = attr_value   
            ### The name and amount of fiducial points and features are the same for all patients
            fiducial = f[group_name]["segments"].attrs['fiducial_order']
            features = f[group_name][f"mean_{group_name}"].attrs['features']
            self.fiducial_order = [f.decode() if isinstance(f, bytes) else f for f in fiducial]
            self.features_names = [f.decode() if isinstance(f, bytes) else f for f in features]

    def windows(self, patient: str, signal = None, all: bool = False):
        
        idx = signal
        patient_fiducials = pd.DataFrame(self.data[patient]["segments"])
        bp_sigs = np.arange(len(patient_fiducials.columns)//2,len(patient_fiducials.columns))
        patient_fiducials = patient_fiducials.drop(bp_sigs,axis=1)
        patient_fiducials.columns = self.ids[patient]
        
        y = patient_fiducials.iloc[:,idx].values
        l = len(y)
        n_fiducials = 16        ### 16 fiducial points
        windows = l//n_fiducials
        x2d = y[:windows * n_fiducials].reshape((windows, n_fiducials))
        df_fiducials = pd.DataFrame(x2d,columns=self.fiducial).dropna(how="all")
        if all:
            return patient_fiducials.T, df_fiducials
        return df_fiducials
    
    def metrics(self):
        print("Metrics:")
        ### Storage variables
        abnormalSP_data = {}
        abnormalHR_data = {}
        wrongOrdFidu = {}
        number_overlapFiducial = {}
        number_overlapWindows = {}
        number_fiducialsDetect = {}
        number_derivativesDetected = {}
        scores = {}
        flagScores = {}
        no_detect = {}
        num_noDetect = {}

        for patient in self.data.keys():

            ### Storage per patient
            print(patient)
            wrongOrdFidu[patient] = {}
            scores[patient] = {}
            flagScores[patient] = {}
            number_derivativesDetected[patient] = {}
            lowsp = []
            anormalHR = []
            no_d = []
            num_noD = []
            overlapNum = []
            ratioFpDetect = []
            flagSignals = 0
    
            idx = 0
            for sig in self.ids[patient]:

                ### Settings
                fs = int(self.demo_info[patient]["SamplingFrequency"])
                nsamples = self.Nsamples[patient][idx]

                ### Adquaring the windows to analice per signal
                df_fiducials = self.windows(patient,idx,all=False)

                ### Initializing dictionaries
                for fidu in df_fiducials.columns:
                    if fidu not in wrongOrdFidu[patient]:
                        wrongOrdFidu[patient][fidu] = []
                    if fidu not in flagScores[patient]:
                        flagScores[patient][fidu] = []

                ### Metrics process (abnormal HR, checking the number of detected peaks, scores)
                metrics = Metrics(df_fiducials,fs,nsamples,self.threshold)

                ### Missing fiducials in the signal, flag the signal if its missing at least 1
                flagND = metrics.checkNA()
                
                if flagND > 0:
                    no_d.append(sig)

                numSP = metrics.checkNumPeaks()
                if numSP == True:
                    lowsp.append(sig)
                
                wrongOrdFidu, numFlags, winFlags, otro = metrics.checkOrder(wrongOrdFidu,patient,sig)
                wNum = len(winFlags.keys())
                fpNum = wNum*16

                if numFlags != 0:
                    flagSignals += 1
                
                na_ratio = (flagND/fpNum)*100
                num_noD.append(na_ratio)

                list_deriv = metrics.list_derivatives
                numD = metrics.numDerivatives
                for d in numD.keys():
                    if d not in number_derivativesDetected[patient]:
                        number_derivativesDetected[patient][d] = []
                    ffp = numD[d]
                    n = wNum*len(list_deriv[d])
                    r = (1 - ffp/n)*100
                    number_derivativesDetected[patient][d].append(r)
            
                ratio = (1 - numFlags/fpNum)*100
                ratioFpDetect.append(ratio)

                winOverlap = 0
                for win, f in winFlags.items():
                    if f != 0:
                        winOverlap+=1
                overlapNum.append(winOverlap)

                flagHR = metrics.checkHR()
                if flagHR:
                    anormalHR.append(sig)
                 
                align, cons = metrics.consistency_alignment()
                comScores = metrics.scoreCombined()
                scores[patient][sig] = comScores
                for fp in comScores.keys():
                    if (comScores[fp] < 90).any():
                        flagScores[patient][fp].append(sig)
                
                idx+=1

            number_overlapWindows[patient] = overlapNum
            number_overlapFiducial[patient] = flagSignals
            abnormalHR_data[patient] = anormalHR      
            abnormalSP_data[patient] = lowsp
            no_detect[patient] = no_d
            num_noDetect[patient] = num_noD
            number_fiducialsDetect[patient] = ratioFpDetect

        self.resultsMetrics = {
                            "checkHR": abnormalHR_data, ### signals (their segment_id) with extreme HR values
                            "checkSP": abnormalSP_data, ### signals that didnt contain enough amount of SP
                            "checkNAvalues": no_detect, ### signals with at least 1 NA values in their fiducials
                            "numberNAvalues": num_noDetect, ### percentage of NA values for each signal
                            "checkOrderFiducials": wrongOrdFidu, ### signals with at least 1 fiducial overlapped or NA value
                            "numberOverlapFiducials": number_overlapFiducial, ### signals with at least 1 overlapped fiducial
                            "numberOverlapWindows": number_overlapWindows, ### number of windows of the signal that has any fiducial point overlapped
                            "flagForScore": flagScores, ### signals that didnt meet the threshold for the score
                            "combinedScore": scores, ### scores divided by each fiducial points per signal
                            "numberProperFiducials":number_fiducialsDetect, ### percentage of fiducials properly detected
                            "numberProperFiducials_byDerivatives": number_derivativesDetected ### percentage for each derivative
                            }
        
        return self.resultsMetrics
    
    def results(self):

        print("Analysis of metrics:")
        ids = self.ids
        remove = ["numberOverlapWindows","checkOrderFiducials","numberOverlapFiducials","flagForScore","checkNAvalues"]
        results = {k: v for k, v in self.resultsMetrics.items() if k not in remove}
        all_patients = list(ids.keys())

        for p in all_patients:
            signal_flags = []
            df = pd.DataFrame(0,index=ids[p],columns=results.keys())
            for metric, values_patient in results.items():
                if type(values_patient[p]) is list:
                    if metric == "numberProperFiducials":
                        idx = 0
                        for signal in ids[p]:
                            ratio = values_patient[p][idx]
                            df.loc[signal,metric] = ratio
                            idx+=1
                    else:       
                        signal_flags = values_patient[p]
                        for signal in ids[p]:
                            if signal in signal_flags:
                                df.loc[signal,metric] = 1
                            else:
                                df.loc[signal,metric] = 0
                elif type(values_patient[p]) is dict:
                    if metric == "numberProperFiducials_byDerivatives":
                        for deriv in values_patient[p].keys():
                            ratios = values_patient[p][deriv]
                            df.loc[:,deriv] = ratios
                    else: 
                        for signal in ids[p]:
                            scoreFP = values_patient[p][signal]
                            score = []
                            for fp in scoreFP.keys():
                                s = np.sum(scoreFP[fp])/len(scoreFP[fp])
                                score.append(s)
                            signalScore = np.sum(score)/len(score)
                            df.loc[signal,metric] = signalScore

            self.df_results[p] = df

        return self.df_results
    
    def report(self):

        print("Report:")
        if self.threshold:
            thresFiducials = self.threshold["thresFiducials"]
            thresScores = self.threshold["thresScores"]
        else:
            thresFiducials = 80
            thresScores = 80
        ids = self.ids
        results = self.df_results
        
        for patient, df in results.items():
            report = pd.Series(pd.NA,index=ids[patient])
            for metric in df.columns:
                if metric == "checkHR":
                    test = df[metric].eq(1)
                    report.loc[df.index[test]] = 1
                elif metric == "checkSP":
                    test = df[metric].eq(1)
                    report.loc[df.index[test]] = 1
                elif metric == "numberProperFiducials":
                    test = df[metric] < thresFiducials
                    report.loc[df.index[test]] = 1
                elif metric == "combinedScore":
                    test = df[metric] < thresScores
                    report.loc[df.index[test]] = 1
                else:
                    pass

            report = report.fillna(0)
            results[patient]["report"] = report
            
        return results
        

    def h5format(self,filename: str):
        print("Saving in h5 file: ",filename)
        all_results = self.df_results
        with h5py.File(filename, 'w') as f:
            for patient, df in all_results.items():
                grp = f.create_group(patient)
                grp.create_dataset("Metrics", data=df.to_numpy())
    
                grp.attrs["metrics"] = np.array(df.columns, dtype="S")
                grp.attrs["ids"] = np.array(list(df.index) ,dtype=float)
        
