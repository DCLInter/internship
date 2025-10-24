import h5py
import pandas as pd
import numpy as np
from metrics_functions import Metrics
class Checker:
    def __init__(self, threshold: dict = {}, data_ext: dict = {}, features_names: list = [], demo_info: dict = {}, samples: dict = {}, ids: dict = {}):

        self.ids = ids.copy()
        self.demo_info = demo_info.copy()
        self.Nsamples = samples.copy()
        self.resultsMetrics = {}
        self.df_results = {}
        self.threshold = threshold
        self.data = data_ext.copy()
        self.fiducial_order = ['on','sp','dn','dp','off','u','v','w','a','b','c','d','e','f','p1','p2']  ### Order of
        self.features_names = features_names

    def windows(self, fiducials: pd.DataFrame, signal = None):

        idx = signal
        patient_fiducials = fiducials
        y = patient_fiducials.iloc[:,idx].values
        l = len(y)
        n_fiducials = 16        ### 16 fiducial points
        windows = l//n_fiducials
        x2d = y[:windows * n_fiducials].reshape((windows, n_fiducials))
        df_fiducials = pd.DataFrame(x2d,columns=self.fiducial_order).dropna(how="all")
    
        return df_fiducials
    
    def metrics(self, patient):
        print("Metrics:")
        # Storage variables for a single patient
        scores = {}
        flagScores = {}
        number_derivativesDetected = {}

        # Process only the specified patient
        lowsp = []
        anormalHR = []
        no_d = []
        num_noD = []
        overlapNum = []
        ratioFpDetect = []
        flagSignals = 0

        idx = 0
        patient_fiducials = pd.DataFrame(self.data[patient]["segments"])
        patient_fiducials.columns = self.ids[patient]
        for sig in self.ids[patient]:
            fs = int(self.demo_info[patient]["SamplingFrequency"])
            nsamples = self.Nsamples[patient]
            
            df_fiducials = self.windows(patient_fiducials, idx)
            
            for fidu in df_fiducials.columns:
                if fidu not in flagScores:
                    flagScores[fidu] = []

            metrics = Metrics(df_fiducials, fs, nsamples, self.threshold)

            flagND = metrics.checkNA()
            if flagND > 0:
                no_d.append(sig)

            numSP = metrics.checkNumPeaks()
            if numSP == True:
                lowsp.append(sig)

            wrongOrdFidu, numFlags, winFlags = metrics.checkOrder(patient, sig)

            wNum = len(winFlags.keys())
            fpNum = wNum * 16

            if numFlags != 0:
                flagSignals += 1

            na_ratio = (flagND / fpNum) * 100
            num_noD.append(na_ratio)

            list_deriv = metrics.list_derivatives
            numD = metrics.numDerivatives
            for d in numD.keys():
                if d not in number_derivativesDetected:
                    number_derivativesDetected[d] = []
                ffp = numD[d]
                n = wNum * len(list_deriv[d])
                r = (1 - ffp / n) * 100
                number_derivativesDetected[d].append(r)
            ratio = (1 - numFlags / fpNum) * 100
            ratioFpDetect.append(ratio)

            winOverlap = 0
            for win, f in winFlags.items():
                if f != 0:
                    winOverlap += 1
            overlapNum.append(winOverlap)

            flagHR = metrics.checkHR()
            if flagHR:
                anormalHR.append(sig)

            align, cons = metrics.consistency_alignment()
            comScores = metrics.scoreCombined()
            scores[sig] = comScores
            for fp in comScores.keys():
                if (comScores[fp] < self.threshold["thresScores"]).any():
                    flagScores[fp].append(sig)
            idx += 1

        number_overlapWindows = overlapNum
        number_overlapFiducial = flagSignals
        abnormalHR_data = anormalHR
        abnormalSP_data = lowsp
        no_detect = no_d
        num_noDetect = num_noD
        number_fiducialsDetect = ratioFpDetect

        resultsMetrics = {
            "checkHR": abnormalHR_data,
            "checkSP": abnormalSP_data,
            "checkNAvalues": no_detect,
            "numberNAvalues": num_noDetect,
            "checkOrderFiducials": wrongOrdFidu,
            "numberOverlapFiducials": number_overlapFiducial,
            "numberOverlapWindows": number_overlapWindows,
            "flagForScore": flagScores,
            "combinedScore": scores,
            "numberProperFiducials": number_fiducialsDetect,
            "numberProperFiducials_byDerivatives": number_derivativesDetected
        }
        self.resultsMetrics[patient] = resultsMetrics.copy()

        return resultsMetrics

    def results(self, patient):

        print("Analysis of metrics:")
        remove = ["numberOverlapWindows","checkOrderFiducials","numberOverlapFiducials","flagForScore","checkNAvalues"]
        results = {k: v for k, v in self.resultsMetrics[patient].items() if k not in remove}
        ids = self.ids[patient]

        df_results = pd.DataFrame(index=ids, columns=results.keys())
        for metrics in results.keys():
            obj = type(results[metrics])

            if obj == list:
                l = results[metrics]
                if metrics == "numberProperFiducials" or metrics == "numberNAvalues":
                    df_results.loc[:,metrics] = l
                elif l:
                    df_results.loc[l,metrics] = 1
            elif metrics == "numberProperFiducials_byDerivatives":
                for deriv in results[metrics].keys():
                    df_results.loc[:,deriv] = results[metrics][deriv]
            elif metrics == "combinedScore":
                for sig in ids:
                    sc = results[metrics][sig]
                    n = sc.shape[0]
                    mscore = sc.sum()/n
                    df_results.loc[sig,metrics] = mscore.sum()/len(mscore)
        df_results = df_results.replace(np.nan,0)
        self.df_results[patient] = df_results
        return df_results
    
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
                df["ids"] = list(df.index)
                grp.create_dataset("Metrics", data=df.to_numpy())
    
                grp.attrs["metrics"] = np.array(df.columns, dtype="S")
        
