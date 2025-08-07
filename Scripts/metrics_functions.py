import numpy as np
import pandas as pd

class Metrics:
    def __init__(self, fiducials: pd.DataFrame, fs: int = 125, samples: int = 1125, thresholds: dict = {}):

        self.fiducials = fiducials
        self.sp = fiducials["sp"]
        self.fs = fs
        self.samples = samples
        self.thresholds = thresholds
        self.time = self.samples/self.fs
        self.IPR, self.Tpp = self.getIPR_TPP()
        self.timeArrays()
    
    def getIPR_TPP(self):
        Tpp = self.sp/self.fs
        Tpp = np.diff(Tpp)
        Tpp = np.round(Tpp,6)
        IPR = 60/Tpp

        return IPR, Tpp
    
    def checkNA(self):
        flag = 0
        for fp in self.fiducials.keys():
            a = self.fiducials[fp]
            if (a.isna()).any():
                flag += len(np.where(a.isna()[0]))
        
        return flag

    def checkNumPeaks(self):
        if self.thresholds:
            limit = self.thresholds["sp_limit"]
        else:
            limit = 2

        SPr = len(self.fiducials["sp"])
        tSignal = self.time ### Length of the signal in seconds
        SPt = (self.IPR/60)*tSignal
        SPt = np.mean(SPt)
        
        if SPt > 0:
                SPt = round(SPt) ### (theorical) Mean amount of peaks in the length of our signal (time)
        if (SPr < SPt - limit):
            flag = True
        else:
            flag = False

        return flag
    
    def checkOrder(self, dic_flags: dict, patient: str, signal):
        lppg = ["on","sp","dn","dp","off"]
        ld1 = ["u","v","w"]
        ld2 = ["a","b","c","d","e","f"]
        ld3 = ["p1","p2"]
        l = [lppg,ld1,ld2,ld3]
        self.list_derivatives = {"ppg": lppg, "d1": ld1, "d2": ld2, "d3": ld3}
        numFlagFidu = 0
        otro = 0
        winOverlap = {}
        flags = {}
        numPerDerivatives = {}
        for listfp in l:
            fld = 0
            for fidu in listfp:
                ind = listfp.index(fidu)
                p0 = self.fiducials[listfp[ind]]
                p = self.fiducials[listfp[ind-1]] if ind > 0 else p0
                p1 = self.fiducials[listfp[ind+1]] if ind < len(listfp)-1 else p0
                
                if (p0.isna()).any():
                    numFlagFidu += len(np.where(p0.isna())[0])
                    flags[fidu] = np.where(p0.isna())[0]
                    dic_flags[patient][listfp[ind]].append(signal)
                    continue
                
                if (p.isna()).any():
                    na_indx = p[p.isna()].index
                    p = p.dropna()
                    p0 = p0.drop(na_indx)
                    p1 = p1.drop(na_indx)
                if (p1.isna()).any():
                    na_indx = p1[p1.isna()].index
                    p1 = p1.dropna()
                    p = p.drop(na_indx)
                    p0 = p0.drop(na_indx)

                if ind == 0:
                    pos = p0 < p1
                elif ind == len(listfp)-1:
                    pos = p < p0
                else:
                    pos = (p < p0) & (p0 < p1)

                flags[fidu] = np.where(pos == False)[0]
                if (pos == False).any():
                    fld += len(np.where(pos == False)[0])
                    numFlagFidu += len(np.where(pos == False)[0])
                    dic_flags[patient][listfp[ind]].append(signal)
                else:
                    pass
            if listfp == lppg:
                numPerDerivatives["ppg"] = fld
            elif listfp == ld1:
                numPerDerivatives["d1"] = fld
            elif listfp == ld2:
                numPerDerivatives["d2"] = fld
            elif listfp == ld3:
                numPerDerivatives["d3"] = fld
        self.numDerivatives = numPerDerivatives
                
        for win in np.arange(self.fiducials.shape[0]):
            cont = 0
            for fp in flags.keys():
                if (flags[fp] == win).any():
                    cont +=1
            winOverlap["win"+str(win)] = cont

        return dic_flags, numFlagFidu, winOverlap, otro
    
    def checkHR(self):

        if self.thresholds:
            bmin = self.thresholds["bmin"]
            bmax = self.thresholds["bmax"]
        else:
            bmin = 50
            bmax = 180
        HRsig = np.mean(self.IPR)
        print(HRsig, self.IPR)
        if bmin < HRsig and HRsig < bmax:
            flag = False
        else:
            flag = True

        return flag

    def timeArrays(self):
        ### We adquire the time between the fiducials (fp)
        self.fiducials_times = {}
        self.fiducials_tdiff = {}
        self.mTFP = {}
        for fp in self.fiducials.keys():
            a = self.fiducials[fp]/self.fs
            ### Checks if any of the fiducials wasnt detected
            if (a.isna()).any():
                continue

            self.fiducials_times[fp] = np.array(a,dtype=float) ### The temporal position of the fp in seconds
            self.fiducials_tdiff[fp] = np.round(np.diff(self.fiducials_times[fp]),6) ### time between the fp
            self.mTFP[fp] = np.mean(self.fiducials_tdiff[fp])

    def consistency_alignment(self):
        self.alignment = {}
        self.consistency = {}
        for fp in self.fiducials_tdiff.keys():
            n = len(self.fiducials_tdiff[fp])
            alig = 1 - abs(self.fiducials_tdiff[fp] - self.Tpp)/self.Tpp
            cons = 1 - abs(self.fiducials_tdiff[fp] - self.mTFP[fp])/self.mTFP[fp]

            self.alignment[fp] = alig*100
            self.consistency[fp] = cons*100

        return self.alignment, self.consistency
    
    def scoreCombined(self):
        if self.thresholds:
            w1 = self.thresholds["w_consistency"]
            w2 = self.thresholds["w_alignment"]
        else:
            w1 = 0.25
            w2 = 0.75

        self.scores = {}
        for fp in self.alignment.keys():
            score_alig_cons = (w1*(self.consistency[fp]) + w2*(self.alignment[fp]))
            self.scores[fp] = score_alig_cons

        return self.scores
    


