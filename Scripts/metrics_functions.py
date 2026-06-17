import numpy as np
import pandas as pd
import scipy.stats as stats

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
    
    def timeArrays(self):
        ### We adquire the time between the fiducials (fp)
        fp = self.fiducials.copy()
        fd_t = fp/self.fs
        fd_td = fd_t.diff().iloc[1:]
        m = fd_td.mean(axis=0)

        self.fiducials_times = fd_t
        self.fiducials_tdiff = fd_td
        self.mTFP = m
    
    def checkNA(self):
        flag = 0
        flag = self.fiducials.isna().sum().sum()
        
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
    
    def extra(self):
        fp_c = np.array(self.fiducials["c"])
        fp_d = np.array(self.fiducials["d"])
        if np.array_equal(fp_c, fp_d):
            return True
    def checkOrder(self, signal):
        lppg = ["on","sp","dn","dp","off"]
        ld1 = ["u","v","w"]
        ld2 = ["a","b","c","d","e","f"]
        ld3 = ["p1","p2"]
        l = [lppg,ld1,ld2,ld3]
        self.list_derivatives = {"ppg": lppg, "d1": ld1, "d2": ld2, "d3": ld3}
        numFlagFidu = 0
        winOverlap = {}
        flags = {}
        numPerDerivatives = {}
        dic_flags = {}
        percentage_flags_perWindow = {}

        for listfp in l:
            fld = 0
            for fidu in listfp:
                if fidu not in dic_flags:
                    dic_flags[fidu] = []

                ind = listfp.index(fidu)
                p0 = self.fiducials[listfp[ind]]
                p = self.fiducials[listfp[ind-1]] if ind > 0 else p0
                p1 = self.fiducials[listfp[ind+1]] if ind < len(listfp)-1 else p0
                
                if (p0.isna()).any():
                    numFlagFidu += len(np.where(p0.isna())[0])
                    flags[fidu] = np.where(p0.isna())[0]
                    dic_flags[listfp[ind]].append(signal)
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
                    dic_flags[listfp[ind]].append(signal)
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
        
        for fp in flags.keys():
            percentage_flags_perWindow[fp] = (len(flags[fp])/self.fiducials.shape[0])*100

        ### dic_flags: dictionary with the fiducial points as keys and the signals that have problems with the order as values
        ### numFlagFidu: total number of fiducial points (including all 16 different fiducials) that have problems with the order in the signal
        ### winOverlap: dictionary with the windows as keys and the number of fiducial points that have problems in that window as values
        return dic_flags, numFlagFidu, winOverlap, percentage_flags_perWindow
    
    def checkHR(self):

        if self.thresholds:
            bmin = self.thresholds["bmin"]
            bmax = self.thresholds["bmax"]
        else:
            bmin = 50
            bmax = 180
        HRsig = np.mean(self.IPR)
        if bmin < HRsig and HRsig < bmax:
            flag = False
        else:
            flag = True

        return flag

    def consistency_alignment(self):
        
        fd_t = self.fiducials_tdiff.copy()

        alig = abs(fd_t.sub(self.Tpp, axis=0))
        alig = alig.div(self.Tpp, axis=0)
        alignment = (1 - alig)*100

        cons = abs(fd_t.sub(self.mTFP, axis=1))
        cons = cons.div(self.mTFP, axis=1)
        consistency = (1 - cons)*100

        self.alignment = alignment
        self.consistency = consistency
        
        return self.alignment, self.consistency
    
    def scoreCombined(self):
        
        if self.thresholds:
            w1 = self.thresholds["w_consistency"]
            w2 = self.thresholds["w_alignment"]
        else:
            w1 = 0.25
            w2 = 0.75
        alig = self.alignment.copy()
        cons = self.consistency.copy()
        scCons = w1*cons
        scAlig = w2*alig

        scores = scAlig.add(scCons,axis=1)
        self.scores = scores
        
        return self.scores
    


