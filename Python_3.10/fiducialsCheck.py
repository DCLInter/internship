import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotmap import DotMap
import json
import csv

from pyPPG import PPG, Fiducials, Biomarkers
from lib_changes import PPG2, Fiducials2, Biomarkers2 

from pyPPG.datahandling import load_data, plot_fiducials, save_data
import pyPPG.preproc as PP
import pyPPG.fiducials as FP
from lib_changes import fiducials2 as FP2 

import pyPPG.biomarkers as BM
from lib_changes import biomarkers2 as BM2 

import pyPPG.ppg_sqi as SQI
import tkinter as tk
from tkinter import simpledialog
from tkinter import filedialog
import scipy.stats

data = {}
segment_ids = {}
demo_info = {}
strange_data = {}
anormal_data = {}
wrongOrdFidu = {}
anormalOrdFP = {}
strangePRV = {}
fpc_prvc = {}
fpc_prvonc = {}
scores = {}
scores_on = {}
scores_mix = {}
alignRef = {}
no_detect = {}
data_path = 'C:/Users/adhn565/OneDrive - City, University of London/Data/patient_data.h5'

if data_path=="":
    sig_path = filedialog.askopenfilename(title='Select signals file', filetypes=[("Input Files", ".h5")])
else:
    sig_path=data_path

with h5py.File(data_path, 'r') as f:
    for group_name in f:
        group = f[group_name]
        dataset_names = list(group.keys())
        dataset_names = dataset_names[:]
        if dataset_names:
            data[group_name] = group[dataset_names[0]][4:].T
            segment_ids[group_name] =  group[dataset_names[0]][1].T 
    for group_name in f.keys():
        group = f[group_name]
        if group_name not in demo_info:
            demo_info[group_name] = {}
        for attr_name, attr_value in group.attrs.items():
            demo_info[group_name][attr_name] = attr_value

data = {k: data[k] for k in ["p000022"] if k in data}
# data = {k: data[k] for k in list(data.keys())[:50] if k in data}

#### Trying to check the position of the fiducial points by comparing their distances with the HR ####
### Starting with the fp extraction from the original data since we need the HR per window]
for i in data.keys():
    print(i)
    wrongOrdFidu[i] = {}
    fpc_prvc[i] = {}
    fpc_prvonc[i] = {}
    strangePRV[i] = {}
    scores[i] = {}
    scores_on[i] = {}
    scores_mix[i] = {}
    alignRef[i] = {}
    lowsp = []
    anormalHR = []
    no_d = []
    scnorm1 = {}
    z = 0

    for sig in np.arange(len(data[i])):
        signal = DotMap()
        signal.filtering = True
        signal.fL=0.5000001
        signal.fH=12
        signal.order=4
        signal.sm_wins={'ppg':50,'vpg':10,'apg':10,'jpg':10}

        corr_on = ['on', 'dn', 'dp', 'v', 'w', 'f']
        correction=pd.DataFrame()
        correction.loc[0, corr_on] = True
        signal.correction=correction

        signal.name = i
        signal.start_sig = 0
        signal.end_sig = len(data[i][sig])
        signal.v = data[i][sig]
        signal.fs = int(demo_info[i]["SamplingFrequency"])
        prep = PP.Preprocess(fL=signal.fL, fH=signal.fH, order=signal.order, sm_wins=signal.sm_wins)
        signal.ppg, signal.vpg, signal.apg, signal.jpg = prep.get_signals(s=signal)

        s = PPG2(signal)

        #### Fiducials
        fpex = FP2.FpCollection(s=s)
        fiducials = fpex.get_fiducials(s=s)
        fp = Fiducials(fp=fiducials)
        
        ### Checking the order of the fiducials
        for fidu in fiducials.columns:
            if fidu not in wrongOrdFidu[i]:
                wrongOrdFidu[i][fidu] = []
            if fidu not in fpc_prvc[i]:
                fpc_prvc[i][fidu] = []
            if fidu not in fpc_prvonc[i]:
                fpc_prvonc[i][fidu] = []
            if (fidu == "sp") or (fidu == "on") and (fidu not in strangePRV[i]):
                strangePRV[i][fidu] = [] 
            if fidu not in scores[i]:
                scores[i][fidu] = []
            if fidu not in scores_on[i]:
                scores_on[i][fidu] = []
            if fidu not in scores_mix[i]:
                scores_mix[i][fidu] = []
            if fidu not in alignRef[i]:
                alignRef[i][fidu] = []
            if fidu not in scnorm1:
                scnorm1[fidu] = []

        lppg = ["on","sp","dn","dp","off"]
        ld1 = ["u","v","w"]
        ld2 = ["a","b","c","d","e","f"]
        ld3 = ["p1","p2"]
        l = [lppg,ld1,ld2,ld3]
        c = 0
        if segment_ids[i][sig] == 116:
            print("aja")
        for listfp in l:
            for fidu in listfp:
                ind = listfp.index(fidu)
                if ind == 0:
                    p0 = fiducials[listfp[ind]]
                    p1 = fiducials[listfp[ind+1]]
                    pos = p0 < p1
                elif ind == len(listfp)-1:
                    p0 = fiducials[listfp[ind-1]]
                    p1 = fiducials[listfp[ind]]
                    pos = p0 < p1
                else:
                    p = fiducials[listfp[ind-1]]
                    p0 = fiducials[listfp[ind]]
                    p1 = fiducials[listfp[ind+1]]
                    pos = (p0 < p1) & (p < p0)
                if segment_ids[i][sig] == 116 and fidu == "f":
                    print("BBBBBBB",fidu,pos,p0,p1)
                if (pos == False).any():
                    c +=1
                    wrongOrdFidu[i][listfp[ind]].append(segment_ids[i][sig])
                else:
                    pass
        if c != 0:
            z+=1    
        else:
            pass   

        #### Features (we only need HR per window, in this case is IPR (Instantaneous Pulse Rate) [bpm])
        bmex = BM2.BmCollection2(s=s, fp=fp)
        bm_defs, bm_vals = bmex.get_biomarkers2(get_stat=False)
        IPR = bm_vals["ppg_features"]["IPR"]
        ### Saving the data that doesn't have enough Systolic peaks (i.e. the data that is not suitable for the analysis)
        SPr = len(fp.sp) ### Amount of systolic peaks detected in the signal
        tSignal = len(s.ppg)/s.fs ### Length of the signal in seconds
        SPt = (IPR/60)*tSignal ### To define the amount of peaks the signals should have we will use the Heart rate
        SPt = np.mean(SPt)
        minbeats = (50/60)*tSignal
        maxbeats = (180/60)*tSignal
        if SPt > 0:
            SPt = round(SPt) ### (theorical) Mean amount of peaks in 8 seconds (the length of our signal)
        if (SPr < SPt-2) or (SPr < minbeats-1) or (SPr > maxbeats):
            lowsp.append(segment_ids[i][sig]) 
        
        IPRt = np.array(60/IPR) ### Changing the IPR [bpm] to times [s] (which is peak-peak times, this is the way the library does the calc for IPR)
        IPRt = np.round(IPRt,6)
        bmin = 50 ### 50bpm
        bmax = 180 ### 180bpm
        Tpi = np.array(bm_vals["ppg_features"]["Tpi"]) ### Time between onsets
        Tpi = np.round(Tpi,6)
        rest = abs(IPRt - Tpi)

        if (bmin < IPR).all() and (IPR < bmax).all():
            ### Now we adquire the time between the fiducials
            fiducials_times = {}
            fiducials_tdiff = {}
            fiducials_PRV = {}
            mTFP = {}
            mPRVfp = {}
            fiducials_errorPRV = {}
            nd = 0
            
            for fp in fiducials.keys():
                a = fiducials[fp]/s.fs
                ### Checks if any of the fiducials wasnt detected
                if (a.isna()).any():
                    nd+=1
                    continue
                fiducials_times[fp] = np.array(a,dtype=float) ### The temporal position of the fp in seconds
                fiducials_tdiff[fp] = np.round(np.diff(fiducials_times[fp]),6) ### time between the fiducial points
                mTFP[fp] = np.mean(fiducials_tdiff[fp])
                fiducials_PRV[fp] = abs(np.diff(fiducials_tdiff[fp])) ### The difference between the times of those fp per window
                mPRVfp[fp] = np.mean(fiducials_PRV[fp])
                
                ### See the absolute difference bewteen each PRV value (of the fiducials) vs the mean.
                fiducials_errorPRV[fp] = []
                for j in np.arange(len(fiducials_PRV[fp])):
                    v = fiducials_PRV[fp][j]
                    error = abs(mPRVfp[fp]-v)
                    fiducials_errorPRV[fp].append(error)

            if nd != 0:
                no_d.append(segment_ids[i][sig])

            ### Consistency and Alignment
            x = abs(IPRt - Tpi)
            referenceAlig = np.sum(x)/len(IPRt)

            for fp in fiducials_tdiff.keys():
                n = len(fiducials_tdiff[fp])
                x = 1 - abs(fiducials_tdiff[fp] - IPRt)/IPRt # SP as reference
                alig1 = np.sum(x)/n
                relE1 = (abs(x/IPRt))

                x = 1 - abs(fiducials_tdiff[fp] - Tpi)/Tpi # Onests as reference
                alig2 = np.sum(x)/n
                relE2 = (abs(x/Tpi))

                x = 1 - abs(fiducials_tdiff[fp] - mTFP[fp])/mTFP[fp]
                cons = np.sum(x)/n

                if alig1 > referenceAlig or alig2 > referenceAlig:
                    alignRef[i][fp].append(segment_ids[i][sig])

                ### Normalized Score combining consistency and alignment
                score_alig_cons = (0.33*(cons) + 0.34*(alig1) + 0.33*(alig2))*100
                scnorm1[fp].append(score_alig_cons)

                if (score_alig_cons < 90).any():
                    scores_mix[i][fp].append(segment_ids[i][sig])
            ### Score
            for fp in fiducials_tdiff.keys():
                x = abs(fiducials_tdiff[fp] - IPRt)
                score = (1 - (x/IPRt))*100

                if (score < 90).any():
                    scores[i][fp].append(segment_ids[i][sig])

                x = abs(fiducials_tdiff[fp] - Tpi)
                score = (1 - (x/Tpi))*100

                if (score < 90).any():
                    scores_on[i][fp].append(segment_ids[i][sig])
            
            
            


            PRV = abs(np.diff(IPRt))
            PRVon = abs(np.diff(Tpi))
            mPRV = np.mean(PRV)
            mPRVon = np.mean(PRVon)
            ### Checking the PRV values for the baselines(PRV of sp and onsets) have normal values
            if (PRV > 0.05).any():
                strangePRV[i]["sp"].append(segment_ids[i][sig])
            if (PRVon > 0.05).any():
                strangePRV[i]["on"].append(segment_ids[i][sig])
            ###  See the absolute difference bewteen each PRV value vs the mean.
            PRVe = []
            PRVone = []
            
            # print(PRV)
            for j in np.arange(len(PRV)):
                v = PRV[j]
                error = abs(mPRV-v)
                PRVe.append(error)
                # print("hola prv",v,error,mPRV)

            for j in np.arange(len(PRVon)):
                v = PRVon[j]
                error = abs(mPRVon-v)
                PRVone.append(error)
            PRVe = np.array(PRVe)
            PRVone = np.array(PRVone)
            ### Seeing if the "error" or difference with the mean of the FP and the PRV (systolic peaks) and onstes are similar
            for fp in fiducials_errorPRV.keys():
                x = fiducials_errorPRV[fp]
                y = PRVe
                errorFPSP = (abs(x - y)/y)*100
                if (errorFPSP > 50).any():
                    fpc_prvc[i][fp].append(segment_ids[i][sig])
                # print(fp,errorFPSP)
            for fp in fiducials_errorPRV.keys():
                x = fiducials_errorPRV[fp]
                y = PRVone
                errorFPON = (abs(x - y)/y)*100
                if (errorFPON > 50).any():
                    fpc_prvonc[i][fp].append(segment_ids[i][sig])
        else:
            anormalHR.append(segment_ids[i][sig])

    anormalOrdFP[i] = z
    anormal_data[i] = anormalHR      
    strange_data[i] = lowsp
    no_detect[i] = no_d
    anormalPerc = []
    ### To compute anormal data for the mixed score of consistency and alignment we do a percentile analysis
    for fp in scnorm1.keys():
        scores_signals = np.array(scnorm1[fp])
        scoresMedian = np.median(scores_signals)
        percentile_90 = np.percentile(scores_signals,90)
        if (scores_signals > percentile_90).any():
            anormalPerc = [scores_signals[scores_signals > percentile_90]]
        
    
df_strange = pd.DataFrame({
'patient': list(strange_data.keys()),
'segment_id': [v for v in strange_data.values()],
'amount': [len(v) for v in strange_data.values()],
'signals_total': [len(data[v]) for v in list(strange_data.keys())],
'%': [(len(s)/len(data[p]))*100 for p,s in strange_data.items()]
})

df_anormal = pd.DataFrame({
'patient': list(anormal_data.keys()),
'segment_id': [v for v in anormal_data.values()],
'amount': [len(v) for v in anormal_data.values()],
'signals_total': [len(data[v]) for v in list(anormal_data.keys())],
'%': [(len(s)/len(data[p]))*100 for p,s in anormal_data.items()]
})

df_nod = pd.DataFrame({
'patient': list(no_detect.keys()),
'segment_id': [v for v in no_detect.values()],
'amount': [len(v) for v in no_detect.values()],
'signals_total': [len(data[v]) for v in list(no_detect.keys())],
'%': [(len(s)/len(data[p]))*100 for p,s in no_detect.items()]
})

row_strangePRV = []
for patient, fiducials in strangePRV.items():
    row_strangePRV.append([patient, "", "", len(data[patient])])
    for fiducial, signals in fiducials.items():
        r = len(signals)/len(data[patient])*100
        row_strangePRV.append(["", fiducial, "; ".join(map(str, signals)), len(signals),r])
df_strangePRV = pd.DataFrame(row_strangePRV, columns=["Patient", "Baseline", "Segment_id", "Amount Signals","%"])

fpc_prvc_rows = []
for patient, fiducials in fpc_prvc.items():
    fpc_prvc_rows.append([patient, "", "", len(data[patient])])
    for fiducial, signals in fiducials.items():
        fpc_prvc_rows.append(["", fiducial, "; ".join(map(str, signals)), len(signals)])
df_fpc_prvc = pd.DataFrame(fpc_prvc_rows, columns=["Patient", "Fiducial", "Segment_id", "Amount Signals"])

fpc_prvonc_rows = []
for patient, fiducials in fpc_prvonc.items():
    fpc_prvonc_rows.append([patient, "", "", len(data[patient])])
    for fiducial, signals in fiducials.items():
        fpc_prvonc_rows.append(["", fiducial, "; ".join(map(str, signals)), len(signals)])
df_fpc_prvonc = pd.DataFrame(fpc_prvonc_rows, columns=["Patient", "Fiducial", "Segment_id", "Amount Signals"])

wrongOrd_rows = []
for patient, fiducials in wrongOrdFidu.items():
    r = anormalOrdFP[patient] / len(data[patient])
    wrongOrd_rows.append([patient, "", "", len(data[patient]), r * 100])
    for fiducial, signals in fiducials.items():
        wrongOrd_rows.append(["", fiducial, "; ".join(map(str, signals)), len(signals), ""])
df_wrongOrd = pd.DataFrame(wrongOrd_rows, columns=["Patient", "Fiducial", "Segment_id", "Amount Signals", "%"])

score_rows = []
for patient, fiducials in scores.items():
    score_rows.append([patient, "", "", len(data[patient])])
    for fiducial, signals in fiducials.items():
        r = len(signals)/len(data[patient])
        score_rows.append(["", fiducial, "; ".join(map(str, signals)), len(signals), r*100])
df_scores = pd.DataFrame(score_rows, columns=["Patient", "Fiducial", "Segment_id", "Amount Signals", "%"])

scoreon_rows = []
for patient, fiducials in scores_on.items():
    scoreon_rows.append([patient, "", "", len(data[patient])])
    for fiducial, signals in fiducials.items():
        r = len(signals)/len(data[patient])
        scoreon_rows.append(["", fiducial, "; ".join(map(str, signals)), len(signals), r*100])
df_scores_on = pd.DataFrame(scoreon_rows, columns=["Patient", "Fiducial", "Segment_id", "Amount Signals", "%"])

scoremix_rows = []
for patient, fiducials in scores_mix.items():
    scoremix_rows.append([patient, "", "", len(data[patient])])
    for fiducial, signals in fiducials.items():
        r = len(signals)/len(data[patient])
        scoremix_rows.append(["", fiducial, "; ".join(map(str, signals)), len(signals), r*100])
df_scoresmix = pd.DataFrame(scoremix_rows, columns=["Patient", "Fiducial", "Segment_id", "Amount Signals", "%"])

align_rows = []
for patient, fiducials in alignRef.items():
    align_rows.append([patient, "", "", len(data[patient])])
    for fiducial, signals in fiducials.items():
        r = len(signals)/len(data[patient])
        align_rows.append(["", fiducial, "; ".join(map(str, signals)), len(signals), r*100])
df_align = pd.DataFrame(align_rows, columns=["Patient", "Fiducial", "Segment_id", "Amount Signals", "%"])

# Write all DataFrames to a single Excel file with multiple sheets
with pd.ExcelWriter("asdasd.xlsx") as writer:
    df_nod.to_excel(writer, sheet_name="NA values", index=False)
    df_strange.to_excel(writer, sheet_name="Low Sytolic Peaks", index=False)
    df_anormal.to_excel(writer, sheet_name="Anormal HR Data", index=False)
    df_strangePRV.to_excel(writer,sheet_name="Anormal PRV Data", index=False)
    df_wrongOrd.to_excel(writer, sheet_name="Fiducial Wrong Order", index=False)
    df_scores.to_excel(writer, sheet_name="Scores Low", index=False)
    df_scores_on.to_excel(writer, sheet_name="Scores Low", index=False, startcol= len(df_scores.columns)+1)
    df_scoresmix.to_excel(writer, sheet_name="ScoresMix Low", index=False)
    df_align.to_excel(writer, sheet_name="Alignment Interval", index=False)
    df_fpc_prvc.to_excel(writer, sheet_name="Fiducial Wrong PRVerror", index= False)
    df_fpc_prvonc.to_excel(writer, sheet_name="Fiducial Wrong PRVerror", index= False, startcol= len(df_fpc_prvc.columns)+1)
def times_diff():
    t1 = np.arange(0, len(IPR), 0.1)
    t2 = np.arange(0, len(Tpi), 0.1)
    t3 = np.arange(0, len(x), 0.1)

    y1 = np.full_like(t1, IPR[0])
    y2 = np.full_like(t2, Tpi[0])
    y3 = np.full_like(t3, x[0])

    plt.figure(figsize=(8, 5))
    plt.plot(t1,y1, label='T HR')
    plt.plot(t2,y2, label='T onsets')
    plt.plot(t3,y3, label= f'T {fp}')

    plt.xlabel('Windows of the signal')
    plt.ylabel('Time')
    plt.title('Time differences')
    plt.legend()
    plt.tight_layout()
    plt.show()

