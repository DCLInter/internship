import h5py
import numpy as np
import pandas as pd

from tkinter import filedialog

data = {}
segment_ids = {}
demo_info = {}
N_samples = {}
data_path = 'C:/Users/adhn565/OneDrive - City, University of London/Python_3.10/pruebaAttrs.h5'

strange_data = {}
anormal_data = {}
wrongOrdFidu = {}
anormalOrdFP = {}
scores = {}
scores_mix = {}
alignRef = {}
no_detect = {}

if data_path=="":
    data_path = filedialog.askopenfilename(title='Select fiducials file', filetypes=[("Input Files", ".h5")])
else:
    pass

# Opens the archive read mode only with h5py
with h5py.File(data_path, 'r') as f:
    for group_name in f:
        group = f[group_name]
        data[group_name] = {}
        N_samples[group_name] = f[group_name]["segments"].attrs["N-samples"]
        segment_ids[group_name] = group["segments"][0]
        for dtset_name in group:
            data[group_name][dtset_name] = group[dtset_name][()]
            if dtset_name == "segments":
                data[group_name][dtset_name] = group[dtset_name][1:]
        if group_name not in demo_info:
            demo_info[group_name] = {}
        for attr_name, attr_value in group.attrs.items():
            demo_info[group_name][attr_name] = attr_value
        
    ### The name and amount of fiducial points and features are the same for all patients
    fiducial = f[group_name]["segments"].attrs['fiducial_order']
    features = f[group_name][f"mean_{group_name}"].attrs['features']
    fiducial = [f.decode() if isinstance(f, bytes) else f for f in fiducial]
    features = [f.decode() if isinstance(f, bytes) else f for f in features]
data = {k: data[k] for k in ["p000022","p000027"] if k in data}
# print(demo_info)
# print(patient_fiducials)

for i in data.keys():
    print(i)
    wrongOrdFidu[i] = {}
    scores[i] = {}
    scores_mix[i] = {}
    alignRef[i] = {}
    lowsp = []
    anormalHR = []
    no_d = []
    z = 0
    patient_fiducials = pd.DataFrame(data[i]["segments"])
    bp_sigs = np.arange(len(patient_fiducials.columns)//2,len(patient_fiducials.columns))
    patient_fiducials = patient_fiducials.drop(bp_sigs,axis=1)
    for sig in patient_fiducials.columns:
        x = patient_fiducials[sig].values
        fs = int(demo_info[i]["SamplingFrequency"])
        samples = N_samples[i][sig]
        l = len(x)
        n_fiducials = 16        ### 16 fiducial points
        windows = l//n_fiducials
        x2d = x[:windows * n_fiducials].reshape((windows, n_fiducials))
        df_fiducials = pd.DataFrame(x2d,columns=fiducial).dropna(how="all") ### droping the rows filled completly with NaN

        Tpp = df_fiducials["sp"]/fs
        Tpp = np.diff(Tpp)
        Tpp = np.round(Tpp,6)
        IPR = 60/Tpp
        SPr = len(df_fiducials["sp"])
        tSignal = samples/fs ### Length of the signal in seconds
        SPt = (IPR/60)*tSignal
        SPt = np.mean(SPt)
        bmin = 50 ### 50bpm
        bmax = 180 ### 180bpm
        minbeats = (bmin/60)*tSignal
        maxbeats = (bmax/60)*tSignal
        
        ### Saving the data that doesn't have enough Systolic peaks (i.e. the data that is not suitable for the analysis)
        if SPt > 0:
                SPt = round(SPt) ### (theorical) Mean amount of peaks in the length of our signal (time)
        if (SPr < SPt-2) or (SPr < minbeats-1) or (SPr > maxbeats):
            lowsp.append(segment_ids[i][sig])
        ### Checking the order of the fiducials
        for fidu in df_fiducials.columns:
            if fidu not in wrongOrdFidu[i]:
                wrongOrdFidu[i][fidu] = []
            if fidu not in scores[i]:
                scores[i][fidu] = []
            if fidu not in scores_mix[i]:
                scores_mix[i][fidu] = []
            if fidu not in alignRef[i]:
                alignRef[i][fidu] = []

        lppg = ["on","sp","dn","dp","off"]
        ld1 = ["u","v","w"]
        ld2 = ["a","b","c","d","e","f"]
        ld3 = ["p1","p2"]
        l = [lppg,ld1,ld2,ld3]
        c = 0
        # print(segment_ids["p000022"])
        for listfp in l:
            for fidu in listfp:
                ind = listfp.index(fidu)

                p0 = df_fiducials[listfp[ind]]
                p = df_fiducials[listfp[ind-1]] if ind > 0 else p0
                p1 = df_fiducials[listfp[ind+1]] if ind < len(listfp)-1 else p0
                
                if (p0.isna()).any():
                    c+=1
                    wrongOrdFidu[i][listfp[ind]].append(segment_ids[i][sig])
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

                if (pos == False).any():
                    c +=1
                    wrongOrdFidu[i][listfp[ind]].append(segment_ids[i][sig])
                else:
                    pass
        if c != 0:
            z+=1    
        else:
            pass
        if (bmin < IPR).all() and (IPR < bmax).all():
             ### Now we adquire the time between the fiducials
            fiducials_times = {}
            fiducials_tdiff = {}
            mTFP = {}
            nd = 0
            
            for fp in df_fiducials.keys():
                a = df_fiducials[fp]/fs
                ### Checks if any of the fiducials wasnt detected
                if (a.isna()).any():
                    nd+=1
                    continue
                fiducials_times[fp] = np.array(a,dtype=float) ### The temporal position of the fp in seconds
                fiducials_tdiff[fp] = np.round(np.diff(fiducials_times[fp]),6) ### time between the fiducial points
                mTFP[fp] = np.mean(fiducials_tdiff[fp])

            if nd != 0:
                no_d.append(segment_ids[i][sig])

            ### Consistency and Alignment

            for fp in fiducials_tdiff.keys():
                n = len(fiducials_tdiff[fp])
                x = 1 - abs(fiducials_tdiff[fp] - Tpp)/Tpp # SP as reference
                alig1 = np.sum(x)/n
                x = 1 - abs(fiducials_tdiff[fp] - mTFP[fp])/mTFP[fp]
                cons = np.sum(x)/n

                if alig1*100 < 90:
                    alignRef[i][fp].append(segment_ids[i][sig])

                ### Normalized Score combining consistency and alignment
                score_alig_cons = (0.5*(cons) + 0.5*(alig1))*100

                if (score_alig_cons < 90).any():
                    scores_mix[i][fp].append(segment_ids[i][sig])
            ### Score
            for fp in fiducials_tdiff.keys():
                x = abs(fiducials_tdiff[fp] - Tpp)
                score = (1 - (x/Tpp))*100

                if (score < 90).any():
                    scores[i][fp].append(segment_ids[i][sig])
            
        else:
            anormalHR.append(segment_ids[i][sig])

    anormalOrdFP[i] = z
    anormal_data[i] = anormalHR      
    strange_data[i] = lowsp
    no_detect[i] = no_d

df_strange = pd.DataFrame({
'patient': list(strange_data.keys()),
'segment_id': [v for v in strange_data.values()],
'amount': [len(v) for v in strange_data.values()],
'signals_total': [data[v]["segments"].shape[1] for v in list(strange_data.keys())],
'%': [(len(s)/data[p]["segments"].shape[1])*100 for p,s in strange_data.items()]
})

df_anormal = pd.DataFrame({
'patient': list(anormal_data.keys()),
'segment_id': [v for v in anormal_data.values()],
'amount': [len(v) for v in anormal_data.values()],
'signals_total': [data[v]["segments"].shape[1] for v in list(anormal_data.keys())],
'%': [(len(s)/data[p]["segments"].shape[1])*100 for p,s in anormal_data.items()]
})

df_nod = pd.DataFrame({
'patient': list(no_detect.keys()),
'segment_id': [v for v in no_detect.values()],
'amount': [len(v) for v in no_detect.values()],
'signals_total': [data[v]["segments"].shape[1] for v in list(no_detect.keys())],
'%': [(len(s)/data[p]["segments"].shape[1])*100 for p,s in no_detect.items()]
})

wrongOrd_rows = []
for patient, fiducials in wrongOrdFidu.items():
    r = anormalOrdFP[patient] / data[patient]["segments"].shape[1]
    wrongOrd_rows.append([patient, "", "", data[patient]["segments"].shape[1], r * 100])
    for fiducial, signals in fiducials.items():
        wrongOrd_rows.append(["", fiducial, "; ".join(map(str, signals)), len(signals), ""])
df_wrongOrd = pd.DataFrame(wrongOrd_rows, columns=["Patient", "Fiducial", "Segment_id", "Amount Signals", "%"])

score_rows = []
for patient, fiducials in scores.items():
    score_rows.append([patient, "", "", data[patient]["segments"].shape[1]])
    for fiducial, signals in fiducials.items():
        r = len(signals)/data[patient]["segments"].shape[1]
        score_rows.append(["", fiducial, "; ".join(map(str, signals)), len(signals), r*100])
df_scores = pd.DataFrame(score_rows, columns=["Patient", "Fiducial", "Segment_id", "Amount Signals", "%"])

scoremix_rows = []
for patient, fiducials in scores_mix.items():
    scoremix_rows.append([patient, "", "", data[patient]["segments"].shape[1]])
    for fiducial, signals in fiducials.items():
        r = len(signals)/data[patient]["segments"].shape[1]
        scoremix_rows.append(["", fiducial, "; ".join(map(str, signals)), len(signals), r*100])
df_scoresmix = pd.DataFrame(scoremix_rows, columns=["Patient", "Fiducial", "Segment_id", "Amount Signals", "%"])

align_rows = []
for patient, fiducials in alignRef.items():
    align_rows.append([patient, "", "", data[patient]["segments"].shape[1]])
    for fiducial, signals in fiducials.items():
        r = len(signals)/data[patient]["segments"].shape[1]
        align_rows.append(["", fiducial, "; ".join(map(str, signals)), len(signals), r*100])
df_align = pd.DataFrame(align_rows, columns=["Patient", "Fiducial", "Segment_id", "Amount Signals", "%"])

# Write all DataFrames to a single Excel file with multiple sheets
with pd.ExcelWriter("asd.xlsx") as writer:
    df_nod.to_excel(writer, sheet_name="NA values", index=False)
    df_strange.to_excel(writer, sheet_name="Low Sytolic Peaks", index=False)
    df_anormal.to_excel(writer, sheet_name="Anormal HR Data", index=False)
    df_wrongOrd.to_excel(writer, sheet_name="Fiducial Wrong Order", index=False)
    df_scores.to_excel(writer, sheet_name="Scores Low", index=False)
    df_scoresmix.to_excel(writer, sheet_name="ScoresMix Low", index=False)
    df_align.to_excel(writer, sheet_name="Alignment Interval", index=False)

