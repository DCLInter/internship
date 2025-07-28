import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import h5py
import matplotlib.pyplot as plt

def plot(signal,peaks,valleys,show: bool =False, delay: int = 1):
    if show:
        plt.figure(figsize=(10, 5))
        plt.plot(signal, label='Data')
        plt.plot(signal[peaks], 'ro', label='Peaks')
        plt.plot(signal[valleys], 'bo', label='Valleys')
        plt.plot(signal[valleys_clean], 'bo', label='Valleys')
        plt.title(f'{patient} - {sig}')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
        plt.pause(delay)
        plt.close()
    else:
        pass

data = {}
segment_ids = {}
data_path = "C:/Users/adhn565/Documents/Data/patient_data.h5"
with h5py.File(data_path, 'r') as f:
    for group_name in f:
        group = f[group_name]
        for dataset in group:
            d = group[dataset][:].T
            segment_ids[group_name] = group[dataset][1]
            d = d[len(d) // 2:]  # Truncate to half length
            data[group_name] = pd.DataFrame(d) 
# data = {k: data[k] for k in ["p000001"]}
bp_values = {}
a = 0
for patient, data_df in data.items():
    print(patient)
    list_bp = []
    for sig in data_df.index:
        s = data_df.iloc[sig]
        ### Know that the index for the points are just the number of the sample (starting in 0)
        signal = s[4:].reset_index(drop=True)
        max_signal = max(signal)
        min_signal = min(signal)
        
        peaks, _ = find_peaks(signal, height= max_signal*0.9)
        valleys, _ = find_peaks(-signal, height= -min_signal*1.1)

        pkdiff = np.diff(peaks)
        vdiff = np.diff(valleys)
        pkd = []
        vd = []
        for p in pkdiff:
            if p < 40:
                idx = np.where(pkdiff == p)[0]
                if len(idx) > 1:
                    for i in idx:
                        p1 = signal[peaks[i]]
                        p2 = signal[peaks[i+1]]
                        if p1 > p2:
                            pkd.append(i+1)
                        else:
                            pkd.append(i)
                else:    
                    p1 = signal[peaks[idx[0]]]
                    p2 = signal[peaks[idx[0]+1]]
                    if p1 > p2:
                        pkd.append(idx[0]+1)
                    else:
                        pkd.append(idx[0])

        for p in vdiff:
            if p < 40:
                idx = np.where(vdiff == p)[0]
                if len(idx) > 1:
                    for i in idx:
                        
                        p1 = signal[valleys[i]]
                        p2 = signal[valleys[i+1]]
                        if p1 > p2:
                            vd.append(i)
                        else:
                            vd.append(i+1)
                else:    
                    p1 = signal[valleys[idx[0]]]
                    p2 = signal[valleys[idx[0]+1]]
                    if p1 > p2:
                        vd.append(idx[0])
                    else:
                        vd.append(idx[0]+1)
    
        peaks_clean = np.delete(peaks,pkd)
        valleys_clean = np.delete(valleys,vd)
        flag = [False,0]

        try:
            if peaks_clean[0] > valleys_clean[0]:
                valleys_clean = np.delete(valleys_clean,0)
        except:
            print("Something went wrong, signal: ",sig,"- number of peaks and valleys: ",len(peaks),len(valleys),"- number after cleaning",len(peaks_clean),len(valleys_clean))
            # flag = [True,sig]
            a += 1
        peaks_values = signal[peaks_clean]
        valleys_values = signal[valleys_clean]
        
        if len(peaks_clean) == 0:
            SBP = np.nan
        elif len(valleys_clean) == 0:
            DBP = np.nan
        else:
            SBP = np.average(peaks_values)
            DBP = np.average(valleys_values)
            
        MAP = np.mean(signal)
        
        list_bp.append([SBP,DBP,MAP])

        show = False
        if flag[0] == True:
            show = True
            print("SBP, DBP, MAP: ",SBP,DBP,MAP)
        plot(signal,peaks_clean,valleys_clean,show=show,delay=10)

    df_bp = pd.DataFrame(list_bp, columns=["SBP","DBP","MAP"])
    bp_values[patient] = df_bp
    
print("Fallo completamente: ", a)
with h5py.File("BP_values.h5", "w") as f:
    for patient, df in bp_values.items():
        group = f.create_group(patient)
        dataset = group.create_dataset("Bp_values",data= df.T.to_numpy())
        
