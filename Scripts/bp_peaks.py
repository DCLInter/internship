import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import h5py
import matplotlib.pyplot as plt

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

for patient, data_df in data.items():
    for sig in data_df.index:
        s = data_df.iloc[sig]
        ### Know that the index for the points are just the number of the sample (starting in 0)
        signal = s[4:].reset_index(drop=True)
        max_signal = max(signal)
        min_signal = min(signal)
        
        peaks, _ = find_peaks(signal, height=max_signal*0.95)
        valleys, _ = find_peaks(-signal, height= -min_signal*1.1)

        pkdiff = np.diff(peaks)
        vdiff = np.diff(valleys)
        mpkdiff = np.mean(pkdiff)
        stdpk = np.std(pkdiff)
        mvdiff = np.mean(vdiff)
        stdv = np.std(vdiff)
        print(valleys,vdiff,mvdiff)
        print(peaks,pkdiff,mpkdiff)
        pkd = []
        vd = []
        for p in pkdiff:
            if p < 40:
                idx = np.where(pkdiff == p)[0]
                pkd.append(idx)

        for p in vdiff:
            if p < 40:
                idx = np.where(vdiff == p)[0]
                vd.append(idx)
        
        peaks_clean = np.delete(peaks,pkd)
        valleys_clean = np.delete(valleys,vd)
        # if peaks[0] > valleys[0]:
        #     peaks_clean = peaks
        #     print(sig,len(peaks),len(valleys))
        #     vd = []
        #     pk = 0
        #     for p in np.arange(len(valleys)-1):
        #         v0 = valleys[p]
        #         v1 = valleys[p+1]
        #         p0 = peaks[pk]
        #         if (abs(p0 - v0)) > (abs(v0 - v1)):
        #             vd.append(p)
        #             pk -= 1
        #         else:
        #             pk += 1
        #     valleys_clean = np.delete(valleys,vd)
        # else:
        #     valleys_clean = valleys
        #     pkd = []
        #     vl = 0
        #     for p in np.arange(len(peaks)-1):
        #         p0 = peaks[p]
        #         p1 = peaks[p+1]
        #         v0 = valleys[vl]
        #         if (abs(p0 - v0)) > (abs(p0 - p1)):
        #             vd.append(p)
        #             vl -= 1
        #         else:
        #             vl += 1
        #     peaks_clean = np.delete(peaks,pkd)

        plt.figure(figsize=(10, 5))
        plt.plot(signal, label='Data')
        plt.plot(signal[peaks], 'ro', label='Peaks')
        plt.plot(signal[peaks_clean], 'bx', label='Pk clean')
        plt.plot(signal[valleys], 'bo', label='Valleys')
        plt.plot(signal[valleys_clean], 'gx', label='Clean')
        plt.title(f'{patient} - {sig}')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
        # plt.pause(2)
        plt.close()
