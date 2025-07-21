import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import h5py
import matplotlib.pyplot as plt

data = {}
data_path = "Data/patient_data/h5"
with h5py.File(data_path, 'r') as f:
    for group in f.keys():
        data[group] = f[group]
        for dataset in f[group].keys():
            data[group][dataset] = f[group][dataset][()]
            data[group][dataset] = f[group][dataset][len(f[group][dataset]) // 2:]  # Truncate to half length

for group, datasets in data.items():
    peaks = find_peaks(datasets, distance=10)
    valleys = find_peaks(-datasets, distance=10)
    peaks_values = datasets[peaks[0]]
    valleys_values = datasets[valleys[0]]
    max = max(peaks_values)
    min = min(valleys_values)
    max_peaks = peaks_values[peaks_values == max]
    min_valleys = valleys_values[valleys_values == min]

    plt.figure(figsize=(10, 5))
    plt.plot(datasets, label='Data')
    plt.plot(max_peaks, 'ro', label='Peaks')
    plt.plot(min_valleys, 'bo', label='Valleys')
    plt.title(f'{group} - {datasets}')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
