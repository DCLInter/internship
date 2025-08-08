import numpy as np
import pandas as pd
import matplotlib as plt
from other_functions import bland_altman_plot
import matplotlib.pyplot as plt
from other_functions_PPG import Others
from fearture_extraction import Feature_Extraction

# x = np.array([10,20,15,12,14,16,19,18,17])
# y = np.array([9,20,16,23,14,15,17,16,19])

# bland_altman_plot(x,y,"test")
# data_path = "C:/Users/adhn565/Documents/Data/patient_data.h5"
# fext = Feature_Extraction(data_path,"none.h5","none.csv")
# data = fext.data
# demo_info = fext.demo_info
# ids = fext.segment_ids
# extra = Others(data=data,demo_info=demo_info,segments_ids=ids)
# extra.signal_analysisPPG("p000001",specific_signal=55)

x = np.array([10,20,15,12,14,16,17,18,13,28,3])
Q1 = np.percentile(x, 25)
Q3 = np.percentile(x, 75)
IQR = Q3 - Q1
outliers = x[(x < Q1 - 1.5 * IQR) | (x > Q3 + 1.5 * IQR)]
median = np.median(x)
lower_whisker = np.min(x[x >= Q1 - 1.5 * IQR])
upper_whisker = np.max(x[x <= Q3 + 1.5 * IQR])
plt.figure(figsize=(9,7))
plt.boxplot(x, vert=False,patch_artist=True, boxprops=dict(facecolor='lightblue'), flierprops=dict(marker='o', markerfacecolor='red', markersize=8))
plt.title("Interquartile Range")
plt.text(lower_whisker, 1.09, "Lower whisker\n(Q1 - 1.5*IQR)",
         ha='center', color='blue', fontweight='bold')
plt.text(Q1, 0.89, "Q1",
         ha='center', color='green', fontweight='bold')
plt.text(Q3, 0.89, "Q3",
         ha='center', color='green', fontweight='bold')
plt.text(median, 1.09, "Median",
         ha='center', color='purple', fontweight='bold')
plt.text(upper_whisker, 1.09, "Upper whisker\n(Q3 + 1.5*IQR)",
         ha='center', color='blue', fontweight='bold')
for val in outliers:
    plt.plot(val, 1, 'ro')  # red dot
    plt.text(val, 1.02, "Outlier", ha='center', color='red', fontweight='bold')
plt.show()