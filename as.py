import numpy as np
import pandas as pd
from Scripts.other_functions import bland_altman_plot
import matplotlib.pyplot as plt
from Scripts.other_functions_PPG import Others
from Scripts.fearture_extraction import Feature_Extraction

# x = np.array([10,20,15,12,14,16,19,18,17])
# y = np.array([9,20,16,23,14,15,17,16,19])

# bland_altman_plot(x,y,"test")
data_path = "C:/Users/adhn565/Documents/Data/completo_conAttrs_16_7_25.h5"
fext = Feature_Extraction(data_path,"none.h5","none.csv")
data = fext.data
demo_info = fext.demo_info
ids = fext.segment_ids
extra = Others(data=data,demo_info=demo_info,segments_ids=ids)
extra.signal_analysisPPG("p000001",)