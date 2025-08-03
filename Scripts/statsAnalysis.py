import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math
import scipy.stats as sc
import tkinter as tk
from tkinter import filedialog
import os
from sklearn.ensemble import IsolationForest
from other_functions import search_feat

class STATS:
    def __init__(self, data: dict, features: list, savefolder: str, segment_ids: dict = None):
        self.data = data
        self.features = features
        self.segment_ids = segment_ids
        self.savefolder = savefolder
        os.makedirs(savefolder, exist_ok=True)
        normal_feats = {}
        for p in data.keys():
            save_folder = os.path.join(savefolder,p)
            os.makedirs(save_folder, exist_ok=True)
            full_data = pd.DataFrame(index=["Kurtosis","Skewness","IQR","STD","Mean","Median","Distribution","pvalue","Samples"])
            out_all = pd.DataFrame(index=["IQR","MAD"])
            print("Analyzing: ",p)
            normal_feats[p] = {"mean":[],"median":[]}
            # Loop through datasets for each patient
            # Here we are only taking the first two datasets for each patient, you can change this
            list_dataset = list(data[p].keys())
            for d in list_dataset[:2]:
                print("Dataset: ",d)
                nf = 0
                save_QQ = os.path.join(save_folder,"QQplots_" + d.replace( "_"+p , ""))
                save_BP = os.path.join(save_folder,"BoxPlots_" + d.replace( "_"+p , ""))
                save_hist = os.path.join(save_folder,"Histograms_" + d.replace( "_"+p , ""))
                
                os.makedirs(save_BP, exist_ok=True)
                os.makedirs(save_QQ, exist_ok=True)
                os.makedirs(save_hist, exist_ok=True)

                self.QQplot_all(p,save_QQ, dataset=d)
                self.Hist_all(p,save_hist,dataset=d)

                # Loop through features for each dataset
                print("Analyzing features for dataset: ",d)
                for f in range(0,len(features)):
                    st_data,ot_data = self.analysis_feat(patient=p,feature=f,dataset=d,stats_data=full_data,outliers_data=out_all, save_BP=save_BP)
                
                st_data.to_csv(os.path.join(save_folder, f"{d}.csv"),index=True)
                ot_data.to_csv(os.path.join(save_folder, f"{d}_outliers.csv"),index=True)
                normal_feats[p][d.replace( "_"+p , "" )].append(nf)

            df_nf = pd.DataFrame(normal_feats)
            df_nf.to_csv(os.path.join(savefolder,"NormalDistributions.csv"))

        self.plot_overPatients(savefolder=savefolder)
        self.isolationForest(data)

    def QQplot_all(self, patient: str, save_folder: str, dataset: str):
        data = self.data
        data_matrix = data[patient][dataset]
        features = self.features
        n_features = data_matrix.shape[0]
        features_per_plot = [range(0, 10), range(10, 20), range(20, n_features)]  # Adjust as needed

        for idx, feature_range in enumerate(features_per_plot, 1):
            n = len(feature_range)
            cols = 5
            rows = math.ceil(n / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))

            for i, f in enumerate(feature_range):
                row = i // cols
                col = i % cols
                ax = axes[row, col] if rows > 1 else axes[col]
                x = data_matrix[f]
                sm.qqplot(x, line="s", ax=ax)
                ax.set_title(f"Q-Q plot - {features[f]}")

            # Hide unused subplots
            for j in range(n, rows * cols):
                row = j // cols
                col = j % cols
                fig.delaxes(axes[row, col] if rows > 1 else axes[col])

            plt.subplots_adjust(hspace=0.3, wspace=0.3)
            save_path = os.path.join(save_folder, f"QQplot_part{idx}.png")
            plt.savefig(save_path)
            plt.close(fig)

    def Hist_all(self, patient: str, save_folder: str, dataset: str):
        data = self.data
        features = self.features
        data_matrix = data[patient][dataset]
        n_features = data_matrix.shape[0]
        features_per_plot = [range(0, 10), range(10, 20), range(20, n_features)]  # Adjust as needed

        for idx, feature_range in enumerate(features_per_plot, 1):
            n = len(feature_range)
            cols = 5
            rows = math.ceil(n / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))

            for i, f in enumerate(feature_range):
                row = i // cols
                col = i % cols
                ax = axes[row, col] if rows > 1 else axes[col]
                x = data_matrix[f]
                ax.hist(x,bins=50, color='skyblue', edgecolor='black')
                ax.set_title(f"Histogram - {features[f]}")

            # Hide unused subplots
            for j in range(n, rows * cols):
                row = j // cols
                col = j % cols
                fig.delaxes(axes[row, col] if rows > 1 else axes[col])

            plt.subplots_adjust(hspace=0.3, wspace=0.3)
            save_path = os.path.join(save_folder, f"Histograms_part{idx}.png")
            plt.savefig(save_path)
            plt.close(fig)

    def hisMedians(self, savefolder:str, dataset:str):
        data = self.data
        features = self.features
        savefolder = os.path.join(savefolder,f"Histogram_{dataset}")
        os.makedirs(savefolder, exist_ok=True)
        for f in features:
            df_feat = []
            image = os.path.join(savefolder,f"Histogram of {f}.png")
            for p in data.keys():
                idxf = search_feat(f)
                x = data[p][f"{dataset}_{p}"][idxf]
                x = np.median(x)
                df_feat.append(x)
            plt.figure(figsize=(6, 6))
            plt.hist(df_feat, bins=60, color='skyblue', edgecolor='black')
            plt.title(f"Histogram of {dataset} of {f} (All patients)")
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(image)
            # plt.show()
            plt.close()

    def boxMedians(self, savefolder:str, dataset:str):
        data = self.data
        features = self.features
        savefolder = os.path.join(savefolder,f"Boxplot_{dataset}")
        os.makedirs(savefolder, exist_ok=True)
        for f in features:
            df_feat = []
            image = os.path.join(savefolder,f"Boxplot of {f}.png")
            for p in data.keys():
                idxf = search_feat(f)
                x = data[p][f"{dataset}_{p}"][idxf]
                x = np.median(x)
                df_feat.append(x)
            df_feat = [v for v in df_feat if not np.isnan(v)]
            plt.figure(figsize=(6, 6))
            plt.boxplot(df_feat, showfliers=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
            plt.title(f"Boxplot of {dataset} of {f} (All patients)")
            plt.ylabel('Value')
            plt.tight_layout()
            plt.savefig(image)
            # plt.show()
            plt.close()

    def plot_overPatients(self, savefolder: str):
        for d in ["mean","median"]:
            self.hisMedians(savefolder=savefolder,dataset=d)
            self.boxMedians(savefolder=savefolder,dataset=d)

    def isolationForest(self):
        data = self.data
        features = self.features
        savefolder = self.savefolder
        # For isolation forest its a really good tool for identifying outliers wit multiple variables (in this case we are inputing all the features)
        ot_isf = pd.DataFrame(index=data.keys())
        for patient in data.keys():
            print(patient)
            datasets = list(data[patient].keys())
            npData = data[patient][datasets[0]]
            x = pd.DataFrame(npData.T,columns=features)
            # Drop NaNs for Isolation Forest
            x_clean = x.dropna()
            isl = IsolationForest(contamination= "auto", random_state=42)
            isl.fit(x_clean)
            labels = isl.predict(x_clean)
            # Create a full labels array with NaN for dropped indices
            full_labels = pd.Series(np.nan, index=x.index)
            full_labels[x_clean.index] = labels  #I got the outliers labels (1,-1) for the original dataset, the nan value mantains but for the plot it doesnt matter (the plot ignores it)
            outliers_iso = x[full_labels == -1]
            amountOut = (len(outliers_iso)/len(x))*100
            ot_isf[patient] = amountOut
        savefile = os.path.join(savefolder, "outliers_isolationForest.csv")
        ot_isf.to_csv(savefile)

        return ot_isf

    def analysis_feat(self, stats_data: pd.DataFrame, outliers_data: pd.DataFrame, save_BP: str, data_analysis = None, patient:str = None ,feature: int = None, dataset: str = None):
        data = self.data
        segment_ids = self.segment_ids
        features = self.features
        
        if data_analysis is not None:
            x = data_analysis

        x = data[patient][dataset+"_"+patient][feature]
        x = pd.Series(x)
        n = len(x)
        print(features[feature],n)

        if (x.isna()).any():
            l = x.isna().sum()
            l_per = l/n
            signal = x[x.isna()].index()
            id = []
            for i in signal:
                id.append(segment_ids[patient][i])
            print(features[feature],"Nan value",x[x.isna()],"number",x.isna().sum(),"percentage",l_per)

        ### Ploting the distribution with a histogram
        plt.figure(figsize=(6, 6))
        plt.hist(x, bins=100, color='skyblue', edgecolor='black')
        plt.title(f"Histogram of feature {features[feature]}")
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
        plt.close()

        ### Scipy normality test, still recommended to check the QQ plots in the case of large data (samples >300)
        if n > 40:
            normt = sc.normaltest(x)
            if normt.pvalue < 1e-3: #Rejects the Null hypothesis
                nt = "Not normal"
                
            else:
                nt = "Normal"
                # nf+=1
        elif n > 3:
            normt = sc.shapiro(x)
            if normt.pvalue < 0.05: #Rejects the Null hypothesis
                nt = "Not normal"
                
            else:
                nt = "Normal"
                # nf+=1
        else:
            nt = "Not enough data: " + str(n)

        ### Other statistics 
        kurt = sc.kurtosis(x)
        skew =  sc.skew(x)
        iqr = sc.iqr(x)
        std = x.std()
        mean = x.mean()
        median = x.median()
        perc_25 = x.quantile(0.25)
        perc_75 = x.quantile(0.75)
        pvalue = normt.pvalue
        other = pd.DataFrame([kurt,skew,iqr,std,mean,median,nt,pvalue,n],index=["Kurtosis","Skewness","IQR","STD","Mean","Median","Distribution","pvalue","Samples"],columns=[f"{features[feature]}"])
        stats_data[other.columns[0]]=other

        ### Outliers (by IQR method, MAD method and Isolation forest)
        low = perc_25 - 1.5*iqr
        high = perc_75 + 1.5*iqr
        outliers_iqr = x[(x<low)|(x>high)]
        amountOut = (len(outliers_iqr)/len(x))*100
        outliers_data.loc["IQR",features[feature]] = amountOut

        MAD = sc.median_abs_deviation(x)
        MAD = (x-median)/MAD 
        outliers_mad = x[abs(MAD)>3]
        amountOut = (len(outliers_mad)/len(x))*100
        outliers_data.loc["MAD",features[feature]] = amountOut
        
        ### Show boxplots
        bxpstatsmean = [{'whishi': mean+std,
                'whislo': mean-std,
                'fliers': [],
                'q1': perc_25,
                'med': mean,
                'q3': perc_75}]

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].boxplot(x, vert=True,showfliers=False,showmeans=True,meanline=True,patch_artist=True, boxprops=dict(facecolor='lightblue'))
        ax[0].set_title(f"Boxplot (IQR and median) of feature {features[feature]}")
        ax[0].set_ylabel('value')

        ax[1].bxp(bxpstatsmean,patch_artist=True, boxprops=dict(facecolor='lightgreen'),whiskerprops=dict(color="red"),capprops=dict(color="red"))
        ax[1].set_title(f"Boxplot (mean and std) of feature {features[feature]}")
        ax[1].set_ylabel('value')

        sp = os.path.join(save_BP,f"{features[feature]}")
        plt.savefig(sp)
        plt.tight_layout()
        # plt.show()
        plt.close()

        return stats_data,outliers_data



data = {}
segment_ids = {}
listaFinal = []
data_path = 'C:/Users/adhn565/Documents/Data/completo_conAttrs_16_7_25.h5'

if data_path=="":
    data_path = filedialog.askopenfilename(title='Select signals file', filetypes=[("Input Files", ".h5")])
else:
    data_path=data_path

# Opens the archive read mode only with h5py
with h5py.File(data_path, 'r') as f:
    for group_name in f:
        group = f[group_name]
        data[group_name] = {}
        for dtset_name in group:
            data[group_name][dtset_name] = group[dtset_name][()]
        segment_ids[group_name] = group["segments"][0]
    ### The name and amount of fiducial points and features are the same for all patients
    fiducial = f["p000001"]["segments"].attrs['fiducial_order']
    features = f["p000001"]["mean_p000001"].attrs['features']
    fiducial = [f.decode() if isinstance(f, bytes) else f for f in fiducial]
    features = [f.decode() if isinstance(f, bytes) else f for f in features]

######################### Stadistical analysis #########################
# Create an instance of the STATS class with the data, features, and save folder
# You can change the savefolder to your desired path
# Just by calling the class it will create the folders and save the results but you can also call the methods individually
# Data is a dictionary with patients as keys and datasets as values
# the datasets are dictionaries with the dataset name as keys and the data as values
# be sure that the first 2 datasets are the ones you want to analyze
# If you want to change the datasets to analyze, you can change the code in the class
# Please enssure that the data is in the correct format and structure

savefolder = "c:/Users/adhn565/Documents/Stats"
stats = STATS(data=data, features=features, savefolder=savefolder, segment_ids=segment_ids)

# Perform the analysis on features and save the results
# If you want to use the function directly you can do it like this:
# you need to pass the data to analize, it can be a list, and array or dataframe (1D)
data_analysis = []
stats_data = pd.DataFrame(index=["Kurtosis","Skewness","IQR","STD","Mean","Median","Distribution","pvalue","Samples"])
outliers_data = pd.DataFrame(index=["IQR","MAD"])
# If you want to save the boxplots you can specify the folder
saveBP = os.path.join(savefolder, "BoxPlots")
os.makedirs(saveBP, exist_ok=True)
stats, outliers = stats.analysis_feat(data_analysis=data_analysis, stats_data=stats_data, outliers_data=outliers_data, save_BP=saveBP)

# savefolder = "c:/Users/adhn565/Documents/Stats"
# normal_feats = {}
# for p in data.keys():
#     save_folder = os.path.join(savefolder,p)
#     os.makedirs(save_folder, exist_ok=True)
#     full_data = pd.DataFrame(index=["Kurtosis","Skewness","IQR","STD","Mean","Median","Distribution","pvalue","Samples"])
#     out_all = pd.DataFrame(index=["IQR","MAD"])
#     print(p)
#     normal_feats[p] = {"mean":[],"median":[]}
#     for d in list(data[p].keys())[:2]:
#         nf = 0
#         save_QQ = os.path.join(save_folder,"QQplots_" + d.replace("_"+p,""))
#         save_BP = os.path.join(save_folder,"BoxPlots_" + d.replace("_"+p,""))
#         save_hist = os.path.join(save_folder,"Histograms_" + d.replace("_"+p,""))
        
#         os.makedirs(save_BP, exist_ok=True)
#         os.makedirs(save_QQ, exist_ok=True)
#         os.makedirs(save_hist, exist_ok=True)

#         QQplot_all(p,save_QQ, dataset=d)
#         Hist_all(p,save_hist,dataset=d)
#         for f in range(0,len(features)):
#             st_data,ot_data = analysis_feat(patient=p,feature=f,dataset=d,stats_data=full_data,outliers_data=out_all)
#         st_data.to_csv(os.path.join(save_folder, f"{d}.csv"),index=True)
#         ot_data.to_csv(os.path.join(save_folder, f"{d}_outliers.csv"),index=True)
#         normal_feats[p][d.replace("_"+p,"")].append(nf)
#     df_nf = pd.DataFrame(normal_feats)
#     df_nf.to_csv(os.path.join(savefolder,"NormalDistributions.csv"))

# plot_overPatients()
# isolationForest(data)