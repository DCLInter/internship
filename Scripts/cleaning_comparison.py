import h5py
import numpy as np
import pandas as pd
import scipy.stats as scStats
import os

data = {}
data_clean = {}
segment_ids = {}
data_path = 'C:/Users/adhn565/Documents/Data/completo_conAttrs_16_7_25.h5'
data_path_clean = 'C:/Users/adhn565/Documents/Data/data_clean_features.h5'

with h5py.File(data_path, 'r') as f:
    for group_name in f:
        group = f[group_name]
        data[group_name] = {}
        for dtset_name in group:
            data[group_name][dtset_name] = group[dtset_name][()]
        segment_ids[group_name] = group["segments"][0]
    
    fiducial = f["p000001"]["segments"].attrs['fiducial_order']
    features = f["p000001"]["mean_p000001"].attrs['features']
    fiducial = [f.decode() if isinstance(f, bytes) else f for f in fiducial]
    features = [f.decode() if isinstance(f, bytes) else f for f in features]

with h5py.File(data_path_clean, 'r') as f:
    for group_name in f:
        group = f[group_name]
        data_clean[group_name] = {}
        for dtset_name in group:
            data_clean[group_name][dtset_name] = group[dtset_name][()]

def to_xslx(data: pd.DataFrame, filename: str):
    with pd.ExcelWriter(filename) as writer:
        data.to_excel(writer, index=True)
        print(f"Data saved to {filename}")

def search_feat(feature:str, features_list: list):
    for i in features_list:
        if i == feature:
            x = features_list.index(i)
    return x

def outliers_IQRandMAD(array):

    median = np.nanmedian(array)
    iqr = scStats.iqr(array, nan_policy="omit")
    quantiles = np.nanquantile(array,[0.25,0.75])
    percentile25 = quantiles[0]
    percentile75 = quantiles[1]
    mad = scStats.median_abs_deviation(array, nan_policy="omit")

    high_limit = percentile75 + iqr*1.5
    low_limit = percentile25 - iqr*1.5
    score_mad = abs((array - median)/mad)

    outliers_iqr = array[(array < low_limit) | (array > high_limit)]
    outliers_mad = array[score_mad > 3]
    
    feature = pd.Series(array)
    outliers_iqr_index = feature[(feature < low_limit) | (feature > high_limit)].index
    ourliers_mad_index = feature[score_mad > 3].index

    df_outliers_iqr = pd.DataFrame(outliers_iqr,columns=["values"],index=outliers_iqr_index)
    df_outliers_mad = pd.DataFrame(outliers_mad,columns=["values"],index=ourliers_mad_index)
    
    outliers = {
        "IQR": df_outliers_iqr,
        "MAD": df_outliers_mad
    }
    
    return outliers


def cleaningAnalysis(before: np.array,after: np.array):
    #### "X" IS THE FEATURE FROM THE NORMAL DATASET, "Y" IS THE FEATURE FROM THE CLEANED DATASET
    x = before
    y = after
    
    ### change in standart statistics
    means = [np.nanmean(x),np.nanmean(y)]
    median = [np.nanmedian(x),np.nanmedian(y)]
    quantiles_before = np.nanquantile(x,[0.25,0.27])
    quantiles_after = np.nanquantile(y,[0.25,0.27])
    percentiles25 = [quantiles_before[0], quantiles_after[0]]
    percentiles75 = [quantiles_before[1], quantiles_after[1]]
    
    means_change = 100*np.diff(means)[0]/means[0] if means[0] != 0 else 0
    median_change = 100*np.diff(median)[0]/median[0] if median[0] != 0 else 0
    percentiles_change = [ 100*np.diff(percentiles25)[0]/percentiles25[0] , 100*np.diff(percentiles75)[0]/percentiles75[0] ]

    iqr = [scStats.iqr(x, nan_policy="omit"), scStats.iqr(y, nan_policy="omit")]
    iqr_change = 100*np.diff(iqr)[0]/iqr[0] if iqr[0] != 0 else 0

    mad = [scStats.median_abs_deviation(x,nan_policy="omit"), scStats.median_abs_deviation(y,nan_policy="omit")]
    mad_change = 100*np.diff(mad)[0]/mad[0] if mad[0] != 0 else 0

    ### Statistical distances test (Kolmogorov-Smirnov test)
    ### In this case the null hypothesis for the KS test is that both samples comes from the same distribution
    ks_test, ks_p = scStats.ks_2samp(x,y)
    ws_test = scStats.wasserstein_distance(x,y)

    stats_changes = {
        "mean": means_change,
        "median": median_change,
        "Q3": percentiles_change[1],
        "Q1": percentiles_change[0],
        "IQR": iqr_change,
        "MAD": mad_change,
        "KS test": ks_test,
        "KS p-value": ks_p,
        "Wasserstein_dist": ws_test
    }

    return stats_changes

def byPatient(data: dict, data_clean: dict, features_names: list):
    change_perPatient = {}
    outliers_perPatient = {}
    change_outliers = {}
    ft_names = features_names

    for patient in data.keys():

        ft = data[patient][f"mean_{patient}"]
        ft_clean = data_clean[patient][f"mean_{patient}"]
        change_perFeature = {}
        outliers_perFeature = {}
        change_outliers[patient] = pd.DataFrame(columns=["IQR_change", "MAD_change"])

        for feat in np.arange(len(ft_clean)):
            x_before = ft[feat]
            if x_before.size > 10:
                x_before = x_before[:len(x_before)//2]
            x_after = ft_clean[feat]
            
            try:
                stat_changes = cleaningAnalysis(before = x_before, after = x_after)
                change_perFeature[ft_names[feat]] = stat_changes

                outliers_original = outliers_IQRandMAD(x_before)
                outliers_clean = outliers_IQRandMAD(x_after)

            except:
                print(f"{patient}, {features[feat]}\nOne of the arrays its empty: \n normal data (size): {x_before.size}\n clean data (size): {x_after.size}")

            outliers_perFeature[ft_names[feat]] = {
            "outliers_original": outliers_original,
            "outliers_clean": outliers_clean,
            }
            
            otl_og_size = outliers_original["IQR"].size
            otl_cl_size = outliers_clean["IQR"].size
            ## Percentage of outliers before and after cleaning
            otl_og_per = (otl_og_size / x_before.size) * 100
            otl_cl_per = (otl_cl_size / x_after.size) * 100
            iqr_change = abs(otl_cl_per - otl_og_per)/otl_og_per * 100 if otl_og_per > 0 else 0
            change_outliers[patient].loc[ft_names[feat],"IQRbef%"] = otl_og_per
            change_outliers[patient].loc[ft_names[feat],"IQRaft%"] = otl_cl_per

            otl_og_size = outliers_original["MAD"].size
            otl_cl_size = outliers_clean["MAD"].size
            otl_og_per = (otl_og_size / x_before.size) * 100
            otl_cl_per = (otl_cl_size / x_after.size) * 100
            mad_change = abs(otl_cl_per - otl_og_per)/otl_og_per * 100 if otl_og_per > 0 else 0
            change_outliers[patient].loc[ft_names[feat],"MADbef%"] = otl_og_per
            change_outliers[patient].loc[ft_names[feat],"MADaft%"] = otl_cl_per

            change_outliers[patient].loc[ft_names[feat],"IQR_change"] = iqr_change
            change_outliers[patient].loc[ft_names[feat],"MAD_change"] = mad_change

        outliers_perPatient[patient] = outliers_perFeature
        change_perPatient[patient] = change_perFeature
    
    return change_perPatient, outliers_perPatient, change_outliers

def byFeature_medians(data: dict, data_clean: dict, features_names: list):
    median_all = {}
    outliers = {}
    outl_changes = pd.DataFrame(columns=["IQR_change", "MAD_change"],index=features_names)
    ft_names = features_names
    for ft in ft_names:
        feat_values = []
        feat_values_clean = []

        for p in data.keys():
            idx_ft = search_feat(ft,ft_names)
            x = data[p][f"mean_{p}"][idx_ft]
            if x.size > 10:
                x = x[:len(x)//2]
            y = data_clean[p][f"mean_{p}"][idx_ft]
        
            if x.size == 0 or y.size == 0:
                continue

            median_x = np.nanmedian(x)
            median_y = np.nanmedian(y)

            feat_values.append(median_x)
            feat_values_clean.append(median_y)

        array_before = np.array(feat_values)
        array_after = np.array(feat_values_clean)

        stat_changes = cleaningAnalysis(before = array_before, after = array_after)
        median_all[ft] = stat_changes

        ### Outliers
        otl_original = outliers_IQRandMAD(array_before)
        otl_clean = outliers_IQRandMAD(array_after)
        
        outliers[ft] = {
            "outliers_original": otl_original,
            "outliers_cleaned": otl_clean,
        }
        otl_og_size = otl_original["IQR"].size
        otl_cl_size = otl_clean["IQR"].size
        outl_changes.loc[ft,"numberIQR"] = otl_og_size
        outl_changes.loc[ft,"numberIQR_aft"] =otl_cl_size
        ## Percentage of outliers compared to the original data
        otl_og_per = 100*(otl_og_size / array_before.size)
        otl_cl_per = 100*(otl_cl_size / array_after.size)
        ## Change in percentage of outliers
        iqr_change = 100*abs(otl_cl_per - otl_og_per)/otl_og_per if otl_og_per > 0 else 0
        outl_changes.loc[ft,"IQRbef%"] = otl_og_per
        outl_changes.loc[ft,"IQRaft%"] =otl_cl_per

        otl_og_size = otl_original["MAD"].size
        otl_cl_size = otl_clean["MAD"].size
        outl_changes.loc[ft,"numberMAD"] = otl_og_size
        outl_changes.loc[ft,"numberMAD_aft"] =otl_cl_size
        otl_og_per = 100*(otl_og_size / array_before.size) 
        otl_cl_per = 100*(otl_cl_size / array_after.size) 
        mad_change = 100*abs(otl_cl_per - otl_og_per)/otl_og_per if otl_og_per > 0 else 0
        outl_changes.loc[ft,"MADbef%"] = otl_og_per
        outl_changes.loc[ft,"MADaft%"] = otl_cl_per

        outl_changes.loc[ft,"IQR_change"] = iqr_change
        outl_changes.loc[ft,"MAD_change"] = mad_change

        print("antes",array_before.size,"despues",array_after.size)
    
    return median_all, outliers, outl_changes

def byFeature_all(data: dict, data_clean: dict, features_names: list):
    signals_all = {}
    outliers = {}
    overlaps = {}
    outl_changes = pd.DataFrame(columns=["IQR_change", "MAD_change"],index=features_names)
    ft_names = features_names
    for ft in ft_names:
        feat_values = []
        feat_values_clean = []
        for p in data.keys():
            idx_ft = search_feat(ft,ft_names)
            x = data[p][f"mean_{p}"][idx_ft]
            x = x[:len(x)//2]
            y = data_clean[p][f"mean_{p}"][idx_ft]

            if x.size == 0 or y.size == 0:
                continue

            feat_values.extend(x)
            feat_values_clean.extend(y)

        array_before = np.array(feat_values)
        array_after = np.array(feat_values_clean)
        stat_changes = cleaningAnalysis(before = array_before, after = array_after)
        signals_all[ft] = stat_changes
        ### Outliers
        outliers_original = outliers_IQRandMAD(array_before)
        outliers_clean = outliers_IQRandMAD(array_after)

        outliers[ft] = {
            "outliers_original": outliers_original,
            "outliers_cleaned": outliers_clean,
        }

        otl_og_size = outliers_original["IQR"].size
        otl_cl_size = outliers_clean["IQR"].size
        outl_changes.loc[ft,"numberIQR"] = otl_og_size
        outl_changes.loc[ft,"numberIQR_aft"] =otl_cl_size
        ## Percentage of outliers compared to the original data
        otl_og_per = 100*(otl_og_size / array_before.size)
        otl_cl_per = 100*(otl_cl_size / array_after.size) 
        ## Change in percentage of outliers
        iqr_change = 100*abs(otl_cl_per - otl_og_per)/otl_og_per if otl_og_per > 0 else 0
        outl_changes.loc[ft,"IQRbef%"] = otl_og_per
        outl_changes.loc[ft,"IQRaft%"] =otl_cl_per

        otl_og_size = outliers_original["MAD"].size
        otl_cl_size = outliers_clean["MAD"].size
        outl_changes.loc[ft,"numberMAD"] = otl_og_size
        outl_changes.loc[ft,"numberMAD_aft"] =otl_cl_size
        otl_og_per = 100*(otl_og_size / array_before.size)
        otl_cl_per = 100*(otl_cl_size / array_after.size)
        mad_change = 100*abs(otl_cl_per - otl_og_per)/otl_og_per if otl_og_per > 0 else 0
        outl_changes.loc[ft,"MADbef%"] = otl_og_per
        outl_changes.loc[ft,"MADaft%"] =otl_cl_per

        outl_changes.loc[ft,"IQR_change"] = iqr_change
        outl_changes.loc[ft,"MAD_change"] = mad_change

        ### Overlaps
        before = np.array(outliers_original["IQR"].index)
        before_m = np.array(outliers_original["MAD"].index)
        after = np.array(outliers_clean["IQR"].index)
        after_m = np.array(outliers_clean["MAD"].index)
        
        ### Jaccard similarity score
        overlap_iqr_score = len(set(before) & set(after)) / len(set(before) | set(after))
        overlap_mad_score = len(set(before_m) & set(after_m)) / len(set(before_m) | set(after_m))

        overlaps[ft] = {
            "overlap_iqr": 100*overlap_iqr_score,
            "overlap_mad": 100*overlap_mad_score
        }

        print("antes",array_before.size,"despues",array_after.size)

    return signals_all, outliers, overlaps, outl_changes

# st_ch_p, outl_p, outl_ch = byPatient(data,data_clean,features)
st_ch_med, outl_med, outl_ch_med = byFeature_medians(data,data_clean,features)
st_ch_all, outl_all, overlap, outl_ch_all = byFeature_all(data,data_clean,features)

df_iqr_p = pd.DataFrame(columns=features)
df_mad_p = pd.DataFrame(columns=features)

# for p, df in outl_ch.items():
#     for ft, otl in df.items():
#         df_iqr_p.loc[p, ft] = otl["IQR_change"]
#         df_mad_p.loc[p, ft] = otl["MAD_change"]

#         df_iqr_p.loc[p+"_1", ft] = otl["IQRbef%"]
#         df_mad_p.loc[p+"_1", ft] = otl["MADbef%"]

#         df_iqr_p.loc[p+"_2", ft] = otl["IQRaft%"]
#         df_mad_p.loc[p+"_2", ft] = otl["MADaft%"]

# filepath = "C:/Users/adhn565/Documents/Data"
# for p, df in outl_ch.items():
#     filename = os.path.join(filepath, f"{p}.xlsx")
#     to_xslx(df, filename)
# print(st_ch_med["Av-Au ratio"])

# Create DataFrames for the results
df_stats_med = pd.DataFrame(index=st_ch_med.keys())
for ft, stat in st_ch_med.items():
    for stk, stv in stat.items():
        df_stats_med.loc[ft, stk] = stv

df_stats_all = pd.DataFrame(index=st_ch_all.keys())
for ft, stat in st_ch_all.items():
    for stk, stv in stat.items():
        df_stats_all.loc[ft, stk] = stv

overlap_df = pd.DataFrame(columns=["IQR(%)", "MAD(%)"], index=overlap.keys())
for ft in overlap.keys():
    overlap_df.loc[ft, "IQR(%)"] = overlap[ft]["overlap_iqr"]
    overlap_df.loc[ft, "MAD(%)"] = overlap[ft]["overlap_mad"]

# Save the results to Excel files
dataframes = {
    "stats_change_medians": df_stats_med,
    "stats_change_all": df_stats_all,
    "outliers_change_only_medians": outl_ch_med,
    "outliers_change_with_all": outl_ch_all,
    "overlap_JaccardSimilarity": overlap_df,
}
filepath = ""
for name, df in dataframes.items():
    filename = os.path.join(filepath, f"{name}.xlsx")
    to_xslx(df, filename)


