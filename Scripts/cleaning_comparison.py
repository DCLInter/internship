import h5py
import numpy as np
import pandas as pd
import scipy.stats as scStats

data = {}
data_clean = {}
segment_ids = {}
data_path = 'C:/Users/adhn565/Documents/Data/completo_conAttrs_16_7_25.h5'
data_path_clean = 'C:/Users/adhn565/Documents/Data/test_data_feat.h5'

with h5py.File(data_path, 'r') as f:
    for group_name in f:
        group = f[group_name]
        data[group_name] = {}
        for dtset_name in group:
            data[group_name][dtset_name] = group[dtset_name][()]
            data[group_name][dtset_name] = data[group_name][dtset_name]
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
        segment_ids[group_name] = group["segments"][0]
    
    fiducial = f["p000001"]["segments"].attrs['fiducial_order']
    features = f["p000001"]["mean_p000001"].attrs['features']
    fiducial = [f.decode() if isinstance(f, bytes) else f for f in fiducial]
    features = [f.decode() if isinstance(f, bytes) else f for f in features]

def search_feat(feature:str, features_list: list):
    for i in features_list:
        if i == feature:
            x = features_list.index(i)
    return x

def outliers_IQRandMAD(array):
    median = np.median(array)
    iqr = scStats.iqr(array)
    percentile25 = np.percentile(array,25)
    percentile75 = np.percentile(array,75)
    mad = scStats.median_abs_deviation(array)

    high_limit = percentile75 + iqr*1.5
    low_limit = percentile25 + iqr*1.5
    score_mad = abs(array - median)/mad

    outliers_iqr = array[(array < low_limit) | (array > high_limit)]
    ourliers_mad = array[score_mad > 3]
    
    feature = pd.Series(array)
    outliers_iqr_index = feature[(feature < low_limit) | (feature > high_limit)].index
    ourliers_mad_index = feature[score_mad > 3].index
    
    outliers_values = {
        "IQR": outliers_iqr,
        "MAD": ourliers_mad
    }
    outliers_indexes = {
        "IQR": list(outliers_iqr_index),
        "MAD": list(ourliers_mad_index)
    }

    return outliers_values, outliers_indexes


def cleaningAnalysis(before: np.array,after: np.array):
    
    #### X IS THE FEATURE FROM THE NORMAL DATASET, Y IS THE FEATURE FROM THE CLEANED DATASET
    x = before
    x = x[:len(x)//2]
    y = after

    ### change in standart statistics
    means = [np.mean(x),np.mean(y)]
    means_change = 100*np.diff(means)[0]/means[0]

    percentiles25 = [np.percentile(x,25), np.percentile(y,25)]
    percentiles75 = [np.percentile(x,75), np.percentile(y,75)]
    percentiles_change = [ 100*np.diff(percentiles25)[0]/percentiles25[0] , 100*np.diff(percentiles75)[0]/percentiles75[0] ]

    iqr = [scStats.iqr(x), scStats.iqr(y)]
    iqr_change = 100*np.diff(iqr)[0]/iqr[0]

    median = [np.median(x),np.median(y)]
    median_change = 100*np.diff(median)[0]/median[0]

    mad = [scStats.median_abs_deviation(x), scStats.median_abs_deviation(y)]
    mad_change = 100*np.diff(mad)[0]/mad[0]

    ### Statistical distances test (Kolmogorov-Smirnov test and Wasserstein distance)
    ### In this case the null hypothesis for the KS test is that both samples comes from the same distribution
    ks_test, ks_p = scStats.ks_2samp(x,y)
    w_dist = scStats.wasserstein_distance(x,y)

    stats_changes = {
        "mean": means_change,
        "median": median_change,
        "Q3": percentiles_change[1],
        "Q1": percentiles_change[0],
        "IQR": iqr_change,
        "MAD": mad_change,
        "KS test": ks_test,
        "KS p-value": ks_p,
        "Wasserstein": w_dist
    }

    return stats_changes

def byPatient(data: dict, data_clean: dict, features_names: list):
    change_perPatient = {}
    ft_names = features_names
    for patient in data.keys():

        ft = data[patient][f"mean_{patient}"]
        ft_clean = data_clean[patient][f"mean_{patient}"]
        change_perFeature = {}

        for feat in np.arange(len(ft_clean)):
            x_before = ft[feat]
            x_after = ft_clean[feat]
            stat_changes = cleaningAnalysis(before = x_before, after = x_after)
            change_perFeature[ft_names[feat]] = stat_changes

        change_perPatient[patient] = change_perFeature
    
    return change_perPatient

def byFeature_medians(data: dict, data_clean: dict, features_names: list):
    median_all = {}
    ft_names = features_names
    for f in ft_names:
        feat_values = []
        feat_values_clean = []
        for p in data.keys():
            idx_ft = search_feat(f,ft_names)
            x = data[p][f"mean_{p}"][idx_ft]
            x = x[:len(x)//2]
            y = data_clean[p][f"mean_{p}"][idx_ft]

            median_x = np.median(x)
            median_y = np.median(y)

            feat_values.append(median_x)
            feat_values_clean.append(median_y)

        array_before = np.array(feat_values)
        array_after = np.array(feat_values_clean)
        stat_changes = cleaningAnalysis(before = array_before, after = array_after)
        median_all[f] = stat_changes

        ### Outliers
        otl_val_x, otl_idx_x = outliers_IQRandMAD(array_before)
        otl_val_y, otl_idx_y = outliers_IQRandMAD(array_after)

        overlap_iqr = np.where(otl_idx_x["IQR"] == otl_idx_y["IQR"])[0]
        overlap_mad = np.where(otl_idx_x["MAD"] == otl_idx_y["MAD"])[0]
        
    
    return median_all

def byFeature_all(data: dict, data_clean: dict, features_names: list):
    signals_all = {}
    ft_names = features_names
    for f in ft_names:
        feat_values = []
        feat_values_clean = []
        for p in data.keys():
            idx_ft = search_feat(f,ft_names)
            x = data[p][f"mean_{p}"][idx_ft]
            x = x[:len(x)//2]
            y = data_clean[p][f"mean_{p}"][idx_ft]

            feat_values.extend(x)
            feat_values_clean.extend(y)

        array_before = np.array(feat_values)
        array_after = np.array(feat_values_clean)
        stat_changes = cleaningAnalysis(before = array_before, after = array_after)
        signals_all[f] = stat_changes

    return signals_all

b = byFeature_medians(data,data_clean,features)
# a = byFeature_all(data,data_clean,features)


