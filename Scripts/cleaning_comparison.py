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
    #### "X" IS THE FEATURE FROM THE NORMAL DATASET, "Y" IS THE FEATURE FROM THE CLEANED DATASET
    x = before
    x = x[:len(x)//2]
    y = after

    ### change in standart statistics
    means = [np.mean(x),np.mean(y)]
    median = [np.median(x),np.median(y)]
    percentiles25 = [np.percentile(x,25), np.percentile(y,25)]
    percentiles75 = [np.percentile(y,75), np.percentile(y,75)]
    
    means_change = 100*np.diff(means)[0]/means[0]
    median_change = 100*np.diff(median)[0]/median[0]
    percentiles_change = [ 100*np.diff(percentiles25)[0]/percentiles25[0] , 100*np.diff(percentiles75)[0]/percentiles75[0] ]

    iqr = [scStats.iqr(x), scStats.iqr(y)]
    iqr_change = 100*np.diff(iqr)[0]/iqr[0]

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
    outliers_perPatient = {}
    number_overlapPatient = {}
    ft_names = features_names
    for patient in data.keys():

        ft = data[patient][f"mean_{patient}"]
        ft_clean = data_clean[patient][f"mean_{patient}"]
        change_perFeature = {}
        outliers_perFeature = {}
        number_overlapFeature = {}

        for feat in np.arange(len(ft_clean)):
            x_before = ft[feat]
            x_after = ft_clean[feat]
            try:
                stat_changes = cleaningAnalysis(before = x_before, after = x_after)
                change_perFeature[ft_names[feat]] = stat_changes

                outliers_val_bef, outliers_idx_bef = outliers_IQRandMAD(x_before)
                outliers_val_aft, outliers_idx_aft = outliers_IQRandMAD(x_after)

                overlap_iqr = np.where(outliers_idx_bef["IQR"] == outliers_idx_aft["IQR"])[0]
                overlap_mad = np.where(outliers_idx_bef["MAD"] == outliers_idx_aft["MAD"])[0]
            except:
                print(f"{patient}, {features[feat]}\nOne of the arrays its empty: \n normal data (size): {x_before.size}\n clean data (size): {x_after.size}")

            outliers_perFeature[ft_names[feat]] = {
            "values_original": outliers_val_bef,
            "values_cleaned": outliers_val_aft,
            "indexes_original": outliers_idx_bef,
            "indexes_cleaned": outliers_idx_aft
            }

            overlap_score_iqr = 0
            if len(overlap_iqr) > 0:
                v1 = outliers_val_bef["IQR"]
                v2 = outliers_val_aft["IQR"]
                total = len(v1) + len(v2)
                if v1.size == 0 or v2.size == 0:
                    pass
                else:
                    n = len(overlap_iqr)
                    overlap_score_iqr = n/total
            overlap_score_mad = 0
            if len(overlap_mad) > 0:
                v1 = outliers_val_bef["MAD"]
                v2 = outliers_val_aft["MAD"]
                if v1.size == 0 or v2.size == 0:
                    pass
                else:
                    m = len(overlap_mad)
                    overlap_score_mad = m/(len(v1) + len(v2))

            number_overlapFeature[ft_names[feat]] = {
                "IQR": overlap_score_iqr,
                "MAD": overlap_score_mad
            }

        number_overlapPatient[patient] = number_overlapFeature
        outliers_perPatient[patient] = outliers_perFeature
        change_perPatient[patient] = change_perFeature
    
    return change_perPatient, number_overlapPatient, outliers_perPatient

def byFeature_medians(data: dict, data_clean: dict, features_names: list):
    median_all = {}
    number_overlap = {}
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

            median_x = np.median(x)
            median_y = np.median(y)

            feat_values.append(median_x)
            feat_values_clean.append(median_y)

        array_before = np.array(feat_values)
        array_after = np.array(feat_values_clean)

        stat_changes = cleaningAnalysis(before = array_before, after = array_after)
        median_all[ft] = stat_changes

        ### Outliers
        otl_val_x, otl_idx_x = outliers_IQRandMAD(array_before)
        otl_val_y, otl_idx_y = outliers_IQRandMAD(array_after)

        overlap_iqr = np.where(otl_idx_x["IQR"] == otl_idx_y["IQR"])[0]
        overlap_mad = np.where(otl_idx_x["MAD"] == otl_idx_y["MAD"])[0]

        outliers = {
            "values_original": otl_val_x,
            "values_cleaned": otl_val_y,
            "indexes_original": otl_idx_x,
            "indexes_cleaned": otl_idx_y
        }

        overlap_score_iqr = 0
        if len(overlap_iqr) > 0:
            v1 = otl_val_x["IQR"]
            v2 = otl_val_y["IQR"]
            total = len(v1) + len(v2)
            if v1.size == 0 or v2.size == 0:
                pass
            else:
                n = len(overlap_iqr)
                overlap_score_iqr = n/total
        overlap_score_mad = 0
        if len(overlap_mad) > 0:
            v1 = otl_val_x["MAD"]
            v2 = otl_val_y["MAD"]
            if v1.size == 0 or v2.size == 0:
                pass
            else:
                m = len(overlap_mad)
                overlap_score_mad = m/(len(v1) + len(v2))

        number_overlap[ft] = {
            "IQR": overlap_score_iqr,
            "MAD": overlap_score_mad
        }
    
    return median_all, number_overlap, outliers

def byFeature_all(data: dict, data_clean: dict, features_names: list):
    signals_all = {}
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

            feat_values.extend(x)
            feat_values_clean.extend(y)

        array_before = np.array(feat_values)
        array_after = np.array(feat_values_clean)
        stat_changes = cleaningAnalysis(before = array_before, after = array_after)
        signals_all[ft] = stat_changes

    return signals_all

st_ch_p, num_p, outl_p = byPatient(data,data_clean,features)
st_ch_med, num_med, outl_med = byFeature_medians(data,data_clean,features)
st_ch_all = byFeature_all(data,data_clean,features)

print(num_p)
# iqrs = {}
# for key, value in num_p.items():
#     liqrs = []
#     for k, v in value.items():
#         for ft, dt in v.items():
#             liqrs.append(dt[])
#     iqrs[key] = liqrs
# print(iqrs)

# print(num_med)

# a = pd.DataFrame(data_clean["p000610"]["mean_p000610"])
# b = pd.DataFrame(data["p000366"]["mean_p000366"])
# print(b)

