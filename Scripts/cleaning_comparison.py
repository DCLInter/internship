import h5py
import numpy as np
import pandas as pd
import scipy.stats as scStats
import os

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

def to_xslx(data: pd.DataFrame, filename: str):
    with pd.ExcelWriter(filename) as writer:
        data.to_excel(writer, index=True)
        print(f"Data saved to {filename}")

def dict_to_df(data: dict):
    """Converts a nested dictionary to a pandas DataFrame.
    If the dictionary contains dictionaries, the keys of the external dictionary is used as the index and
    the keys of the internal dictionaries are used as columns.
    The values are converted to pd.Series and the added to the Dataframe.

    The function will fail if the internal dictionaries contains lists or numpy arrays of different lengths,
    it will also fail if the internal dictionaries has of values more nested dictionaries, lists or numpy arrays.
    """
    df = pd.DataFrame()
    for key, value in data.items():
        if isinstance(value, dict):
            # Converts the first key of the dictionary to the index
            df.index = pd.Index(list(value.keys()), name=key)
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, list) or isinstance(sub_value, np.ndarray):
                    df.loc[key,sub_key] = pd.Series(sub_value)
                else:
                    df.loc[key,sub_key] = pd.Series([sub_value])
        else:
            df[key] = pd.Series(value)

    return df

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

    outliers = pd.DataFrame(columns=["IQR_val","MAD_val","IQR_idx","MAD_idx"])
    outliers["IQR_val"] = outliers_iqr
    outliers["MAD_val"] = ourliers_mad
    outliers["IQR_idx"] = outliers_iqr_index
    outliers["MAD_idx"] = ourliers_mad_index
    print("aaaa", outliers)

    outliers_values = {
        "IQR": outliers_iqr,
        "MAD": ourliers_mad
    }
    outliers_indexes = {
        "IQR": list(outliers_iqr_index),
        "MAD": list(ourliers_mad_index)
    }

    return outliers_values, outliers_indexes, outliers


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
            x_after = ft_clean[feat]
            try:
                stat_changes = cleaningAnalysis(before = x_before, after = x_after)
                change_perFeature[ft_names[feat]] = stat_changes

                outliers_val_bef, outliers_idx_bef, outliers_original = outliers_IQRandMAD(x_before)
                outliers_val_aft, outliers_idx_aft, outliers_clean = outliers_IQRandMAD(x_after)

            except:
                print(f"{patient}, {features[feat]}\nOne of the arrays its empty: \n normal data (size): {x_before.size}\n clean data (size): {x_after.size}")

            outliers_perFeature[ft_names[feat]] = {
            "outliers_original": outliers_original,
            "outliers_clean": outliers_clean,
            }

            otl_og_size = outliers_original["IQR_val"].size
            otl_cl_size = outliers_clean["IQR_val"].size
            ## Percentage of outliers before and after cleaning
            otl_og_per = (otl_og_size / x_before.size) * 100 if x_before.size > 0 else 0
            otl_cl_per = (otl_cl_size / x_after.size) * 100 if x_after.size > 0 else 0
            ## Change in percentage of outliers
            iqr_change = abs(otl_cl_per - otl_og_per)/otl_og_per * 100 if otl_og_per > 0 else 0

            otl_og_size = outliers_original["MAD_val"].size
            otl_cl_size = outliers_clean["MAD_val"].size
            otl_og_per = (otl_og_size / x_before.size) * 100 if x_before.size > 0 else 0
            otl_cl_per = (otl_cl_size / x_after.size) * 100 if x_after.size > 0 else 0
            mad_change = abs(otl_cl_per - otl_og_per)/otl_og_per * 100 if otl_og_per > 0 else 0

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

            median_x = np.median(x)
            median_y = np.median(y)

            feat_values.append(median_x)
            feat_values_clean.append(median_y)

        array_before = np.array(feat_values)
        array_after = np.array(feat_values_clean)

        stat_changes = cleaningAnalysis(before = array_before, after = array_after)
        median_all[ft] = stat_changes

        ### Outliers
        otl_val_x, otl_idx_x, otl_original = outliers_IQRandMAD(array_before)
        otl_val_y, otl_idx_y, otl_clean = outliers_IQRandMAD(array_after)

        outliers[ft] = {
            "outliers_original": otl_original,
            "outliers_cleaned": otl_clean,
        }

        otl_og_size = otl_original["IQR_val"].size
        otl_cl_size = otl_clean["IQR_val"].size
        ## Percentage of outliers before and after cleaning
        otl_og_per = (otl_og_size / array_before.size) * 100 if array_before.size > 0 else 0
        otl_cl_per = (otl_cl_size / array_after.size) * 100 if array_after.size > 0 else 0
        ## Change in percentage of outliers
        iqr_change = abs(otl_cl_per - otl_og_per)/otl_og_per * 100 if otl_og_per > 0 else 0

        otl_og_size = otl_original["MAD_val"].size
        otl_cl_size = otl_clean["MAD_val"].size
        otl_og_per = (otl_og_size / array_before.size) * 100 if array_before.size > 0 else 0
        otl_cl_per = (otl_cl_size / array_after.size) * 100 if array_after.size > 0 else 0
        mad_change = abs(otl_cl_per - otl_og_per)/otl_og_per * 100 if otl_og_per > 0 else 0

        outl_changes.loc[ft,"IQR_change"] = iqr_change
        outl_changes.loc[ft,"MAD_change"] = mad_change
    
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
            if x.size > 10:
                x = x[:len(x)//2]
            y = data_clean[p][f"mean_{p}"][idx_ft]

            feat_values.extend(x)
            feat_values_clean.extend(y)

        array_before = np.array(feat_values)
        array_after = np.array(feat_values_clean)
        stat_changes = cleaningAnalysis(before = array_before, after = array_after)
        signals_all[ft] = stat_changes
        ### Outliers
        _, _, outliers_original = outliers_IQRandMAD(array_before)
        _, _, outliers_clean = outliers_IQRandMAD(array_after)

        outliers[ft] = {
            "outliers_original": outliers_original,
            "outliers_cleaned": outliers_clean,
        }

        otl_og_size = outliers_original["IQR_val"].size
        otl_cl_size = outliers_clean["IQR_val"].size
        ## Percentage of outliers before and after cleaning
        otl_og_per = (otl_og_size / array_before.size) * 100 if array_before.size > 0 else 0
        otl_cl_per = (otl_cl_size / array_after.size) * 100 if array_after.size > 0 else 0
        ## Change in percentage of outliers
        iqr_change = abs(otl_cl_per - otl_og_per)/otl_og_per * 100 if otl_og_per > 0 else 0

        otl_og_size = outliers_original["MAD_val"].size
        otl_cl_size = outliers_clean["MAD_val"].size
        otl_og_per = (otl_og_size / array_before.size) * 100 if array_before.size > 0 else 0
        otl_cl_per = (otl_cl_size / array_after.size) * 100 if array_after.size > 0 else 0
        mad_change = abs(otl_cl_per - otl_og_per)/otl_og_per * 100 if otl_og_per > 0 else 0

        outl_changes.loc[ft,"IQR_change"] = iqr_change
        outl_changes.loc[ft,"MAD_change"] = mad_change

        ### Overlaps
        if outliers_original.empty or outliers_clean.empty:
            overlap_iqr = []
            overlap_mad = []
        else:
            overlap_iqr = np.where([outliers_original["IQR_idx"] == outliers_clean["IQR_idx"]])[0]
            overlap_mad = np.where([outliers_original["MAD_idx"] == outliers_clean["MAD_idx"]])[0]
        
        ### Overlap IQR and MAD indexes of the signals and percentages
        overlap_iqr_idx = outliers_original["IQR_idx"][overlap_iqr]
        overlap_mad_idx = outliers_original["MAD_idx"][overlap_mad]
        overlap_iqr_percentage = len(overlap_iqr_idx) / len(outliers_original["IQR_idx"]) * 100 if len(outliers_original["IQR_idx"]) > 0 else 0
        overlap_mad_percentage = len(overlap_mad_idx) / len(outliers_original["MAD_idx"]) * 100 if len(outliers_original["MAD_idx"]) > 0 else 0

        overlaps[ft] = {
            "overlap_iqr_idx": overlap_iqr_idx,
            "overlap_mad_idx": overlap_mad_idx,
            "overlap_iqr_percentage": overlap_iqr_percentage,
            "overlap_mad_percentage": overlap_mad_percentage
        }

    return signals_all, outliers, overlaps, outl_changes

# st_ch_p, num_p, outl_p = byPatient(data,data_clean,features)
st_ch_med, outl_med, outl_ch_med = byFeature_medians(data,data_clean,features)
st_ch_all, outl_all, overlap, outl_ch_all = byFeature_all(data,data_clean,features)
# Create DataFrames for the results
df_stats_med = dict_to_df(st_ch_med)
df_stats_all = dict_to_df(st_ch_all)

overlap_df = pd.DataFrame(columns=["IQR_percentage", "MAD_percentage"], index=overlap.keys())
for ft in overlap.keys():
    overlap_df.loc[ft, "IQR_percentage"] = overlap[ft]["overlap_iqr_percentage"]
    overlap_df.loc[ft, "MAD_percentage"] = overlap[ft]["overlap_mad_percentage"]

# Save the results to Excel files
dataframes = {
    "stats_change_medians": df_stats_med,
    "stats_change_all": df_stats_all,
    "outliers_change_only_medians": outl_ch_med,
    "outliers_change_with_all": outl_ch_all,
    "overlap_change": overlap_df,
}
filepath = "C:/Users/adhn565/Documents/Data"
for name, df in dataframes.items():
    filename = os.path.join(filepath, f"{name}.xlsx")
    to_xslx(df, filename)


