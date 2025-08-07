import h5py
import numpy as np
import pandas as pd
import scipy.stats
from dotmap import DotMap

from pyPPG import PPG, Fiducials, Biomarkers
##Modified (Fiducials2 and Biomarkers2 are the same as the originals)
from lib_changes import PPG2, Fiducials2, Biomarkers2

import pyPPG.preproc as PP
import pyPPG.fiducials as FP
from lib_changes import fiducials2 as FP2 ##Modified

import pyPPG.biomarkers as BM
from lib_changes import biomarkers2 as BM2 ##Modified

from other_functions_PPG import Others ### Class with some other functions

class Feature_Extraction():
    def __init__(self, data_path: str, h5name: str, csvname: str, data_ext: dict = None):
        
        self.data = {}
        self.segment_ids = {}
        self.demo_info = {}
        self.samples = {}
        self.signal_dict = {}
        self.mean = {}
        self.median = {}
        self.empty = {}

        self.data_path = data_path
        self.filename_save = h5name
        self.filename_csv = csvname

        # Opens the archive read mode only with h5py
        with h5py.File(data_path, 'r') as f:
            #This works if the archive has only 1 dataset for each group
            for group_name in f:
                group = f[group_name]
                dataset_names = list(group.keys())
                dataset_names = dataset_names[:]
                if dataset_names:
                    ### [4:] used to eliminate the first 4 columns since they werent needed for the features
                    self.data[group_name] = group[dataset_names[0]][4:].T
                    self.segment_ids[group_name] =  group[dataset_names[0]][1].T
            # Aquires the attributes of each group
            for group_name in f.keys():
                group = f[group_name]
                if group_name not in self.demo_info:
                    self.demo_info[group_name] = {}
                for attr_name, attr_value in group.attrs.items():
                    self.demo_info[group_name][attr_name] = attr_value

        if data_ext is not None:
            self.data = data_ext
            self.demo_info = {}

        #self.data = {k: self.data[k] for k in ["p000010"] if k in self.data} # This is to change the amount of data you want to analyze

    def  save_h5(self, fiducials: dict, means: dict, medians: dict, fiducials_names: list, filename: str):
        demo_info = self.demo_info
        samples = self.samples
        with h5py.File(filename, 'w') as f:
            ### Creates a group per patient (or for the keys in fiducials)
            for patient_id, fiducials_list in fiducials.items():
                grp = f.create_group(patient_id)
                # Saves the demographic info in the attributes of each group
                if demo_info is not None:
                    for attr_name, attr_value in demo_info[patient_id].items():
                        grp.attrs[attr_name] = attr_value
                fiducials_list = fiducials_list.replace({pd.NA: np.nan})
                ### Creates 3 datasets: segments,mean_patient and median_patient
                ### the attributes of each dataset are their respective columns
                dset = grp.create_dataset(f'segments', data=fiducials_list.to_numpy(dtype = float, na_value = np.nan))
                dset.attrs['fiducial_order'] = np.array(fiducials_names, dtype= 'S')
                # As an additional attribute for "segments" the number of samples were added for each signal
                dset.attrs["N-samples"] = samples[patient_id]
                stats1 = means[patient_id]
                me = grp.create_dataset(f"mean_{patient_id}",data=stats1.to_numpy(dtype = float, na_value = np.nan))
                me.attrs["features"] = np.array(stats1.index.tolist(), dtype= 'S')
                stats2 = medians[patient_id]
                med = grp.create_dataset(f"median_{patient_id}",data=stats2.to_numpy(dtype = float, na_value = np.nan))
                med.attrs["features"] = np.array(stats2.index.tolist(), dtype= 'S')

    def stats_features(self, features: pd.DataFrame):
        X = pd.DataFrame(columns=features.columns)
        Med = pd.DataFrame(columns=features.columns)
        for ft in features.columns:
            X.loc[0,ft] = features[ft].mean()
            Med.loc[0,ft] = features[ft].median()

        return X, Med

    def feature_extraction(self):
        data = self.data
        segment_ids = self.segment_ids
        demo_info = self.demo_info
        samples = self.samples
        signal_dict = self.signal_dict
        mean = self.mean
        median = self.median
        empty = self.empty
        
        for i in data.keys():
            signal = DotMap()
            signal.filtering = True # whether or not to filter the PPG signal
            signal.fL=0.5000001 # Lower cutoff frequency (Hz)
            signal.fH=12 # Upper cutoff frequency (Hz)
            signal.order=4 # Filter order
            signal.sm_wins={'ppg':50,'vpg':10,'apg':10,'jpg':10} # smoothing windows in millisecond for the PPG, PPG', PPG", and PPG'"
        
            # Initialise the correction for fiducial points
            corr_on = ['on', 'dn', 'dp', 'v', 'w', 'f']
            correction=pd.DataFrame()
            correction.loc[0, corr_on] = True
            signal.correction=correction

            #Initialise cycling storage variables
            fp_pt = pd.DataFrame()
            ft_pt_mean = pd.DataFrame()
            ft_pt_median = pd.DataFrame()
            fp_pt_list = []
            fp_col = []
            empty[i] = []
            samples[i] = []

            print(f"patient: {i}")
            for sig in np.arange(len(data[i])): # Processing each signal
                signal.name = sig
                signal.start_sig = 0
                signal.end_sig = len(data[i][sig])
                signal.v = data[i][sig]
                signal.fs = int(demo_info[i]["SamplingFrequency"])

                #### Preprocess the signal with pyPPG (filtering and acquires the derivatives)
                prep = PP.Preprocess(fL=signal.fL, fH=signal.fH, order=signal.order, sm_wins=signal.sm_wins)
                signal.ppg, signal.vpg, signal.apg, signal.jpg = prep.get_signals(s=signal)

                # Create a PPG class
                s = PPG2(signal)
                samples[i].append(len(s.ppg))

                # Acquire the fiducial points
                fpex = FP2.FpCollection(s=s)
                fiducials = fpex.get_fiducials(s=s)

                # Create a fiducials class
                fp = Fiducials(fp=fiducials)

                # Just saving the names of the fiducials for later use in the h5 file
                df = pd.DataFrame(fiducials)
                if  sig == len(data[i])-1:
                    fp_col = df.columns
                else:
                    pass

                # Saving the extracted fiducials (flattened so each row will be a singular signal)
                df = df.values.flatten()
                fp_pt_list.append(df)

                # Init the biomarkers package (the features)
                # Using the modified version for personal selected features
                bmex = BM2.BmCollection2(s=s, fp=fp)

                # Extract biomarkers
                bm_defs, bm_vals = bmex.get_biomarkers2(get_stat=False)
                bm_df = bm_vals["ppg_features"]
                df = bm_df.drop(columns="TimeStamp")
                Tpp = bm_df["Tpp"]
                # if self.segment_ids[i][sig] == 55:
                #     print(df["IPR"],np.mean(df["IPR"]))
                # PRV (pulse rate variability), we need the data of the whole segment to calculate the PRV (change in the time between beats)
                # The array needs to be bigger have more than 1 element or it wont work
                if len(Tpp) > 0:
                    PRV = np.diff(Tpp, prepend=Tpp[0])
                    PRV = PRV*1000 # to ms
                    df["PRV"] = PRV 

                    # interquartil range, standart deviation of PRV
                    sdPRV = pd.Series(PRV).std()
                    IQR = scipy.stats.iqr(PRV)
                else:
                    # empty will be a csv containing the signals in which Tpp is to small/empty meaning that either
                    # the signal was too small or the library couldnt pick up properly the peaks and fiducials
                    empty[i].append(segment_ids[i][sig])
                    df["PRV"] = np.nan
                    sdPRV = np.nan
                    IQR = np.nan

                # Instead of saving the values of the features (since we have values of the features by beat to beat windows)
                # we will save the mean and median of the features as the finale feature datasets.
                x,y = self.stats_features(df)

                # Then we will add some additional features that are obtained not from windows but from the whole signal/segment of the patient
                # Adding Kurtosis and Skewness of the whole segment to the finale features dataset
                k = scipy.stats.kurtosis(s.ppg)
                sk = scipy.stats.skew(s.ppg)
                x["FullKurt"], y["FullKurt"] = [k,k]
                x["FullSkew"], y["FullSkew"] = [sk,sk]
                # Adding the PRV stadistics of the whole segement to the finale features dataset
                x["sdPRV"], y["sdPRV"] = [sdPRV,sdPRV]
                x["IQR_PRV"], y["IQR_PRV"] = [IQR,IQR]
                
                ft_pt_mean = pd.concat([ft_pt_mean,x])
                ft_pt_median = pd.concat([ft_pt_median,y])
            
            fp_pt = pd.DataFrame(fp_pt_list)
            fp_pt.insert(0,"segment_ID",segment_ids[i])
            signal_dict[i] = fp_pt.T
            fp_col.insert(0,"segment_ID")
            # Saving the mean and median of the features per signal for each patient
            mean[i] = ft_pt_mean.T
            median[i] = ft_pt_median.T

        #### If you want to save the data when it changes between patients just move it inside the loop, it will work
        print("Saving in: ",self.filename_save)
        self.save_h5(signal_dict,mean,median,fiducials_names=fp_col, filename=self.filename_save)
        
        # Saving the signals that could not be processed due to empty fiducials
        df_empty = pd.DataFrame({
        'patient': list(empty.keys()),
        'segment_id': [v for v in empty.values()],
        'amount': [len(v) for v in empty.values()],
        'signals_total': [len(data[v]) for v in list(empty.keys())],
        '%': [(len(s)/len(data[p]))*100 for p,s in empty.items()]
        })
        df_empty.to_csv(self.filename_csv,index=False)

        return mean,median,df_empty,signal_dict
