import h5py
import pandas as pd
import numpy as np

class Cleaner:
    def __init__(self, datapath: str):
        self.path = datapath
        self.data = {}
        self.ids = {}

        with h5py.File(datapath, 'r') as f:
            for patient in f.keys():
                grp = f[patient]
                data = grp["Metrics"][:]
                columns = [c.decode() for c in grp.attrs["metrics"]]
                index = [i for i in grp.attrs["ids"]]
                self.ids[patient] = index
                self.data[patient] = pd.DataFrame(data, columns=columns, index=index)
            self.metrics = columns
    
    def detect(self):

        print("Detecting flagged signals:")
        remove = {}
        for patient in self.data.keys():
            report = self.data[patient]["report"]
            test = report.eq(1)
            r = report.index[test]
            remove[patient] = r.tolist()
        self.remove = remove

        return remove
    
    def clean(self,original_datapath: str):

        print("Cleaning process:")
        self.original_data = {}
        original_ids = {}
        self.clean_data = {}
        self.demo_info = {}

        with h5py.File(original_datapath, 'r') as f:
            for group_name in f:
                group = f[group_name]
                dataset_names = list(group.keys())
                dataset_names = dataset_names[:]
                if dataset_names:
                    original_ids[group_name] = group[dataset_names[0]][1]
                    original_ids[group_name] = original_ids[group_name][:len(original_ids[group_name])//2]
                    self.original_data[group_name] = group[dataset_names[0]][:].T
                    self.original_data[group_name] = self.original_data[group_name][:len(self.original_data[group_name])//2]
                    self.original_data[group_name] = pd.DataFrame(self.original_data[group_name],index=original_ids[group_name])

                if group_name not in self.demo_info:
                    self.demo_info[group_name] = {}
                for attr_name, attr_value in group.attrs.items():
                    self.demo_info[group_name][attr_name] = attr_value

        remove = self.remove
        clean_data = self.original_data.copy()

        for patient in remove.keys():
            print(patient)
            sig_rem = np.array(remove[patient])
            ids = np.array(self.ids[patient])
            target = np.isin(ids,sig_rem)
            clean_data[patient] = clean_data[patient].drop(clean_data[patient].index[target])

        self.clean_data = clean_data

        return clean_data


    def saveh5(self,filename):
        print("Saving in: ",filename)
        data = self.clean_data
        with h5py.File(filename, 'w') as f:
            for patient, df in data.items():
                grp = f.create_group(patient)
                grp.create_dataset("Signals", data=df.T.to_numpy())
                # Save column names and index as attributes
                for attrs, value in self.demo_info[patient].items():
                    if isinstance(value, str):
                        dt = h5py.string_dtype(encoding="utf-8")
                        grp.attrs[attrs] = np.array(value,dtype=dt)
                    else:
                        grp.attrs[attrs] = value
    
    def csvReport(self, filename: str):
        metrics = ["signals","age","gender","weight [kg]","height [m]","BMI [kg/m^2]","checkHR","checkSP","numberProperFiducials","std_fp","numPPG","numD1","numD2","numD3","combinedScore","std_s","report","eliminate"]
        metrics_data = self.data
        atributes = self.demo_info
        remove = self.remove
        
        report_dfcsv = pd.DataFrame(index=metrics_data.keys(),columns=metrics)

        for patient, df in metrics_data.items():
            hr = df["checkHR"].sum()
            hr = round(100*hr/len(df["checkHR"]),2)

            sp = df["checkSP"].sum()
            sp = round(100*sp/len(df["checkSP"]),2)
            
            report = df["report"].sum()
            report = round(100*report/len(df["report"]),2)

            f = df["numberProperFiducials"].mean()
            stdFD = df["numberProperFiducials"].std()

            f0 = df["ppg"].mean()
            stdf0 = df["ppg"].std()

            f1 = df["d1"].mean()
            stdf0 = df["d1"].std()

            f2 = df["d2"].mean()
            stdf0 = df["d2"].std()

            f3 = df["d3"].mean()
            stdf0 = df["d3"].std()

            sc = df["combinedScore"].mean()
            stdS = df["combinedScore"].std()

            numSignals = len(self.original_data[patient])
            numRemove = len(remove[patient])

            age = atributes[patient]["Age"][0]
            gender = atributes[patient]["Gender"][0]
            weight = atributes[patient]["Weight"][0]
            height = atributes[patient]["Height"][0]/100
            bmi = weight/(height**2)
            
            report_dfcsv.loc[patient] = [numSignals,age,gender,weight,height,bmi,hr,sp,f,stdFD,f0,f1,f2,f3,sc,stdS,report,numRemove]
        
        report_dfcsv.to_csv(filename)

