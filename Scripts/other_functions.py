import h5py
import numpy as np
import pandas as pd
import os

class Extras:
    def __init__():
        pass

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