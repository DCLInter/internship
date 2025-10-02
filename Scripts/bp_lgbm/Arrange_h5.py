import h5py
import numpy as np
from pathlib import Path

def convert_h5_structure(old_path: str, new_path: str):
    """
    Convert HDF5 file structure:
    - Flatten "PPG_features/Features" into "PPG_Features"
    - Keep all other datasets as they are
    """
    with h5py.File(old_path, "r") as f_old, h5py.File(new_path, "w") as f_new:
        for key in f_old.keys():
            if key == "PPG_features":
                # Extract Features dataset inside the group
                features = f_old["PPG_features"]["Features"][:]
                f_new.create_dataset("PPG_Features", data=features, dtype=features.dtype)
                print(f"Created dataset PPG_Features with shape {features.shape}")
            else:
                # Copy other datasets directly
                data = f_old[key][:]
                f_new.create_dataset(key, data=data, dtype=data.dtype)
                print(f"Copied dataset {key} with shape {data.shape}")

    print(f"\n✅ Converted file saved at: {new_path}")


if __name__ == "__main__":
    old_file = Path(r"C:\Users\addp972\OneDrive - City, University of London\3.PhD\9. Experiments\PulseDB\SupplementarySubsets\Features_VitalDB_CalFree_Test_Subset.h5")
    new_file = Path(r"C:\Users\addp972\OneDrive - City, University of London\3.PhD\9. Experiments\PulseDB\SupplementarySubsets\Features_VitalDB_CalFree_Test_Subset_converted.h5")

    convert_h5_structure(old_file, new_file)
