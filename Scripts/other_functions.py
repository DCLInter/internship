import h5py
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def to_xslx(data: pd.DataFrame, filename: str):
    with pd.ExcelWriter(filename) as writer:
        data.to_excel(writer, index=True)
        print(f"Data saved to {filename}")
        
def search_feat(feature:str, features_list: list):
    for i in features_list:
        if i == feature:
            x = features_list.index(i)
    return x
def bland_altman_plot(true, pred, label):
    mean = (true + pred) / 2                # X
    diff = pred - true                      # Y
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)

    plt.figure(figsize=(6, 4))
    plt.scatter(mean, diff, alpha=0.5)
    plt.axhline(mean_diff, color='red', linestyle='--', label=f"Mean: {mean_diff:.2f}")
    plt.axhline(mean_diff + 1.96 * std_diff, color='gray', linestyle='--', label='+ 1.96*SD')
    plt.axhline(mean_diff - 1.96 * std_diff, color='gray', linestyle='--', label='- 1.96*SD')
    plt.title(f"Bland-Altman Plot: {label}")
    plt.xlabel('Mean (Pred - True)')
    plt.ylabel('Difference (Pred - True)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{label}_bland_atlman.png")
    plt.show()
    plt.close()