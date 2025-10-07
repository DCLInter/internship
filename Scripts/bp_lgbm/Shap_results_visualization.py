############ VISUALIZATION SHAPLEY VALUES #####################
#                                                             #
# Here, all the results from shap will be visualized          #
#                                                             #
###############################################################

import pandas as pd
import shap_analysis as sa
from local_paths import SHAP_RESULTS_PAPER
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # =========================================================
    # Load all the needed shapley results
    #==========================================================
    targets = ["SBP", "DBP", "MAP"]
    res_path = SHAP_RESULTS_PAPER/ "Clean_80_DS"

    for target in targets:
        #------ RANK PLOTS --------------------------------------
        top_k = 10
        df_rank_matrix = pd.read_csv(res_path/f"{target}_rank_matrix.csv")
        feature_names = df_rank_matrix.columns.to_list()
        
        sa.plot_rank_boxplot(rank_matrix=df_rank_matrix,
                            feature_names= feature_names,
                            top_k=top_k,
                            show=False,
                            sort_by="hybrid",
                            alpha=1,
                            save_path=res_path/f"top_{top_k}_rank_features_{target}.png")
        
        sa.plot_mean_std_scatter(rank_matrix=df_rank_matrix,
                                feature_names=feature_names,
                                top_k_labels= 28,
                                show=False,
                                save_path=res_path/f"feature_rank_dispersion_{target}.png")
        #------ ABS SHAP PLOTS ------------------------------------
        abs_df = pd.read_csv(res_path/f"{target}_abs_shap_matrix.csv")
        sa.plot_shap_magnitude(shap_abs_matrix= abs_df,
                                    show=False,
                                    save_path=res_path/f"mean_feature_shap_contribution_{target}.png")

        #------ RAW SHAP PLOTS ------------------------------------
        raw_df = pd.read_csv(res_path/f"{target}_raw_shap_matrix.csv")
        sa.plot_shap_directionality(shap_signed_matrix=raw_df,
                                    show=False,
                                    save_path=res_path/f"feature_direction_{target}.png")