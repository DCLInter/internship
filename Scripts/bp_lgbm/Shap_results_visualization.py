############ VISUALIZATION SHAPLEY VALUES #####################
#                                                             #
# Here, all the results from shap will be visualized          #
#                                                             #
###############################################################

import pandas as pd
import shap_analysis as sa
from local_paths import SHAP_RESULTS_PAPER
from pathlib import Path

if __name__ == "__main__":

    # =========================================================
    # Load all the needed shapley results
    #==========================================================
    res_path = SHAP_RESULTS_PAPER/ "Test"
    top_k = 10
    df_rank_matrix = pd.read_csv(res_path/"SBP_rank_matrix.csv")
    feature_names = df_rank_matrix.columns.to_list()
    print(feature_names)
    
    sa.plot_rank_boxplot(rank_matrix=df_rank_matrix,
                         feature_names= feature_names,
                         top_k=top_k,
                         show=False,
                         sort_by="hybrid",
                         alpha=1,
                         save_path=res_path/f"top_{top_k}_rank_features_SBP.png")
    
    sa.plot_mean_std_scatter(rank_matrix=df_rank_matrix,
                             feature_names=feature_names,
                             top_k_labels= 28,
                             show=False,
                             save_path=res_path/f"feature_rank_dispersion_SBP.png")