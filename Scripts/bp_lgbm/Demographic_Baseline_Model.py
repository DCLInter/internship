############ BASELINE MODEL ###################################
#                                                             #
# Here, a baseline model based on linear regression will be   #
#  trained for SBP, DBP and MAP as baseline comparison        #
#                                                             #
###############################################################

import preprocessing
import eval
from pathlib import Path
from local_paths import PULSE_DB_SUP_DIR
from data import load_PulseDB_sup_ds
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pprint import pprint

if __name__ == "__main__":

    # =========================================================
    # Paths
    #==========================================================
    train_original_path = PULSE_DB_SUP_DIR / "Features_complete_VitalDB_Train_Subset.h5"
    test_original_path = PULSE_DB_SUP_DIR / "Features_VitalDB_CalFree_Test_Subset.h5"

    # =========================================================
    # Load Data
    #==========================================================
    feature_names = ["IPR", "Tsp", "TWRRF25", "TWRRF50", "Tsw25", 
                 "Tsw50", "Tsw75", "Tdw25", "Tdw50", "Tdw75", 
                 "AUCpi", "IPA",  "Av-Au_ratio", "Ab-Aa_ratio", "Ac-Aa_ratio", 
                 "Ad-Aa_ratio", "Ap2-Ap1_ratio", "AGI", "Kurtosis", "Skewness", 
                 "L-H_ratio", "ShannonEntropy", "Tpp", "PRV", "FullKurt", 
                 "FullSkew", "sdPRV", "IQR_PRV"]
    df_train = load_PulseDB_sup_ds(train_original_path, feature_names=feature_names)
    df_test = load_PulseDB_sup_ds(test_original_path, feature_names=feature_names)
    
    print(df_train.head())
    print(df_test.head())

    # =========================================================
    # Check NaNs and fill them
    #==========================================================
    """
    print("************************************************************")
    print("Check NaNs")
    print("************************************************************")
    
    # Check if any NaN at all
    print("Is there any NaN in: df_features_mean?")
    print(df_test.isna().any().any())
    # Count total number of NaNs
    print(df_test.isna().sum())
    """
    # Fill X NaNs
    df_train = preprocessing.median_impute_patientwise(df_train, patient_col= "Subject")
    df_test = preprocessing.median_impute_patientwise(df_test, patient_col= "Subject")

    # =========================================================
    # Initialize model
    #==========================================================
    # --- Preprocessing ---
    num_features = ["Age", "BMI", "Height", "Weight"]
    cat_features = ["Gender"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(drop="first"), cat_features),
        ]
    )
    
    # --- Build pipeline ---
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    # drop ID columns before training
    demographics_col = ["Subject", "Age", "Gender", "Height", "Weight", "BMI"]
    targets = ["SBP", "DBP", "MAP"]
    X_train = df_train[demographics_col]

    X_test = df_test[demographics_col]
    print(X_train.head())

    Y_train = df_train[targets[1]]
    Y_test = df_test[targets[1]]
    print(Y_train.head())

    # --- Train/test split ---
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, test_size=0.2, random_state=42, shuffle=True, stratify=X_train["Subject"]
    )

    # --- Fit model ---
    model.fit(X_train, Y_train)

    # --- Predict ---
    Y_pred_val = model.predict(X_val)
    Y_pred_test = model.predict(X_test)

    # =========================================================
    # Training and Eval
    #==========================================================
    Bland_Altman_Path = Path(r"C:\Users\addp972\OneDrive - City, University of London\3.PhD\9. Experiments\2.LightGBM_SHAP\ICASSP_Submission\Bland_Altman")
    metrics_eval = eval.evaluate(Y_val, Y_pred_val, Bland_Altman_Path /f"val_original_{targets[1]}_baseline.png")
    metrics_test = eval.evaluate(Y_test, Y_pred_test, Bland_Altman_Path /f"test_original_{targets[1]}_baseline.png")
    
    print("***************** VALIDATION ***********************")
    pprint(metrics_eval, sort_dicts=False)
    print("***************** TESTING **************************")
    pprint(metrics_test, sort_dicts=False)
