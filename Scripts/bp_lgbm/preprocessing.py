######## PREPROCESSING SCRIPT ################################################################
#                                                                                            #
# This script handles the splitting, standarization and things like that.                    #
#                                                                                            #
#                                                                                            #
##############################################################################################

import pandas as pd
from typing import Dict, List

def split_patients_by_signal_share(
    meta_df: pd.DataFrame,
    threshold: float = 0.80,             # target fraction (0–1)
    margin: float = 0.02,                # allowed tolerance
    patient_col: str = "Patient",        # your DF’s column name
    signals_col: str = "Total_signals",  # your DF’s column name
    prefer: str = "desc",                # "desc" = biggest patients first, "asc" = smallest first
) -> Dict[str, object]:
    """
    Greedy splitter: selects patients until cumulative share of signals
    falls within [threshold - margin, threshold + margin].

    If adding a patient would overshoot threshold + margin, that patient is skipped.
    """

    # total number of signals across all patients
    total_signals = int(meta_df[signals_col].sum())
    if total_signals == 0:
        return {
            "selected_ids": [],
            "remaining_ids": meta_df[patient_col].tolist(),
            "selected_share": 0.0,
            "note": "No signals in dataset."
        }

    # sort patients
    df = meta_df[[patient_col, signals_col]].copy()
    df = df.sort_values(signals_col, ascending=(prefer == "asc"))

    # target band
    low = (threshold - margin) * total_signals
    high = (threshold + margin) * total_signals

    selected: List[str] = []
    selected_signals = 0

    for _, row in df.iterrows():
        pid, sigs = row[patient_col], int(row[signals_col])
        tentative = selected_signals + sigs

        if tentative > high:  # overshoot → skip this patient
            continue

        selected.append(pid)
        selected_signals = tentative

        if low <= selected_signals <= high:  # stop once inside band
            break

    selected_share = selected_signals / total_signals
    remaining = [pid for pid in df[patient_col].tolist() if pid not in selected]

    return {
        "selected_ids": selected,
        "remaining_ids": remaining,
        "selected_share": selected_share,
        "selected_signals": selected_signals,
        "remaining_signals": total_signals - selected_signals,
        "total_signals": total_signals,
        "note": (
            "Within margin."
            if low <= selected_signals <= high
            else "Could not hit band exactly; stopped at closest under upper bound."
        ),
    }
