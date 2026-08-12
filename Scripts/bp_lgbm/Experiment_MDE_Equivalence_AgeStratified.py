############ MDE + EQUIVALENCE — AGE >=60 CONTRAST ############################
#                                                                              #
# Post-processing only: no model training, no re-inference. Computes minimum  #
# detectable effect (MDE) and an equivalence test for the null-result age     #
# contrast (Age>=60 stratum-specific model vs the full-dataset model),        #
# reported in the manuscript as a non-significant difference in test R2.     #
#                                                                              #
# Unlike the filtering contrast, this one is NOT paired: Bootstrap/ppg        #
# (full model) and Bootstrap/age_stratified/age_gte60 each pre-allocate       #
# their own resamples list independently, over different subject pools of    #
# different sizes (full test set vs the age>=60 subset) — row i of one has    #
# no relationship to row i of the other. Treating the two bootstrap          #
# distributions as independent, Var(diff) = Var(A) + Var(B), so:              #
#   sd_diff = sqrt(sd_gte60^2 + sd_full^2)                                    #
# This also means there is no 1,000-row paired-difference series to take      #
# empirical percentiles of, so CIs use a normal (Wald) approximation          #
# instead — consistent with the mde = (1.96+0.84)*sd formula, which is        #
# itself a normal-approximation quantity.                                     #
#                                                                              #
# Saves:                                                                      #
#   Bootstrap/age_stratified/age_gte60/MDE_Equivalence_AgeStratified.csv      #
###############################################################################

import numpy as np
import pandas as pd

from local_paths import PERFORMANCE_RESULTS_PAPER

TARGETS  = ["SBP", "DBP", "MAP"]
CONTRAST = "age_gte60_vs_full"

Z_MDE = 2.8     # (1.96 + 0.84), two-sided alpha=0.05, 80% power
Z_95  = 1.96    # two-sided 95% normal quantile
Z_90  = 1.645   # two one-sided 5% tests -> 90% CI normal quantile

EQUIV_DELTA = 2.0   # equivalence margin, pp of R2, declared a priori


def load_point_r2(res_dir, target):
    pt = pd.read_csv(res_dir / f"PointEstimates_{target}.csv")
    return float(pt.loc[pt["metric"] == "R2", "value"].iloc[0])


def load_sd_pp(res_dir, target):
    dist = pd.read_csv(res_dir / f"Distribution_{target}.csv")
    return float(np.std(dist["R2"].to_numpy(), ddof=1)) * 100.0


if __name__ == "__main__":

    dir_full  = PERFORMANCE_RESULTS_PAPER / "Bootstrap" / "ppg"
    dir_gte60 = PERFORMANCE_RESULTS_PAPER / "Bootstrap" / "age_stratified" / "age_gte60"

    rows = []
    for target in TARGETS:
        sd_full_pp  = load_sd_pp(dir_full,  target)
        sd_gte60_pp = load_sd_pp(dir_gte60, target)

        # Independent bootstraps -> variances add, SD does not.
        sd_diff_pp = float(np.sqrt(sd_gte60_pp**2 + sd_full_pp**2))

        observed_diff_pp = (
            load_point_r2(dir_gte60, target) - load_point_r2(dir_full, target)
        ) * 100.0

        mde_pp = Z_MDE * sd_diff_pp

        # No shared resample index between the two bootstraps, so there is no
        # paired-difference distribution to percentile off of (unlike the
        # filtering contrast). CI is instead the normal-approximation Wald
        # interval observed_diff +/- z * sd_diff, using the two-sample SD above.
        ci95_lower_pp = observed_diff_pp - Z_95 * sd_diff_pp
        ci95_upper_pp = observed_diff_pp + Z_95 * sd_diff_pp
        ci90_lower_pp = observed_diff_pp - Z_90 * sd_diff_pp
        ci90_upper_pp = observed_diff_pp + Z_90 * sd_diff_pp

        equivalent = bool((ci90_lower_pp > -EQUIV_DELTA) and (ci90_upper_pp < EQUIV_DELTA))

        rows.append({
            "contrast":         CONTRAST,
            "target":           target,
            "observed_diff_pp": observed_diff_pp,
            "sd_pp":            sd_diff_pp,
            "mde_pp":           mde_pp,
            "ci95_lower_pp":    ci95_lower_pp,
            "ci95_upper_pp":    ci95_upper_pp,
            "ci90_lower_pp":    ci90_lower_pp,
            "ci90_upper_pp":    ci90_upper_pp,
            "equivalent":       equivalent,
        })

    result_df = pd.DataFrame(rows)

    # =========================================================
    # Sanity checks — stop before saving if any fail
    # =========================================================
    for row in rows:
        target = row["target"]

        ci95_width = row["ci95_upper_pp"] - row["ci95_lower_pp"]
        ci90_width = row["ci90_upper_pp"] - row["ci90_lower_pp"]
        if not (ci90_width < ci95_width):
            raise RuntimeError(
                f"SANITY CHECK FAILED ({target}): 90% CI width ({ci90_width:.3f} pp) "
                f"is not strictly narrower than 95% CI width ({ci95_width:.3f} pp)."
            )

        if not (row["mde_pp"] > 0):
            raise RuntimeError(f"SANITY CHECK FAILED ({target}): MDE is not positive.")

        # sqrt(a^2+b^2) must exceed each individual SD (quadrature property).
        if not (row["sd_pp"] >= max(
            load_sd_pp(dir_full, target), load_sd_pp(dir_gte60, target)
        )):
            raise RuntimeError(
                f"SANITY CHECK FAILED ({target}): combined sd_pp is smaller than "
                f"one of its inputs — quadrature sum is broken."
            )

    print("Sanity checks passed: 90% CIs are narrower than 95% CIs, all MDEs are "
          "positive, combined SD exceeds each individual bootstrap SD.\n")

    # =========================================================
    # Save + print
    # =========================================================
    out_path = dir_gte60 / "MDE_Equivalence_AgeStratified.csv"
    result_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    print("\n=== MDE / Equivalence - age>=60 vs full contrast (pp of R2) ===")
    with pd.option_context("display.float_format", lambda x: f"{x:.2f}"):
        print(result_df.to_string(index=False))
