############ MDE + EQUIVALENCE — FILTERING CONTRAST ###########################
#                                                                              #
# Post-processing only: no model training, no re-inference. Computes minimum  #
# detectable effect (MDE) and an equivalence test for the null-result         #
# filtering contrast (filtered vs unfiltered training set), reported in the   #
# manuscript as a non-significant difference in test R2.                     #
#                                                                              #
# There is no pre-saved paired-difference distribution file. The paired diff  #
# is reconstructed here from the two per-model Distribution_{target}.csv      #
# files under Bootstrap_Filtering/, which are row-aligned by construction     #
# (Experiment_Bootstrap_Filtering.py allocates one `resamples` list and       #
# reuses it for both models in the same loop). Reconstructing this way and    #
# taking the 2.5th/97.5th percentiles reproduces the manuscript's published   #
# 95% CIs exactly — checked below as a sanity gate before anything is saved.  #
#                                                                              #
# Saves:                                                                      #
#   Bootstrap_Filtering/MDE_Equivalence_Filtering.csv                         #
###############################################################################

import numpy as np
import pandas as pd

from local_paths import PERFORMANCE_RESULTS_PAPER

TARGETS  = ["SBP", "DBP", "MAP"]
CONTRAST = "filtered_vs_unfiltered"

Z_MDE       = 2.8    # (1.96 + 0.84), two-sided alpha=0.05, 80% power
EQUIV_DELTA = 2.0     # equivalence margin, pp of R2, declared a priori

# Manuscript-reported 95% CIs for this contrast (pp of R2), used as a sanity
# gate on the reconstructed paired-difference distribution.
MANUSCRIPT_CI95_PP = {
    "SBP": (-1.92, 1.21),
    "DBP": (-1.62, 1.64),
    "MAP": (-1.91, 1.36),
}
SANITY_TOL_PP = 0.02


def load_point_r2(res_dir, target):
    pt = pd.read_csv(res_dir / f"PointEstimates_{target}.csv")
    return float(pt.loc[pt["metric"] == "R2", "value"].iloc[0])


if __name__ == "__main__":

    res_root = PERFORMANCE_RESULTS_PAPER / "Bootstrap_Filtering"

    rows = []
    for target in TARGETS:
        dist_filtered = pd.read_csv(res_root / "ppg_filtered" / f"Distribution_{target}.csv")
        dist_original = pd.read_csv(res_root / "ppg_original" / f"Distribution_{target}.csv")

        if len(dist_filtered) != len(dist_original):
            raise RuntimeError(
                f"{target}: row-count mismatch between ppg_filtered "
                f"({len(dist_filtered)}) and ppg_original ({len(dist_original)}) "
                f"distributions — files are not paired by resample index."
            )

        # Paired difference: same resample index -> same subject draw for both
        # models (allocated once and reused across both loops upstream).
        diff_pp = (dist_filtered["R2"] - dist_original["R2"]).to_numpy() * 100.0

        observed_diff_pp = (
            load_point_r2(res_root / "ppg_filtered", target)
            - load_point_r2(res_root / "ppg_original", target)
        ) * 100.0

        sd_pp  = float(np.std(diff_pp, ddof=1))
        mde_pp = Z_MDE * sd_pp

        ci95_lower_pp, ci95_upper_pp = np.percentile(diff_pp, [2.5, 97.5])
        ci90_lower_pp, ci90_upper_pp = np.percentile(diff_pp, [5, 95])

        equivalent = bool((ci90_lower_pp > -EQUIV_DELTA) and (ci90_upper_pp < EQUIV_DELTA))

        rows.append({
            "contrast":        CONTRAST,
            "target":          target,
            "observed_diff_pp": observed_diff_pp,
            "sd_pp":           sd_pp,
            "mde_pp":          mde_pp,
            "ci95_lower_pp":   float(ci95_lower_pp),
            "ci95_upper_pp":   float(ci95_upper_pp),
            "ci90_lower_pp":   float(ci90_lower_pp),
            "ci90_upper_pp":   float(ci90_upper_pp),
            "equivalent":      equivalent,
        })

    result_df = pd.DataFrame(rows)

    # =========================================================
    # Sanity checks — stop before saving if any fail
    # =========================================================
    for row in rows:
        target = row["target"]
        man_lo, man_hi = MANUSCRIPT_CI95_PP[target]
        d_lo = abs(row["ci95_lower_pp"] - man_lo)
        d_hi = abs(row["ci95_upper_pp"] - man_hi)
        if d_lo > SANITY_TOL_PP or d_hi > SANITY_TOL_PP:
            raise RuntimeError(
                f"SANITY CHECK FAILED ({target}): reconstructed 95% CI "
                f"[{row['ci95_lower_pp']:.2f}, {row['ci95_upper_pp']:.2f}] pp "
                f"does not match manuscript [{man_lo:.2f}, {man_hi:.2f}] pp "
                f"(tol={SANITY_TOL_PP} pp)."
            )

        ci95_width = row["ci95_upper_pp"] - row["ci95_lower_pp"]
        ci90_width = row["ci90_upper_pp"] - row["ci90_lower_pp"]
        if not (ci90_width < ci95_width):
            raise RuntimeError(
                f"SANITY CHECK FAILED ({target}): 90% CI width ({ci90_width:.3f} pp) "
                f"is not strictly narrower than 95% CI width ({ci95_width:.3f} pp)."
            )

        if not (row["mde_pp"] > 0):
            raise RuntimeError(f"SANITY CHECK FAILED ({target}): MDE is not positive.")

    print("Sanity checks passed: reconstructed 95% CIs match manuscript values, "
          "90% CIs are narrower than 95% CIs, all MDEs are positive.\n")

    # =========================================================
    # Save + print
    # =========================================================
    out_path = res_root / "MDE_Equivalence_Filtering.csv"
    result_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    print("\n=== MDE / Equivalence - filtering contrast (pp of R2) ===")
    with pd.option_context("display.float_format", lambda x: f"{x:.2f}"):
        print(result_df.to_string(index=False))
