"""
Correctness check for the refactored pipeline: reruns run_pipeline at the
90% thresholds on one subset and compares the result against the
already-existing Clean_Features_<subset>_90.h5 produced by the old
Scripts/new_checker.py + cleaning.py code.

This is NOT a unit test in the pytest sense - it's a one-off script you run
to convince yourself the refactor reproduces the original pipeline's
output, per the task's "validate correctness" requirement. It prints a
pass/fail summary and exits non-zero if the comparison fails.

Run from inside this folder:
    ..\..\.venv310\Scripts\python.exe validate_against_clean_subset.py
"""

import sys

import h5py
import numpy as np

from run_pipeline import DATA_DIR, OUTPUT_DIR, SUBSETS, run_subset

# Which subset + which existing clean file to validate against. The repo
# had pre-existing Clean_Features_*_90.h5 files for both subsets at the
# time this script was written; either works. Train is used by default
# because it also has a matching Fiducial_Points file (see ASSUMPTIONS.txt
# for the CalFree_Test subset's missing fiducials file).
SUBSET_NAME = "VitalDB_Train_Subset"
REFERENCE_CLEAN_PATH = DATA_DIR / f"Clean_Features_{SUBSET_NAME}_90.h5"

# compute_raw_metrics() is a plain per-signal Python loop; the full Train
# subset has 465,480 signals and can take a long time end to end (see
# README.txt). MAX_SIGNALS limits this validation run to a prefix of the
# subset so it finishes quickly - since signals are processed independently
# and in order, a match on the first MAX_SIGNALS signals is strong evidence
# the refactor is correct without waiting for the full run. Pass
# --full on the command line to validate the entire subset instead.
MAX_SIGNALS = 20000


def load_reference(path):
    with h5py.File(path, "r") as f:
        return {name: f[name][()] for name in f}


def main():
    if not REFERENCE_CLEAN_PATH.exists():
        print(f"Reference file not found: {REFERENCE_CLEAN_PATH}")
        sys.exit(1)

    full_run = "--full" in sys.argv
    max_signals = None if full_run else MAX_SIGNALS
    print(f"Validating on {'the full subset' if full_run else f'the first {max_signals} raw signals'}.")

    paths = SUBSETS[SUBSET_NAME]
    result = run_subset(SUBSET_NAME, paths["fiducials"], paths["features"], OUTPUT_DIR,
                         max_signals=max_signals)

    print("\n=== Validation against", REFERENCE_CLEAN_PATH, "===")
    reference = load_reference(REFERENCE_CLEAN_PATH)
    ours = load_reference(result["clean_path"])

    n_ours = ours["PPG_Features"].shape[1]
    if not full_run:
        # The reference file was cleaned from the FULL subset, so we can't
        # compare against it wholesale when we only processed a prefix of
        # the raw signals. But cleaning is per-signal independent and
        # order-preserving: the surviving signals originally at raw indices
        # 0..max_signals-1 must appear, in the same order, as exactly the
        # first `n_ours` entries of the reference's kept signals (nothing
        # kept from raw indices >= max_signals can sort before them).
        # So slicing the reference's first n_ours kept entries gives an
        # exact expected match for our truncated run.
        reference = {name: values[..., :n_ours] for name, values in reference.items()}
        print(f"(sliced reference to its first {n_ours} kept signals for this partial comparison)")

    ok = True
    n_ref = reference["PPG_Features"].shape[1]
    print(f"Signals kept - reference: {n_ref}, ours: {n_ours}")
    if n_ref != n_ours:
        ok = False
        print("  MISMATCH in signal count.")

    if n_ref == n_ours:
        feat_close = np.allclose(reference["PPG_Features"], ours["PPG_Features"], equal_nan=True)
        print("PPG_Features values match:", feat_close)
        ok &= feat_close

        ref_subjects = np.array([s.decode() if isinstance(s, bytes) else s
                                  for s in reference["Subject"][0]])
        our_subjects = np.array([s.decode() if isinstance(s, bytes) else s
                                  for s in ours["Subject"][0]])
        subjects_match = np.array_equal(ref_subjects, our_subjects)
        print("Subject id sequence matches:", subjects_match)
        ok &= subjects_match

        for field in ["SBP", "MAP", "Age"]:
            close = np.allclose(reference[field], ours[field], equal_nan=True)
            print(f"{field} values match:", close)
            ok &= close

    print("\nRESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
