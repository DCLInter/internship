FIDUCIAL QUALITY FILTERING PIPELINE
====================================

This folder is a refactor of the original fiducial-quality-checking code
that used to live at the root of Scripts/ (checker.py, cleaning.py,
metrics_functions.py, new_checker.py). It covers ONLY the quality-check /
signal-discarding stage of the project - it assumes pyPPG fiducial points
have already been extracted (that happens upstream, in
Scripts/new_extraction_fiducials.py, and is NOT part of this folder).

Where this fits in the bigger picture:

    [pyPPG fiducial extraction]  (Scripts/new_extraction_fiducials.py, upstream, not here)
                |
                v
    Fiducial_Points_<subset>.h5  +  Features_<subset>.h5
                |
                v
    === everything in this folder ===
                |
                v
    Clean_Features_<subset>_<threshold>.h5   -->  feeds Layer 2 (Scripts/bp_lgbm/)


FILES
-----

metrics.py
    SignalMetrics class + free functions. Computes quality-check numbers
    for ONE signal at a time: missing fiducials, peak count, heart rate,
    fiducial order, and the alignment/consistency matrices used for
    criterion 4. Deliberately does NOT apply alpha/beta weights or
    thresholds - see combine_scores() and the checker.py docstring for why.

checker.py
    QualityChecker class. Runs SignalMetrics over every signal in a group
    (compute_raw_metrics - the expensive step), then separately combines
    the cached alignment/consistency into a score and applies thresholds
    (score_and_report - the cheap step). This split is what makes an
    alpha/beta/threshold sweep fast: call compute_raw_metrics() once, then
    call score_and_report() many times with different parameters.

cleaner.py
    SignalCleaner class. Reads a saved quality report (.h5), figures out
    which signal ids fail the report (detect()), drops them from any
    matching dataset (clean()), and can produce a descriptive per-criterion
    percentage summary WITHOUT dropping anything (csv_report()).

run_pipeline.py
    The script you actually run. Configuration (data paths, alpha/beta,
    thresholds) lives at the top of the file - edit it there. Loads a
    subset's fiducials + features, runs the checker, saves the quality
    report, saves the cleaned features, and saves three summary files (see
    OUTPUTS below). This replaces Scripts/new_checker.py.

validate_against_clean_subset.py
    Reruns run_pipeline at the 90% thresholds on the Train subset and
    compares the result byte-for-byte (PPG_Features values, Subject order,
    SBP/MAP/Age) against the pre-existing Clean_Features_VitalDB_Train_
    Subset_90.h5 that the OLD pipeline produced. This is the correctness
    check requested for this refactor.

ASSUMPTIONS.txt
    Everything that had to be inferred or decided without checking with
    the project owner. Read this before trusting the numbers.


INPUTS (per subset, from PulseDB SupplementarySubsets folder)
---------------------------------------------------------------

Fiducial_Points_<subset>.h5
    Group "PPG_fiducial_points", dataset "Fiducials", shape
    (n_windows_per_signal * 16, n_signals). Each column is one signal;
    every 16 consecutive rows are one window's fiducial indices, in the
    order defined by metrics.FIDUCIAL_ORDER.

Features_<subset>.h5
    Flat datasets, each shaped (1, n_signals) except PPG_Features which is
    (28, n_signals): Age, BMI, DBP, Gender, Height, MAP, PPG_Features, SBP,
    SF, Subject, Weight. n_signals and column order must match the
    fiducials file exactly (they are joined purely by column position).


OUTPUTS (written to fiducial_quality_filtering/outputs/ by default)
---------------------------------------------------------------------

metrics_<subset>.h5
    The full quality report: one row per signal, columns = checkHR,
    checkSP, numberProperFiducials, combinedScore, ppg/d1/d2/d3 (percent of
    each derivative's fiducials properly ordered), report (1 = discard).
    This is the input cleaner.py reads.

Problems_Fiducials_<subset>.xlsx
    Mean percent of windows flagged as problematic, per fiducial point
    (on, sp, dn, ... p2), across the whole subset. Descriptive, pre-drop.

criteria_report_<subset>.csv
    THE per-criterion percentages/scores artifact, computed before
    anything is dropped: % failing HR check, % failing peak-count check,
    mean/std % fiducials detected, mean/std combined score, % flagged for
    removal, and (once detect() has run) the count actually removed.

Clean_Features_<subset>_<threshold>.h5
    Same shape/format as Features_<subset>.h5, with flagged signals
    dropped. threshold in the filename is DISCARD_THRESHOLDS["thres_
    fiducials"] from run_pipeline.py's config (90 by default).

Demographic_Info_<subset>.xlsx
    Per-subject total/removed signal counts and percentage removed.
    Descriptive, computed AFTER dropping (this one necessarily reflects the
    drop, since it's reporting on the drop itself).


HOW TO RUN
----------

From inside this folder, using the project's Python 3.10 virtual
environment (see ASSUMPTIONS.txt about the state of that environment):

    ..\..\.venv310\Scripts\python.exe run_pipeline.py
    ..\..\.venv310\Scripts\python.exe validate_against_clean_subset.py

Edit the CONFIGURATION block at the top of run_pipeline.py first if your
data lives somewhere other than the OneDrive SupplementarySubsets path
hardcoded there, or if you want different alpha/beta/thresholds.


SWEEPING ALPHA/BETA OR THE THRESHOLDS
--------------------------------------

Don't call run_pipeline.run_subset() in a loop - it reloads the data and
reruns the expensive per-signal metrics every time. Instead:

    checker = QualityChecker(...)
    checker.compute_raw_metrics("Full_set")          # run ONCE
    for w_consistency, w_alignment in weight_grid:
        for thres_fid, thres_score in threshold_grid:
            df = checker.score_and_report(
                "Full_set", w_consistency, w_alignment, thres_fid, thres_score
            )
            # df["report"] now reflects this parameter combination -
            # inspect / aggregate as needed.

See checker.py's module docstring for why this split is safe (alignment
and consistency don't depend on the weights or thresholds at all).
