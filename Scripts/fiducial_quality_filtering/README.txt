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

raw_metrics_cache.py
    Cacheable, vectorized version of the expensive per-signal loop. Same
    numbers as checker.py (reuses metrics.py directly, no duplicated
    logic), but stored as plain numpy arrays instead of per-signal pandas
    DataFrames, so they can be (a) saved to a single .npz file and (b)
    scored for ANY alpha/beta/thresholds with a few vectorized numpy ops
    over ALL signals at once - no per-signal Python loop. Validated against
    checker.py's per-signal path (exact report match, score match to
    float32 precision) before being used for the sweeps below.

build_cache.py
    Runs raw_metrics_cache.compute_raw_metrics_cache() once per subset and
    saves RawMetricsCache_<subset>.npz into the PulseDB SupplementarySubsets
    folder (next to the source data, NOT in outputs/ - it's an input to the
    sweep scripts, not a result). Only needs re-running if the underlying
    fiducial data changes. Takes ~60-90 min for the Train subset (465k
    signals, ~2GB RAM); the Test subset is ~8x smaller and much faster.

sweep_common.py
    Shared helpers for the two sweep scripts below: turning a per-signal
    discard mask into the summary stats and the reusable dropped-signal
    JSON (see OUTPUTS below).

sweep_alpha_beta.py
    First-pass sensitivity sweep over alpha (w_consistency), with
    beta = 1 - alpha (a 1D sweep, NOT a full alpha x beta grid).
    Thresholds held fixed at 90/90. Loads the cache from build_cache.py -
    does NOT recompute raw metrics.

sweep_thresholds.py
    Same idea, sweeping thres_fiducials = thres_score together (a 1D
    sweep - NOT thres_fiducials and thres_score varied independently).
    alpha/beta held fixed at 0.25/0.75.


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


SWEEP OUTPUTS (written to the Filtering_Sensitivity folder, one subfolder
per subset - see ASSUMPTIONS.txt for the exact path)
---------------------------------------------------------------------------

<subset>/alpha_beta_sweep_summary.csv
<subset>/threshold_sweep_summary.csv
    One row per grid point (alpha value, or threshold value): the swept
    parameter(s), samples dropped (raw count + % of the full subset),
    subjects with >=1 dropped sample (raw count + %), and subjects
    ENTIRELY dropped - 100% of their samples gone (raw count + %). Both
    subject-drop definitions are reported side by side, since they can
    diverge a lot (see ASSUMPTIONS.txt).

<subset>/dropped_signals/alpha_beta/alpha_<value>.json
<subset>/dropped_signals/threshold/thres_<value>.json
    One JSON file per grid point: which exact signals were dropped at that
    parameter value. Schema:
        {
          "subset": "...",
          "parameters": {"w_consistency": 0.3, "w_alignment": 0.7,
                          "thres_fiducials": 90, "thres_score": 90},
          "n_total_signals": 465480,
          "n_dropped": 276707,
          "dropped": {"p000001_1": [3, 7, 12, ...], "p000003_1": [0, 1, ...]}
        }
    "dropped" maps subject id -> list of dropped LOCAL sample indices (the
    signal's position within that subject's own block of signals, in
    Features_<subset>.h5 file order - not a global index). This is exactly
    what you need to build a per-subject keep/drop boolean mask when
    loading the full dataframe for training, without re-running any of
    this pipeline.


HOW TO RUN
----------

From inside this folder, using the project's Python 3.10 virtual
environment (see ASSUMPTIONS.txt about the state of that environment):

    ..\..\.venv310\Scripts\python.exe run_pipeline.py
    ..\..\.venv310\Scripts\python.exe validate_against_clean_subset.py

Edit the CONFIGURATION block at the top of run_pipeline.py first if your
data lives somewhere other than the OneDrive SupplementarySubsets path
hardcoded there, or if you want different alpha/beta/thresholds.

For a sweep, run once (expensive, ~60-90 min for Train):

    ..\..\.venv310\Scripts\python.exe build_cache.py

then as many times as you like (cheap, seconds, since it only loads the
cache and does vectorized numpy - no per-signal loop):

    ..\..\.venv310\Scripts\python.exe sweep_alpha_beta.py
    ..\..\.venv310\Scripts\python.exe sweep_thresholds.py

Edit ALPHA_GRID / THRESHOLD_GRID / FIXED_THRESHOLDS / FIXED_WEIGHTS at the
top of those two scripts to change the grid or the held-fixed parameter.


WRITING YOUR OWN SWEEP (a genuine alpha x beta x threshold grid, a
different subject-drop rule, etc.)
---------------------------------------------------------------------

Load the cache once, then score as many times as you like - each call is
vectorized numpy over every signal at once, no Python loop:

    from raw_metrics_cache import load_cache, vectorized_report
    cache = load_cache("RawMetricsCache_<subset>.npz")   # from build_cache.py
    discard_mask = vectorized_report(cache, w_consistency, w_alignment,
                                      thres_fiducials, thres_score)
    # discard_mask: bool array, one per signal, True = drop.
    # Feed it to sweep_common.summarize_drop()/dropped_signals_payload()
    # for the same stats/JSON shape the two sweep scripts produce.

If you don't have a cache yet and only need ONE parameter combination
(not a sweep), checker.QualityChecker still works standalone - see
checker.py's module docstring. It's the same math, just per-signal pandas
instead of vectorized numpy, and not cacheable.
