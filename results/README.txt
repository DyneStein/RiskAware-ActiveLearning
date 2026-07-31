===============================================================================
results/  --  the raw output of every experiment
===============================================================================

This is the unprocessed record. Nothing here has been summarised or filtered.
If you want the conclusions instead, read RESULTS.txt in the repository root.


-------------------------------------------------------------------------------
HOW IT IS ORGANISED
-------------------------------------------------------------------------------

    results/experiments/<name>/

One folder per experiment, 36 in total. The folder name tells you what the
experiment was:

    resnet50_entropy_dual_metric
    |        |       |
    |        |       +-- the decision rule
    |        +---------- how uncertainty was measured
    +------------------- the network

  network        resnet50, densenet169, efficientnet_b4
  uncertainty    entropy, mc_dropout, margin, least_confidence
  decision rule  dual_metric        = ours (uncertainty AND danger)
                 uncertainty_only   = the standard rule (uncertainty alone)

Folders named <network>_baseline_<method> are the four published methods we
compared against: coreset, badge, clue, vaal.


-------------------------------------------------------------------------------
WHAT IS INSIDE EACH EXPERIMENT FOLDER
-------------------------------------------------------------------------------

  results.csv
      The main file. One row per round, 15 rows. Open it in Excel.
      Columns include:
        round                 which round, 1 to 15
        accuracy              how often the model was right, on the test set
        f1_macro              accuracy that treats rare diseases as equally
                              important as common ones
        fn_rate_malignant     share of cancers the model called harmless
        fn_rate_melanoma      the same, for melanoma only
        unsafe_auto_accepts   dangerous cases waved through with no review
                              -- this is the number the project is about
        queries_this_round    how many cases were sent to the doctor
        labeled_size          how many labelled images the model had by then

  full.json
      The same information as one blob, plus the settings used.

  environment.json
      Which GPU, which CUDA, which library versions, which code commit.
      Present on the newer runs. The original 24 runs predate this file and
      their Colab environments were not recorded -- we say so rather than
      invent it.

  pool_predictions/
      One CSV per round. For every unlabelled image that round: what the model
      predicted, how uncertain it was, how dangerous it judged the case, and
      whether it was waved through or sent to the doctor. This is the file that
      lets anyone recompute our headline number from scratch.

  plots/
      Confusion matrices and a learning curve for that single experiment.


-------------------------------------------------------------------------------
NOT IN THIS REPOSITORY
-------------------------------------------------------------------------------

results/checkpoints/ holds the trained networks themselves -- 2.6 GB. Git is
not a good home for files that size, so they are kept outside and published
separately. See HOW_TO_RUN.txt.

===============================================================================
