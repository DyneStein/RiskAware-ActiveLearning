===============================================================================
analysis/  --  everything computed after the experiments finished
===============================================================================

The experiments in results/ produced raw numbers. This folder holds what we
worked out from them: the summary tables, the statistical tests, and the
figures.

Rebuild all of it with:

    python analysis/build_analysis.py


-------------------------------------------------------------------------------
figures/   9 charts -- the first look at the results
-------------------------------------------------------------------------------

  01_unsafe_auto_accepts_total.png
        The headline chart. Dangerous cases waved through, our rule versus the
        standard one, for every network and uncertainty method. Our bar is
        shorter in every single pairing.

  02_fn_rate_malignant_final.png     cancers missed at the end, on the test set
  03_accuracy_f1_final.png           proof the safety gain did not cost accuracy
  04_query_cost_total.png            the price, in cases sent to the doctor
  05..07_*_trajectory_by_model.png   the same three things, round by round
  08_risk_score_auroc_trend.png      how good the danger score was, each round
  09_headline_summary.png            the one-glance version of all of it


-------------------------------------------------------------------------------
tables/   the three summary spreadsheets
-------------------------------------------------------------------------------

  master_summary.csv
        One row per experiment. Final accuracy, F1, missed-cancer rate, total
        dangerous cases waved through, total doctor queries used, and how good
        the danger score was. Start here.

  dual_vs_uncertainty_comparison.csv
        Our rule against the standard one, head to head, with every difference
        already worked out.

  risk_auroc_by_experiment.csv
        How well the danger score separated cancer from non-cancer, per run.


-------------------------------------------------------------------------------
  rigor/   rigorous statistical and robustness checks
-------------------------------------------------------------------------------

This is the part that turns a working prototype into a defensible study.

  figures/       32 charts covering everything below
  tables/        26 spreadsheets, one per check
  predictions/   what each trained model predicted for each test image, saved
                 so that image-by-image comparisons can be recomputed by anyone
  predictions_robustness/
                 the same, but after the test images were degraded on purpose
                 (blur, noise, JPEG compression) to see what breaks first

What gets checked:

  statistics        Is the difference real, or could it be luck? Because we ran
                    many comparisons at once, the results are adjusted so that
                    finding something by chance is accounted for.
  calibration       When the model says "90% sure", is it right about 90% of
                    the time? A safety story built on a badly calibrated score
                    would be hollow.
  ablation          Turn each half of the decision rule off and see what
                    breaks. This is where the "both signals are needed" result
                    comes from.
  per-class scores  How the model does on each of the 7 diagnoses separately,
                    with honest confidence intervals -- including a note saying
                    which classes have too few test images to be trusted.
  robustness        What happens to the results when the images get worse.
  runtime           What the extra check actually costs in compute.
  threshold sweep   What happens if the danger threshold is moved.
  Grad-CAM          Heatmaps showing where in the image the model was looking.

The code that produces all of this is in evaluation/rigor/. Each script is a
single file named after what it does.

===============================================================================
