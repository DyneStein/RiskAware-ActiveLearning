===============================================================================
paper/  --  the figures and tables in publication form
===============================================================================

Same numbers as in analysis/, but selected, tidied and exported at print
quality. If you want a figure to put in a slide or a manuscript, take it from
here.


-------------------------------------------------------------------------------
03_FIGURES/main_paper/     the 11 figures that carry the argument
-------------------------------------------------------------------------------

  01_unsafe_auto_accepts_total.png     the headline safety result
  09_headline_summary.png              the whole study on one chart
  10_al_efficiency_accuracy_vs_labels.png
                                       accuracy against labels spent -- this is
                                       the chart that shows the safety gain is
                                       bought with extra labels, not free
  14_ablation_both_signals_needed.png  what breaks when you remove either half
  16_safety_cost_pareto.png            safety against cost; upper-left is best
  17_reliability_classification.png    is the model's confidence honest?
  18_reliability_risk_head.png         is the danger score's confidence honest?
  22_auc_per_class_with_ci.png         per-disease performance, with the
                                       unreliable classes marked as unreliable
  25_significance_heatmap.png          which differences are real
  (Grad-CAM panel removed -- see XAI_results/ for the updated visual analysis)
  33_robustness_by_model.png           what happens as image quality drops

03_FIGURES/supplementary/  23 more figures -- everything above broken down
further, plus the checks that support them.


-------------------------------------------------------------------------------
04_TABLES/     21 spreadsheets
-------------------------------------------------------------------------------

The ones most likely to be wanted:

  master_summary.csv                one row per experiment, all key numbers
  dual_vs_uncertainty_comparison.csv  our rule versus the standard one
  significance_image_level.csv      the statistical tests that count
  ablation_decision_level.csv       the "both signals are needed" evidence
  al_efficiency_budget_matched.csv  the honest cost accounting
  per_class_auc.csv                 performance on each of the 7 diagnoses
  calibration_metrics.csv           whether stated confidence is trustworthy
  robustness_summary.csv            performance under degraded images
  risk_threshold_sweep.csv          what changes if the threshold moves
  runtime_per_experiment.csv        how long everything took

Others cover the risk-head design comparison and significance tested at other levels.


-------------------------------------------------------------------------------
COMPARISON/    the comparison against published methods
-------------------------------------------------------------------------------

Its own self-contained folder, with its own README.txt. This is the part to
show someone who asks "but how does it compare to what already exists?"


-------------------------------------------------------------------------------
XAI_results/   Sample Grad-CAM++ / EigenCAM / Score-CAM heatmap visualisations
-------------------------------------------------------------------------------

These are some of the results from the XAI analysis. 40 representative heatmap
images are included, spread across all three XAI methods and all three
architectures (ResNet-50, DenseNet-169, EfficientNet-B4).

Images are organised into subfolders by method:

  XAI_results/gradcam++/   14 images -- Grad-CAM++ (gradient-based, class-discriminative)
  XAI_results/eigenCAM/    13 images -- EigenCAM   (gradient-free, PCA-based)
  XAI_results/scoreCam/    13 images -- Score-CAM  (gradient-free, perturbation-based)

The majority of images show correct predictions.

Filename convention:
  <architecture>__<strategy>__<variant>__<image_id>__<CORRECT|FAILURE>__<Method>.png

Example:
  efficientnet_b4__entropy__dual_metric__ISIC_0068778__CORRECT__GradCAM++.png


-------------------------------------------------------------------------------
ISIC2019_OOD_evaluation/   Out-of-Distribution evaluation against ISIC 2019
-------------------------------------------------------------------------------

Contains the complete evaluation of the 24 models on the external ISIC 2019
test set. 

Fairness Preprocessing applied to ensure strict OOD generalisation:
1. HAM10000 has 7 classes, but ISIC2019 introduces an 8th. The 8th class
   was completely removed to fairly test the models' learned representations.
2. ISIC2019 is a superset containing all HAM10000 images. Every overlapping
   HAM10000 image was removed, guaranteeing evaluation solely on unseen patients.

Subdirectories:
  results/              Raw JSON metrics and predictions per model.
  Compared_Results/     Side-by-side plots of Dual-Metric vs Baseline models.
  All_Compared_Results/ Aggregate CSV/LaTeX tables and charts of all 24 models.


-------------------------------------------------------------------------------
A NOTE ON WORDING, IF YOU QUOTE THESE
-------------------------------------------------------------------------------

Two different things are measured in this project and they must not be mixed:

  The safety result is measured on the pool of unlabelled images -- the cases
  the decision rule actually decides about.

  The accuracy and missed-cancer results are measured on the held-out test set
  of 1,905 images the model never trained on.

The first is a statement about a decision rule. The second is a statement about
a finished model. They are both real and they answer different questions. Any
sentence quoting the safety number should say "on the unlabelled pool".

===============================================================================
