# ISIC 2019 Out-of-Distribution (OOD) Evaluation Results

This folder contains the complete Out-of-Distribution (OOD) evaluation results of our 24 models against the ISIC 2019 test set.

## Dataset Preparation & Fairness

To ensure a rigorously fair and strictly Out-of-Distribution evaluation, two critical preprocessing steps were applied to the ISIC 2019 dataset before running these tests:

1. **Class Alignment**: The original HAM10000 training dataset contains 7 diagnostic classes. The ISIC 2019 dataset introduces an 8th class. To maintain a fair evaluation of the models' learned representations, all images belonging to this novel 8th class were removed from the test set.
2. **Leakage Prevention**: The ISIC 2019 dataset is a superset that includes all of the original HAM10000 images. To prevent data leakage and ensure this is a true *out-of-distribution* evaluation on unseen patients, every single HAM10000 image was completely scrubbed and removed from the ISIC 2019 test set. 

Only completely novel, unseen images from the ISIC 2019 dataset were used to generate the results in this folder.

## Folder Structure

The results are organized into three distinct tiers of analysis:

* **`results/`**: 
  Contains the raw, individual outputs for each of the 24 models evaluated. This includes image-level predictions, individual confusion matrices, and detailed metrics (F1, Accuracy, False-Negative rates) stored as JSON files.

* **`Compared_Results/`**: 
  Contains side-by-side performance comparisons. The framework pairs each Dual-Metric model against its corresponding Baseline (Uncertainty-Only) model. Inside, you will find direct visual comparisons of their Clinical Safety (False Negative Rates) and Classification Performance (F1/Accuracy) on the OOD data.

* **`All_Compared_Results/`**: 
  Contains the grand summary of the OOD evaluation. This includes master summary tables (`.csv` and LaTeX format) and aggregate bar charts comparing the performance and safety of all 24 models simultaneously.
