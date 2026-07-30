# 🔬 Researcher Guide — How to Run Experiments

> **This guide is your step-by-step reference for running, resuming, and interpreting every experiment in the Risk-Aware Active Learning framework.**

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [All CLI Commands](#all-cli-commands)
  - [Running a Single Experiment](#1-running-a-single-experiment)
  - [Running All 24 Experiments](#2-running-all-24-experiments)
  - [Resuming Interrupted Experiments](#3-resuming-interrupted-experiments)
  - [Regenerating Plots](#4-regenerating-plots-from-saved-results)
- [Understanding the Experiment Matrix](#understanding-the-experiment-matrix)
- [Per-Round Calibrated Thresholds](#per-round-calibrated-thresholds)
- [Dynamic Class Weights (Ablation)](#dynamic-class-weights-ablation)
- [What Happens During an Experiment](#what-happens-during-an-experiment)
- [Output Files Reference](#output-files-reference)
  - [Logs](#logs)
  - [Checkpoints](#checkpoints)
  - [Plots](#plots)
  - [Tables](#tables)
- [How to Read the Results](#how-to-read-the-results)
- [Key Metrics Explained](#key-metrics-explained)
- [Common Scenarios](#common-scenarios)
- [Tips for Colab / Cloud GPUs](#tips-for-colab--cloud-gpus)

---

## Prerequisites

1. **Python 3.8+** with the following packages installed:
   ```
   torch, torchvision, numpy, pandas, scikit-learn, matplotlib, tqdm, Pillow
   ```

2. **Dataset**: The HAM10000 dataset must be available in a sibling directory named exactly `Ham-1000000 Dataset for skin`. It MUST contain the two image part folders and the metadata.
   ```text
   Parent Folder/
   ├── Ham-1000000 Dataset for skin/
   │   ├── HAM10000_images_part_1/     # Folder with ~5,000 images (ISIC_*.jpg)
   │   ├── HAM10000_images_part_2/     # Folder with ~5,000 images (ISIC_*.jpg)
   │   └── HAM10000_metadata.csv       # Original dataset metadata file
   │
   └── RiskAware-ActiveLearning/       # This codebase
       ├── active_learning/
       ├── config.py
       └── ...
   ```

3. **Seed Data**: Already prepared in the `Seed Data/` folder (490 images, 70 per class).

4. **GPU**: A CUDA-compatible GPU is strongly recommended (tested on NVIDIA T4).

---

## Quick Start

```bash
# Run the default experiment (EfficientNet-B4 + Entropy + Dual-Metric)
python main.py

# Run all 24 experiments
python main.py --run-all

# Resume after a crash
python main.py --run-all --resume
```

---

## All CLI Commands

### 1. Running a Single Experiment

**Syntax:**
```bash
python main.py --model <MODEL> --uncertainty <METHOD> --policy <POLICY> [--rounds N] [--risk-threshold T]
```

**Available options:**

| Parameter           | Choices                                              | Default            |
|---------------------|------------------------------------------------------|--------------------|  
| `--model`           | `efficientnet_b4`, `resnet50`, `densenet169`         | `efficientnet_b4`  |
| `--uncertainty`     | `entropy`, `mc_dropout`, `margin`, `least_confidence`| `entropy`          |
| `--policy`          | `uncertainty_only`, `dual_metric`                    | `dual_metric`      |
| `--rounds`          | Any integer (1–50)                                   | `15`               |
| `--risk-threshold`  | Any float — **manual override**, skips risk calibration (leave unset to recalibrate every round) | unset (calibrated) |
| `--use-dynamic-weights` | Flag (no value) — **no-op**, weighting is already on by default | on |
| `--no-dynamic-weights` | Flag (no value) — forces weighting OFF, for the unweighted ablation | — |

> **Note:** the *uncertainty* threshold has no CLI flag — it is always recalibrated every round (see [Per-Round Calibrated Thresholds](#per-round-calibrated-thresholds) below). Only the *risk* threshold can be manually pinned, for the threshold-sensitivity ablation sweep.

**Examples:**

```bash
# EfficientNet-B4 with MC Dropout and baseline policy
python main.py --model efficientnet_b4 --uncertainty mc_dropout --policy uncertainty_only

# ResNet-50 with Margin sampling and our dual-metric policy
python main.py --model resnet50 --uncertainty margin --policy dual_metric

# DenseNet-169 with Least Confidence, only 5 rounds (for quick testing)
python main.py --model densenet169 --uncertainty least_confidence --policy dual_metric --rounds 5

# Same experiment but with a stricter risk threshold (0.2 instead of default 0.3)
python main.py --model efficientnet_b4 --uncertainty entropy --policy dual_metric --risk-threshold 0.2

# Run with all defaults (EfficientNet-B4 + Entropy + Dual-Metric, 15 rounds, risk=0.3)
python main.py
```

**What happens:**
1. A fresh `PoolManager` is created — splits 9,525 remaining images into 80% unlabeled pool (~7,620) and 20% test set (~1,905).
2. The model trains on the 490 seed images, scores the unlabeled pool, applies the escalation policy, and repeats for `N` rounds.
3. Results are saved to the `results/` folder (see [Output Files Reference](#output-files-reference)).

---

### 2. Running All 24 Experiments

```bash
python main.py --run-all
```

This runs every combination in the full experiment matrix:

```
3 models × 4 uncertainty methods × 2 policies = 24 experiments
```

Each experiment gets a fresh `PoolManager` and model. After all 24 finish, comparative plots are automatically generated.

**With custom rounds:**
```bash
python main.py --run-all --rounds 10
```

---

### 3. Resuming Interrupted Experiments

If your T4 disconnects, laptop shuts down, or Colab runtime resets — just add `--resume`:

```bash
# Resume a specific experiment
python main.py --model resnet50 --uncertainty mc_dropout --policy dual_metric --resume

# Resume the full 24-experiment run
python main.py --run-all --resume
```

**What `--resume` does:**

| Scenario | Behavior |
|---|---|
| **Experiment fully completed** (has `_full.json`) | Skips entirely, loads saved results |
| **Experiment partially completed** (has checkpoint) | Loads model weights + pool state + accumulated metrics from the last completed round and continues from the next round |
| **Experiment never started** (no checkpoint) | Starts from scratch |

**Example output when resuming:**
```
======================================================================
EXPERIMENT: resnet50_mc_dropout_dual_metric
  Model: resnet50
  Uncertainty: mc_dropout
  Policy: dual_metric
  Rounds: 15
======================================================================

  ⟳ Resuming from checkpoint: round 8
  Checkpoint loaded: results/checkpoints/resnet50_mc_dropout_dual_metric/round_8/model.pt
  Pool state loaded from results/checkpoints/.../round_8/pool_state
  ⟳ Will continue from round 9

--- Round 9/15 ---
  ...
```

---

### 4. Regenerating Plots from Saved Results

If experiments are already complete and you just want to regenerate visualizations:

```bash
python main.py --plot-only
```

This loads results from `results/logs/all_experiments.json` (or individual `*_full.json` files) and regenerates all 7 plots + the comparison table.

---

## Understanding the Experiment Matrix

The full experiment matrix compares **every combination** of model, uncertainty method, and escalation policy:

| Dimension | Options | Purpose |
|---|---|---|
| **Models** | EfficientNet-B4, ResNet-50, DenseNet-169 | Test generalizability across architectures |
| **Uncertainty** | Entropy, MC Dropout, Margin, Least Confidence | Compare uncertainty estimation approaches |
| **Policy** | Uncertainty-Only (baseline), Dual-Metric (ours) | **This is the core comparison** |

Each experiment ID follows the pattern: `{model}_{uncertainty}_{policy}`

For example: `efficientnet_b4_entropy_dual_metric`

When using a non-default risk threshold, it is automatically appended to the ID:

`efficientnet_b4_entropy_dual_metric_rt0.2`

This means results from different thresholds are saved separately and never overwrite each other.

---

## Per-Round Calibrated Thresholds

Thresholds used to be hardcoded (`UNCERTAINTY_THRESHOLD = 0.5`, `RISK_THRESHOLD = 0.3` in `config.py`). They no longer drive the actual decisions — they're kept only as documented fallback constants.

**What happens now:** at the start of **every** round, the model scores its **current labelled set** with itself. The **90th-percentile** uncertainty score and **90th-percentile** risk score become that round's escalation thresholds. This is done separately for whichever uncertainty method the experiment is using.

> ⚠️ **This changed, and the old behaviour was a real bug.** The first design calibrated **once** in round 1 against the 490 seed images and froze those values. A rapidly-improving model then stopped producing scores anywhere near a bar set against its weakest self, and escalation collapsed: **558 images → 1 → 0** by round 3, with the labelled set frozen for the remaining 12 rounds while training accuracy ran to 99% and test accuracy plateaued. Recalibrating every round, against the *current* labelled set, is what keeps the policy live as the model improves. Any document still describing round-1-only calibration is out of date.

**Why this replaces a fixed 0.5/0.3:** uncertainty scores are not squeezed into a shared `[0, 1]` range — entropy can go up to `ln(7) ≈ 1.95`, MC-dropout variance sits much closer to `0`. A single constant like `0.5` meant something completely different (and often nonsensical) depending on which uncertainty method was active. Calibrating per-method, per-round fixes that, and mirrors how you'd actually calibrate a clinical deployment: run the current model on a known labelled batch and see what "unusually high" looks like *for that model as it stands*.

**Where it's logged:** each round's `calibrated_unc_threshold_this_round` and `calibrated_risk_threshold_this_round` are written into `results.csv` and into that round's checkpoint `meta.json`. Because they are recomputed each round, they are a **trajectory, not a constant** — the reproducibility table should quote the per-round values (or their range), not a single number per experiment.

**`--risk-threshold` still works as a manual override** — pass it to skip risk calibration and pin a specific value, which is exactly what the threshold-sensitivity sweep (see [Common Scenarios](#common-scenarios)) needs. The uncertainty threshold is always calibrated; there's no manual override for it.

**Resuming:** thresholds are **not** reloaded from `meta.json` on `--resume`. They are recalculated fresh for the round being run, which is exactly what an uninterrupted run would have done — so a resumed run reproduces an uninterrupted one. (This simplified the resume path: there is no threshold state to restore.)

---

## Dynamic Class Weights (Ablation)

HAM10000 is heavily imbalanced (≈67% of images are benign nevi, `nv`). Inverse-frequency class weighting is therefore **ON by default** (`config.USE_DYNAMIC_CLASS_WEIGHTS = True`) and is applied to **both** the classification head and the risk head. Weights are recomputed from the **current labelled set** every round (as more images get queried the class balance shifts, so the weights shift with it):

```
weight[class] = n_labeled_total / (num_classes × count[class])
```

This is the same formula as scikit-learn's `class_weight='balanced'`. A class with few labeled examples gets a proportionally bigger weight in the loss.

```bash
# Baseline (unweighted) — this is also the default with no flag
python main.py --model efficientnet_b4 --uncertainty entropy --policy dual_metric

# Ablation: same config, with dynamic class weights
python main.py --model efficientnet_b4 --uncertainty entropy --policy dual_metric --no-dynamic-weights
```

Runs with the flag get a `_dynw` suffix appended to their experiment ID (e.g. `efficientnet_b4_entropy_dual_metric_dynw`), so weighted and unweighted results are never overwritten by each other and can be compared directly.

---

## What Happens During an Experiment

Each active learning round (default: 15 per experiment) follows this sequence:

| Step | Action | Details |
|------|--------|---------|
| 1 | **Train** | Train the model on the current labeled set (starts at 490 seed images); uses dynamic class weights by default (both heads); `--no-dynamic-weights` disables them |
| 1b | **Calibrate** *(every round)* | Score the **current labelled set** with the just-trained model; take the 90th-percentile uncertainty/risk scores as **this round's** thresholds — see [Per-Round Calibrated Thresholds](#per-round-calibrated-thresholds) |
| 2 | **Score** | Run inference on the entire unlabeled pool — compute uncertainty + risk scores |
| 3 | **Decide** | Apply the escalation policy (uncertainty-only or dual-metric 2×2 grid) using the calibrated thresholds |
| 4 | **Query** | Move escalated images from unlabeled → labeled set (simulated oracle) |
| 5 | **Track Safety** | Count "unsafe auto-accepts" — high-risk images the model auto-accepted |
| 6 | **Evaluate** | Test on the fixed held-out test set (~1,905 images) |
| 7 | **Log** | Record all metrics for this round |
| 8 | **Checkpoint** | Save model weights + pool state + results so far |

---

## Output Files Reference

All outputs are saved under the `results/` directory:

```
results/
├── checkpoints/                          # Model & state snapshots
│   └── {experiment_id}/
│       └── round_{N}/
│           ├── model.pt                  # PyTorch model weights
│           ├── pool_state/               # Data pool state
│           │   ├── labeled.csv
│           │   ├── unlabeled.csv
│           │   ├── test.csv
│           │   └── query_history.csv
│           └── meta.json                 # Round number + accumulated results
│
├── logs/                                 # Metrics & raw predictions
│   ├── {experiment_id}_results.csv       # Per-round metrics (updated live)
│   ├── {experiment_id}_full.json         # Complete results (saved at end)
│   │                                 # (no _calibration.json — thresholds now live in results.csv)
│   ├── {experiment_id}/
│   │   ├── round_1_pool_predictions.csv  # Every image scored in round 1
│   │   ├── round_2_pool_predictions.csv
│   │   └── ...
│   └── all_experiments.json              # Combined results (--run-all)
│
├── plots/                                # Generated visualizations
│   ├── accuracy_vs_queries.png
│   ├── fn_rate_over_rounds.png
│   ├── unsafe_auto_accepts.png
│   ├── per_class_f1.png
│   ├── annotation_efficiency.png
│   └── {experiment_id}/                  # Per-experiment plots
│       └── uncertainty_vs_risk_scatter_*.png
│
└── tables/                               # LaTeX-ready comparison tables
    └── comparison_table.csv
```

### Logs

| File | Description | When Created |
|---|---|---|
| `{id}_results.csv` | One row per round with all metrics, including `uncertainty_threshold_used`, `risk_threshold_used`, and `use_dynamic_weights`. **Updated after every round** — survives crashes. | Continuously during experiment |
| `{id}_full.json` | Complete experiment configuration + all round results in JSON. | At experiment completion |
| ~~`{id}_calibration.json`~~ | **No longer written.** Thresholds are recalibrated every round, so a single locked-in pair no longer exists. The per-round values are columns `calibrated_unc_threshold_this_round` / `calibrated_risk_threshold_this_round` in `results.csv`, and are also stored in each round's checkpoint `meta.json`. | — |
| `{id}/round_N_pool_predictions.csv` | For every image in the unlabeled pool: `image_id`, `true_label`, `predicted_label`, `uncertainty_score`, `risk_score`, `decision`, `category`. | After each round |
| `all_experiments.json` | Combined JSON of all 24 experiments (for plotting). | After `--run-all` completes |

### Checkpoints

| File | Description |
|---|---|
| `model.pt` | Full model `state_dict` for the round |
| `pool_state/labeled.csv` | Which images are in the labeled set at this point |
| `pool_state/unlabeled.csv` | Which images remain in the unlabeled pool |
| `pool_state/test.csv` | The fixed test set (same every round, saved for completeness) |
| `pool_state/query_history.csv` | History of how many images were queried per round |
| `meta.json` | JSON with `completed_round`, `experiment_id`, `round_results` array |

> **Note:** To save disk space, only the **latest** round's checkpoint is kept. The previous round's checkpoint is automatically deleted after a new one is saved.

### Plots

| Plot | Filename | What It Shows |
|---|---|---|
| Accuracy / F1 vs Queries | `accuracy_vs_queries.png` | Learning curves — how accuracy/F1 improves as more images are labeled |
| FN Rate Over Rounds | `fn_rate_over_rounds.png` | Safety metric — does the false-negative rate on malignant classes decrease? |
| Unsafe Auto-Accepts | `unsafe_auto_accepts.png` | Bar chart — how many high-risk images were incorrectly auto-accepted |
| Per-Class F1 | `per_class_f1.png` | Grouped bar chart — F1 score for each of the 7 skin lesion classes |
| Annotation Efficiency | `annotation_efficiency.png` | Accuracy gained per oracle query — measures efficiency |
| Uncertainty vs Risk Scatter | `uncertainty_vs_risk_scatter_*.png` | 2×2 grid visualization for dual-metric experiments (round 1 and final round) |

### Tables

| File | Description |
|---|---|
| `comparison_table.csv` | Final-round metrics for all experiments, ready for copy-paste into a paper |

---

## How to Read the Results

### The `_results.csv` File (Most Important)

Open `results/logs/{experiment_id}_results.csv` to see a table like:

| round | labeled_count | unlabeled_count | queries_this_round | total_queries | auto_accepted_this_round | unsafe_auto_accepts | accuracy | f1_macro | fn_rate_malignant |
|-------|--------------|-----------------|-------------------|---------------|-------------------------|--------------------|-----------|---------|--------------------|
| 1     | 620          | 7490            | 130               | 130           | 7490                    | 12                 | 0.6821   | 0.5234  | 0.3421             |
| 2     | 755          | 7355            | 135               | 265           | 7220                    | 8                  | 0.7156   | 0.5678  | 0.2987             |
| ...   | ...          | ...             | ...               | ...           | ...                     | ...                | ...       | ...     | ...                |

### The `round_N_pool_predictions.csv` File

This gives you image-level detail for each round:

| image_id | true_label | predicted_label | uncertainty_score | risk_score | decision | category |
|---|---|---|---|---|---|---|
| ISIC_0027419 | mel | nv | 0.23 | 0.41 | escalate | low_unc_high_risk |
| ISIC_0025030 | nv | nv | 0.08 | 0.05 | auto_accept | low_unc_low_risk |

The `category` column is only meaningful for `dual_metric` experiments — it tells you which quadrant of the 2×2 grid each image fell into.

---

## Key Metrics Explained

| Metric | What It Measures | Why It Matters |
|---|---|---|
| `accuracy` | Overall classification accuracy on the test set | General model performance |
| `f1_macro` | Macro-averaged F1 across all 7 classes | Handles class imbalance better than accuracy |
| `fn_rate_malignant` | **False-negative rate on high-risk classes** (mel, bcc, akiec) | **PRIMARY SAFETY METRIC** — missed cancers |
| `fn_rate_melanoma` | False-negative rate specifically for melanoma | Most dangerous class |
| `unsafe_auto_accepts` | Number of high-risk images that were auto-accepted (not sent to oracle) | Measures how often the policy lets a dangerous image through |
| `queries_this_round` | How many images were sent to the oracle this round | Measures annotation cost |
| `f1_{class}` | Per-class F1 (e.g., `f1_mel`, `f1_bcc`, etc.) | Performance on individual diagnoses |

### The Key Comparison

The paper's central claim is:

> **Dual-metric policy should have a lower `fn_rate_malignant` and fewer `unsafe_auto_accepts` than uncertainty-only, without significant loss in `accuracy` or `f1_macro`.**

When analyzing results, compare these metrics between the two policies for the same model + uncertainty method.

---

## Common Scenarios

### "I just want to quickly test if everything works"

```bash
python main.py --model resnet50 --uncertainty entropy --policy dual_metric --rounds 2
```

This runs only 2 rounds with ResNet-50 — should finish in a few minutes on a GPU.

### "I want to compare the two policies head-to-head for one model"

```bash
# Run baseline
python main.py --model efficientnet_b4 --uncertainty entropy --policy uncertainty_only

# Run ours
python main.py --model efficientnet_b4 --uncertainty entropy --policy dual_metric
```

Then compare `efficientnet_b4_entropy_uncertainty_only_results.csv` vs `efficientnet_b4_entropy_dual_metric_results.csv`.

### "I want to run a threshold sensitivity analysis"

This is a powerful way to strengthen the paper — show that the dual-metric policy is robust across different risk thresholds:

```bash
# Run the same model+uncertainty combo at multiple risk thresholds
python main.py --model efficientnet_b4 --uncertainty entropy --policy dual_metric --risk-threshold 0.1
python main.py --model efficientnet_b4 --uncertainty entropy --policy dual_metric --risk-threshold 0.2
python main.py --model efficientnet_b4 --uncertainty entropy --policy dual_metric --risk-threshold 0.3
python main.py --model efficientnet_b4 --uncertainty entropy --policy dual_metric --risk-threshold 0.4
python main.py --model efficientnet_b4 --uncertainty entropy --policy dual_metric --risk-threshold 0.5
```

Each run saves to a separate experiment ID (e.g., `..._rt0.1`, `..._rt0.2`, etc.), so results never overwrite each other. You can then compare the `fn_rate_malignant` and `unsafe_auto_accepts` across thresholds to find the optimal tradeoff between safety and annotation cost.

### "Run ALL 24 experiments at a custom threshold"

```bash
python main.py --run-all --risk-threshold 0.2 --resume
```

This runs the full 24-experiment matrix but with a risk threshold of 0.2 instead of the default 0.3.

### "My Colab disconnected at experiment 14/24"

```bash
python main.py --run-all --resume
```

Experiments 1–13 (completed) will be skipped. Experiment 14 will resume from its last checkpoint. Experiments 15–24 will run from scratch.

### "I made changes to the visualization code and want to re-plot"

```bash
python main.py --plot-only
```

---

## Tips for Colab / Cloud GPUs

1. **Always use `--resume`** when re-running on cloud GPUs. It's safe to add even on a fresh run (it will start from scratch if no checkpoint exists).

2. **Mount Google Drive** and update paths in `config.py` so results survive runtime resets:
   ```python
   # In Colab cell:
   from google.colab import drive
   drive.mount('/content/drive')

   # Update config.py paths:
   RESULTS_DIR = '/content/drive/MyDrive/RiskAware-AL/results'
   ```

3. **MC Dropout experiments take ~3× longer** than single-pass methods (30 forward passes per scoring round). Plan accordingly:
   - Entropy / Margin / Least Confidence: ~15 min per experiment
   - MC Dropout: ~45 min per experiment
   - **Total for all 24**: approximately 10–12 hours on a T4

4. **Check `_results.csv` files** for intermediate progress — they update after every round, even before the experiment finishes.

5. **Disk space**: Checkpoints auto-clean (only the latest round is kept), but the logs and pool prediction CSVs accumulate. Budget ~2–3 GB for the full 24-experiment run.
