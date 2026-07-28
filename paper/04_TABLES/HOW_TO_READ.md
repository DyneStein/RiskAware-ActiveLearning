# HOW TO READ — 04_TABLES

## The short version

- 21 spreadsheets, plain CSV — open in Excel, Sheets, or `pandas.read_csv`.
- **Start with `master_summary.csv`** — all 24 experiments, one row each. That's the whole study.
- **The p-values live in `significance_configuration_level.csv`.** Use the `wilcoxon_p_holm`
  column, not `wilcoxon_p`.
- **The best result is in `ablation_decision_level.csv`** — the four-rule comparison.
- **Signs matter:** for unsafe auto-accepts a **negative** difference is an **improvement**; for
  query counts a positive difference is a **cost**.
- For the numbers already computed and cross-checked, see
  `01_PAPER_WRITING_KIT/KEY_NUMBERS.md` instead.

**Everything below is the detailed column-by-column reference**, for when a specific column is
unclear.

---

## Conventions used in every file

| Column pattern | Meaning |
|---|---|
| `experiment_id` | `<model>_<method>_<policy>`, e.g. `resnet50_entropy_dual_metric`. **The key that joins every table.** |
| `model` | `resnet50`, `densenet169`, `efficientnet_b4` |
| `method` | `entropy`, `margin`, `least_confidence`, `mc_dropout` |
| `policy` | `dual_metric` (proposed) or `uncertainty_only` (baseline) |
| `_dual` suffix | value under the dual-metric policy |
| `_unc` / `_uncertainty_only` suffix | value under the baseline |
| `_delta` / `_diff` | dual **minus** baseline. **Sign matters** — see the box below |
| `_pp` | percentage **points** |
| `_pct` | a percentage |
| `ci_low`, `ci_high` | 95% confidence interval bounds (percentile bootstrap) |
| `_p`, `_p_holm` | raw p-value, and Holm–Bonferroni corrected p-value |
| Probabilities | stored as decimals (0.8898 = 88.98%), not percentages |

> ### ⚠️ Which direction is good?
> **Lower is better** for: `unsafe_auto_accepts`, `fn_rate_*`, `ece`, `mce`, `brier`, `nll`,
> `total_queries` (it is a cost), and all runtime columns.
> **Higher is better** for: `accuracy`, `f1_macro`, `auc`/`auroc`, `pr_auc`, `recall`,
> `high_risk_catch_rate`, and all `*_retained` columns.
>
> So a **negative** `mean_difference` on `unsafe_auto_accepts` is a **good** result, and a
> **positive** difference on `total_queries` is a **cost**. Read the metric before reading the sign.

---

# The tables, grouped by purpose

## A. Start here — the overview tables

### `master_summary.csv` — 24 rows, one per experiment
The single most useful file. Every experiment's final state.

| Column | Meaning |
|---|---|
| `rounds_completed`, `complete` | Should be 15 and `True` for all 24 rows |
| `final_labeled_count` | Images labelled by the end |
| `total_queries` | Oracle labels requested across all 15 rounds — **the cost** |
| `total_unsafe_auto_accepts` | **The primary endpoint.** Malignant pool images auto-accepted, summed over rounds |
| `final_accuracy`, `final_f1_macro` | Test-set quality at round 15 |
| `final_fn_rate_malignant` | Malignant test images predicted benign (any malignant class counts as a catch) |
| `final_fn_rate_melanoma` | Stricter: melanoma not identified *as melanoma* |
| `mean_fn_rate_malignant` | Averaged over all 15 rounds rather than final only |
| `auroc_round1`, `auroc_final`, `auroc_mean` | Risk head's own discrimination — evidence the risk signal is real independent of any policy |

### `dual_vs_uncertainty_comparison.csv` — 12 rows, one per matched pair
The head-to-head comparison with the deltas pre-computed: `unsafe_reduction_pct`,
`fn_rate_reduction_pct`, `accuracy_delta_pp`, `f1_delta_pp`, `extra_queries_pct`.
**Use this for the head-to-head comparison without recomputing anything.**

### `risk_auroc_by_experiment.csv` — 24 rows
The risk head's AUROC per experiment (round 1, final, mean). Behind figure 08.

---

## B. Statistics — where the p-values come from

### `significance_configuration_level.csv` — **the paper's primary statistical table**
6 rows, one per metric. n = 12 paired configurations.

| Column | Meaning |
|---|---|
| `mean_difference` | dual − baseline |
| `diff_ci_low`, `diff_ci_high` | 95% bootstrap CI. **If it crosses zero, "no difference" cannot be excluded** |
| `pairs_dual_higher` | How many of the 12 favoured dual. `12` or `0` means unanimous |
| `wilcoxon_p` | Raw paired Wilcoxon signed-rank p-value |
| `sign_test_p`, `permutation_p` | Two alternative tests — agreement between them is reassuring |
| `rank_biserial_effect` | Effect size. **−1.0 or +1.0 = every single pair moved the same way** |
| `cohens_dz` | Standardised effect size for paired data |
| **`wilcoxon_p_holm`** | **Quote this one.** Corrected for the six-metric family |
| `significant_holm_0.05` | `True` / `False` verdict |

### `significance_image_level.csv` — 12 rows, one per configuration
Compares the two policies **image by image** on the shared 1,905-image test set.

| Column | Meaning |
|---|---|
| `mcnemar_accuracy_p` | McNemar's test on accuracy. Uses **only** the images where the two policies disagreed — every image both got right, or both got wrong, carries no information about which is better |
| `mcnemar_disc_unc_only_correct` / `_dual_only_correct` | The discordant counts. These two numbers *are* the test |
| `*_ci_excludes_zero` | `True` = that difference is individually resolvable |
| `*_p_holm`, `*_sig_holm` | Corrected values and verdicts |

> **The apparent contradiction, and its resolution.** Accuracy is significant across the 12
> configurations (Holm p = 0.014) but individually resolvable in only **1 of 12** here. Both are
> correct — they ask different questions. Each configuration's gap is smaller than the noise from a
> 1,905-image test set, but **11 of 12 gaps point the same way**. One coin landing heads proves
> nothing; eleven of twelve is strong evidence. Address this in the paper before a reviewer raises
> it.

### `significance_ablation_level.csv` — n = 24 experiments
The same testing machinery applied to the ablation replay. **The strongest numbers in the study:**
mean difference −1,666.25, Holm p = 9.5 × 10⁻⁷, rank-biserial **−1.0** (all 24 improved).

---

## C. The ablation

### `ablation_decision_level.csv` — 96 rows = 24 experiments × 4 rules
Each logged round replayed under an alternative escalation rule **with the model held fixed**.

| Column | Meaning |
|---|---|
| `rule` | `dual_metric`, `uncertainty_only`, `risk_only`, `random_matched` |
| `n_escalated` | Labels the rule would have spent |
| `unsafe_auto_accepts` | Dangerous images it would have missed |
| `high_risk_caught` / `high_risk_total` | Numerator and denominator of the catch rate |
| `high_risk_catch_rate` | The headline column |

**`random_matched` is the control** — it escalates the *same number* of images as dual-metric, but
chosen at random. That is what makes "uncertainty barely beats random" a fair statement.

> **Aggregation note for the paper.** Averaging the per-experiment `high_risk_catch_rate` gives
> 29.32 / 12.58 / 17.24 / 10.21. Pooling all images first and then taking the ratio gives
> 30.59 / 11.47 / 19.64 / 10.14. The ordering and the near-additivity conclusion are identical.
> **State which convention is used.**

### `risk_threshold_sweep.csv`
The same replay across risk thresholds 0.05 → 1.01. The final value **disables the risk route
entirely**, recovering the baseline. Behind figures 15 and 16.

---

## D. Efficiency

### `al_efficiency_budget_matched.csv` — 12 rows
**The table behind the paper's most important negative result.**

`matched_budget` is the largest label count both policies actually reached, so the comparison is
like-for-like. Every metric appears as a `_dual` / `_unc` pair plus a `_delta_pp`.
**Mean `acc_delta_pp` = −0.35** — the method is *behind* at equal spend.

### `labels_to_reach_accuracy.csv`
For each accuracy target, how many labels each policy needed. `labels_saved_by_dual` is **negative**
throughout — dual needs roughly 300 *more*. Blunt, and honest.

---

## E. Discrimination

### `per_class_auc.csv` — 7 classes × 24 experiments
| Column | Meaning |
|---|---|
| `target`, `target_name` | Class code and full name |
| `n_positive`, `prevalence` | How many test images belong to this class |
| `auc`, `auc_ci_low`, `auc_ci_high` | One-vs-rest ROC-AUC with 2,000-resample bootstrap CI |
| `pr_auc` | Precision–recall AUC — fairer for rare classes |
| `kind` | `class_probability` (from the classifier) or the risk-head variant |

**Melanoma: AUC 0.9454 [0.9286, 0.9601], PR-AUC 0.7812.** `df` (9 images) and `vasc` (14) have very
wide intervals — do not over-interpret them.

### `auc_summary_by_policy.csv`
The same, split by policy, for checking whether the policy changed discrimination. It did not
meaningfully.

### `risk_head_decoupling.csv` — **tests whether the two-head redesign was worth it**
Compares the risk head against simply summing the malignant class probabilities, on nested
populations.

| `population` | Result |
|---|---|
| `all_images` | 0.9520 vs 0.9524 — **a tie** |
| `misclassified` | 0.408 vs 0.386 |
| `confidently_misclassified` | 0.363 vs 0.358 |
| `false_negatives_only` | flagged at 0.5: **5.57% vs 0.62%** — the risk head rescues ~9× more |

> ⚠️ **The sub-chance AUCs are a selection artefact**, not a finding. Conditioning on classifier
> error defines the population *by* the classifier being wrong, mechanically depressing any
> correlated score. **Compare the two columns to each other, never to 0.5.**

---

## F. Calibration

### `calibration_metrics.csv` — 24 rows
Classification-head columns: `accuracy`, `mean_confidence`, `overconfidence_gap`, `ece`,
`ece_adaptive`, `mce`, `brier_multiclass`, `nll`.
Risk-head columns (`risk_` prefix): `risk_ece`, `risk_ece_adaptive`, `risk_mce`, `risk_brier`,
`risk_mean_score`, `risk_base_rate`.
Temperature-scaling columns: `temperature`, `ece_heldout_before_T`, `ece_heldout_after_T`.

- `ece` uses **equal-width** bins; `ece_adaptive` uses **equal-mass** bins. Reporting both shows the
  result is not a binning artefact.
- `mce` is the **worst** bin, not the average.
- Temperature is fitted on **half** the test set and evaluated on the other half — so
  `ece_heldout_after_T` is a fair out-of-sample number, not self-graded.
- `risk_base_rate` (0.183) is the true malignant prevalence; compare it against `risk_mean_score`
  (0.1825) — they nearly match, which is a good sign for the risk head.

---

## G. Robustness

### `robustness_summary.csv` — 6 experiments × 5 conditions
Conditions: `clean`, `blur_1.5`, `brightness_0.7`, `contrast_0.7`, `gaussian_noise_0.05`.

Each metric appears three ways: the corrupted value, the `_clean` baseline, and a `_retained` ratio
(corrupted ÷ clean, where **1.0 = no degradation**).

**The two findings this table supports:**
1. `risk_auroc_retained` (90.3%) > `accuracy_retained` (85.2%) — the safety signal survives better
   than the diagnosis.
2. Filtering to `efficientnet_b4` + `gaussian_noise_0.05` gives accuracy **0.0079** — below random.
   **Always report per model.**

*Note: `jpeg_q30` is defined in the pipeline but has not been run — see
`06_STATUS_AND_OPEN_ITEMS/OPEN_ITEMS.md`.*

---

## H. Runtime

### `runtime_per_experiment.csv` — 24 rows
`total_seconds`, `mean_round_seconds`, `total_hours` as actually logged on the T4. Sums to
**94.1 GPU-hours**.

### `runtime_components_measured.csv` — 3 rows
Direct microbenchmarks: `ms_per_image_inference`, `ms_per_image_mc_dropout`,
`ms_per_image_train_step`.
> **Absolute milliseconds are CPU thread-count dependent** (the thread count is recorded in
> `analysis/rigor/runtime_benchmark.json`). **Ratios are stable — report ratios.**

### `runtime_round_composition.csv`
Per-round decomposition into `train_seconds` / `query_seconds` / `test_eval_seconds`, computed by
multiplying measured per-image costs by real set sizes.
`modelled_total_seconds` vs `logged_round_seconds` lets a reader check the model against reality.

> **Why these are microbenchmarks and not a regression:** the obvious approach — regressing logged
> round time on labelled and unlabelled counts — is **unidentifiable here**, because the pool is
> closed and `labeled_count + unlabeled_count = 8110` in every row. Perfect collinearity with the
> intercept. It produced *negative* query times, which is how the problem was found. See
> `02_METHODS_AND_MATH/METHODS.md` §10.

---

## I. Explainability

### `gradcam_cases_<experiment>.csv` — 3 files
The specific images shown in the figure-28 panels.

| Column | Meaning |
|---|---|
| `image_id` | HAM10000 identifier, e.g. `ISIC_0027776` |
| `true_label`, `predicted_label` | Ground truth and prediction |
| `confidence` | The classifier's probability for its own prediction |
| `prob_akiec` … `prob_vasc` | Full 7-class distribution |
| `risk_score` | The risk head's independent P(malignant) |
| `case_type` | `CAUGHT` / `RESCUED` / `MISSED` — why this case was selected |
| `age`, `sex`, `localization`, `dx_type`, `lesion_id` | Patient metadata |

`dx_type = histo` means histopathologically confirmed — the most reliable label grade. The
case-study image `ISIC_0027776` is one of these.

---

## Fastest way to check any single number

```python
import pandas as pd
df = pd.read_csv("significance_configuration_level.csv")
print(df[["metric", "mean_difference", "wilcoxon_p_holm", "significant_holm_0.05"]])
```

Or in Excel: open, freeze the header row, and filter on `experiment_id`.

**Every number quoted in the paper should trace back to one of these files.**
`01_PAPER_WRITING_KIT/KEY_NUMBERS.md` gives the pre-computed values with the source file named
beside each one.
