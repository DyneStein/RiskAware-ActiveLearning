# Key numbers — every quotable figure, with its source

**Every number on this page was recomputed directly from the CSV named beside it.** If a number
appears anywhere in the paper, it should appear here first. If the two disagree, this page wins.

Unless stated otherwise, a value is the **mean across all 24 experiments** (or across the 12 matched
configuration pairs, where the comparison is paired).

---

## 1. Scale of the study

| Quantity | Value |
|---|---|
| Total images (HAM10000) | 10,015 |
| Test set (fixed, shared by all 24 runs) | **1,905** |
| Active-learning pool | **8,110** |
| Initial labelled seed set | **490** (70 per class) — *a subset of the 8,110 pool, not additional* |
| Classes | 7 |
| Active-learning rounds | 15 |
| Experiments | **24** = 3 backbones × 4 uncertainty measures × 2 policies |
| Matched comparison pairs | **12** |
| Malignant prevalence in test set | **18.3%** (349 of 1,905) |
| Test-split checksum (md5 prefix) | `feff50b2fec0` — identical across all 24 runs |

---

## 2. Primary results — configuration level (n = 12 paired configurations)

Source: `04_TABLES/significance_configuration_level.csv`
Test: Wilcoxon signed-rank, paired; Holm–Bonferroni across the six metrics; bootstrap 95% CIs.

| Metric | Mean difference (dual − uncertainty) | 95% CI | Raw p | **Holm p** | Configs favouring dual | Significant |
|---|---|---|---|---|---|---|
| **Unsafe auto-accepts** | **−4,030.5** | [−4,319.5, −3,715.2] | 0.00049 | **0.0029** | **12 / 12** | ✅ |
| **Total oracle queries** | **+382.4** | [+315.2, +447.4] | 0.00049 | **0.0029** | **12 / 12** | ✅ (cost) |
| Final accuracy | **+0.60 pp** | [+0.28, +1.00] | 0.0034 | **0.0137** | 11 / 12 | ✅ |
| Final F1-macro | +0.86 pp | [−0.22, +1.97] | 0.233 | 0.305 | 8 / 12 | ❌ |
| Final missed-cancer rate | −1.19 pp | [−2.46, +0.21] | 0.152 | 0.305 | 3 / 12 | ❌ |
| Final melanoma recall | +2.43 pp | [+0.44, +4.39] | 0.057 | 0.170 | 8 / 12 | ❌ |

**Relative reduction in unsafe auto-accepts: ≈ 43%** (dual mean 5,503.9 vs uncertainty-only mean
9,534.4). Relative increase in queries: **+9.1%**.

> ⚠️ **Quote the Holm column.** Raw p-values here are uncorrected for the six-metric family.
> Melanoma recall in particular looks borderline at 0.057 raw but is **0.170 corrected — not
> significant.**

---

## 3. Ablation — the strongest evidence (n = 24 experiments)

Source: `04_TABLES/ablation_decision_level.csv`
Method: replay each logged round under an alternative escalation rule with the model held fixed.

| Rule | High-risk catch rate | Oracle labels spent | Unsafe auto-accepts |
|---|---|---|---|
| Random, cost-matched | **10.21%** | 6,201 | 7,832 |
| Uncertainty only (standard baseline) | **12.58%** | 4,339 | 7,716 |
| Risk only | **17.24%** | **1,917** | 7,004 |
| **Dual-metric (proposed)** | **29.32%** | 6,201 | **6,050** |

**The three sentences to write from this table:**

1. Uncertainty sampling catches **12.58%** of high-risk images versus **10.21%** for random —
   it is barely a safety mechanism at all, despite spending 4,339 labels.
2. Risk alone reaches **17.24%** at **under half** the annotation cost of the baseline.
3. 12.58 + 17.24 = 29.82, and the combination achieves **29.32** — **near-additive**, so the two
   routes flag largely *different* images. This is the ablation's payload.

> **Methodological footnote for the paper:** these are means of the per-experiment catch rates.
> Pooling all images before taking the ratio gives 11.47 / 19.64 / 30.59 / 10.14 — the ordering and
> the near-additivity conclusion are unchanged. State which convention is used.

Significance across the 24 experiments (`significance_ablation_level.csv`): unsafe auto-accepts
mean difference **−1,666.25**, CI [−2,205.6, −1,119.1], Wilcoxon p = 1.19 × 10⁻⁷,
Holm p = 9.54 × 10⁻⁷, rank-biserial effect size **−1.0** (every one of the 24 improved).

---

## 4. Label efficiency — the negative result

Source: `04_TABLES/al_efficiency_budget_matched.csv`, `labels_to_reach_accuracy.csv`

| At matched annotation budget | Value |
|---|---|
| Accuracy difference (dual − uncertainty) | **−0.35 pp** (dual is behind) |
| Extra labels needed to reach a given accuracy | **≈ +300** |

**Framing:** the +0.60 pp accuracy gain at round 15 is *bought* with 382 extra labels, not *earned*
per label. This is a safety intervention with a price, not an efficiency improvement.

---

## 5. Per-class discrimination

Source: `04_TABLES/per_class_auc.csv` (mean over 24 experiments; CIs are bootstrap, 2,000 resamples)

| Class | AUC | 95% CI | PR-AUC |
|---|---|---|---|
| Dermatofibroma (df) | 0.9961 | [0.9869, 1.0000] | 0.9153 |
| Vascular lesions (vasc) | 0.9946 | [0.9829, 1.0000] | 0.9302 |
| Basal cell carcinoma (bcc) | 0.9935 | [0.9872, 0.9979] | 0.9318 |
| Melanocytic nevi (nv) | 0.9697 | [0.9620, 0.9766] | 0.9854 |
| Benign keratosis (bkl) | 0.9666 | [0.9542, 0.9773] | 0.8371 |
| Actinic keratoses (akiec) | 0.9629 | [0.9364, 0.9839] | 0.7068 |
| **Melanoma (mel)** | **0.9454** | **[0.9286, 0.9601]** | **0.7812** |

**Melanoma is the hardest class** — expected, and worth stating plainly. The AUC 0.945 vs PR-AUC
0.781 gap is the normal signature of a rare, hard class; report both.

**Risk score's own discrimination:** AUROC **0.962** averaged across all rounds and experiments
(`risk_auroc_by_experiment.csv`) — evidence the risk head learned a real signal independent of any
policy that consumes it.

---

## 6. Calibration

Source: `04_TABLES/calibration_metrics.csv` (mean of all 24 rows)

| Metric | Classification head | Risk head |
|---|---|---|
| Accuracy | 0.8858 | — |
| Mean confidence | 0.9577 | 0.1825 (mean score) |
| **ECE (equal-width)** | **0.0728** | **0.0563** |
| ECE (equal-mass / adaptive) | 0.0719 | — |
| MCE (worst bin) | 0.4241 | 0.4611 |
| Brier score | 0.1880 (7-class) | 0.0651 (binary) |
| NLL | 0.5099 | — |

**Temperature scaling** (fitted on half the test set, evaluated on the other half):
**T ≈ 2.16**, held-out ECE **0.0732 → 0.0233** — a **68% reduction** from a single scalar,
with accuracy unchanged by construction.

**The two sentences to write:** the models are overconfident by about **7 percentage points**
(claiming 95.8%, delivering 88.6%); the **risk head is the better-calibrated of the two**
(0.056 vs 0.073), which matters because the escalation threshold is applied to the risk score.

> Note: `05_RESULTS/RESPONSE_TO_REQUESTED_CHANGES.md` breaks calibration down **per policy**
> (dual: ECE 0.0719, T 2.166, held-out 0.0727→0.0222; uncertainty-only: 0.0736, 2.144,
> 0.0737→0.0245). Those are the same data split two ways, not a disagreement.

---

## 7. Robustness

Source: `04_TABLES/robustness_summary.csv` (6 experiments × 4 corruptions, no retraining)

**Retention = corrupted ÷ clean.** Higher is better.

| Corruption | Accuracy retained | Risk AUROC retained | Melanoma recall retained |
|---|---|---|---|
| Brightness × 0.7 | 99.6% | 99.4% | 101.3% |
| Contrast × 0.7 | 99.1% | 99.4% | 86.5% |
| Gaussian blur σ 1.5 | 89.5% | 94.7% | **44.1%** |
| Gaussian noise σ 0.05 | **52.7%** | 67.7% | **11.0%** |

**Headline (averaged over all corruptions):** the **risk head degrades more slowly than the
classifier — 90.3% vs 85.2% retained.** The system loses its grip on *which* disease before it
loses the sense that something is *dangerous*, so it escalates rather than confidently
mis-diagnosing. This is a genuine win for the two-head design.

**Per-model accuracy — report this breakdown, the mean conceals it:**

| Model | Clean | Blur | Brightness | Contrast | **Gaussian noise** |
|---|---|---|---|---|---|
| ResNet-50 | 0.8961 | 0.7816 | 0.8913 | 0.8866 | 0.7100 |
| DenseNet-169 | 0.8887 | 0.8039 | 0.8853 | 0.8824 | 0.6924 |
| **EfficientNet-B4** | 0.8625 | 0.7843 | 0.8593 | 0.8541 | **0.0079** |

**EfficientNet-B4 collapses to 0.0079 under mild noise — far below the ~0.14 of random guessing.**
The identical corruption applied to the other two backbones gives an orderly ~20-point drop, so
this is architecture-specific, not a bug in the test. Likely worsened by training B4 at 224 px
instead of its native 380 px.

---

## 8. Runtime

Source: `04_TABLES/runtime_per_experiment.csv`, `runtime_components_measured.csv`

| Quantity | Value |
|---|---|
| Total compute across all 24 experiments | **94.1 GPU-hours** |
| Mean per experiment — entropy | 2.83 h |
| Mean per experiment — least-confidence | 2.88 h |
| Mean per experiment — margin | 2.96 h |
| **Mean per experiment — MC-dropout** | **7.00 h (≈ 2.4× the others)** |

**Measured component costs** (CPU, 8 threads — absolute values are thread-dependent, *ratios* are
not):

| Model | Inference / image | MC-dropout (30 passes) / image | Training step / image |
|---|---|---|---|
| EfficientNet-B4 | 115.1 ms | 3,368.9 ms | 584.3 ms |
| ResNet-50 | 132.6 ms | 4,004.5 ms | 423.8 ms |
| DenseNet-169 | 145.6 ms | 4,377.0 ms | 522.7 ms |

**The number that matters for the paper: the escalation rule itself costs ≈ 8 ms per round.**
The dual-metric policy's cost is annotation, not computation.

---

## 9. The two-head design, tested directly

Source: `04_TABLES/risk_head_decoupling.csv`

| Population | Risk head AUC | Summed malignant probabilities AUC |
|---|---|---|
| All 1,905 test images | **0.9520** | **0.9524** |
| Misclassified images only | 0.4078 | 0.3862 |
| Confidently misclassified | 0.3630 | 0.3576 |

**On the full test set the redesign is a tie.** But on the classifier's **false negatives**, the
risk head still flags **5.57%** as high-risk versus **0.62%** for the summed probabilities —
roughly **9× more rescues**. It works where it was designed to work, on a small fraction of cases.

> ⚠️ **Do not over-read the sub-chance AUCs (0.41, 0.36).** Conditioning on classifier error is a
> selection effect — the population is defined by the classifier being wrong, which mechanically
> depresses any score correlated with it. **Compare the two columns to each other, not to 0.5.**

---

## 10. The single sentence that summarises the study

> Dual-metric escalation reduced unsafe auto-accepts by **4,030 per run (≈43%)** in **12 of 12**
> configurations (Holm p = 0.003), at a cost of **+382 oracle labels (+9.1%)** (Holm p = 0.003),
> with **no significant change** in F1-macro, missed-cancer rate, or melanoma recall.

---

## Numbers NOT to quote

- **Any label-efficiency claim.** At matched budget the method is 0.35 pp behind.
- **Melanoma recall +2.43 pp as significant.** Holm p = 0.170.
- **Missed-cancer rate −1.19 pp as significant.** Holm p = 0.305, and only 3 of 12 improved.
- **The 52.7% mean noise robustness on its own.** It hides EfficientNet-B4's collapse.
- **The sub-chance decoupling AUCs as evidence of anything absolute.** Selection artefact.
