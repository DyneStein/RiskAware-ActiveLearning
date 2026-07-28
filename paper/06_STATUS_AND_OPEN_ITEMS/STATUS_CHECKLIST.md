# Status checklist — every request, with evidence

**12 of 13 complete.** Only external validation genuinely needs new work.

---

## Message 1 requests

| # | Request | Status | Result | Where it is |
|---|---|---|---|---|
| 1 | Expected Calibration Error (ECE) | ✅ | ECE 0.073 — overconfident by ~7 pp. Risk head better at 0.056 | Fig 19; `calibration_metrics.csv` |
| 2 | Brier score | ✅ | 0.188 (7-class), 0.065 (risk head) | Fig 20; same table |
| 3 | Reliability diagrams | ✅ | Both heads; points sag below the diagonal | Figs 17, 18 |
| 4 | Accuracy vs number of labelled samples | ✅ | **−0.35 pp at matched budget** — the method is behind | Figs 10–13; `al_efficiency_budget_matched.csv` |
| 5 | AUC per lesion class, especially melanoma | ✅ | Melanoma 0.945 [0.929, 0.960], PR-AUC 0.781 | Figs 21–23; `per_class_auc.csv` |
| 6 | p-values | ✅ | Safety and cost both Holm p = 0.003; three metrics n.s. | Figs 24, 25; `significance_*.csv` |
| 7 | Confidence intervals | ✅ | 95% percentile bootstrap on every reported difference | Same |
| 8 | Training time | ✅ | 94.1 GPU-hours total; MC-dropout ≈ 2.4× | Figs 26, 27; `runtime_*.csv` |
| 9 | Inference time | ✅ | 115–146 ms/image (CPU, 8 threads) | `runtime_components_measured.csv` |
| 10 | Query time | ✅ | Escalation rule ≈ 8 ms/round | Same |

---

## Message 2 requests

| # | Request | Status | Result | Where it is |
|---|---|---|---|---|
| 1 | External validation on ISIC 2019/2020 | 🔧 **Code ready, needs a ~3 GB download** | See the contamination warning below | `external_validation_isic.py` |
| 2 | Statistical significance analysis | ✅ | Three levels: 12 configurations, 1,905 images, 24 experiments | `significance_*.csv` |
| 3 | Ablation studies | ✅ | 4-rule comparison + threshold sweep. **The strongest evidence in the study** | Figs 14–16; `ablation_decision_level.csv` |
| 4 | Calibration metrics | ✅ | As above, plus temperature scaling (T ≈ 2.16, ECE 0.073 → 0.023) | Figs 17–20 |
| 5 | Formal mathematical definitions | ✅ | Full specification **plus two proved propositions** | `02_METHODS_AND_MATH/METHODS.md` |
| 6 | Robustness experiments | ✅ **4 of 5 corruptions** (JPEG outstanding) | Risk head degrades more slowly; EfficientNet-B4 collapses under noise | Figs 29, 30, 33; `robustness_summary.csv` |
| 7 | Explainability analysis | ✅ | Grad-CAM for all 3 backbones, **both heads** | Fig 28 ×3; `gradcam_cases_*.csv` |

---

## ⚠️ The ISIC contamination warning — raise this before anyone runs it

**ISIC 2019 physically contains HAM10000.** It was assembled from BCN20000 + **HAM10000** + MSK,
and the HAM10000 images retain their original `ISIC_xxxxxxx` filenames.

**Testing on ISIC 2019 as downloaded means testing on our own training data.** The scores would look
excellent and be worthless, and a reviewer who knows these datasets would spot it instantly.

**The safeguard in place:** `external_validation_isic.py` removes overlapping images by filename,
reports how many it removed, and **refuses to run unless the measured overlap is zero**.

**Recommendation: use ISIC 2020 as the primary external test.** Different year, different patients,
no overlap — and its benign/malignant labels map directly onto what the risk head predicts.

**Expect the numbers to drop.** Different cameras, sites and populations always cost something.
That is the normal, reportable outcome.

---

## What is genuinely still outstanding

| Item | Cost | Blocked on |
|---|---|---|
| External validation on ISIC 2020 | ~3 GB download + one inference pass | The download; best run in Colab |
| JPEG corruption (the 5th robustness condition) | ~40 min CPU | Nothing — can run any time |
| Multi-seed replication | ~60 GPU-hours (3–4 configs × 5 seeds) | GPU access |

Details and commands: `06_STATUS_AND_OPEN_ITEMS/OPEN_ITEMS.md`.

---

## Two points worth making explicitly

**1. The efficiency plot request was correct and changed a claim.**
Plotting accuracy against *labels* rather than *rounds* revealed that the method is 0.35 pp behind
at matched budget. Without that plot, the paper would have claimed label efficiency — which is the
first thing a reviewer would have checked, and it would not have survived.

**2. Several results do not favour the method, and all are being reported.**
Not label-efficient; missed-cancer rate not significant; the two-head redesign a tie on overall AUC;
EfficientNet-B4 collapsing under noise. They are presented alongside the strong safety result and a
single shared-backbone mechanism that explains why findings 2 and 3 came out that way. Volunteering
these is what makes the strong result credible.
