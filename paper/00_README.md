# Risk-Aware Active Learning for Skin Lesion Diagnosis
## Complete material for the research paper

**Method:** Dual-metric escalation policy with an independent clinical-risk head
**Dataset:** HAM10000 — 10,015 dermoscopic images, 7 lesion classes
**Experiments:** 24 completed runs = 3 backbones × 4 uncertainty measures × 2 escalation policies
**Status:** all 24 runs completed all 15 active-learning rounds. Analysis complete.
**Code:** `github.com/DyneStein/RiskAware-ActiveLearning`

---

## The short version

- **Everything needed to write the paper is in this folder**, in six numbered parts.
- **Every folder contains a `HOW_TO_READ.md`** explaining what is inside, what each figure and
  column means, and every abbreviation and symbol used.
- **To write the paper:** `01_PAPER_WRITING_KIT/PAPER_OUTLINE.md` is a section-by-section plan
  naming the figure, table and number for each part.
- **To check any number:** `01_PAPER_WRITING_KIT/KEY_NUMBERS.md`. Every value there was recomputed
  from the source CSV named beside it.
- **To see current status:** `06_STATUS_AND_OPEN_ITEMS/STATUS_CHECKLIST.md`.

---

## Contents

| Folder | Contents |
|---|---|
| **01_PAPER_WRITING_KIT** | Section-by-section outline, draft abstract and contribution statements, every quotable number with its source, the claim→evidence map, and the limitations section. |
| **02_METHODS_AND_MATH** | Formal mathematical specification, complete notation and abbreviation reference, and the full experimental setup with the hyperparameter table. |
| **03_FIGURES** | All 34 figures — 11 selected for the main paper, 23 supplementary — with a figure-by-figure reading guide. |
| **04_TABLES** | All 21 result tables as CSV, with every column decoded. |
| **05_RESULTS** | The two full prose analyses: overall findings, and the response to the requested additional analyses. |
| **06_STATUS_AND_OPEN_ITEMS** | Request-by-request status, what remains outstanding, and how to regenerate every number from scratch. **Includes `POOL_VS_TEST_FRAMING.md` — read this before writing the abstract.** |
| **COMPARISON** | ⭐ **New.** The self-contained head-to-head comparison against CoreSet, BADGE, CLUE and VAAL at matched label cost: 4 figures, 6 tables, 2 generated LaTeX tables, and a plain-English `README.md` explaining every term. Built by `python -m tools.build_comparison_package`. |

---

## The study in brief

A model classifies a dermoscopic image into one of 7 lesion categories. Under **active learning**,
it also decides per image whether to accept its own prediction or escalate the case to an expert
annotator, since expert labelling is the binding resource constraint.

Standard acquisition functions escalate the cases the model is **most uncertain** about. This
optimises for information gain but is indifferent to clinical consequence: a model that is
confidently wrong about a melanoma is never queried, and the case is auto-accepted without review.

This work adds a **second, independent signal** — a dedicated risk head, trained alongside the
classification head on a shared backbone, that estimates *how dangerous this case is if the model
is wrong* rather than *how confused the model is*. A case is escalated if **either** signal exceeds
its per-round calibrated threshold.

---

## Principal results

**1. Primary endpoint — unsafe auto-accepts.**
Malignant pool images auto-accepted without review fell by **4,030 per run (≈43%)**, in **12 of 12**
matched configurations, Holm-corrected **p = 0.003**.

**2. The cost.**
**+382 oracle labels (+9.1%)**, in **12 of 12** configurations, Holm-corrected **p = 0.003**.

**3. Ablation — the strongest evidence.**
Escalating on uncertainty alone captures **12.6%** of high-risk pool cases, against **10.2%** for
cost-matched random selection. Risk alone reaches **17.2%** at under half the annotation cost. The
two combined reach **29.3%** — near-additive, indicating the two routes flag largely disjoint cases.

**4. Two structural guarantees.**
The dual-metric escalation set is a superset of the uncertainty-only set at fixed scores, so unsafe
auto-acceptance is **monotonically non-increasing** — which is why the improvement held in 24 of 24
experiments with no exceptions. The additional annotation cost equals exactly the risk-route-only
flagged set. The method is therefore a **controlled trade with a tunable dial**, not an empirical
heuristic.

---

## The framing to use consistently

> Dual-metric escalation is a **safety intervention on a decision rule, with a quantified price**:
> on the decisions it governs, a large and statistically significant reduction in unsafe
> auto-accepts, in exchange for approximately 9% more oracle labels and no meaningful change in
> classification quality. We did not demonstrate a significant improvement in missed-cancer rate on
> held-out patients, and we explain why.

🚨 **The single most important editorial rule in this package.** The headline safety number is
measured on the **unlabelled acquisition pool**; the held-out test-set safety number did **not**
reach significance. Both are true, they answer different questions, and they must never be blurred
into one claim. Read `06_STATUS_AND_OPEN_ITEMS/POOL_VS_TEST_FRAMING.md` before writing a single
sentence of the abstract — it contains the argument and ready-to-adapt paragraphs for every section.

**The method should not be presented as a label-efficiency improvement.** At a matched annotation
budget it is 0.35 percentage points *behind* on accuracy and requires roughly 300 additional labels
to reach any given accuracy target. This is reported in the Results, not only in the Limitations —
see `01_PAPER_WRITING_KIT/LIMITATIONS_AND_FUTURE_WORK.md` §L6.

---

## Results that do not favour the method

All four are reported, and are accompanied by a single mechanism that explains two of them.

1. **Not label-efficient** — 0.35 pp behind at matched budget.
2. **Missed-cancer rate did not significantly improve** — Holm p = 0.305; only 3 of 12
   configurations improved.
3. **The two-head redesign is a tie on overall AUC** — 0.9520 vs 0.9524 against simply summing the
   malignant class probabilities. Its benefit is confined to the classifier's error region, where
   it flags 5.6% of false negatives against 0.6%.
4. **EfficientNet-B4 collapses under mild Gaussian noise** — accuracy 0.008, below random guessing,
   while the other two backbones degrade in an orderly ~20-point drop.

**The unifying mechanism:** the two heads have independent parameters but a **shared backbone**.
They are decoupled in parameters, not in features, so a failure of the shared representation
propagates to both. This explains why the pool-level escalation decision improved substantially
while the test-set outcome did not, and it identifies backbone separation as the direct next step.

---

## Two methodological notes worth reading

**A dataset contamination trap was identified and guarded against.** ISIC 2019 was assembled from
BCN20000 + **HAM10000** + MSK, and the HAM10000 images retain their original `ISIC_xxxxxxx`
identifiers. Evaluating on ISIC 2019 as distributed would be evaluating on training data. The
external-validation script excludes overlapping images by filename and refuses to run unless the
measured overlap is zero. **ISIC 2020 is the recommended external test set** — independent
patients, and binary labels that map directly onto the risk head. Full detail in
`06_STATUS_AND_OPEN_ITEMS/STATUS_CHECKLIST.md`.

**An analysis method was discarded as mathematically unsound rather than reported.** Separating
training time from query time by regressing logged round times on set sizes is unidentifiable here:
the pool is closed, so labelled + unlabelled = 8,110 in every round, making the predictors
perfectly collinear with the intercept. It produced *negative* query times, which is how the
problem was detected. Direct microbenchmarking replaced it. Documented in
`02_METHODS_AND_MATH/METHODS.md` §10.

---

## Reproducibility

All analysis runs off saved model checkpoints — no retraining and no GPU required. One command
regenerates every figure and table:

```bash
python -m evaluation.rigor.run_all
```

The reload path is verified: recomputing round-15 accuracy from a checkpoint gives `0.8987` against
the logged `0.8986876640419947`. Full instructions in
`06_STATUS_AND_OPEN_ITEMS/HOW_TO_REGENERATE.md`.

---

## Outstanding work

| Item | Cost | Blocked on |
|---|---|---|
| External validation on ISIC 2020 | ~3 GB download + one inference pass | The download |
| Score-CAM / three-panel XAI figure | ~2 h | Confirming which dataset the existing XAI images came from |
| `LICENSE` names the authors generically | 2 min | The authors' legal names |
| Checkpoint upload (1.8 GB) to a GitHub Release | ~30 min | Repository owner |

**Completed since the last revision:** the 12 cost-matched baseline runs (CoreSet, BADGE, CLUE,
VAAL × 3 backbones), the JPEG corruption condition, the EfficientNet-B4 noise diagnostic, and the
rare-class CI diagnostic. **Multi-seed replication is deliberately not being run** — seed 42
throughout, on supervisor direction; the frozen test split substitutes image-level testing
(n = 1,905) for replicate-level testing. **No GPU work remains.**

Details in `06_STATUS_AND_OPEN_ITEMS/OPEN_ITEMS.md`.
