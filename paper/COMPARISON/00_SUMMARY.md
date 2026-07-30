# Comparison against published active-learning methods — summary

**Risk-Aware Active Learning for Skin Lesion Diagnosis** · HAM10000 · seed 42 throughout

---

## What was run

Twelve new acquisition runs: **3 backbones × 4 published baselines**, 15 rounds each, added to the
existing 24-experiment matrix (36 runs total).

| Baseline | Published |
|---|---|
| CoreSet | Sener & Savarese, ICLR 2018 |
| BADGE | Ash et al., ICLR 2020 |
| CLUE | Prabhu et al., ICCV 2021 |
| VAAL | Sinha et al., ICCV 2019 |

**The comparison is cost-matched.** The baselines are *acquisition strategies* that receive a label
budget; our method is an *escalation policy* that chooses its own. Compared naively, whichever method
requests more labels wins, and the result would measure budget rather than selection quality. Each
baseline was therefore given **exactly** the number of labels our policy spent in that same round on
that same backbone.

This was verified exact for all 12 runs (ResNet-50 4,678 / DenseNet-169 4,773 / EfficientNet-B4
3,976 labels). The analysis script re-checks it and refuses to produce any table if a single run
deviates.

All 36 runs share **one frozen 1,905-image test split** (confirmed by hashing every run's split
file — exactly one distinct value), which is what licenses the paired image-level statistics below.

---

## Result 1 — Safety: a clean sweep

Cumulative high-risk cases auto-accepted **without human review**, over 15 rounds. Lower is better.

| Backbone | **Ours** | CoreSet | BADGE | CLUE | VAAL | Uncertainty-only |
|---|---|---|---|---|---|---|
| ResNet-50 | **4,945** | 9,575 | 8,194 | 8,481 | 12,628 | 9,327 |
| DenseNet-169 | **4,495** | 8,543 | 6,947 | 7,397 | 11,308 | 8,275 |
| EfficientNet-B4 | **7,362** | 9,893 | 10,963 | 11,873 | 12,745 | 12,346 |

**15 of 15 comparisons favour the dual-metric policy, by 25.6% to 60.8%. No exceptions.**

→ `figures/fig1_safety_headline.png`, `tables/02_safety_scoreboard.csv`

---

## Result 2 — Accuracy: no penalty paid

Image-level McNemar over the 1,905 shared test images, Holm-corrected across all 15 comparisons.

| Comparison | Δ accuracy | Verdict |
|---|---|---|
| vs **VAAL** (3/3 backbones) | +2.6 to +5.4 pp | significantly **better** (p ≤ 0.0092) |
| vs **Uncertainty-only** (EfficientNet-B4) | +2.41 pp | significantly **better** (p = 0.0053) |
| vs **CoreSet, BADGE, CLUE** (all backbones) | −0.68 to +1.52 pp | **no detectable difference** |
| vs Uncertainty-only (ResNet-50, DenseNet-169) | +0.21 to +0.52 pp | no detectable difference |

**The eleven non-significant rows are the intended result, not a shortfall.** The claim is *safety
gained at no accuracy cost*; narrow confidence intervals straddling zero are the evidence for "no
cost". CoreSet, BADGE and CLUE are explicitly optimised to maximise learning per label, so matching
them at identical budget while substantially outperforming them on safety is the substantive finding.

→ `figures/fig2_safety_accuracy_tradeoff.png`, `figures/fig4_accuracy_significance.png`

---

## Result 3 — Held-out cancer detection: honest null

Of 349 malignant test images, the policy detects 264 / 272 / 280 depending on backbone. Restricted to
the 209 melanomas, it detects 143 / 152 / 157.

The direction favours the policy in **13 of 15** comparisons, but only **2 of 15** reach significance
after correction, both against VAAL. Against CoreSet and BADGE the difference is one to two cases —
indistinguishable from chance.

**This is reported as a limitation, not a result.** With 209 melanomas, and the two models typically
disagreeing on only 15–30 of them, the study is not powered to detect differences of this size.

---

## The distinction that governs how this is written up

The safety result (Result 1) is measured on the **unlabelled acquisition pool**. The cancer-detection
result (Result 3) is measured on the **held-out test set**. They answer different questions and are
kept separate throughout.

The defence for measuring safety on the pool is substantive rather than convenient: this is an
intervention on a **decision rule**, not on the model's weights, so the correct place to evaluate
"did fewer dangerous cases get waved through?" is the set of cases that were waved through. What has
*not* been shown is that the additional labels make the final model safer on unseen patients, and the
mechanism is identified — the two heads carry separate parameters but share a backbone, so they fail
on the same images.

Full treatment, with paragraphs drafted for each section of the paper:
`../06_STATUS_AND_OPEN_ITEMS/POOL_VS_TEST_FRAMING.md`

---

## A note on statistical power

A paired test across the three backbones has n = 3, where the smallest attainable two-sided p-value
is 2/2³ = **0.250** — significance is arithmetically unreachable regardless of effect size. All
p-values in this package therefore come from **image-level** McNemar tests (n = 1,905), which the
frozen test split makes valid. `tables/04_direction_across_backbones.csv` deliberately reports **win
counts and no p-values**, so 0.250 cannot be misread as a failed test.

---

## Folder contents

| File | Purpose |
|---|---|
| `README.md` | Full explanation of every term, method and design decision, written for a non-specialist reader |
| `figures/fig1_safety_headline.png` | The headline safety result |
| `figures/fig2_safety_accuracy_tradeoff.png` | Safety against accuracy; upper-left is better |
| `figures/fig3_learning_curves.png` | Accuracy against labels spent |
| `figures/fig4_accuracy_significance.png` | Accuracy differences with confidence intervals |
| `tables/01_main_comparison.csv` | Main results table, all methods, all backbones |
| `tables/02_safety_scoreboard.csv` | Safety reductions, absolute and relative |
| `tables/03_significance_image_level.csv` | McNemar tests, Holm-adjusted |
| `tables/04_direction_across_backbones.csv` | Consistency across backbones (win counts only) |
| `tables/05_learning_curves_per_round.csv` | Per-round data behind figure 3 |
| `tables/06_run_provenance.csv` | GPU, CUDA, library versions, git commit and seed, per run |
| `tables/main_comparison.tex` | Main table as LaTeX, generated from the CSV |
| `tables/safety_reduction.tex` | Safety table as LaTeX |

The two LaTeX tables are **generated** from the same CSVs the figures were built from, so the
manuscript tables cannot drift from the underlying data.

**Reproducible in two commands** from the repository root:

```
python -m evaluation.rigor.baseline_comparison
python -m tools.build_comparison_package
```

Code and results: `github.com/DyneStein/RiskAware-ActiveLearning` (commit `058e46d`).
