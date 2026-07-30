# The Comparison Package — what's in here and what it means

This folder holds **only** the head-to-head comparison evidence: the material
that answers the one question every reviewer asks first.

> *"Fine, your method works. But is it better than what already exists — and
> did you measure that fairly?"*

Everything else about the project (calibration, Grad-CAM heatmaps, robustness
to noise, timing, ablations) lives in `analysis/rigor/` and is not repeated
here.

This README is written to be readable without a machine-learning background.
Every technical term is explained the first time it appears.

---

## Part 1 — The vocabulary, in plain English

Read this once and the tables will make sense.

**Active learning (AL).** Labelling medical images is expensive: a
dermatologist has to look at each one. Active learning is the idea that the
model should *choose* which images are worth a doctor's time, instead of
labelling everything or picking at random. It works in **rounds**: train,
pick some images, get them labelled, retrain, repeat. We ran **15 rounds**.

**The pool.** Our 10,015 skin-lesion images are split three ways:
- **490 seed labelled** — the small starting set the model learns from first.
- **7,620 unlabelled pool** — the big pile the model picks from each round.
- **1,905 test set** — held back the entire time, never learned from, used
  only for measuring. Think of it as the final exam.

**Oracle.** The stand-in for the human doctor. When the model asks for a
label, we look up the true answer from the dataset. This is standard
practice — it means no real clinicians were needed.

**Query / label budget.** How many images the model is allowed to send to
the doctor in a round. Cost, essentially.

**Escalation.** Our system does something extra. For each image it decides:
*auto-accept* my own answer, or *escalate* this to a human. Escalating is
safe but costs a doctor's time; auto-accepting is free but risky.

**Uncertainty.** "How confused is the model?" Measured by **entropy** — high
entropy means the model is spreading its bet across several diagnoses.

**Clinical risk.** "How dangerous is this case if I'm wrong?" A separate
output of the model estimating the chance the lesion is malignant (cancerous).

**The two-head architecture ("dual-metric").** Our model has one shared
image-understanding backbone feeding **two separate outputs**:
- a **classification head** predicting which of 7 diagnoses it is;
- a **risk head** predicting whether it's malignant.

These are separate sets of parameters, so the danger signal is not just a
rearrangement of the diagnosis probabilities. It's an independent opinion.

**Why two signals matter.** The dangerous case is *confident but dangerous*:
the model is sure, so a normal AL system auto-accepts it — but it's a
melanoma. Uncertainty alone never flags it. Risk catches it. This is the
entire point of the paper.

**Unsafe auto-accept.** The key safety number. A case that was genuinely
high-risk (melanoma, basal cell carcinoma, or actinic keratosis) that the
system auto-accepted *without* sending it to a human. In a real clinic this
is a cancer that nobody double-checked. **Lower is better.**

**FN-mal (false-negative rate, malignant).** Of the truly malignant lesions
in the final exam, what fraction did the model call benign (harmless)? A
missed cancer. **Lower is better.**

**Mel-rec (melanoma recall).** Of the real melanomas, what fraction did the
model correctly find? **Higher is better.** Melanoma is the deadliest skin
cancer, so this gets its own column.

**Accuracy.** Fraction of the 1,905 test images given the right diagnosis
out of 7. **Higher is better.**

**F1-macro.** Accuracy is misleading here, because 67% of our images are
one common harmless class (`nv`, ordinary moles). A lazy model that guessed
"mole" every time would score 67% accuracy while being clinically useless.
F1-macro instead scores each of the 7 classes separately and averages them,
so the rare classes count as much as the common one. **Higher is better**,
and it's the more honest headline number.

**Backbone.** The image-recognition network underneath. We used three —
ResNet-50, DenseNet-169, EfficientNet-B4 — so that a result isn't an
accident of one architecture.

**Seed.** A number fixing all the randomness so a run can be repeated
exactly. Every run here used **seed 42**.

---

## Part 2 — Who we compared against, and why those four

We compare against four **published, well-known** methods. Naming
recognisable methods is important: reviewers trust a comparison against
papers they've read.

| Method | Published | The one-line idea |
|---|---|---|
| **CoreSet** | Sener & Savarese, ICLR 2018 | Pick images that *cover* the data best — spread out, no clumps. Ignores confusion entirely. |
| **BADGE** | Ash et al., ICLR 2020 | Combine "confused" and "different from each other" in one step, via gradient directions. **The** standard reference point in this field. |
| **CLUE** | Prabhu et al., ICCV 2021 | Cluster the images, weighting confusing ones more, then take one from each cluster. |
| **VAAL** | Sinha et al., ICCV 2019 | Train a second network to spot which images "don't look like" the already-labelled ones. |

We also keep **Uncertainty-only** in the tables. That isn't a literature
method — it's *our own system with the risk head switched off*. It answers
"does the second head actually do anything?", which is a different and
equally necessary question.

---

## Part 3 — The fairness problem, and how we solved it

**This is the most important methodological point in the package.** If you
explain one thing to your supervisor, explain this.

The four baselines and our method are not the same *kind* of thing:

- The baselines are **acquisition strategies**. They answer: *"you may have
  k labels — which k images do you want?"* The budget k is given to them.
- Ours is an **escalation policy**. It answers: *"which images are unsafe for
  me to auto-accept?"* It therefore **decides its own budget** each round.

Compare them naively and the comparison is meaningless. Whichever method
happens to ask for more labels will look better, and you'd be measuring
*budget*, not *intelligence*.

**The fix: cost-matching.** In every round, each baseline was handed exactly
the number of labels our method spent in that same round on that same
backbone — read directly from our run's own log file.

So for ResNet-50, every method spent labels in this pattern:

```
round:  1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
      810  584  401  528  417  365  272  209  169  155  152  155  156  153  152   = 4,678 total
```

Not just the same total — the same amount **in each round**, which matters
because a label is worth more early on than late.

**This is verified, not assumed.** `evaluation/rigor/baseline_comparison.py`
re-checks the match before it prints anything, and **refuses to produce the
tables at all** if any run is off by a single label. It currently passes
12/12.

| Backbone | Labels every method spent |
|---|---|
| ResNet-50 | 4,678 |
| DenseNet-169 | 4,773 |
| EfficientNet-B4 | 3,976 |

**One honest exception.** *Uncertainty-only* is the run our budgets were
derived from comparison against, and it chose its own smaller budget
(4,284 / 4,484 / 3,356). It is therefore **not** cost-matched, and the
learning-curve figure says so on its face. The four literature baselines
are matched exactly.

---

## Part 4 — Statistics: why some numbers have p-values and some don't

A **p-value** answers "could this gap be luck?" Below 0.05 is the usual bar
for "convincingly real."

Here's the trap we had to design around, and it's worth understanding
because it would have quietly sunk the results section.

### The n=3 problem

The obvious test is: compare our method to a baseline on each of the three
backbones, and test those three paired differences. That's a **Wilcoxon
signed-rank test** with n=3.

With 3 pairs, the smallest p-value that test can *ever* produce is
2/2³ = **0.250**. Not "we didn't find significance" — significance is
**arithmetically impossible**, no matter how enormous the effect is.

| Pairs (n) | Best possible p-value |
|---|---|
| 3 | 0.250 |
| 5 | 0.0625 |
| 6 | 0.031 |
| 8 | 0.0078 |

You need at least 6 pairs before p<0.05 is even reachable. Reporting
"p=0.25" from an n=3 test invites a reader to conclude the effect failed a
test, when in truth no test was possible.

### The fix: test at the image level instead

All 36 runs share **one identical test set of 1,905 images**. We verified
this directly — hashing every run's test-split file gives exactly one
distinct value across all 36 runs. (This holds because the split is
controlled by `SPLIT_SEED`, deliberately frozen and kept separate from the
training seed.)

That lets us compare two models **image by image** with **McNemar's test**,
where n = 1,905 instead of 3.

**McNemar's test in plain English.** Take two models on the same exam. Ignore
every question they both got right and every question they both got wrong —
those tell you nothing about which is better. Look only at the questions
where they **disagreed**: A right and B wrong, or B right and A wrong. If
the two models were equally good, those disagreements should split roughly
50/50, like coin flips. A lopsided split is real evidence.

**Holm correction.** Running many tests means some will look significant by
chance alone. Roll a die enough times and you'll get a six. The Holm
correction tightens the threshold to account for the number of tests run.
Every p-value in these tables is Holm-adjusted.

### So we report three levels, and label them honestly

| Level | n | What it gives you | Where |
|---|---|---|---|
| **Descriptive** | — | Effect sizes. How big is the gap? | `01_main_comparison.csv` |
| **Image-level** | 1,905 | Real p-values. Is the gap real? | `03_significance_image_level.csv` |
| **Run-level** | 3 | Direction only. Does it hold on all 3 backbones? | `04_direction_across_backbones.csv` |

The run-level file deliberately contains **no p-values at all** — only win
counts and mean gaps — precisely so nobody can misread a 0.250.

---

## Part 5 — What's in the files

### Figures

| File | What it shows |
|---|---|
| `fig1_safety_headline.png` | Unsafe auto-accepts, all 6 methods × 3 backbones. **The headline result.** Our bar is dramatically the shortest in every group. |
| `fig2_safety_accuracy_tradeoff.png` | Accuracy vs safety. Up-and-left is better. Shows we didn't simply trade accuracy away to buy safety. |
| `fig3_learning_curves.png` | Accuracy against labels spent. Curves that overlap = equal learning per label. |
| `fig4_accuracy_significance.png` | Accuracy difference with confidence intervals. A bar crossing zero means "no detectable accuracy difference" — which here is the *good* outcome. |

### Tables

| File | What it is |
|---|---|
| `01_main_comparison.csv` | The main results table. Every method, every backbone, final round. No p-values by design. |
| `02_safety_scoreboard.csv` | Unsafe auto-accepts, ours vs each method, absolute and percentage reduction. |
| `03_significance_image_level.csv` | McNemar tests on 1,905 paired images, Holm-adjusted. **The real evidence.** |
| `04_direction_across_backbones.csv` | Does the result hold on all 3 backbones? Win counts only. |
| `05_learning_curves_per_round.csv` | The raw per-round numbers behind fig3. |
| `main_comparison.tex` | The main table as LaTeX, generated from the CSV. Paste into the paper. |
| `safety_reduction.tex` | The safety table as LaTeX. |

The two `.tex` files are **generated**, never hand-typed, so they cannot
disagree with the CSVs.

### Reading the confidence intervals

A **95% confidence interval** like `[-0.63, +1.73]` means: the true
difference is plausibly anywhere in that range. Because it **includes zero**,
we cannot claim a difference exists. Because it's **narrow**, we can say any
real difference is small — at most about 1.7 percentage points either way.
"Narrow and containing zero" is a much stronger statement than "we found
nothing"; it's evidence of genuine equivalence.

---

## Part 6 — How to regenerate all of this

Two commands, in order, from the repository root:

```bash
python -m evaluation.rigor.baseline_comparison   # computes tables + figures
python -m tools.build_comparison_package         # assembles this folder
```

The first needs per-image prediction dumps to exist. If they don't:

```bash
python -m evaluation.rigor.dump_test_predictions
```

That reloads each finished model and runs one forward pass over the test set.
It trains nothing, runs on a laptop CPU, and skips anything already done —
about 9 minutes per model.

---

## Part 7 — The honest limitations

Stating these yourself is strictly better than having a reviewer find them.
It also costs you very little: every one of them has a defensible answer.

1. **Single seed (42).** No error bars from repeated runs. Per supervisor
   direction. Mitigated by the image-level tests, which get statistical power
   from 1,905 images rather than from repeated runs — and by the result
   holding on all three backbones. Worth stating as a limitation explicitly.

2. **`unsafe auto-accepts` is measured on the unlabelled pool, not the
   held-out test set.** It counts decisions made during the active-learning
   process. It is a large, consistent, cost-matched effect — but it is a
   statement about the *labelling process*, not about deployment on unseen
   patients. The test-set missed-cancer numbers are the deployment-facing
   claim, and they are **weaker**. Do not let the abstract blur these two.

3. **The main safety result is partly structural.** Our escalation set is a
   mathematical superset of the uncertainty-only set (Proposition 1), so it
   *cannot* auto-accept more unsafe cases than uncertainty-only. The
   *direction* of that particular comparison is guaranteed by construction,
   not discovered. Say so — a reviewer who notices it unaided will trust
   nothing else in the paper. Note this argument does **not** apply to the
   four literature baselines, where the comparison is genuinely empirical.

4. **Simulated oracle.** Labels come from the dataset, not a live
   dermatologist. Real clinicians disagree with each other and with the
   dataset.

5. **One dataset.** Everything here is HAM10000. External validation on
   ISIC 2020 is scripted but not yet run. (Not ISIC 2019 — it *contains*
   HAM10000, so testing on it would mean testing on training images.)

6. **EfficientNet-B4 at 224px**, not its native 380px, for compute reasons.

---

## Part 8 — The results, in numbers

### Safety (on the pool) — a clean sweep

Cumulative unsafe auto-accepts over 15 rounds, at identical label cost:

| Backbone | Ours | CoreSet | BADGE | CLUE | VAAL | Unc-only |
|---|---|---|---|---|---|---|
| ResNet-50 | **4,945** | 9,575 | 8,194 | 8,481 | 12,628 | 9,327 |
| DenseNet-169 | **4,495** | 8,543 | 6,947 | 7,397 | 11,308 | 8,275 |
| EfficientNet-B4 | **7,362** | 9,893 | 10,963 | 11,873 | 12,745 | 12,346 |

**15 of 15 comparisons favour ours, by 25.6% to 60.8%.** No exceptions.

### Accuracy (on the test set) — indistinguishable, which is the point

Image-level McNemar on the 1,905 shared test images, Holm-corrected across all
15 comparisons. **4 of 15 significant:**

| Comparison | Δ accuracy | Holm p | Verdict |
|---|---|---|---|
| vs **VAAL** (all 3 backbones) | +2.6 to +5.4 pp | 0.0092 → <0.0001 | **ours significantly better** |
| vs **Uncertainty-only** (EfficientNet-B4) | +2.41 pp | 0.0053 | **ours significantly better** |
| vs **CoreSet, BADGE, CLUE** (all backbones) | −0.68 to +1.52 pp | all ≥ 0.26 | **no detectable difference** |
| vs Uncertainty-only (ResNet-50, DenseNet-169) | +0.21 to +0.52 pp | 1.0000 | no detectable difference |

**This is the desired result, not a disappointing one.** The claim is *safety
gained at no accuracy cost*. Eleven non-significant rows with narrow intervals
straddling zero are exactly the evidence for "no cost". Beating VAAL outright is
a bonus.

### Missed cancers (on the test set) — honest null

Of 349 malignant test images, ours catches 264 (EfficientNet-B4), 272
(ResNet-50), 280 (DenseNet-169). Only **2 of 15** comparisons are significant,
both against VAAL. Against CoreSet, BADGE, CLUE and uncertainty-only there is
**no significant difference**. Report this as a limitation, not a result — see
Part 7, item 2.

### The one-paragraph version

> Across three backbones and four recent acquisition baselines, all matched to
> an identical per-round labelling budget, dual-metric escalation reduced unsafe
> auto-accepts on the acquisition pool — high-risk cases accepted without human
> review — in **15 of 15 comparisons**, by between 26% and 61%. On the held-out
> test set, classification accuracy was statistically indistinguishable from
> CoreSet, BADGE and CLUE (paired 95% confidence intervals straddling zero;
> image-level McNemar, Holm-corrected), and significantly higher than VAAL on
> all three backbones, confirming the safety gain was not bought by degrading
> diagnostic performance. Held-out missed-cancer rate moved in the expected
> direction but reached significance only against VAAL; with 349 malignant test
> cases the study is not powered to detect the effect sizes involved, and we
> report this as a limitation rather than a result.
