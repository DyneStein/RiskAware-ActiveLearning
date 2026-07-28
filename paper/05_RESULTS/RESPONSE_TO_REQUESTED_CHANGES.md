# Response to the supervisor's requested changes

**Date:** 2026-07-26 · **Status:** in progress, updated as jobs finish
**Code:** `RiskAware-ActiveLearning/evaluation/rigor/` · **Outputs:** `analysis/rigor/`

All 24 experiments are now complete (15/15 rounds each), so everything below is computed
on the full matrix.

**The key enabler:** every experiment saved its final model weights *and* the exact test
split it used, in `results/checkpoints/<experiment>/round_15/`. That means almost
everything requested can be computed by **reloading the trained models and running
inference** — no retraining, no new GPU-hours. The one exception is external validation,
which needs a dataset download.

Correctness check on that reload path: recomputing round-15 accuracy for
`resnet50_entropy_dual_metric` from the reloaded model gives 0.8986876640419947, matching
the value the experiment logged to four-teen decimal places. The reload is faithful.

---

## Status at a glance

| # | Requested | Status | Where |
|---|---|---|---|
| 1 | Expected Calibration Error (ECE) | ✅ done | `tables/calibration_metrics.csv` |
| 2 | Brier score | ✅ done | same |
| 3 | Reliability diagrams | ✅ done | `figures/17–18` |
| 4 | Accuracy vs number of labelled samples | ✅ done | `figures/10–13` |
| 5 | AUC per lesion class (esp. melanoma) | ✅ done | `figures/21–23`, `tables/per_class_auc.csv` |
| 6 | Statistical tests: p-values | ✅ done | `tables/significance_*.csv` |
| 7 | Confidence intervals | ✅ done | `figures/24–25` |
| 8 | Runtime: training / inference / query | ✅ done | `figures/26–27` |
| 9 | Ablation studies | ✅ done (decision-level) | `figures/14–16` |
| 10 | Formal maths of the dual-metric policy | ✅ done | `RiskAware-ActiveLearning/METHODS.md` |
| 11 | Explainability analysis | ✅ done (all 3 backbones) | `figures/28_gradcam_panel_*` |
| 12 | Robustness experiments | ✅ done (4 of 5 corruptions) | `figures/29–30, 33` |
| 13 | External validation on ISIC 2019/2020 | 🔧 code ready, needs the dataset | `external_validation_isic.py` |

---

## 1–3. Calibration: ECE, Brier score, reliability diagrams ✅

Computed on the held-out test set for **both** output heads, since the system uses both and they
can fail independently: the 7-class classification confidence, and the risk head's P(malignant) —
the one that actually drives escalation.

| | Uncertainty-only | Dual-metric |
|---|---|---|
| Accuracy | 0.8828 | 0.8888 |
| Mean confidence | 0.9553 | 0.9601 |
| **Over-confidence gap** | **+0.0725** | **+0.0713** |
| Classification ECE | 0.0736 | 0.0719 |
| Adaptive ECE (equal-mass bins) | 0.0725 | 0.0713 |
| MCE (worst bin) | 0.4126 | 0.4356 |
| Multi-class Brier | 0.1923 | 0.1837 |
| NLL | 0.5198 | 0.5001 |
| **Risk-head ECE** | 0.0579 | **0.0547** |
| Risk-head Brier | 0.0673 | 0.0630 |

**The finding: the models are overconfident by about 7 percentage points.** They claim ~96%
confidence and are right ~88% of the time. This is the single most common weakness in modern
neural networks and it is exactly what the supervisor suspected — accuracy alone cannot reveal it.

Two things soften it, and both should be reported:

1. **The risk head is better calibrated than the classifier** (ECE 0.055 vs 0.072). Since the risk
   head is what the escalation threshold is applied to, the score the safety mechanism actually
   depends on is the more trustworthy of the two.
2. **Most of the miscalibration is the cheap, fixable kind.** Fitting a single temperature
   parameter on half the test set and evaluating on the other half:
   **T ≈ 2.15, held-out ECE 0.0737 → 0.0245** — a ~67% reduction from one scalar. That means the
   problem is over-sharp probabilities, not a structurally broken model.

Dual-metric is very slightly better calibrated than the baseline on every measure, but the
difference is small and we make no claim about it.

Reliability diagrams: `figures/17_reliability_classification.png` (curve sags below the diagonal =
overconfident) and `figures/18_reliability_risk_head.png`.

---

## 5. AUC per lesion class, especially melanoma ✅

One-vs-rest ROC-AUC per class, averaged over all 24 experiments, with 95% percentile-bootstrap
confidence intervals over test images (2,000 resamples). PR-AUC is reported alongside because
ROC-AUC flatters rare classes when negatives vastly outnumber positives.

| Lesion class | n | ROC-AUC | 95% CI | PR-AUC |
|---|---|---|---|---|
| **Melanoma** | 209 | **0.9454** | [0.929, 0.960] | 0.7812 |
| Actinic keratoses | 51 | 0.9629 | [0.936, 0.984] | 0.7068 |
| Benign keratosis | 206 | 0.9666 | [0.954, 0.977] | 0.8371 |
| Melanocytic nevi | 1,327 | 0.9697 | [0.962, 0.977] | 0.9854 |
| Basal cell carcinoma | 89 | 0.9935 | [0.987, 0.998] | 0.9318 |
| Vascular lesions | 14 | 0.9946 | [0.983, 1.000] | 0.9302 |
| Dermatofibroma | 9 | 0.9961 | [0.987, 1.000] | 0.9153 |

**Melanoma is the hardest class of the seven** — which is honest and expected; it is genuinely the
one most easily confused with an ordinary mole, and it is the class the project exists for. The gap
between its ROC-AUC (0.945) and PR-AUC (0.781) is the usual signature of a rare, hard class and is
worth reporting rather than hiding behind the ROC number.

### An uncomfortable result the supervisor did not ask for, but should see

Scoring "malignant vs benign" two ways on the same test set:

| Scoring rule | ROC-AUC | 95% CI | PR-AUC |
|---|---|---|---|
| Risk head (the two-head redesign) | 0.9520 | [0.939, 0.963] | 0.8654 |
| Summed malignant class probabilities (the ORIGINAL design) | 0.9524 | [0.940, 0.964] | 0.8671 |

**A tie.** On the full test set, giving the risk head its own parameters bought nothing over simply
adding up the classifier's malignant-class probabilities.

But that was never the claim. The argument for the redesign was about **decoupling**: when the
classifier is confidently wrong, a risk score computed *from* its probabilities is necessarily
wrong too. So the comparison is repeated on the subset where the classifier fails:

| Population | n | Risk head | Summed probs | Δ |
|---|---|---|---|---|
| All test images | 1,905 | 0.9520 | 0.9524 | −0.0003 |
| Classifier got it wrong | 218 | 0.4078 | 0.3862 | **+0.0217** |
| Classifier *confidently* wrong | 108 | 0.3630 | 0.3576 | +0.0053 |

⚠️ **How to read those last two rows.** Conditioning on "the classifier was wrong" selects malignant
cases it called benign *together with* benign cases it called malignant — so **any** score
correlated with the classifier is pushed below chance there, by construction. The absolute level is
a selection artefact and must **not** be reported as "the risk score is worse than random". Both
rules are subject to identical selection, so only the **difference** between them is meaningful.

The cleanest, artefact-free number is on the missed cancers themselves — true malignant cases the
classifier called benign (~80 per experiment):

| | Mean score on missed cancers | Still flagged at 0.5 |
|---|---|---|
| Risk head | 0.1138 | **5.6%** |
| Summed class probabilities | 0.0974 | 0.6% |

So the redesign does what it was designed to do — it rescues roughly **9× more** missed cancers than
the original formulation — but **5.6% is still a small fraction.** The two heads share a backbone,
so when the shared features are fooled, both heads are fooled together. They are decoupled in
*parameters* but not in *features*.

**This is the most scientifically valuable finding of the analysis**, because it explains the
results mechanistically: it is precisely why `unsafe_auto_accepts` improved enormously and
significantly (the risk route catches many dangerous images uncertainty ignores) while the test-set
missed-malignant rate did **not** reach significance (the specific images the classifier gets wrong
are the ones the risk head also gets wrong). It also points directly at the next design step: a
separate backbone, or differently-supervised features, for the risk pathway.

See `figures/32_risk_head_decoupling.png` and `tables/risk_head_decoupling.csv`.

---

## 11. Explainability analysis ✅

Grad-CAM heat-maps showing which pixels drove each decision, generated for **both heads
separately** — "why did you call it melanoma?" beside "why did you call it dangerous?" — which is
only possible because of the two-head architecture. Panels exist for all three backbones
(`figures/28_gradcam_panel_<experiment>.png`).

Cases were chosen to be informative rather than flattering: correctly caught melanomas, melanomas
the classifier **missed**, and a case both heads missed.

**The headline case (ResNet-50, `ISIC_0027776`):** a melanoma the classifier labelled benign
keratosis. The classification heat-map shows it attending to the **dark corners of the image** — a
camera/vignetting artefact, exactly the kind of shortcut dermoscopy datasets are notorious for.
The risk head, on the same image, attends to the **actual lesion** and returns 0.83, which
escalates it for review.

**This single figure is the paper's argument in one image**, and it is not a cherry-pick: the same
image is misclassified as benign keratosis by DenseNet-169 too, and its risk head also flags it
(0.60). The failure and the rescue both reproduce across architectures.

An honest failure case is included as well: a melanoma where *both* heads attend to corner
artefacts and both miss it — consistent with the shared-backbone limitation described above.

---

## 12. Robustness experiments ✅

Each final model was re-evaluated on the **same** held-out test set after mild, clinically
plausible degradation — sensor noise, defocus blur, dim lighting, low contrast. No retraining: the
shipped model simply meets a worse image, as it would in a real clinic. (6 experiments = all three
backbones × both policies. JPEG re-compression was still running when the job was interrupted and
is the one gap.)

### Finding 1 — the safety signal degrades more slowly than the diagnosis ✅

| Corruption | Accuracy retained | Risk-AUROC retained | Melanoma recall |
|---|---|---|---|
| Dim (×0.7) | 99.6% | 99.4% | 0.71 |
| Low contrast | 99.1% | 99.4% | 0.61 |
| Defocus blur | 89.5% | **94.7%** | 0.31 |
| Noise σ=0.05 | 52.7% | **67.7%** | **0.08** |
| **Average** | **85.2%** | **90.3%** | — |

This is the desired behaviour from a safety architecture, and it was a stated hypothesis before
the run rather than a post-hoc observation: under degradation the system loses its grip on *which*
disease it is faster than it loses the sense that the case is *dangerous*. A system that degrades
this way escalates to a clinician instead of silently mis-diagnosing. The two-head design gets
credit here that the overall-AUC comparison in §5 denied it.

### Finding 2 — EfficientNet-B4 collapses completely under noise ⚠️

The 52.7% average above is not a graceful decline; it is one architecture falling off a cliff and
dragging the mean with it:

| Accuracy | Clean | Under noise σ=0.05 |
|---|---|---|
| DenseNet-169 | 0.889 | 0.692 |
| ResNet-50 | 0.896 | 0.710 |
| **EfficientNet-B4** | **0.863** | **0.008** |

Accuracy of **0.008 is far below random guessing** (1/7 ≈ 0.14) — the model collapses to predicting
essentially one wrong class for every image. Its risk head goes with it (AUROC 0.937 → 0.400,
below chance). The corruption code is not at fault: the identical transform applied to the other
two backbones produces the orderly ~20-point drop above.

This is a **serious, publishable robustness failure** and it must be reported per-model. Quoting
only the 52.7% mean would conceal it. Two things likely contribute: EfficientNet-B4 is being run at
224×224 rather than its native 380×380 (already a declared limitation), and its heavy use of
batch-norm statistics makes it sensitive to input-distribution shift.

### Finding 3 — melanoma recall is fragile ⚠️

Melanoma sensitivity falls from **0.70 clean → 0.31 under blur → 0.08 under noise.** Even for the
two robust backbones, a slightly out-of-focus dermatoscope costs most of the melanoma detection
rate. This belongs in the Limitations section: the safety claims in this paper are established on
clean, curated images, and image quality control would be a hard requirement for any deployment.

See `figures/29_robustness_degradation.png`, `30_robustness_heads_compared.png`, and
`33_robustness_by_model.png`.

---

## 4. Active-learning efficiency — accuracy vs number of labelled samples ✅

This one changed how we should describe the result, so it is worth reading carefully.

Plotting against *round number* flatters the dual-metric policy, because the two policies
do not buy the same number of labels per round — the risk route is uncapped, so by round 15
dual-metric has spent **+382 more oracle labels on average** (95% CI [+315, +447], every one
of the 12 configurations, Holm-adjusted p = 0.003). Judged per round it looks better partly
just for having asked more questions.

Re-plotted against labels consumed, and compared at a **matched annotation budget**:

| Measured at equal labels | dual − uncertainty-only |
|---|---|
| Accuracy | **−0.35 pp** (dual wins 4/12 configurations) |
| F1-macro | −0.91 pp |
| Melanoma recall | −0.71 pp (dual wins 6/12) |
| Extra labels needed to reach 80% accuracy | dual needs **+304 more** |

**So: the dual-metric policy is not more label-efficient. It is slightly less.** It spends
labels on cases that are *dangerous* rather than cases that are *informative*, and those are
not the same images — informative-but-safe images teach the classifier more per label.

This does not weaken the paper; it sharpens it. The contribution is a **safety
intervention with a quantified price**, not a free improvement. The honest one-line
statement is:

> Dual-metric escalation reduces unsafe auto-accepts by a large and statistically
> significant margin, at a cost of ~8% more oracle labels and no meaningful change in
> classification quality.

Claiming label-efficiency instead would be the first thing a reviewer checked, and it would
not survive. Figures 10–13 make the trade-off explicit.

---

## 6–7. Statistical tests: p-values, confidence intervals, effect sizes ✅

**The honest constraint first.** The matrix was run at one seed per configuration. That
rules out the textbook design (repeat each configuration over k seeds, test across seeds).
Rather than fake it, two tests the data genuinely supports are run, each labelled with what
it can and cannot conclude. Multi-seed replication remains the recommendation and is stated
as a limitation in every output.

### A. Configuration-level (n = 12 paired configurations)

Each (model, uncertainty-method) combination was run under both policies, so the 12 pairs
can be tested with a Wilcoxon signed-rank test, corroborated by an exact sign test and a
paired permutation test, with Holm-Bonferroni correction across metrics.

| Metric | Mean Δ (dual − baseline) | 95% CI | Wilcoxon p | Holm p | |
|---|---|---|---|---|---|
| **Unsafe auto-accepts (total)** | **−4,030.5** | [−4,319, −3,715] | 0.0005 | **0.0029** | ** |
| **Final accuracy** | **+0.60 pp** | [+0.28, +1.00] | 0.0034 | **0.0137** | * |
| Final F1-macro | +0.86 pp | [−0.22, +1.97] | 0.233 | 0.305 | ns |
| Final missed-malignant rate | −1.19 pp | [−2.46, +0.21] | 0.152 | 0.305 | ns |
| Final melanoma recall | +2.43 pp | [+0.44, +4.39] | 0.057 | 0.170 | ns |
| **Total oracle queries** | **+382.4** | [+315, +447] | 0.0005 | **0.0029** | ** |

Reading this honestly:

- **The safety result is solid.** Unsafe auto-accepts fall in **12 of 12** configurations,
  with the CI far from zero. This is the paper's headline and it survives correction.
- **The cost is equally solid** and in the same direction every time: more labels.
- **Accuracy is not harmed** — mildly *better* at round 15, significant after correction
  (though §4 shows that gain is bought with the extra labels, not earned per label).
- **Melanoma recall (+2.4 pp) and missed-malignant rate (−1.2 pp) do not reach
  significance.** The CI for melanoma recall excludes zero while the Wilcoxon does not —
  they test different things (mean shift vs rank consistency), and at n = 12 that
  disagreement means "promising, not established". It should be reported as such, not
  rounded up into a claim.

### B. Image-level (n = 1,905 paired test images)

All 24 experiments were evaluated on a byte-identical test split (verified by checksum), so
within a pair the two policies predict on the *same* images and can be paired per image.
McNemar's exact test is the correct test for paired binary outcomes; paired bootstrap over
images gives CIs on the metric differences. Holm-corrected across the 12 configurations.

| Configuration | Δ accuracy | 95% CI | McNemar p | Holm p | |
|---|---|---|---|---|---|
| **efficientnet_b4 + entropy** | **+2.41 pp** | [+1.05, +3.83] | 0.0004 | **0.0049** | ** |
| densenet169 + least_confidence | +1.00 pp | [−0.26, +2.20] | 0.132 | 1.000 | ns |
| efficientnet_b4 + mc_dropout | +0.73 pp | [−0.63, +1.99] | 0.324 | 1.000 | ns |
| efficientnet_b4 + margin | +0.68 pp | [−0.53, +1.89] | 0.344 | 1.000 | ns |
| densenet169 + mc_dropout | +0.68 pp | [−0.58, +1.89] | 0.329 | 1.000 | ns |
| resnet50 + entropy | +0.52 pp | [−0.68, +1.68] | 0.450 | 1.000 | ns |
| densenet169 + margin | +0.47 pp | [−0.79, +1.84] | 0.536 | 1.000 | ns |
| resnet50 + least_confidence | +0.37 pp | [−0.84, +1.58] | 0.626 | 1.000 | ns |
| resnet50 + mc_dropout | +0.31 pp | [−0.94, +1.57] | 0.696 | 1.000 | ns |
| densenet169 + entropy | +0.21 pp | [−1.05, +1.57] | 0.816 | 1.000 | ns |
| resnet50 + margin | +0.10 pp | [−1.21, +1.36] | 0.937 | 1.000 | ns |
| efficientnet_b4 + least_confidence | −0.31 pp | [−1.47, +0.84] | 0.661 | 1.000 | ns |

**Significant accuracy difference after Holm correction: 1 of 12. Significant
malignant-detection difference: 0 of 12.**

**This does not contradict §A — the two tests answer different questions, and both answers are
true.** Individually, on 1,905 test images, almost every gap is smaller than test-set noise; only
EfficientNet-B4 + entropy separates. But **11 of 12 differences point the same way**, and it is that
consistency of *direction* across independent configurations which the configuration-level test
detects and calls significant. A single coin landing heads proves nothing; eleven of twelve coins
landing heads is evidence.

The honest statement for the paper: *the accuracy effect is small, positive, and consistent in
direction, but not individually resolvable per configuration at this test-set size.* The safety
effect, by contrast, is large enough to be unambiguous at every level of analysis.

### C. Ablation-level (n = 24 experiments) — the strongest evidence in the study

| Comparison | Δ unsafe auto-accepts | Wilcoxon p | Holm p | |
|---|---|---|---|---|
| dual-metric vs uncertainty-only | −1,666 | 1.2e−07 | 9.5e−07 | *** |
| dual-metric vs risk-only | −954 | 1.8e−05 | 1.8e−05 | *** |
| dual-metric vs random (cost-matched) | −1,782 | 1.2e−07 | 9.5e−07 | *** |

---

## 9. Ablation studies ✅ (decision-level)

**Method, and its limits.** Every round, each experiment logged the per-image uncertainty
score, risk score and true label for the entire pool, plus that round's calibrated
thresholds. The escalation rule is a pure function of those numbers, so each round can be
*replayed* under a different rule and we can count exactly which images each rule would have
caught. This is an ablation of the **decision rule with the model held fixed** — cheap
(seconds, no GPU) and exactly controlled, since every rule sees identical scores from an
identical model. It is a one-step counterfactual: it does not capture how different
labelling would have changed the *next* round's model. A full retraining ablation is a
separate GPU-cost item, listed at the bottom.

Averaged over all 24 experiments, all 15 rounds:

| Rule | Unsafe auto-accepts | Oracle labels | High-risk cases caught |
|---|---|---|---|
| Random (cost-matched to dual) | 7,832 | 6,201 | 10.2% |
| Uncertainty only (baseline) | 7,716 | 4,339 | 12.6% |
| Risk only | 7,004 | **1,917** | 17.2% |
| **Dual-metric (ours)** | **6,050** | 6,201 | **29.3%** |

Three things fall out of this, and they are the most useful results of the night:

1. **Uncertainty alone barely beats random at catching danger** (12.6% vs 10.2%) — despite
   costing 4,339 labels. Uncertainty sampling is simply not a safety mechanism; it was never
   designed to be one. This is the clearest possible motivation for the paper.
2. **Risk alone is by far the most label-efficient** — 17.2% of dangerous cases caught for
   only 1,917 labels, a third of the baseline's spend. Worth reporting prominently as a
   deployment option when annotation budget is the binding constraint.
3. **The two signals are complementary, not redundant.** Combined they catch 29.3%, which
   is close to additive (12.6 + 17.2 = 29.8) — meaning the two routes flag *largely
   different images*. That is the empirical justification for the whole dual-metric design,
   and it is exactly what an ablation is supposed to establish.

Figures 14 (ablation bars), 15 (risk-threshold sweep) and 16 (safety-vs-cost frontier)
cover this. The sweep varies the risk threshold from 0.05 to "disabled" and traces the
safety/cost dial, which is the tuning curve a deployer would need.

---

## 10. Formal mathematical definitions ✅

`RiskAware-ActiveLearning/METHODS.md` — written by reading the code rather than the prose
docs, so the maths matches what actually ran. It covers: notation and the risk partition;
the two-head model; the training objective with the per-round inverse-frequency class
weights; all four uncertainty functionals with their exact ranges; the risk score;
per-round 90th-percentile threshold calibration; and the two escalation policies as set
algebra:

$$\mathcal{E}^{\text{unc}}_{t} = \mathcal{A}_t
\qquad
\mathcal{E}^{\text{dual}}_{t} = \mathcal{A}_t \cup \mathcal{B}_t$$

with $\mathcal{A}_t = \operatorname{Top}_K(u) \cup \{i : u_i > \tau^u_t\}$ (budgeted) and
$\mathcal{B}_t = \{i : r_i > \tau^r_t\}$ (uncapped).

It also states and proves two structural properties that turned out to explain the
empirical pattern:

- **Proposition 1 (monotonicity).** At fixed scores, $\mathcal{E}^{\text{dual}} \supseteq
  \mathcal{E}^{\text{unc}}$, therefore $S^{\text{dual}}_t \le S^{\text{unc}}_t$ *always*.
  This is why the decision-level ablation improves in 24/24 experiments with no exceptions —
  it is arithmetic, not luck. It does **not** transfer to the full experiment, because once
  the policies request different labels they train on different data and their scores
  diverge; there, 12/12 is genuine evidence.
- **Proposition 2 (the price).** The extra annotation cost is exactly
  $|\mathcal{B}_t \setminus \mathcal{A}_t|$ — the images the risk route flags that
  uncertainty would have missed, never more.

Together these frame the contribution precisely: a **controlled trade**, strictly more
safety for strictly more annotation, with the exchange rate set by $\tau^r$ and traced
empirically by the threshold sweep.

---

## 8. Runtime ✅ — including one method that had to be thrown out

**A dead end worth recording.** The obvious approach is to regress the logged
`round_time_seconds` on |labelled| and |pool| across the 360 logged rounds and read off the
training and query coefficients. **That does not work here.** The pool is closed, so

$$|\mathcal{L}_t| + |\mathcal{U}_t| = 8{,}110 \quad \text{exactly, every round, all 24 experiments}$$

The two predictors are perfectly collinear with each other and with the intercept; the fit
is rank-deficient and returns arbitrary coefficients. It announced itself by producing
**negative query times**. Only the combined slope is identifiable — never the split. This is
documented in `METHODS.md` §10 so nobody re-derives it later and trusts it.

**What is reported instead:** each component timed in isolation on real model objects at the
real input size (one training step, one inference pass, a 30-pass MC-dropout inference, and
the escalation rule itself on a realistic pool array), then multiplied by actual per-round
set sizes.

**Logged wall-clock (Colab T4, exactly as run):**

- **94.1 GPU-hours** across all 24 experiments; mean 3.92 h each (range 2.37–7.39 h)
- entropy 2.83 h · least-confidence 2.88 h · margin 2.96 h · **MC-dropout 7.00 h**
- MC-dropout overhead factor **2.42×**, measured from wall-clock, not modelled

**Measured per-image component costs.** These are laptop-CPU numbers and the *absolute*
milliseconds depend on the CPU thread count, so the benchmark records the device and thread count
alongside them — the authoritative values live in `tables/runtime_components_measured.csv`. The
*ratios* below (MC-dropout vs inference, training vs inference, and the per-round split) are
thread-invariant and are what the conclusions rest on. Figures shown are from a 4-thread run:

| Model | Inference | MC-dropout inference | Training step |
|---|---|---|---|
| EfficientNet-B4 | 148.6 ms | 4,445 ms | 726.6 ms (4.9× inference) |
| ResNet-50 | 194.4 ms | 5,806 ms | 606.7 ms (3.1× inference) |
| DenseNet-169 | 214.7 ms | 6,589 ms | 703.9 ms (3.3× inference) |

**The benchmark validates itself:** MC-dropout comes out at **29.9–30.7× plain inference** across
all three architectures — exactly the 30 stochastic forward passes the method specifies. A
decomposition that failed to recover a known quantity would not be trustworthy; this one does.

**Query time is the escalation rule's scoring pass, not the rule itself.** The decision logic runs
in **8.4 ms for dual-metric vs 4.5 ms for uncertainty-only** on a 6,000-image pool — per *round*.
Completely negligible. The uncapped risk route adds essentially zero computational cost; what it
costs is oracle labels, not compute.

**Where a round's time actually goes** (measured per-image costs × real per-round set sizes):

| Uncertainty method | Training | Querying | Test eval |
|---|---|---|---|
| entropy | 94.3% | 4.1% | 1.6% |
| least-confidence | 94.4% | 4.1% | 1.6% |
| margin | 94.7% | 3.8% | 1.5% |
| **mc_dropout** | **59.5%** | **39.8%** | 0.7% |

For three of the four uncertainty methods, active learning's selection step is nearly free —
training dominates at ~95%. **MC-dropout is the outlier**: querying balloons to ~40% of every
round. That is a concrete, practical argument against MC-dropout for this application, since the
ablation shows it buys no safety advantage over the cheaper measures.

Cross-check: this CPU runs 31.7–35.2× slower than the Colab T4 the experiments used, consistent
across all three architectures — which is a plausible CPU/GPU ratio and a sign the component model
is not wildly off.

---

## What still needs a GPU session or a download

| Item | What it needs | Why it's worth it |
|---|---|---|
| **Multi-seed replication (5 seeds)** | ~470 GPU-hours for the full matrix | The only thing that turns "consistent across 12 configurations" into "robust to run-to-run noise". **Recommend narrowing to 3–4 configurations × 5 seeds (~60 h)** rather than the full matrix — the safety effect is what needs replicating, not every cell. |
| **Full retraining ablation** | ~12 GPU-hours | Complements the decision-level ablation by capturing the downstream effect of different labelling. |
| **External validation (ISIC 2020)** | ~3 GB download + ~2 GPU-hours | Code is ready and includes a mandatory, audited exclusion of HAM10000 overlap — see the warning below. |
| **Robustness suite** | ~2 hours CPU locally, minutes on GPU | Code ready; 5 corruptions × 6 experiments. |
| **Grad-CAM panel** | minutes | Code ready. |

### ⚠️ Contamination warning for the ISIC request

**ISIC 2019 contains HAM10000.** The ISIC 2019 training set is the union of BCN20000,
HAM10000 and MSK, and the HAM10000 images keep their original `ISIC_xxxxxxx` identifiers.
Every model here was trained on HAM10000 — so evaluating on ISIC 2019 as downloaded is
**not external validation**, and the numbers would be inflated and indefensible.

`external_validation_isic.py` therefore always excludes, by image id, every image present in
`HAM10000_metadata.csv`, reports how many it removed, and asserts zero remaining overlap
before proceeding.

**Recommendation: use ISIC 2020 as the primary external test.** It is a genuinely
independent challenge year with no HAM10000 overlap, and its binary benign/malignant labels
map *exactly* onto what the risk head predicts — so it tests the paper's central claim
directly rather than by proxy. ISIC 2019 (after filtering) can serve as a secondary 7-class
check.

Expected outcome, written down in advance so it cannot be quietly reframed afterwards:
**performance will drop.** Different scanners, sites and populations always cost something.
The reviewer's question is how much, and whether the *safety* signal degrades faster than
the diagnosis.
