# Paper outline — section by section

**Target format:** full journal paper (IEEE JBHI / Computers in Biology and Medicine style).
A workshop-length version is a trim of this, not a separate write.

Each section below gives: **what to write**, **which figures**, **which tables**, and **which
numbers**. Every number cited here is verified against the source CSV in `04_TABLES/`.

Suggested length ≈ 8,000–9,000 words total.

---

## Title

Working title:

> **Risk-Aware Active Learning for Skin Lesion Diagnosis: A Dual-Metric Escalation Policy for
> Reducing Unsafe Auto-Acceptance**

Keep "unsafe auto-acceptance" in the title. It names the thing that actually improved, and it is
the honest claim. Avoid any title implying label efficiency.

---

## Abstract (~250 words)

Draft ready in `ABSTRACT_AND_CONTRIBUTIONS.md`. Structure: problem → gap → method → experiments →
results with numbers → the cost → limitation.

---

## 1. Introduction (~900 words)

**What to write:**

1. Skin cancer context: melanoma is the dangerous minority class; early detection matters.
2. Deep models do well on dermoscopic images but need large labelled datasets, and expert
   dermatologist labelling is the bottleneck.
3. **Active learning** addresses this by labelling selectively — usually by picking the images the
   model is most uncertain about.
4. **The gap, stated sharply:** uncertainty sampling optimises for *information*, not for *safety*.
   A model that is confidently wrong about a melanoma is never queried, and the case is auto-accepted.
   Confidence and danger are different questions, and no uncertainty measure asks the second one.
5. **The contribution:** an independent risk head, and an escalation policy that fires on either
   signal.
6. Summarise findings with the headline numbers, **including the cost**.

**The motivating number to put in the Introduction** — this is the strongest sentence available:

> Escalating on uncertainty alone captures only **12.6%** of high-risk pool images, compared with
> **10.2%** for escalation chosen at random — despite spending 4,339 oracle labels to do it.

**Figure:** none, or a schematic of the two-head architecture if space allows.

---

## 2. Related Work (~800 words)

Four short subsections:

1. **Deep learning for skin lesion classification** — HAM10000/ISIC benchmark work.
2. **Active learning** — uncertainty sampling (entropy, margin, least-confidence), Bayesian /
   MC-dropout approaches, diversity and core-set methods.
3. **Cost-sensitive and risk-aware learning** — asymmetric misclassification costs in medical ML.
4. **Selective prediction / learning to defer** — models that abstain and refer to a human.

**The positioning sentence:** prior work treats acquisition and deferral as *uncertainty*
problems. This work separates **epistemic uncertainty** (how confused is the model) from
**clinical risk** (how costly is an error here), estimates the second with a dedicated head, and
combines them at the decision level.

---

## 3. Methods (~1,600 words)

**Source: `02_METHODS_AND_MATH/METHODS.md` — this is written and formal. Adapt it directly.**

Subsections, following that document:

| Subsection | Source | Content |
|---|---|---|
| 3.1 Problem setup and notation | METHODS §1 | pool, labelled set, oracle, rounds |
| 3.2 Two-head architecture | METHODS §2 | shared backbone, classification head, independent risk head |
| 3.3 Training objective | METHODS §3 | dual cross-entropy, inverse-frequency class weights |
| 3.4 Uncertainty functionals | METHODS §4 | entropy, margin, least-confidence, MC-dropout — with ranges |
| 3.5 Risk score | METHODS §5 | the risk head; contrast with summing malignant class probabilities |
| 3.6 Threshold calibration | METHODS §6 | 90th percentile of the current labelled set, recalibrated every round |
| 3.7 Escalation policies | METHODS §7 | both policies as set algebra |
| 3.8 Theoretical properties | METHODS §9 | **Propositions 1 and 2 — include these, they are load-bearing** |

**Proposition 1 (monotonic safety).** The dual-metric escalation set is a superset of the
uncertainty-only set at fixed scores, therefore unsafe auto-accepts can never increase. This is
*why* the improvement appeared in 24 of 24 experiments with no exceptions — it is arithmetic, not
luck. Say so explicitly; it converts an empirical result into a guaranteed one.

**Proposition 2 (exact cost).** The additional annotation cost equals exactly the set of images
the risk route flags that the uncertainty route missed. Never more.

**Together these two propositions make the method a controlled trade with a dial**, which is a
much stronger claim than "our method performed better."

**Figure:** the two-head architecture schematic (draw this — it does not exist yet; see
`03_FIGURES/HOW_TO_READ.md`, "Figures still to draw").

---

## 4. Experimental Setup (~700 words)

**Source: `02_METHODS_AND_MATH/EXPERIMENTAL_SETUP.md` — complete, with every hyperparameter.**

Cover: dataset and class distribution; the fixed 80/20 split with the checksum-verified shared test
set; the 490-image seed set; the 24-experiment matrix; training hyperparameters; hardware; seed.

**Table:** the hyperparameter table (from `EXPERIMENTAL_SETUP.md`) — this is the reproducibility
table reviewers look for.

**State plainly here:** all 24 runs share one byte-identical test split of 1,905 images, verified
by checksum (md5 prefix `feff50b2fec0`). This is what licenses the paired image-level statistics
in §6.

---

## 5. Results (~2,000 words)

The spine of the paper. Recommended order — **safety first, then cost, then the ablation**:

### 5.1 Primary outcome — unsafe auto-accepts
Figure `01_unsafe_auto_accepts_total.png`; table `significance_configuration_level.csv`.
−4,030 per run (≈43%), 12/12 configurations, Holm p = 0.003.

### 5.2 The annotation cost
Figure `09_headline_summary.png` (or `04_query_cost_total.png` from supplementary).
+382 labels (+9.1%), 12/12 configurations, Holm p = 0.003.

### 5.3 Classification quality is unchanged
Accuracy +0.60 pp (Holm p = 0.014); F1-macro +0.86 pp (**not significant**, Holm p = 0.30).
The honest reading: no meaningful change.

### 5.4 Label efficiency — the negative result
Figures `10_al_efficiency_accuracy_vs_labels.png`, and `13_annotation_efficiency.png`
(supplementary). At matched budget the method is **0.35 pp behind** and needs ~300 more labels to
reach any given accuracy target. **Report this in the Results, not buried in Limitations.**

### 5.5 Ablation — are both signals needed?
Figure `14_ablation_both_signals_needed.png`; table `ablation_decision_level.csv`.
The four-rule comparison and the near-additivity argument. **This is the section a reviewer will
find most convincing.**

### 5.6 Threshold sensitivity
Figures `15_risk_threshold_sweep.png`, `16_safety_cost_pareto.png`. Shows the safety/cost dial is
continuous and controllable, not a single lucky operating point.

### 5.7 Per-class discrimination
Figure `22_auc_per_class_with_ci.png`; table `per_class_auc.csv`.
All 7 classes with bootstrap CIs. Melanoma is hardest at 0.945 [0.929, 0.960], PR-AUC 0.781.

### 5.8 Calibration
Figures `17`, `18`, `19`, `20`; table `calibration_metrics.csv`.
ECE 0.073, overconfident by ~7 pp; risk head better calibrated (0.056); temperature T ≈ 2.16 cuts
held-out ECE from 0.073 to 0.023.

### 5.9 Robustness
Figure `33_robustness_by_model.png`; table `robustness_summary.csv`.
Risk head degrades more slowly than the classifier (90.3% vs 85.2% retained). Report
EfficientNet-B4's collapse **per model** — the mean conceals it.

### 5.10 Explainability
Figure `28_gradcam_panel_resnet50_entropy_dual_metric.png`.
The case where the classifier attended to a camera artefact and the risk head attended to the
lesion and rescued it — plus the honest failure case where both missed.

### 5.11 Runtime
Figures `26`, `27` (supplementary); tables `runtime_*.csv`.
94.1 GPU-hours total; MC-dropout ≈ 2.4× the others; **the escalation rule itself costs ≈ 8 ms per
round** — the method's cost is annotation, not compute.

---

## 6. Statistical Analysis (~600 words)

Either its own section or folded into §4. Cover: the two levels of analysis and what each can and
cannot conclude; the tests used and why each was chosen; Holm correction across the metric family;
bootstrap CIs; effect sizes.

**Include the apparent contradiction and its resolution:** accuracy is significant across the 12
configurations but individually resolvable in only 1 of 12. Both are correct; they ask different questions. The honest sentence:

> The accuracy effect is small, positive, and consistent in direction, but not individually
> resolvable per configuration at this test-set size.

Addressing this pre-emptively is far better than having a reviewer raise it.

---

## 7. Discussion (~1,000 words)

**Lead with the mechanism, not a summary.** The most valuable insight in the analysis:

> The two heads have independent parameters but **share a backbone**. They are decoupled in
> parameters, not in features. When the shared representation fails — unusual lesion, poor
> lighting, camera artefact — both heads fail together.

This single mechanism explains two otherwise-awkward results:

- ✅ Unsafe auto-accepts improved greatly — "dangerous" and "confusing" are genuinely different
  questions over the *pool*.
- ❌ Test-set missed-cancer rate barely moved — the images the classifier gets wrong are largely
  the same images the risk head gets wrong.

Supporting evidence: on the full test set the risk head (AUC 0.9520) and the summed malignant
probabilities (0.9524) are a tie — but on the classifier's *missed cancers*, the risk head still
flags **5.6%** versus **0.6%**, roughly 9× more. It works where designed to, on a small fraction.

**The direct implication, and the paper's strongest future-work sentence:** give the risk head its
own backbone so the two can fail independently.

Also discuss: the clinical operating-point argument (the threshold is a policy dial, and §5.6 shows
the achievable frontier); why `unsafe_auto_accepts` is the mechanistically honest primary endpoint
while `fn_rate_malignant` is downstream and noisier.

---

## 8. Limitations (~500 words)

**Source: `LIMITATIONS_AND_FUTURE_WORK.md` — complete and ready.**

Non-negotiable inclusions: single seed; simulated oracle; no external validation yet; HAM10000 skin
tone skew; EfficientNet-B4 at 224 px rather than native 380 px; not label-efficient; the
decision-level ablation is a one-step counterfactual, not a retraining ablation.

**Volunteer all of these.** The safety result is strong enough to survive them, and stating them
first is what makes the rest credible.

---

## 9. Conclusion (~300 words)

Restate the trade, the guarantee (Proposition 1), the price (Proposition 2), and the one
prescriptive recommendation: separate backbones for the two heads.

---

## Appendices / Supplementary

- Full 24-experiment results table (`master_summary.csv`)
- The remaining 23 figures
- The complete formal specification (`METHODS.md`)
- Reproducibility statement and code release

---

## Writing rules for this specific paper

1. **Never claim label efficiency.** Repeat: at matched budget the method is behind.
2. **Always quote the Holm-corrected p-value.** Twelve tests were run; one would pass by luck.
3. **Say "not significant" plainly** for F1-macro, missed-cancer rate and melanoma recall. Use
   "suggestive, pending multi-seed replication."
4. **Report robustness per model.** The 52.7% mean hides a total collapse.
5. **State the single-seed limitation early**, not only in §8.
6. **Distinguish the two safety metrics every time.** `unsafe_auto_accepts` is measured on the pool
   and reflects the escalation decision; `fn_rate_malignant` is measured on the test set and
   reflects the trained model. They are not interchangeable.
