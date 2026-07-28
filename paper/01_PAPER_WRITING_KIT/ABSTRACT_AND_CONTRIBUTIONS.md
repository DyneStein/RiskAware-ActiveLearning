# Draft abstract and contribution statements

---

## Draft abstract (248 words)

> Active learning reduces the annotation burden of medical image classification by selecting which
> cases a clinician should label. Standard acquisition functions rank cases by model uncertainty,
> which optimises for information gain but is indifferent to clinical consequence: a model that is
> confidently wrong about a melanoma is never queried, and the case is auto-accepted without review.
> We propose a **dual-metric escalation policy** for skin lesion diagnosis that separates two
> distinct questions — *how uncertain is the model?* and *how dangerous is this case if the model is
> wrong?* — and estimates the second with a dedicated risk head trained alongside the classification
> head on a shared backbone. A case is escalated to the oracle if **either** signal exceeds its
> per-round calibrated threshold. We prove that the resulting escalation set is a superset of the
> uncertainty-only set, so unsafe auto-acceptance cannot increase, and that the additional
> annotation cost is exactly the set of cases flagged by the risk route alone. Across 24 experiments
> on HAM10000 (3 backbones × 4 uncertainty measures × 2 policies, 15 acquisition rounds, a fixed
> 1,905-image test split), the policy reduced unsafe auto-accepts by 43% in 12 of 12 matched
> configurations (Holm-corrected p = 0.003) at a cost of 9.1% more oracle labels (p = 0.003), with
> no significant change in F1-macro. A decision-level ablation shows uncertainty sampling captures
> only 12.6% of high-risk pool cases versus 10.2% for random selection, while the two signals
> combined reach 29.3% — near-additive, indicating they flag largely disjoint cases. We further
> report calibration, per-class AUC, robustness and explainability analyses, and identify shared
> backbone features as the mechanism limiting further gains.

---

### Notes on the abstract

- It leads with the **gap**, not the method. The "confidently wrong about a melanoma" sentence is
  the strongest line available — keep it early.
- It states **the cost in the same breath as the benefit**. Reviewers trust this.
- It includes the **ablation contrast (12.6% vs 10.2%)**, which is the single most persuasive number.
- It ends on the **mechanism and its limit**, not on a triumphal note.
- It does **not** claim label efficiency, improved missed-cancer rate, or improved melanoma recall —
  none of those are significant.

**If a shorter version is needed (150 words):** keep sentences 1–4 (gap and method), the headline
result with its cost, and the ablation contrast. Drop the propositions and the closing analyses.

---

## Contribution statements

Use four to five, phrased as things done rather than things claimed:

1. **A dual-metric escalation policy** that separates epistemic uncertainty from clinical risk,
   estimating the latter with an independent risk head rather than deriving it from the
   classifier's own class posteriors.

2. **Two structural guarantees.** We prove that the dual-metric escalation set contains the
   uncertainty-only set at fixed scores — so unsafe auto-acceptance is monotonically
   non-increasing — and that the extra annotation cost equals exactly the risk-route-only
   flagged set. The method is therefore a controlled trade with a tunable dial, not an empirical
   heuristic.

3. **A comprehensive empirical study**: 24 experiments spanning 3 backbones × 4 uncertainty
   measures × 2 policies over 15 acquisition rounds, all sharing one checksum-verified test split,
   with paired statistical testing at both the configuration level (n = 12) and the image level
   (n = 1,905), Holm-corrected throughout.

4. **A decision-level ablation** demonstrating that uncertainty sampling is a weak safety mechanism
   (12.6% high-risk capture versus 10.2% for random selection) and that the two signals are
   near-additive, establishing that they are complementary rather than redundant.

5. **A mechanistic account of the method's limit.** The two heads are decoupled in parameters but
   share a backbone, so they fail on the same inputs. We show this explains why pool-level unsafe
   auto-accepts improve substantially while test-set missed-cancer rate does not, and we identify
   backbone separation as the direct next step.

---

## The framing sentence to reuse verbatim

Put a version of this in the abstract, the end of the introduction, the start of the discussion,
and the conclusion:

> Dual-metric escalation is a safety intervention with a quantified price: a large and
> statistically significant reduction in unsafe auto-accepts, in exchange for approximately 9% more
> oracle labels and no meaningful change in classification quality.

Consistency across those four locations is what makes a paper read as though it knows what it
found.

---

## Keywords

active learning; medical image classification; skin lesion; melanoma; selective prediction;
uncertainty estimation; risk-aware learning; annotation efficiency; model calibration; HAM10000
