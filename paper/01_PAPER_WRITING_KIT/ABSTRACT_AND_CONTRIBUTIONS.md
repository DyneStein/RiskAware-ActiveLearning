# Draft abstract and contribution statements

> **Framing rule for this whole paper.** Our headline safety number is measured
> on the **unlabelled pool**; the held-out test-set safety number did **not**
> reach significance. Those are two different claims and they must never appear
> in the same sentence as though they were one. Every section below is written
> to keep them distinct. See `../06_STATUS_AND_OPEN_ITEMS/POOL_VS_TEST_FRAMING.md`
> for the full argument and the sentences to reuse.

---

## Draft abstract (298 words)

> Active learning reduces the annotation burden of medical image classification
> by selecting which cases a clinician should label. Standard acquisition
> functions rank cases by model uncertainty, which optimises for information
> gain but is indifferent to clinical consequence: a model that is confidently
> wrong about a melanoma is never queried, and the case is auto-accepted
> without review. We propose a **dual-metric escalation policy** for skin
> lesion diagnosis that separates two distinct questions — *how uncertain is
> the model?* and *how dangerous is this case if the model is wrong?* — and
> estimates the second with a dedicated risk head trained alongside the
> classification head on a shared backbone. A case is escalated to the oracle
> if **either** signal exceeds its per-round calibrated threshold. We prove
> that the resulting escalation set is a superset of the uncertainty-only set,
> so unsafe auto-acceptance cannot increase, and that the additional annotation
> cost is exactly the set of cases flagged by the risk route alone. We evaluate
> on HAM10000 over 36 acquisition runs of 15 rounds each — 3 backbones × 4
> uncertainty measures × 2 policies, plus 3 backbones × 4 recent acquisition
> baselines (CoreSet, BADGE, CLUE, VAAL) matched round-by-round to our own
> label expenditure — all sharing one frozen 1,905-image test split.
> **The intervention is on a decision rule, so we evaluate it on the decisions
> it makes:** on the unlabelled pool, the policy reduced unsafe auto-accepts in
> **15 of 15** cost-matched comparisons against published baselines (26–61%
> fewer) and in **12 of 12** matched configurations against uncertainty-only
> sampling (Holm-corrected p = 0.003), for 9.1% more oracle labels and no
> significant change in F1-macro. **The corresponding held-out metric —
> missed-cancer rate on unseen patients — improved in direction but did not
> reach significance (p = 0.305), and we report this as a limitation rather
> than a result.** A decision-level ablation shows uncertainty sampling
> captures only 12.6% of high-risk pool cases versus 10.2% for random
> selection, while both signals combined reach 29.3% — near-additive,
> indicating they flag largely disjoint cases. We identify shared backbone
> features as the mechanism limiting transfer to held-out safety.

---

### Notes on the abstract

- It leads with the **gap**, not the method. The "confidently wrong about a
  melanoma" sentence is the strongest line available — keep it early.
- **It states where the headline is measured, in the headline sentence itself.**
  This is the single most important edit in the paper. The pool-level result and
  the test-set null are two sentences apart, in the abstract, in our own words.
  A reviewer who discovers this ordering themselves rejects the paper; a
  reviewer who is told it up front reads the rest as careful.
- It states **the cost in the same breath as the benefit**. Reviewers trust this.
- It includes the **ablation contrast (12.6% vs 10.2%)**, the single most
  persuasive number, because it shows the standard method barely beats random
  at the safety job.
- It names the four baselines. Reviewers trust comparisons against papers they
  have read, and "cost-matched" pre-empts the obvious objection.
- It ends on the **mechanism and its limit**, not on a triumphal note.
- It does **not** claim label efficiency, improved missed-cancer rate, or
  improved melanoma recall — none of those are significant.

**If a shorter version is needed (150 words):** keep the gap and method, the
pool-level headline *with its "on the unlabelled pool" qualifier intact*, the
one-clause statement that the held-out metric did not reach significance, and
the ablation contrast. Drop the propositions and the baseline names. **Never
drop the qualifier to save words** — an unqualified "reduced unsafe
auto-accepts by 43%" is the version that gets the paper rejected.

---

## Contribution statements

Use four to five, phrased as things done rather than things claimed:

1. **A dual-metric escalation policy** that separates epistemic uncertainty
   from clinical risk, estimating the latter with an independent risk head
   rather than deriving it from the classifier's own class posteriors.

2. **Two structural guarantees.** We prove that the dual-metric escalation set
   contains the uncertainty-only set at fixed scores — so unsafe
   auto-acceptance is monotonically non-increasing — and that the extra
   annotation cost equals exactly the risk-route-only flagged set. The method
   is therefore a controlled trade with a tunable dial, not an empirical
   heuristic. *(State plainly that this makes the direction of the
   uncertainty-only comparison structural rather than discovered; it does not
   apply to the four literature baselines, where the comparison is genuinely
   empirical.)*

3. **A cost-matched comparison against four recent acquisition methods.**
   CoreSet, BADGE, CLUE and VAAL are acquisition strategies that receive a
   budget; ours is an escalation policy that chooses its own. We resolve the
   mismatch by giving each baseline exactly the label count our policy spent in
   that same round on that same backbone, verified exact for all 12 runs, so
   the comparison isolates *which* images are chosen from *how many*.

4. **A decision-level ablation** demonstrating that uncertainty sampling is a
   weak safety mechanism (12.6% high-risk capture versus 10.2% for random
   selection) and that the two signals are near-additive, establishing that
   they are complementary rather than redundant.

5. **An explicit account of what the result does and does not show, and why.**
   We distinguish the pool-level decision metric from the held-out patient
   metric, report that only the former improves significantly, and give the
   mechanism: the two heads are decoupled in parameters but share a backbone,
   so they fail on the same inputs. We identify backbone separation as the
   direct next step.

---

## The framing sentence to reuse verbatim

Put a version of this in the abstract, the end of the introduction, the start
of the discussion, and the conclusion:

> Dual-metric escalation is a safety intervention on a decision rule, with a
> quantified price: on the decisions it governs, a large and statistically
> significant reduction in unsafe auto-accepts, in exchange for approximately
> 9% more oracle labels and no meaningful change in classification quality. We
> did not demonstrate a significant improvement in missed-cancer rate on
> held-out patients, and we explain why.

Consistency across those four locations is what makes a paper read as though it
knows what it found.

### The two sentences that must always travel together

Never let these be separated by a page break, a section boundary, or a table:

> On the unlabelled pool — the set of cases the policy actually decides — unsafe
> auto-accepts fell by 26–61% against every baseline tested, at matched label
> cost.
>
> On the held-out test set, the corresponding missed-cancer rate moved in the
> same direction but did not reach statistical significance; we report the
> mechanism for this gap in Section [Discussion].

---

## Wording to avoid

These are all true-ish sentences that a reviewer will read as overclaiming.

| Do not write | Write instead |
|---|---|
| "reduces unsafe auto-accepts by 43%" | "reduces unsafe auto-accepts on the unlabelled pool by 43%" |
| "improves patient safety" | "reduces the number of high-risk cases auto-accepted without review" |
| "reduces missed cancers" | "reduces high-risk cases accepted without review; the held-out missed-cancer rate did not change significantly" |
| "outperforms state-of-the-art active learning" | "reduces unsafe auto-accepts relative to four recent acquisition methods at matched label cost, with comparable accuracy" |
| "is more label-efficient" | *(do not claim this at all — at matched budget the policy is −0.35 pp accuracy and needs ~300 more labels to hit any accuracy target)* |
| "the risk head outperforms the summed-probability baseline" | "the risk head matches it on overall AUROC (0.9555 vs 0.9558) and wins only on the subset where the classifier errs" |

---

## Keywords

active learning; medical image classification; skin lesion; melanoma; selective
prediction; uncertainty estimation; risk-aware learning; annotation efficiency;
model calibration; HAM10000
