# HOW TO READ — 02_METHODS_AND_MATH

## The short version

- The precise, formal description of the method — the source material for §3 and §4 of the paper.
- **`METHODS.md`** is the complete specification. **Three sections carry the argument:**
  - **§7 — the method in one line.** The baseline escalates set 𝒜; the proposed policy escalates
    **𝒜 ∪ ℬ**. That union is the entire method.
  - **§9 — two proved propositions.** Proposition 1: the proposed escalation set is a superset of
    the baseline's, so unsafe auto-acceptance is monotonically non-increasing — the direction of
    the primary result is **guaranteed, not empirical**. Proposition 2: the extra annotation cost
    equals exactly the risk-route-only set.
  - **§10 — an analysis that was attempted and discarded** as mathematically unidentifiable.
- **`NOTATION_AND_ABBREVIATIONS.md`** — every symbol, every abbreviation, all 7 class codes with
  test-set counts.
- **`EXPERIMENTAL_SETUP.md`** — dataset, splits, the 24-experiment matrix, the full hyperparameter
  table, compute budget, and a draft reproducibility statement.

---

## `METHODS.md`

Written by reading the source code rather than the earlier design documents, which had drifted out
of date. It is therefore the authoritative description of what the implementation actually does.

| Section | Contents |
|---|---|
| §1 | Notation and problem setup |
| §2 | The two-head model |
| §3 | Training objective, with inverse-frequency class weighting |
| §4 | The four uncertainty functionals, with their natural ranges |
| §5 | The risk score |
| §6 | Per-round threshold calibration |
| §7 | **The two escalation policies, as set algebra** |
| §8 | Evaluation metrics, formally |
| §9 | **Propositions 1 and 2, with proofs** |
| §10 | What the logs cannot identify |
| §11 | Hyperparameter table |

### Why §9 matters to the paper

Proposition 1 converts the primary result from an empirical observation into a structural
guarantee. Because $\mathcal{E}^{\text{dual}} \supseteq \mathcal{E}^{\text{unc}}$ at fixed scores,
the auto-accept set can only shrink — which is why the improvement appeared in **24 of 24**
experiments with zero exceptions. Proposition 2 bounds the cost exactly.

Together they establish the method as a **controlled trade with a tunable dial**, which is a
considerably stronger position than a demonstrated performance improvement.

### Why §10 is included

A reviewer may reasonably ask how training time was separated from query time. The obvious
approach — regressing logged per-round wall-clock times on the labelled and unlabelled counts — is
**unidentifiable for this design**. The pool is closed, so
$|\mathcal{L}_t| + |\mathcal{U}_t| = 8{,}110$ in every round, making the two predictors perfectly
collinear with the intercept. It produced *negative* query times, which is how the problem was
detected.

The method was abandoned and replaced with direct microbenchmarking of each component. Retaining
this as a short methodological note in the paper both documents the standard applied and pre-empts
the suggestion.

---

## Recurring symbols

Complete table in `NOTATION_AND_ABBREVIATIONS.md`. The ones that appear constantly:

| Symbol | Meaning |
|---|---|
| $\mathcal{U}_t$ / $\mathcal{L}_t$ | Unlabelled / labelled sets at round $t$ |
| $u_i$ | Uncertainty score of image $i$ — larger means less certain |
| $r_i$ | Risk score of image $i$ — larger means more dangerous |
| $\tau^u_t$, $\tau^r_t$ | The two thresholds at round $t$ (90th percentile, recalibrated each round) |
| $\mathcal{A}_t$ / $\mathcal{B}_t$ | The uncertainty route / the risk route (uncapped) |
| $\mathcal{E}$ / $\mathcal{S}$ | Escalation set / auto-accept set |
| $\cup$, $\setminus$, $\supseteq$ | union, set difference, superset |

---

## Two distinctions the paper must preserve

**1. `unsafe_auto_accepts` vs `fn_rate_malignant`.**
The first is measured on the unlabelled pool every round and reflects the **escalation decision** —
the quantity the risk score directly controls. The second is measured on the held-out test set at
the end and reflects **what the final model learned**. The first improved by ≈43% (Holm p = 0.003);
the second did not reach significance (Holm p = 0.305). They are not interchangeable, and each
mention should state which is meant.

**2. Percentage vs percentage point.**
Accuracy moving from 88.5% to 89.1% is **+0.60 pp**. Writing "+0.6%" denotes a relative increase,
which is a different quantity.
