# The pool-vs-test-set problem, and how this paper handles it

**This is the most dangerous criticism available against this paper.** It is
also entirely answerable — but only if we raise it ourselves, in our own words,
early. This document is the single source of truth for how that is done.

---

## 1. The problem, stated as a reviewer would state it

> *"The authors report a large safety improvement measured on the unlabelled
> training pool, and no significant safety improvement on held-out patients.
> The clinical claim is therefore unsupported."*

That sentence is fair. Both halves of it are true.

### The two numbers

| | `unsafe_auto_accepts` | `fn_rate_malignant` |
|---|---|---|
| **What it counts** | High-risk cases the policy auto-accepted without review | Truly malignant test images the model called benign |
| **Measured on** | The **unlabelled pool** — data the model is still learning from, and which shrinks every round | The **frozen 1,905-image test set** — never trained on |
| **Where computed** | `al_loop.py`, step 6 (`unsafe_count`) | `al_loop.py`, step 7 (`compute_all_metrics`) |
| **Result vs uncertainty-only** | −4,030 cases, Holm **p = 0.003**, **12/12** configurations | Holm **p = 0.305**, only **3/12** favourable |
| **Result vs 4 baselines** | 26–61% fewer, **15/15** comparisons | not significant |

In a normal paper layout those two numbers sit three pages apart. That gap is
exactly what a reviewer hunts for.

---

## 2. The defence — and it is an honest one

The defence is not spin. It rests on a real distinction about *what kind of
thing* was intervened on.

### 2.1 This is an intervention on a decision rule, not on the weights

The dual-metric policy does not change how the network is trained, its
architecture, or its loss. It changes **which images a clinician is asked to
look at**.

So ask: where should you measure "did fewer dangerous cases get waved through?"

You measure it **on the cases that were waved through**. That set is the pool.
It is not a proxy for the real quantity — it *is* the real quantity for this
claim. Measuring it there is not a workaround; it is the correct measurement.

An analogy that survives scrutiny: if a hospital changes its triage rule, you
evaluate the rule by auditing the patients it triaged. You do not evaluate it by
measuring whether the doctors downstream got better at diagnosis. Those are
different questions, and only the first is about the rule.

### 2.2 The test-set result is a genuinely positive finding — read correctly

The policy escalates substantially more images. A reasonable fear is that this
distorts the training distribution and *degrades* the model.

It does not. Accuracy and F1-macro are unchanged, and against the four
literature baselines accuracy is statistically indistinguishable with paired
95% confidence intervals containing zero.

So the correct reading is: **far fewer dangerous cases slip past, and the model
is just as good.** That is a real result. It is simply not the same result as
"fewer patients are misdiagnosed."

### 2.3 State plainly what was not shown, and why

We did **not** show that the extra labels make the final model safer on unseen
patients. We know the mechanism:

- The classification head and risk head have **separate parameters** but a
  **shared backbone**.
- Measured consequence: on images the classifier gets wrong, the risk head
  gains +0.025 AUC over the summed-probability baseline — but **both fall below
  chance** (AUC ≈ 0.35–0.39) on that subset.
- Meaning: **the safety net is correlated with the thing it is backing up.**
  When the backbone's features are wrong, both heads are wrong together.
- Consequence: 15 rounds of extra labels concentrated on hard cases is not
  enough to move held-out melanoma recall.
- The fix is architectural — separate the backbones — and is named as future
  work, not attempted here.

There is also a straightforward power argument: the test set contains only
**349 malignant images**, of which the models catch 264–280. Shifting that count
by the handful of cases the policy affects is not detectable at n=349.

---

## 3. What this means for each section of the paper

| Section | What to do |
|---|---|
| **Title** | Do not use "safer", "reduces missed cancers", or "improves patient outcomes". Prefer framing around *escalation*, *human review*, or *selective prediction*. |
| **Abstract** | Qualify the headline **in the headline sentence** ("on the unlabelled pool"). State the held-out null within two sentences. Both are already in the draft. |
| **Introduction** | Close with the framing sentence: an intervention on a decision rule, with a quantified price. |
| **Methods** | Define both metrics explicitly, side by side, naming the set each is computed over. Do not let the reader infer it. |
| **Results** | Two separate subsections with unambiguous headings: *"Safety of the escalation decisions (pool)"* and *"Diagnostic performance on held-out patients (test set)"*. Never one table mixing them. |
| **Discussion** | Lead with the gap and the shared-backbone mechanism. This is where the paper earns trust. |
| **Limitations** | Restate it as a numbered limitation, even though it is already in the discussion. Redundancy here is protective. |
| **Conclusion** | Repeat the framing sentence. Do not upgrade the claim in the last paragraph — a very common and very visible failure. |

---

## 4. Paragraphs written out, ready to adapt

### For the Methods section

> We report two safety quantities, computed over different sets, and we
> distinguish them throughout. **Unsafe auto-accepts** is the number of
> high-risk lesions (melanoma, basal cell carcinoma, or actinic keratosis) that
> the escalation policy accepted without referring them for review; it is
> evaluated on the unlabelled pool, since that is the set over which the policy
> makes decisions. **Missed-cancer rate** is the proportion of truly malignant
> lesions in the held-out test set assigned a benign class by the final model;
> it is evaluated on the frozen 1,905-image split and is a property of the
> trained classifier rather than of the escalation rule. The first measures the
> intervention directly; the second measures whether the intervention
> propagates into the learned model.

### For the Results section

> On the unlabelled pool, dual-metric escalation reduced unsafe auto-accepts in
> all 12 matched configurations (mean −4,030 cases, Holm-corrected p = 0.003)
> and in all 15 cost-matched comparisons against CoreSet, BADGE, CLUE and VAAL,
> by between 26% and 61%. Over the same runs, final-round accuracy and
> F1-macro were statistically indistinguishable from the strongest baselines,
> with paired 95% bootstrap confidence intervals containing zero. On the
> held-out test set, missed-cancer rate moved in the expected direction but did
> not reach significance (Holm-corrected p = 0.305, favourable in 3 of 12
> configurations), and melanoma recall was likewise unchanged.

### For the Discussion section

> The asymmetry between our two safety measurements is the central finding to
> interpret, and we do not regard it as a null result to be explained away. The
> policy demonstrably alters which cases reach a human: on the decisions it
> governs, dangerous auto-acceptances fall by a large and consistent margin at
> matched annotation cost. What it does not do, over 15 acquisition rounds, is
> translate that into measurably safer predictions on unseen patients. The
> mechanism is visible in our own decoupling analysis: although the
> classification and risk heads carry separate parameters, they read a shared
> backbone, and on the images where the classifier errs both scores fall below
> chance. The risk signal is therefore correlated with the failure it is
> intended to catch. This bounds what any purely head-level intervention can
> achieve and identifies backbone separation — rather than further threshold
> tuning — as the necessary next step. We also note that the held-out split
> contains 349 malignant cases, which limits the detectable effect size
> independently of the mechanism.

### For the Limitations section

> **The primary safety result is measured on the acquisition pool, not on
> held-out patients.** Unsafe auto-accepts quantifies the behaviour of the
> escalation rule over the cases it decides, which we argue is the appropriate
> target for an intervention on a decision rule. It is not evidence that the
> resulting model misdiagnoses fewer unseen patients; the corresponding
> held-out metric did not reach significance. Additionally, the direction of
> the comparison against uncertainty-only sampling follows structurally from
> Proposition 1 rather than being an empirical discovery; the magnitude, and
> all comparisons against the four literature baselines, are empirical.

---

## 5. The one-line test before submission

Read the abstract aloud. If a listener could come away believing **"this method
reduces missed cancers in patients"**, the abstract is still wrong — regardless
of how carefully the results section is worded.

Every claim must be traceable to the set it was measured on, in the sentence
that makes it.
