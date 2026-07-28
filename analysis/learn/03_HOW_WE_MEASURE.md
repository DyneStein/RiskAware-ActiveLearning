# 3 — How we measure things

*Read documents 1 and 2 first.*

Every measurement word your supervisor used, explained. This is the longest document but it's the
one that unlocks everything else.

---

## 3.1 The four outcomes (everything is built on this)

Take one question: **"is this melanoma?"** There are exactly four things that can happen:

|  | **Truly melanoma** | **Truly not melanoma** |
|---|---|---|
| **Model says melanoma** | ✅ True Positive | ❌ **False Positive** |
| **Model says not melanoma** | 🚨 **False Negative** | ✅ True Negative |

- **False Positive (FP)** = a false alarm. Model shouts cancer, it's a harmless mole.
  **Cost: an unnecessary check-up.** Annoying, not dangerous.
- **False Negative (FN)** = a **miss**. Model says "you're fine", it's cancer.
  **Cost: an undetected cancer.** This can kill someone.

> **The single most important asymmetry in your whole project: a false negative is vastly worse
> than a false positive.** Everything your project does is about trading more false alarms for
> fewer misses.

A table of these four counts across all diseases is called a **confusion matrix**. Your project
saves one for every round.

### False-negative rate

> **FN rate = of all the truly cancerous photos, what fraction did the model call harmless?**

Lower is better. Your project reports two versions:

- **`fn_rate_malignant`** — counts a miss only if a malignant lesion was called *benign*. If a
  melanoma is mistaken for basal cell carcinoma, that is **not** counted as a miss — both are
  malignant, so the patient still gets referred. This is the clinically sensible definition.
- **`fn_rate_melanoma`** — stricter: any melanoma not identified *as melanoma* counts as a miss.

---

## 3.2 Precision and recall

Two words that get confused constantly. Here they are for melanoma:

**Recall** (also called **sensitivity**)
> Of all the real melanomas, how many did we catch?

100% recall = caught every cancer. **Recall is the safety number.**
*(Recall and false-negative rate are two sides of one coin: recall = 100% − FN rate.)*

**Precision**
> When the model shouts "melanoma!", how often is it right?

100% precision = never a false alarm.

**They fight each other.** A model that screams "melanoma!" at every photo has perfect recall
(caught them all!) and terrible precision (endless false alarms). One that only shouts when
absolutely certain has great precision and misses lots of cancers.

**In cancer screening, recall matters more.** A false alarm costs a check-up; a miss costs a life.

### F1 score

**F1** combines precision and recall into one number, in a way that punishes you if either is bad.
You can't get a good F1 by acing one and failing the other.

Two flavours you'll see:

- **F1-macro** — average the F1 of all 7 diseases, **each counting equally**. So melanoma (209
  photos) counts exactly as much as ordinary moles (1,327). **This is the honest one for your
  imbalanced dataset.**
- **F1-weighted** — average weighted by how common each disease is, so moles dominate. Flattering
  and less informative.

When someone says "F1" in your project, they mean **F1-macro**.

---

## 3.3 AUC — the one your supervisor specifically asked for

Forget the full name ("Area Under the ROC Curve"). It tells you nothing. Remember **this sentence**:

> ## AUC = if I show the model one cancer photo and one healthy photo at random, how often does it give the cancer the higher danger score?

That's it. That's the whole concept.

| AUC | Meaning |
|---|---|
| 0.5 | Coin flip. Useless. |
| 0.7 | Poor |
| 0.9 | Good — right 90% of the time |
| 0.95 | Strong |
| 1.0 | Perfect |

**Why AUC and not just accuracy?** Because AUC doesn't care about *where you set the threshold* —
it measures whether the model *ranks* dangerous above safe. It's also unaffected by class imbalance,
which accuracy very much is not.

### Why "per lesion class"

Your supervisor asked for AUC **per class**, and he was right. One overall number would let a model
that has completely given up on melanoma still look fine, because 70% of the data is ordinary moles.

Your actual results:

| Disease | AUC | Verdict |
|---|---|---|
| Dermatofibroma | 0.996 | excellent |
| Vascular lesions | 0.995 | excellent |
| Basal cell carcinoma | 0.994 | excellent |
| Ordinary moles | 0.970 | good |
| Benign keratosis | 0.967 | good |
| Actinic keratoses | 0.963 | good |
| **Melanoma** | **0.945** | **the hardest — as expected** |

Melanoma being last is honest, expected, and worth stating plainly: it's genuinely the hardest to
tell apart from a normal mole. That's *why* the disease is dangerous.

### PR-AUC

A companion measure that is fairer when a class is **rare**. Regular AUC can look flattering when
healthy cases hugely outnumber cancers; PR-AUC doesn't let you off so easily.

Melanoma: AUC 0.945 but **PR-AUC 0.781**. That gap is the normal signature of a rare, hard class.
Reporting both is more honest than quoting only the nicer number.

### Confidence interval

You'll see AUC written as **0.945 [0.929, 0.960]**. The bit in brackets is a **confidence
interval** — "our best estimate is 0.945, but the true value is probably somewhere between 0.929
and 0.960." Document 4 explains where that range comes from.

---

## 3.4 Calibration — your supervisor's #1 request

### The idea, via weather

A forecaster says **"70% chance of rain"** on 100 different days. If they're any good, it should
rain on roughly **70** of them.

If it only rains on 40, the forecaster is **overconfident** — their numbers are inflated and you
can't act on them. Note they might still be *useful* (rainy days do get higher numbers), but the
numbers themselves don't mean what they claim.

**Calibration = do the model's stated percentages match reality?**

### Why this matters enormously for *your* project

Your entire safety mechanism is:

> *"escalate this photo if the risk score is above the threshold"*

That only makes sense if the risk score means something real. **If "0.8" doesn't actually
correspond to 80% danger, you're thresholding a meaningless number** and the safety story is
decoration.

And here's the kicker: **accuracy cannot detect this problem.** A model can be 90% accurate and
wildly overconfident at the same time. That's exactly why your supervisor asked for it separately.

### ECE — Expected Calibration Error

**One number for "how far off are the confidence claims, on average."**

How it's built: sort all predictions into buckets by confidence (all the 60–70% ones together, the
70–80% ones, etc.). In each bucket, compare *claimed* confidence to *actual* accuracy. ECE is the
average gap, weighted by bucket size.

- **0.00** = perfect
- **0.05** = "on average, stated confidence is off by 5 percentage points"
- Lower is better

**Your result: ECE ≈ 0.073.** Mean confidence 95.8%, actual accuracy 88.6% — **overconfident by
about 7 points.** This is real and worth reporting. It's also extremely common; nearly all modern
neural networks are overconfident.

**Good news within it:** your **risk head is better calibrated than the classifier** (ECE 0.056 vs
0.073). Since the escalation threshold is applied to the risk score, the number your safety
mechanism actually depends on is the more trustworthy of the two.

**MCE** — same idea but reports the *worst* single bucket instead of the average. Worst case rather
than typical case.

### Brier score

Think of it as **a golf score for probability predictions — lower is better.**

It's the squared difference between what you predicted and what actually happened, averaged over
everything.

Its nice property (the jargon is **"proper scoring rule"**) is that it punishes **both** being
wrong **and** being overconfident. So you can't cheat it by hedging everything at 50% — that scores
badly too. Your results: ~0.18 for the 7-class head, ~0.065 for the risk head (better, because
2 options is an easier job than 7).

### Reliability diagram

**The picture of the calibration check.** X-axis: what the model claimed. Y-axis: what actually
happened.

- Perfect calibration = points sit on the **diagonal line**
- Points **below** the diagonal = **overconfident** (claimed more than it delivered)
- Points **above** = underconfident

Yours sag below the line — the visual version of that ECE number.
See `analysis/rigor/figures/17` and `18`.

### Temperature scaling — the fix

The standard remedy. You divide the raw logits by one single number **T** before the softmax,
which "cools down" over-hot confidence. It changes the percentages but **not** which disease is
predicted — so accuracy is untouched.

We fit T on half the test set and checked it on the other half (so it's a fair test, not
self-graded). **T ≈ 2.16, and held-out ECE dropped from 0.073 to 0.023** — about a 67% improvement
from one single number.

That's a great thing to report: **we found the problem AND showed it's cheaply fixable.**

---

## 3.5 Two safety numbers that measure different things

Your project has two safety metrics and it's important not to mix them up.

**`unsafe_auto_accepts`** — measured on the **pool**, every round
> Cancerous photos the system waved through without a doctor looking.

This measures the **escalation decision itself**. It's direct: it counts exactly what the policy
did or didn't catch. **This is the metric that improved dramatically (−43%).**

**`fn_rate_malignant`** — measured on the **test set**, at the end
> Of all cancers in the locked-away test set, what fraction did the final model call harmless?

This measures the **trained model's behaviour**. It's indirect — it depends on what the model
learned from whichever photos happened to get labelled. Noisier. **This did not improve
significantly.**

Both matter. But if you only get to cite one number for "the risk score changed the system's
behaviour", `unsafe_auto_accepts` is the mechanically honest one, because it measures the thing the
risk score directly controls.

Document 5 explains *why* one improved and the other didn't — and the explanation is one of the
more interesting findings in your work.

---

## The five things to remember

1. **False negative = a missed cancer.** Far worse than a false alarm. Everything trades one for
   the other.
2. **Recall = of all real cancers, how many did we catch.** This is the safety number.
   **F1-macro** treats all 7 diseases equally, which is the honest choice for imbalanced data.
3. **AUC = show it one cancer and one healthy photo; how often does it rank the cancer higher?**
   0.5 is a coin flip, 1.0 is perfect. Yours for melanoma: 0.945.
4. **Calibration = when it says 90%, is it right 90% of the time?** Yours is **overconfident by
   ~7 points** (ECE 0.073), but one number (temperature ≈ 2.16) fixes two-thirds of it.
5. **`unsafe_auto_accepts` measures the decision; `fn_rate` measures the model.** The first
   improved a lot, the second didn't.

---

➡️ **Next: `04_STATISTICS.md`** — how we know a result is real and not luck.
