# Everything explained in plain English (reference sheet)

> ## 🛑 Don't start here.
> This page assumes you already know roughly what a model, a test set and a probability are.
>
> **If you're starting from zero, go to [`learn/00_START_HERE.md`](learn/00_START_HERE.md)** — a
> six-part course that builds up from "what is machine learning" to your results, in order, with
> nothing assumed. About 1 hour 40 minutes total.
>
> **Come back here afterwards.** This page is the quick-reference version, and **Part 6 at the
> bottom is the draft reply you can send your supervisor.**

Every term the supervisor used, and every term used in the analysis: what it means, why anyone
cares, and what this project actually found.

---

# Part 1: What your project actually does (the 60-second version)

You have an AI that looks at a photo of a skin lesion (a mole, a spot, a lump) and tries to say
which of 7 diseases it is. One of those 7 is **melanoma** — a skin cancer that kills people if
it's missed.

Normally you'd need a doctor to label thousands of photos to train such an AI, which is
expensive. **Active learning** is the trick where the AI looks at unlabelled photos and says
"I'm not sure about *this* one, please have a doctor label it" — so you only pay for the labels
that actually teach it something.

The standard way to pick which photos to send to the doctor is: **send the ones the AI is most
confused about.** That's called uncertainty sampling.

**Your project's idea:** that's not good enough for medicine. Being confused isn't the only
reason to ask a doctor. Sometimes the AI is *totally confident* — and *wrong* — about something
dangerous. A confident mistake on a melanoma is the worst possible outcome, and confusion-based
selection will never catch it, because there was no confusion.

So you added a **second signal**: how *dangerous* this case is, regardless of how confident the
AI feels. Then you send a photo to the doctor if **either** the AI is confused **or** the case
looks dangerous. That's the **dual-metric policy** — "dual" = two signals, "metric" = measurement.

That's the whole paper. Everything else is proving it works.

---

# Part 2: The words from YOUR project

### Backbone
The big shared part of the neural network that looks at the image and turns it into a list of
numbers describing what it sees (texture, colour, shape, borders). Think of it as the **eyes**.
You used three different well-known backbones: ResNet-50, DenseNet-169, EfficientNet-B4. They're
just three different brands of eyes.

### Head
A small piece bolted onto the end of the backbone that turns "what I see" into an actual answer.
Think of it as a **mouth**. The backbone sees; the head speaks.

### Two-head design (this is your architecture)
**One pair of eyes, two mouths.**

- **Mouth 1 — the classification head:** says *"this is melanoma"* (picks one of the 7 diseases).
- **Mouth 2 — the risk head:** says *"this is dangerous"* (just a yes/no on malignant, plus how sure).

**Why two mouths instead of one?** The original design had only the classification mouth, and
calculated danger by adding up the probabilities of the three dangerous diseases. The problem:
if the classification mouth is confidently wrong — says "harmless mole, 99%" about a melanoma —
then the danger number computed *from* that is automatically wrong too. They fail together.
Giving danger its own separate mouth means it *can* disagree with the diagnosis.

(Whether it actually *does* disagree enough is one of our findings — see Part 5, finding 3.)

### Logits → softmax → probabilities
The network's raw output is a list of meaningless-scale numbers called **logits** (e.g. 4.2, -1.3, 0.8).
**Softmax** is the maths that squashes them into percentages that add up to 100%
(e.g. 91%, 2%, 7%). Those percentages are the **probabilities**. That's all softmax is: a
converter.

### Uncertainty (4 flavours)
A number for "how confused is the AI". You tested four ways to measure it:
- **Entropy** — how spread out the 7 percentages are. All on one disease = certain. Spread evenly = confused.
- **Least confidence** — just `1 − (the biggest percentage)`. If the top guess is only 40%, uncertainty is 0.6.
- **Margin** — how close the top two guesses are. If it's torn between melanoma 45% and mole 44%, that's confused.
- **MC-dropout** — run the same photo through the network 30 times, randomly switching off different
  bits of the network each time. If the answer keeps changing, it's not confident. (This is why
  MC-dropout runs were 2.4× slower — 30 passes instead of 1.)

### Risk score
A number 0 to 1 from the risk head = *"probability this lesion is malignant."* 0.9 means
"90% sure this is dangerous."

### Escalate / auto-accept
- **Escalate** = send it to the doctor for a real label.
- **Auto-accept** = don't bother the doctor, just take the AI's answer.

### Oracle
The fake doctor. Since you already have the true labels in the dataset, "asking the doctor"
just means looking up the real answer. Standard practice — it's free, ethical, and reproducible.

### Pool / labelled / unlabelled / test set
- **Pool** (8,110 photos) — the photos the AI is allowed to ask about.
- **Labelled set** — the ones it has already asked about and now knows the answers to. Starts at 490, grows.
- **Unlabelled set** — the rest of the pool, still unknown.
- **Test set** (1,905 photos) — locked in a vault. Never trained on, never asked about. Used **only**
  to grade the AI at the end. This is what stops you fooling yourself.

### Round
One cycle: train the AI → let it look at the unlabelled pool → escalate some to the doctor →
add those to the labelled set → repeat. You did **15 rounds** per experiment.

### Threshold, and "recalibrated every round"
A **threshold** is the bar a score has to clear to trigger escalation. You don't use a fixed
number — every round you take the **90th percentile** of the current scores as the bar
(i.e. "the top 10% most uncertain / most dangerous"). Why: as the AI improves, its scores shrink.
A bar set against a dumb early model would soon be so high that nothing ever clears it, and
escalation would silently stop happening. Re-setting the bar every round keeps it meaningful.

### Top-K budget (K = 150)
"Always send at least the 150 most-confused photos each round, even if none clear the bar."
It's a **floor, not a ceiling** — if 300 photos clear the bar, all 300 go.

### Uncapped risk route
The danger signal has **no** budget limit. If 500 photos look dangerous, all 500 get escalated.
The reasoning: never skip a dangerous case just because you'd already used up your quota that round.

### Unsafe auto-accepts ← **your most important number**
The count of photos that were **genuinely dangerous** (true label is melanoma / BCC / actinic keratoses)
that the system **waved through without a doctor looking**. Every one of these is a potential missed cancer.
Lower = safer. This is the number that improved the most in your results.

### False-negative rate (FN rate)
Of all the truly-cancerous photos in the test set, what fraction did the AI call harmless?
A "false negative" = "you're fine" when you're not. In cancer this is the dangerous direction
of error — a false *positive* just means an unnecessary check-up.

### Seed
A number that fixes all the randomness so a run repeats identically. You used seed 42 for everything.
**This matters a lot** — see "multi-seed" in Part 4.

---

# Part 3: What your supervisor asked for — every term decoded

## 3.1 "Calibration analysis"

**The plain idea:** when the AI says it's 90% sure, is it actually right 90% of the time?

Think of a weather forecaster. If they say "70% chance of rain" on 100 days, it should rain on
about 70 of them. If it only rains on 40, the forecaster is **overconfident** — their numbers are
inflated and you can't trust them.

**Why your supervisor cares:** your entire safety mechanism is *"escalate if risk score > threshold."*
That only makes sense if the risk score means something real. If "0.8" doesn't actually correspond to
80% danger, you're thresholding a meaningless number and the safety story is decoration. Accuracy
cannot detect this problem — a model can be 90% accurate and wildly overconfident at the same time.

### ECE — Expected Calibration Error
**One number for "how badly are the confidence numbers off, on average."**

How it's built: sort predictions into buckets by confidence (all the 60-70% ones together, etc.).
In each bucket compare *claimed* confidence to *actual* accuracy. ECE is the average gap.

- 0.00 = perfect
- 0.05 = "on average, stated confidence is off by 5 percentage points"
- Lower is better.

**What we found: ECE ≈ 0.072.** Mean confidence was **96%** but actual accuracy was **88.6%** — so
your models are **overconfident by about 7 points.** That's a real, reportable weakness. Very
common — almost all modern neural networks are overconfident.

Good news within that: the **risk head is better calibrated than the classifier** (ECE 0.055 vs
0.072). Since the escalation threshold is applied to the risk score, the number your safety
mechanism actually depends on is the more trustworthy of the two.

### Brier score
**Like a golf score for probability predictions — lower is better.**

It's the squared difference between what you predicted and what actually happened, averaged.
The nice property (jargon: "proper scoring rule") is that it punishes **both** being wrong **and**
being overconfident, so you can't cheat it by hedging everything at 50%.

**What we found:** ~0.18 for the 7-class head, ~0.065 for the risk head (risk head does better —
it has an easier job, only 2 options instead of 7).

### Reliability diagram
**The picture of the calibration check.** X-axis = what the AI claimed. Y-axis = what actually
happened. Perfect calibration = a straight diagonal line. Your curve sagging *below* the diagonal
= overconfident.

**What we found:** figures 17 and 18 — sags below the line, confirming the ECE number visually.

### Temperature scaling (bonus we added)
The standard fix. You divide the raw scores by one single number T before the softmax, which
"cools down" over-hot confidence. We fit T on half the test set and checked it on the other half.

**T ≈ 2.15, and it cut held-out ECE from 0.0737 to 0.0245** — a ~67% improvement from one single
number. This is a great thing to tell your supervisor: *we found the problem AND showed it's
cheaply fixable.*

---

## 3.2 "Active Learning Efficiency: plot Accuracy vs Number of labeled samples"

**The plain idea:** doctor time is the expensive thing. So the fair question isn't "who's more
accurate after 15 rounds" — it's **"for the same number of doctor labels, who's more accurate?"**

**Why he asked:** he spotted that your two methods don't spend the same amount. Your dual-metric
policy has that uncapped risk route, so it asks for more labels. Plotting against *rounds* hides
that — it would look better partly just for asking more questions. Plotting against *labels spent*
removes the advantage.

**What we found — and this one is important:**
Dual-metric spent **382 more labels** on average. When we compare both at the *same* label budget,
dual-metric is **0.35 percentage points behind** on accuracy, and needs about **300 more labels**
to reach any given accuracy target.

**So: your method is NOT more label-efficient. It's slightly less.**

This is not a disaster — it's the correct, defensible story. Your method deliberately spends labels
on cases that are *dangerous* rather than cases that are *informative*, and those aren't the same
photos. It buys **safety**, not efficiency. Saying otherwise is the first thing a reviewer would
check, and it wouldn't survive.

---

## 3.3 "AUC per lesion class, especially melanoma"

### What AUC actually means (the clearest way to think about it)
**AUC = "if I show the AI one cancer photo and one healthy photo at random, how often does it give
the cancer the higher danger score?"**

- **0.5** = coin flip. Useless.
- **0.9** = gets it right 90% of the time. Good.
- **1.0** = perfect.

The full name is "Area Under the ROC Curve", which tells you nothing useful, so just remember the
sentence above.

**Why "per class":** your dataset is **67% ordinary moles**. An AI that gave up entirely on the
rare diseases would still score high overall accuracy. One combined number hides exactly the
failure that matters. Melanoma is only ~11% of the data and it's the one that kills people —
so it needs its own number.

**What we found:**

| Lesion | AUC | 95% CI | Notes |
|---|---|---|---|
| Dermatofibroma | 0.996 | [0.987, 1.000] | excellent |
| Vascular lesions | 0.995 | [0.983, 1.000] | excellent |
| Basal cell carcinoma | 0.994 | [0.987, 0.998] | excellent |
| Ordinary moles | 0.970 | [0.962, 0.977] | good |
| Benign keratosis | 0.967 | [0.954, 0.977] | good |
| Actinic keratoses | 0.963 | [0.936, 0.984] | good |
| **Melanoma** | **0.945** | **[0.929, 0.960]** | **the hardest class — as expected** |

Melanoma being lowest is honest and expected; it's genuinely the hardest to distinguish from a
normal mole. 0.945 is still strong.

### PR-AUC (we also report this)
A companion measure that's fairer when one class is rare. Regular AUC can look flattering when
there are far more negatives than positives; PR-AUC doesn't. Melanoma PR-AUC = **0.781** — the gap
between 0.945 and 0.781 is the normal signature of a rare, hard class, and it's more honest to
report both than to quote only the ROC number.

---

## 3.4 "Statistical tests: p-values and confidence intervals"

**The plain idea:** you got a better number. Was that real, or luck?

### p-value
**"If there were genuinely no difference at all, how often would pure luck produce a gap this big?"**

- p = 0.30 → happens by luck 30% of the time. Means nothing.
- p = 0.05 → happens by luck 1 time in 20. The usual "probably real" line.
- p = 0.0005 → 1 time in 2,000. Very likely real.

**Smaller = more convincing.** "Statistically significant" conventionally means p < 0.05.

### Confidence interval (CI)
**The range the true value probably sits in.** "+2.4 percentage points, 95% CI [+0.4, +4.4]"
means: best guess +2.4, but plausibly anywhere from +0.4 to +4.4.

**The trick for reading them:** if the interval **crosses zero**, you can't rule out "no difference
at all". If it stays entirely on one side of zero, the effect is real.

### The tests we used, and why
- **Wilcoxon signed-rank** — compares paired results without assuming a bell curve. Safe for small
  samples like our 12 pairs.
- **McNemar's test** — the correct test when both methods are graded on the *exact same photos*.
  It only looks at photos where the two methods disagreed, because photos they both got right tell
  you nothing about which is better.
- **Bootstrap** — resample your test set thousands of times with replacement and see how much the
  answer wobbles. That wobble *is* your confidence interval. It's a computational shortcut that
  avoids needing scary formulas.
- **Holm-Bonferroni correction** — if you run 12 tests at p<0.05, roughly one will "pass" by pure
  luck. This adjusts for that. **Always quote the corrected p-value**; not doing so is a classic
  reviewer complaint.
- **Effect size** — p-value says "is it real?", effect size says "is it *big enough to care about*?"
  You need both. A tiny difference can be statistically significant with enough data and still be
  clinically pointless.

### ⚠️ The honest limitation we have to declare
You ran **one seed** per configuration. Same starting randomness, once. The textbook way to prove
something isn't luck is to run it **5 times with 5 different seeds** and show it holds every time.
We can't do that from existing data — it needs new GPU runs.

So we ran the two tests the data *does* support, and labelled exactly what each can and can't conclude.
That's the honest move, and it's better than faking the ideal test.

**What we found:**

| What | Result | Verdict |
|---|---|---|
| Unsafe auto-accepts | −4,030 (12 of 12 configs better) | **p = 0.003 — significant** ✅ |
| Extra labels used | +382 (12 of 12) | **p = 0.003 — significant** ✅ |
| Accuracy | +0.60 pp | p = 0.014 — significant ✅ (but see 3.2) |
| F1-macro | +0.86 pp | p = 0.23 — **not significant** ❌ |
| Missed-cancer rate | −1.19 pp | p = 0.15 — **not significant** ❌ |
| Melanoma recall | +2.43 pp | p = 0.057 — **borderline** ⚠️ |

Read that honestly: **the safety result and the cost are rock solid. The clinical-outcome results
are promising but not proven.** Don't claim the bottom three until you have multiple seeds.

### One nuance that looks like a contradiction but isn't

We ran the tests two ways and got two different-looking answers:

- **Across the 12 configurations:** accuracy is up +0.60pp, *significant* (p = 0.014).
- **Within each configuration separately** (comparing the two methods photo-by-photo on the same
  1,905 test images): only **1 of 12** is significant.

Both are correct, because they ask different questions. Individually, each gap is smaller than the
random wobble you'd get from having only 1,905 test photos — you can't resolve it. But **11 of the
12 gaps point the same direction.** One coin landing heads proves nothing; eleven of twelve coins
landing heads is real evidence.

So the honest sentence is: *"the accuracy effect is small, positive, and consistent in direction,
but not individually resolvable per configuration at this test-set size."* The safety effect, by
contrast, is big enough to show up clearly at every level.

*(F1-macro = an average of precision and recall treating all 7 diseases as equally important —
so rare diseases count as much as common ones. "Recall" for melanoma = of all real melanomas,
what fraction did we catch.)*

---

## 3.5 "Runtime: training time, inference time, query time"

**The plain idea:** how long does it take, and where does the time go? Reviewers want to know
if your method is practical.

- **Training time** — teaching the model on the labelled photos.
- **Inference time** — how long to make one prediction on one photo (matters for deployment:
  can a clinic use this live?).
- **Query time** — how long to score the whole unlabelled pool and decide who to escalate.
  This is the overhead *your method specifically adds*.

**What we found:** 94.1 GPU-hours total for all 24 experiments. MC-dropout runs took **2.42×**
longer than the others (30 forward passes instead of 1).

Where each round's time actually goes:

| Method | Training | Querying (picking who to escalate) | Grading |
|---|---|---|---|
| entropy / margin / least-confidence | ~94% | ~4% | ~1.6% |
| **MC-dropout** | **59.5%** | **39.8%** | 0.7% |

So for three of the four methods, **your active-learning step is nearly free** — training eats
almost everything. MC-dropout is the exception: picking who to escalate swallows 40% of every
round. Combined with the ablation (MC-dropout buys no extra safety), that's a solid practical
argument for dropping it.

**The escalation rule itself is basically instant** — 8.4 milliseconds per round for dual-metric
vs 4.5 for the baseline, on a 6,000-photo pool. So your method costs *doctor labels*, not compute.

**A nice self-check:** the benchmark measured MC-dropout at 29.9–30.7× the cost of normal
inference — and MC-dropout is defined as exactly 30 passes. Recovering a known number you didn't
feed in is how you know a measurement method is trustworthy.

**One thing worth telling him** (it shows real rigor): the obvious way to split training vs query
time is a regression on the logged round times. **It's mathematically impossible here.** Your pool
is closed — every photo the doctor labels leaves the unlabelled pool and joins the labelled set —
so `labelled + unlabelled = 8,110` **exactly, every round.** The two quantities are perfectly
locked together, so no maths can separate their costs. It gave *negative* query times, which is
how we caught it. We measured each component directly instead.

---

## 3.6 "Ablation studies"

**The plain idea:** remove one ingredient and see if the cake still works.

You claim two signals are better than one. An ablation proves it by testing: uncertainty alone,
risk alone, both together, and a random baseline. If "both" isn't better than each alone, your
combination isn't doing anything.

**What we found — this is your best result:**

| Method | Dangerous cases caught | Doctor labels spent |
|---|---|---|
| Random picking | 10.2% | 6,201 |
| **Uncertainty only (the standard approach)** | **12.6%** | 4,339 |
| Risk only | 17.2% | **1,917** |
| **Both together (yours)** | **29.3%** | 6,201 |

Three things jump out:

1. **The standard approach is barely better than picking at random** (12.6% vs 10.2%) at catching
   danger — despite spending 4,339 labels. Uncertainty sampling was never designed to be a safety
   mechanism, and this proves it isn't one. **This is your motivation, in one line.**
2. **Risk-only is incredibly cheap** — 17.2% caught for only 1,917 labels, less than half the
   baseline's spend. Worth mentioning as a budget option.
3. **12.6 + 17.2 = 29.8 ≈ 29.3.** The two signals are almost perfectly **additive**, which means
   they're catching **different photos**. That's the proof that both are needed — exactly what an
   ablation is for.

### Threshold sweep
We also turned the risk threshold dial from strict to permissive and traced the trade-off. That
gives a curve showing "spend more labels → catch more danger", so a hospital could pick its own
operating point. The standard approach sits *below* this curve — meaning it's beaten at every price.

---

## 3.7 "Formal mathematical definitions of the dual-metric policy"

**The plain idea:** right now your method is described in words and code. For a journal, it has to
be written in maths, precisely enough that a stranger could re-implement it exactly.

**What we did:** `RiskAware-ActiveLearning/METHODS.md`, written by reading your actual code (not the
docs — they'd drifted). It defines everything in symbols, and proves two small things:

- **Proposition 1:** dual-metric escalates a *superset* of what uncertainty-only escalates
  (it does everything the baseline does, plus more). Therefore it **mathematically cannot** be less
  safe, at fixed scores. This isn't a lucky result, it's arithmetic — and it explains why the
  improvement showed up in 24 out of 24 experiments with zero exceptions.
- **Proposition 2:** the extra cost is exactly the set of photos the risk route flags that the
  uncertainty route missed. Never more.

Together these say your method is a **controlled trade**: strictly more safety for strictly more
labels, with a dial to set the exchange rate. That's a much better story than "our thing is better".

---

## 3.8 "Robustness experiments"

**The plain idea:** real clinics don't have perfect photos. Does it still work if the image is
blurry, noisy, badly lit, or heavily compressed?

We take the finished model and feed it deliberately degraded versions of the same test photos —
no retraining, just a worse image, like the real world would give it.

**The interesting bit for your project:** you have two heads, so we can ask which one breaks first.
If the *diagnosis* degrades but the *danger signal* survives, that's actually ideal behaviour —
the system gets unsure what the disease is while still knowing it's dangerous, so it escalates
instead of silently guessing wrong.

**What we found — three things:**

**1. Good news: the danger signal survives better than the diagnosis.** Averaged over the
degradations, accuracy keeps 85.2% of its clean value but the risk-head AUROC keeps **90.3%**.
That's exactly the behaviour you want — the system gets unsure *what* the disease is faster than it
loses the sense that it's *dangerous*, so it escalates rather than confidently guessing wrong.
This is a win for your two-head design that the plain AUC comparison didn't give it.

**2. Bad news: EfficientNet-B4 falls apart under mild noise.**

| Accuracy | Clean | With slight sensor noise |
|---|---|---|
| DenseNet-169 | 0.889 | 0.692 |
| ResNet-50 | 0.896 | 0.710 |
| **EfficientNet-B4** | **0.863** | **0.008** |

0.008 is *below random guessing* (which would be about 0.14) — it collapses to predicting basically
one wrong class for everything. It's not a bug in our corruption code: the exact same noise applied
to the other two models gives the sensible ~20-point drop you'd expect. It's specific to that
architecture. **Report this per-model** — quoting only the average (52.7%) would hide it completely.

**3. Melanoma detection is fragile.** Melanoma recall goes 0.70 clean → 0.31 with blur → 0.08 with
noise. Even for the two robust models, a slightly out-of-focus camera costs most of your melanoma
detection. That's a Limitations-section point: your safety results are established on clean,
curated photos.

**Status:** 4 of 5 corruptions done (JPEG compression was still running when the job stopped).

---

## 3.9 "Explainability analysis"

**The plain idea:** *where was the model looking?* Skin-lesion datasets are infamous for
"shortcuts" — rulers, pen marks, hair, dark corners from the camera lens. A model can score well by
learning "photos with rulers are cancer" instead of learning actual medicine. That model will fail
catastrophically in a real clinic.

**Grad-CAM** is the standard tool: it produces a heat-map over the photo showing which pixels drove
the decision. Red = "this is what convinced me."

**What we found (figure 28):** because you have two heads, we made heat-maps for *both* — "why did
you say melanoma?" next to "why did you say dangerous?". And it found a genuinely great example:

> A melanoma the classifier got **wrong** (called it a benign keratosis) — the heat-map shows the
> classifier was looking at the **dark corners of the image**, a camera artefact. Meanwhile the risk
> head was looking at the **actual lesion** and gave it a danger score of 0.83, which would have
> escalated it to a doctor.

That single picture is your whole paper's argument. Put it in the paper.

We also included an honest failure: one melanoma where *both* heads looked at the corners and both
missed it.

---

## 3.10 "External validation on ISIC2019/2020"

**The plain idea:** your model was trained and tested on HAM10000 — one dataset, from a limited set
of hospitals and cameras. Does it work on **someone else's** photos? This is the strongest evidence
that a medical AI actually generalises, and reviewers increasingly demand it.

### ⚠️ THE TRAP — tell your supervisor about this
**ISIC 2019 literally contains HAM10000 inside it.** The ISIC 2019 dataset was built by combining
three sources: BCN20000 + **HAM10000** + MSK. The HAM10000 photos keep their original `ISIC_xxxxxxx`
filenames.

**So if you just download ISIC 2019 and test on it, you're testing on your own training data.**
The scores would look great and be completely meaningless — and a reviewer who knows the datasets
would catch it instantly. This is a serious, credibility-destroying mistake, and it's an easy one
to make.

Our script **always** removes every overlapping photo by filename, reports how many it removed, and
refuses to continue unless the overlap is zero.

### The recommendation
**Use ISIC 2020 as the main external test.** It's a different challenge year, different patients,
no overlap with HAM10000 — genuinely independent. And conveniently its labels are just
benign/malignant, which is **exactly** what your risk head predicts. So it tests your core claim
directly.

**Expect the numbers to drop.** Different cameras, different hospitals, different patient
populations — performance always falls. That's the normal, reportable result. The question a
reviewer asks is *how much*, and whether the safety signal falls faster than the diagnosis.
We wrote that expectation down *before* running, so nobody can quietly re-spin a bad result later.

**Status:** code ready. Needs a ~3 GB download, best done in Colab.

---

# Part 4: A few more terms you'll hear

- **Multi-seed** — running the same experiment several times with different randomness, to prove
  the result isn't a fluke. You have 1 seed; the gold standard is 5. This is the single biggest
  remaining gap in your rigor.
- **Baseline** — the standard existing method you're comparing against. Yours is uncertainty-only.
- **Held-out** — data locked away and not used for training, so grading is honest.
- **Generalisation** — does it work on data it's never seen.
- **Overfitting** — memorising the training photos instead of learning the disease. Shows up as
  great training scores and poor test scores.
- **Class imbalance** — 67% of your data is ordinary moles. Left alone, models just learn to always
  guess "mole". Your project fights this with **class weights** (making mistakes on rare diseases
  count for more in the training penalty).
- **Precision vs recall** — *precision*: when it says cancer, how often is it right. *Recall*: of all
  real cancers, how many did it catch. In cancer screening **recall matters more** — a false alarm
  costs a check-up, a miss costs a life.
- **Proper scoring rule** — a scoring method that can't be gamed by hedging. Brier is one.
- **Prevalence / base rate** — how common something is. Malignant cases are 18.3% of your test set.

---

# Part 5: What we actually found — the honest summary

### ✅ The strong result
Adding the risk signal cuts **unsafe auto-accepts** (dangerous photos waved through without a
doctor) by **43%**, in **12 out of 12** configurations, p = 0.003 after correction. The risk score
on its own scores **0.96 AUC** at telling dangerous from safe. This is solid and defensible.

### 💰 The price
**+382 more doctor labels** (~9% more), also in 12 of 12 configurations, equally significant.
The safety is bought, not free.

### ⚠️ Three uncomfortable findings that actually make the paper better

**1. It's not more label-efficient.** Matched at equal labels, it's slightly *behind* on accuracy.
Frame it as a safety intervention with a known price, not an efficiency gain.

**2. The clinical outcome didn't significantly improve.** The test-set missed-cancer rate went down
but not significantly (p = 0.15). Don't claim it yet.

**3. The two-head redesign is roughly a tie with the old method** on overall danger-ranking
(0.9520 vs 0.9524). It wins only where it was designed to — on cases the classifier gets wrong,
and on missed cancers it still flags **5.6% vs 0.6%** (about 9× better, but still a small
fraction).

**Why 2 and 3 are actually one finding:** the two heads **share a backbone** (same eyes). So when
the eyes are fooled — bad lighting, weird angle, an unusual-looking lesion — *both* mouths are
fooled together. The safety net is correlated with the thing it's supposed to catch. That's why
unsafe auto-accepts improved hugely (the risk route catches lots of dangerous photos uncertainty
would miss) while the missed-cancer rate barely moved (the specific photos the classifier gets
wrong are also the ones the risk head gets wrong).

**This is a genuinely good finding.** It explains your results mechanistically, it's honest, and it
points straight at the obvious next design: give the risk head its own backbone, or train it
differently, so the two can fail independently. Reviewers respect this far more than "everything
worked perfectly".

---

# Part 6: What to actually say to your supervisor

You can send something like this:

> Thanks — we've now added most of these. Summary:
>
> **Calibration:** ECE, Brier, and reliability diagrams are done for both the classification head
> and the risk head. The models are overconfident (96% mean confidence vs 88.6% accuracy,
> ECE ≈ 0.072); the risk head is the better-calibrated of the two (ECE 0.055), which matters since
> that's the score escalation thresholds. We also showed temperature scaling fixes most of it —
> a single parameter T ≈ 2.15 drops held-out ECE from 0.074 to 0.025.
>
> **AL efficiency:** plotted against labels consumed, not rounds. Important finding: at a matched
> annotation budget the dual-metric policy is **not** more label-efficient — it's ~0.35pp behind on
> accuracy and needs ~300 more labels to hit the same target. It buys safety, not efficiency, and
> we've reframed the claim accordingly.
>
> **Per-class AUC:** all 7 classes with 95% bootstrap CIs. Melanoma is the hardest at 0.945
> [0.929, 0.960] (PR-AUC 0.781); the risk head scores 0.952 for malignant-vs-benign.
>
> **Statistics:** Wilcoxon signed-rank across the 12 paired configurations plus McNemar's exact test
> on the shared test set, with Holm-Bonferroni correction and bootstrap CIs. The safety result
> (−43% unsafe auto-accepts, 12/12 configurations) and the label cost (+382, 12/12) are both
> significant at p = 0.003. F1, missed-cancer rate and melanoma recall are **not** significant —
> we're not claiming those until we have multiple seeds.
>
> **Runtime:** 94.1 GPU-hours total; MC-dropout is 2.42× the others. Component-level timing shows
> training accounts for ~94% of each round for three of the four uncertainty methods, but querying
> rises to ~40% for MC-dropout. The escalation rule itself costs ~8 ms per round on a 6,000-image
> pool, i.e. the uncapped risk route is essentially free in compute — its cost is annotation.
>
> **Ablations:** uncertainty-only catches only 12.6% of high-risk pool images versus 10.2% for
> random selection, risk-only catches 17.2% at a third of the label cost, and the two combined reach
> 29.3% — almost exactly additive, which shows the two signals flag different images.
>
> **Formal definitions:** written up in METHODS.md, including a proof that the dual-metric
> escalation set is a superset of the baseline's, so it cannot be less safe at fixed scores.
>
> **Explainability:** Grad-CAM for both heads, including a case where the classifier was misled by
> a camera artefact and the risk head correctly flagged the lesion at 0.83.
>
> **Robustness:** all three backbones re-evaluated on noise-, blur-, brightness- and
> contrast-degraded versions of the test set. The risk head degrades more slowly than the
> classifier (90.3% vs 85.2% of clean performance retained), which supports the two-head design.
> Two cautions though: EfficientNet-B4 collapses entirely under mild Gaussian noise (accuracy
> 0.86 → 0.008, below chance, while ResNet-50 and DenseNet-169 only drop to ~0.70), and melanoma
> recall falls from 0.70 to 0.31 under defocus blur. Both go in Limitations.
>
> **Still outstanding:** external validation and multi-seed replication. On external validation — note that ISIC 2019 contains HAM10000, so testing on it
> directly would be train-set leakage; we're excluding the overlap by image ID and would suggest
> ISIC 2020 as the primary external set since it's independent and its binary labels match our risk
> head. For multi-seed, the full matrix at 5 seeds is ~470 GPU-hours, so we'd suggest 3–4
> configurations × 5 seeds (~60 hours) targeting the safety result specifically.

**Two things that will impress him if you mention them:**
1. You caught that ISIC 2019 overlaps HAM10000 (many people don't).
2. You found and reported results that *don't* favour your method, instead of only the good ones.
   That's what separates a study from a sales pitch.
