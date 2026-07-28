# 4 — Statistics: how we know a result is real

*Read documents 1–3 first.*

---

## 4.1 The problem: "our number is bigger" proves nothing

Suppose your method scores 89.1% and the baseline scores 88.5%. You're 0.6 points ahead. Did you
win?

**You have no idea yet.** That gap could be:

- a real improvement, **or**
- pure luck in which photos landed in the test set, **or**
- pure luck in how the training randomness fell that day.

Flip a coin 10 times, get 6 heads. Is the coin biased? Obviously not — 6/10 happens all the time by
chance. Statistics is just the machinery for asking that question properly.

**This is the single biggest difference between "I did an experiment" and "I did research."**

---

## 4.2 p-value

> ## A p-value answers: "if there were genuinely NO difference at all, how often would pure luck alone produce a gap this big?"

| p-value | Means | Verdict |
|---|---|---|
| 0.50 | happens by luck half the time | nothing |
| 0.30 | happens by luck 3 times in 10 | nothing |
| **0.05** | 1 time in 20 | **the usual "probably real" line** |
| 0.01 | 1 time in 100 | convincing |
| 0.0005 | 1 time in 2,000 | very convincing |

**Smaller = more convincing.** Below 0.05, people say the result is **statistically significant**.

Three things that trip everyone up:

- **p is not "the chance my method is better."** It's the chance of seeing a gap this big *if there
  were no difference*. Subtly but importantly different.
- **"Not significant" does not mean "no difference."** It means *you don't have enough evidence to
  tell.* Absence of proof isn't proof of absence.
- **0.05 is an arbitrary convention.** p = 0.049 and p = 0.051 are basically the same evidence.
  Don't treat it as a magic wall.

---

## 4.3 Confidence interval

A p-value tells you *whether* there's an effect. A **confidence interval** tells you **how big**
it probably is — which is usually the more useful question.

> **"+2.4 percentage points, 95% CI [+0.4, +4.4]"**
> means: best guess +2.4, but the true value is plausibly anywhere from +0.4 to +4.4.

**The trick for reading them instantly:**

- Interval **crosses zero** (e.g. `[−1.2, +3.4]`) → can't rule out "no difference at all" ❌
- Interval **entirely on one side** (e.g. `[+0.4, +4.4]`) → the effect is real ✅

Confidence intervals are generally **more informative than p-values** because they show the size of
the effect, not just its existence. Report them wherever you can.

---

## 4.4 The tests we used, and why each one

You don't need to be able to perform these. You need to recognise the names and know why each was
chosen.

### Wilcoxon signed-rank test
Compares **paired** results without assuming the data forms a bell curve. "Paired" means each
result has a natural partner — here, each of your 12 configurations was run under *both* policies,
so the two results are a matched pair.

**Why this one:** with only 12 pairs, tests that assume a bell curve are unreliable. Wilcoxon
works on ranks instead, so it doesn't need that assumption. It's the standard safe choice at small
sample sizes.

### McNemar's test
The correct test when **both methods are graded on the exact same items.** All 24 of your
experiments used the identical 1,905 test photos, so for any pair we can compare them
photo-by-photo.

**The clever part:** it *ignores* every photo both methods got right and every photo both got
wrong. Those tell you nothing about which is better. It only looks at the photos where the two
**disagreed**. That's where all the information is.

### Bootstrap
A brilliantly simple trick for getting a confidence interval without scary formulas:

1. Take your 1,905 test photos.
2. Randomly draw 1,905 of them **with replacement** (so some appear twice, some not at all).
3. Compute your score on that fake sample.
4. Do this 2,000 times.
5. Look at the spread of the 2,000 answers. **That spread is your confidence interval.**

It's essentially asking "what if I'd happened to collect a slightly different test set?" —
simulated by resampling the one you have.

### Holm-Bonferroni correction
**This one really matters.** If you run 12 tests each at the p < 0.05 level, then roughly **one of
them will look significant by pure luck**, even if nothing is going on. Run enough tests and you'll
always find something.

Holm-Bonferroni adjusts the p-values to account for how many tests you ran.

> **Always quote the corrected p-value.** Not doing so is one of the most common — and most
> criticised — mistakes in applied research papers.

### Effect size
A p-value says *"is it real?"* An effect size says *"is it big enough to care about?"*

You need both. With enough data, a completely trivial difference becomes "statistically
significant" — but it might be clinically worthless. A significant improvement of 0.01% in cancer
detection helps nobody.

---

## 4.5 Your project's honest limitation: one seed

Remember from document 1: a **seed** locks the randomness so a run repeats identically. You used
seed 42 for everything, and ran each configuration **once**.

**The textbook approach** is to run each configuration **5 times with 5 different seeds** and check
your result holds every time. That directly measures "how much does this wobble just from
randomness?"

**You can't do that from the data you have.** It would need new GPU runs (roughly 470 GPU-hours for
the full matrix — the practical plan is 3–4 configurations × 5 seeds, about 60 hours, aimed at the
safety result specifically).

**So what did we do instead?** Rather than fake the ideal test, we ran the two tests your existing
data genuinely supports, and labelled exactly what each can and cannot conclude:

**Test A — across the 12 configurations.** Each (backbone + uncertainty measure) combination was
run under both policies. Testing across those 12 pairs asks: *"across many different setups, does
the policy systematically change the outcome?"*
**Cannot conclude:** anything about seed-to-seed wobble. Nothing here measures that.

**Test B — across the 1,905 test photos.** Within one pair, both policies predict on the same
photos, so we can pair them photo-by-photo.
**Cannot conclude:** anything about training randomness either. It measures uncertainty from having
a finite test set.

Neither replaces multi-seed replication. **Being upfront about that is the correct move** — a
reviewer who spots you overclaiming will distrust everything else in the paper.

---

## 4.6 The result that looks like a contradiction (but isn't)

This one is genuinely confusing until you work through it slowly, and your supervisor may well ask
about it.

We tested accuracy two ways and got two different-looking answers:

- **Test A (across the 12 configurations):** accuracy up +0.60 points — **significant**, p = 0.014
- **Test B (within each configuration, photo-by-photo):** only **1 of 12** was significant

Both are correct. **They ask different questions.**

Individually, each configuration's gap is **smaller than the random wobble** you get from having
only 1,905 test photos. You genuinely cannot resolve it — like trying to measure a hair's width
with a ruler marked in centimetres.

**But 11 of the 12 gaps point the same direction.**

> One coin landing heads proves nothing.
> **Eleven out of twelve coins landing heads is strong evidence.**

Test B measures each coin. Test A measures the pattern across all twelve. Both are true, and
together they give the honest sentence for your paper:

> *"The accuracy effect is small, positive, and consistent in direction, but not individually
> resolvable per configuration at this test-set size."*

The safety effect, by contrast, is big enough to show up clearly at **every** level of analysis.
That's what a solid result looks like.

---

## 4.7 Your actual statistical results

| What we measured | Change | 95% CI | Corrected p | Verdict |
|---|---|---|---|---|
| **Unsafe auto-accepts** | −4,030 (12/12 configs) | [−4,319, −3,715] | **0.003** | ✅ **solid** |
| **Extra labels used** | +382 (12/12 configs) | [+315, +447] | **0.003** | ✅ **solid** |
| Accuracy | +0.60 pts (11/12) | [+0.28, +1.00] | 0.014 | ✅ significant |
| F1-macro | +0.86 pts (8/12) | [−0.22, +1.97] | 0.30 | ❌ not significant |
| Missed-cancer rate | −1.19 pts (3/12) | [−2.46, +0.21] | 0.30 | ❌ not significant |
| Melanoma recall | +2.43 pts (8/12) | [+0.44, +4.39] | 0.17 | ⚠️ borderline |

**How to read this honestly:**

- **The safety result and its price are rock solid.** Both happened in 12 out of 12 configurations
  with tiny p-values. Claim these confidently.
- **The clinical-outcome results are promising but unproven.** Missed-cancer rate and melanoma
  recall did not reach significance. **Do not claim them yet.** Say "suggestive, pending multi-seed
  replication."

Notice the missed-cancer row: only **3 of 12** configurations improved, and the interval crosses
zero. That's close to a coin flip. Being straight about this is what makes the rest of your paper
believable.

---

## The five things to remember

1. **"Our number is bigger" is not a result.** It might be luck. Statistics is how you tell.
2. **p-value = how often pure luck would produce a gap this big.** Below 0.05 = "probably real".
   "Not significant" means *not enough evidence*, not *no difference*.
3. **Confidence interval = the range the true value probably sits in.** If it crosses zero, you
   can't rule out "no difference". CIs are usually more useful than p-values.
4. **Always quote the Holm-corrected p-value.** Running 12 tests means one will pass by luck.
5. **Your safety result is solid (12/12, p = 0.003). Your clinical-outcome results are not
   significant.** Say so — it's what makes the rest credible.

---

➡️ **Next: `05_WHAT_WE_FOUND.md`** — your actual results.
