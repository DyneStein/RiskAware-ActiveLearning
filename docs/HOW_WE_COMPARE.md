# How the Comparison Actually Works

*What we compare against what, how we prove a difference is real, and one
problem in the plan that has to be fixed before the runs start.*

Every technical term is explained the first time it appears.

---

## Part 1 — The problem nobody warns you about

Here is the trap, and it is the reason a naive comparison would be thrown
out by a reviewer.

**The four baselines answer a different question from ours.**

| | The question it answers |
|---|---|
| CoreSet, BADGE, CLUE, VAAL | *"You have budget for 300 labels. Which 300?"* |
| **Our dual-metric policy** | *"Which images are unsafe to wave through?"* |

Ours does **not** take a budget. It works out how many to escalate as a
*consequence* of what it finds — the risk route is deliberately uncapped, so
that a dangerous case is never skipped just because a quota was already
full. That is the whole point of the design.

So if you just run both and compare the final numbers, you get a meaningless
answer **in whichever direction you set it up**:

- Give the baselines a flat 150 per round while ours escalates ~310? We win
  partly because **we asked more questions**. A reviewer spots that in
  seconds and the result is dead.
- Cap ours at 150 to match them? Then we have **switched off the uncapped
  risk route** — the exact thing the paper is about. We would be testing a
  crippled version of our own method.

### The fix: cost-matching

**Cost-matched** means: in round 7, each baseline is handed **exactly** the
number of labels our method spent in round 7, on that same model. Not an
average, not an estimate — the real per-round number, read straight out of
that finished run's own `results.csv`.

```
Round:            1     2     3     4    ...
Ours escalated:  312   340   298   355   ...
BADGE gets:      312   340   298   355   ...   ← identical, every round
CoreSet gets:    312   340   298   355   ...
```

Both methods now spend an identical annotation budget on an identical
schedule. **The only thing that differs is which images each one chose.**
That is a fair fight, and it is the only version a reviewer will accept.

The code does this automatically. If the reference run is missing it stops
immediately with an error rather than quietly falling back to a flat
budget — because a comparison that *looks* cost-matched in the paper but
isn't would be far worse than one that failed loudly.

---

## Part 2 — What we actually measure

We report **two separate scoreboards**, because the methods are optimising
different things and pretending otherwise would be dishonest.

### Scoreboard A — Learning (how good does the model get?)

| Metric | Plain English |
|---|---|
| **Accuracy** | Out of 1,905 exam images, what fraction did it get right? |
| **F1-macro** | Accuracy that treats all 7 diagnoses as equally important. Matters here because ordinary moles are two-thirds of the data — a model could score 67% accuracy by answering "mole" every time, and F1-macro exposes that. |
| **Labels to reach X%** | How many questions did it need to hit a given accuracy? Fewer is better. |

**This is the axis the four baselines were designed to win.** They exist to
extract maximum learning per label.

### Scoreboard B — Safety (how many dangerous cases slip through?)

| Metric | Plain English |
|---|---|
| **Unsafe auto-accepts** | A genuinely dangerous image the system waved through with no human review. In a clinic, a missed cancer. |
| **Missed-cancer rate** | Of all the malignant cases in the exam set, what fraction did the final model call benign? |
| **Melanoma recall** | Of all the real melanomas, how many did it catch? |

**None of the four baselines has any notion of clinical consequence.** BADGE
optimises expected information gain. CoreSet optimises coverage. VAAL does
not even look at the classifier. Not one of them contains the idea that a
melanoma is worse to miss than a mole.

**That is the argument of the paper.** Not "we are more accurate" — "we are
comparably accurate *and* we do not wave dangerous cases through, and the
methods that beat us on learning have no mechanism for that at all."

### Being blunt about the likely outcome

**BADGE may well beat us on accuracy.** It is a strong method that has held
up for years. Our current accuracy edge over the old baseline is only
+0.60 percentage points, and at a matched budget we are already 0.35 points
*behind*.

If that happens, it is **not a failure** — provided the paper is framed
around safety from the abstract onwards. What *would* be a failure is
building the paper around an accuracy claim and then having BADGE take it
away in review.

---

## Part 3 — Proving a difference is real, not luck

If our method scores 89.1% and BADGE scores 88.7%, is that a real
difference or did we get lucky? That is what statistical testing answers.

### The vocabulary

| Term | What it means |
|---|---|
| **p-value** | The probability of seeing a gap this big **if there were really no difference at all**. Small p = "luck is a poor explanation". Below **0.05** is the usual bar. |
| **Paired comparison** | Compare like with like — ResNet-ours against ResNet-BADGE, never ResNet-ours against DenseNet-BADGE. Removes "that architecture is just better" from the comparison. |
| **Statistical power** | Whether your experiment is even *capable* of detecting a real difference. Too few comparisons and the answer is no — regardless of how big the real effect is. |
| **Multiple-comparison correction** | Test 20 things and roughly one will look "significant" by chance alone. Correction raises the bar to compensate. We use **Holm**, a standard method. |

### 🔴 The problem I found while writing this

Here is something that has to be said before you spend 40 GPU-hours.

To compare **our method against BADGE**, the natural approach is one paired
observation per model:

```
ResNet-50:        ours vs BADGE     ← pair 1
DenseNet-169:     ours vs BADGE     ← pair 2
EfficientNet-B4:  ours vs BADGE     ← pair 3
```

That is **n = 3**.

The standard test for this (**Wilcoxon signed-rank** — a test that asks
whether differences consistently point the same way, without assuming the
data forms a neat bell curve) has a hard floor on how small its p-value can
get. With n pairs all favouring you, the smallest possible two-sided
p-value is 2 ÷ 2ⁿ:

| Pairs | Best possible p | Under 0.05? |
|---|---|---|
| **n = 3** | **0.250** | ❌ **Impossible** |
| n = 5 | 0.063 | ❌ Impossible |
| n = 6 | 0.031 | ✅ Just |
| n = 8 | 0.008 | ✅ Comfortably |

> **With 12 baseline runs, we can never demonstrate statistical significance
> against any individual baseline at the run level. Even if we beat BADGE on
> all three models, the best p-value obtainable is 0.25 — five times the
> threshold.**

This is the same trap as the 5-seed problem, one level down. It is not a
reason to change the plan, but it **is** a reason to know now what the
12 runs can and cannot support, rather than discovering it while writing the
results section.

### The solution: test at the image level instead

There is a second, much more powerful way to compare, and the code already
does it for the existing policies.

Instead of 3 paired *runs*, use **1,905 paired images**. Every model — ours
and every baseline — sits the identical exam. So for each individual
photograph you can ask: did our model get it right, and did BADGE?

|  | BADGE right | BADGE wrong |
|---|---|---|
| **Ours right** | both fine | **we win this image** |
| **Ours wrong** | **they win this image** | both fail |

**McNemar's test** looks only at the two disagreement boxes. If we win 60
images and lose 30, that lopsidedness is very unlikely to be chance, and it
gives a genuine p-value with real power behind it — because n is 1,905, not 3.

This is only valid because `SPLIT_SEED` is frozen, so **every model in the
entire project sits the byte-identical exam**. That decision, made for a
different reason, is what makes this test possible.

### So the plan is three levels

| Level | What it compares | Sample size | What it can claim |
|---|---|---|---|
| **1. Descriptive** | Main results table, all methods, per model | 3 models | "Ours has the lowest unsafe auto-accepts on all 3" |
| **2. Image-level** | Our final model vs each baseline's, per model | **1,905 images** | Real p-values with real power ✅ |
| **3. Run-level** | Aggregated across models | 3 | Direction only — **explicitly labelled underpowered** |

**Level 3 must be reported honestly as descriptive.** Writing "p = 0.25, not
significant" invites the reader to conclude there is no effect, when the
truth is the test could never have found one. The correct sentence is:
*"with three backbones, a run-level paired test cannot reach significance by
construction; we therefore report run-level results descriptively and test
at the image level."*

Naming that limit yourself reads as competence. Having a reviewer name it
for you reads as carelessness.

### If you want run-level significance too

Run the baselines at **3 seeds instead of 1**: 3 models × 3 seeds = **n = 9**
pairs, minimum p = 0.004. That costs 24 extra runs ≈ 80 GPU-hours.

**My recommendation: do not.** The image-level test already gives you real
statistical evidence, and that 80 hours is far better spent on the multi-seed
replication of the *main* result, which is a bigger hole. Revisit only if a
reviewer explicitly asks.

---

## Part 4 — What the results section will look like

### The main table

Every row at the same cost-matched budget, so the columns are comparable.

| Method | Accuracy | F1-macro | Missed-cancer rate | **Unsafe auto-accepts** | Labels used |
|---|---|---|---|---|---|
| Uncertainty-only *(old baseline)* | | | | | |
| CoreSet *(ICLR 2018)* | | | | | |
| CLUE *(ICCV 2021)* | | | | | |
| BADGE *(ICLR 2020)* | | | | | |
| VAAL *(ICCV 2019)* | | | | | |
| **Dual-metric *(ours)*** | | | | | |

Each cell is the mean across the 3 models, with the spread shown.

**The expected shape of it:** the accuracy columns will be close, and BADGE
may top them. The **unsafe auto-accepts** column should show a clear gap —
because it is the only column that measures something none of the other five
methods is trying to do.

### The figure that makes the argument

A **trade-off plot**: annotation cost on the horizontal axis, unsafe
auto-accepts on the vertical, one point per method.

The reader should be able to see in one glance that the baselines cluster
together — spending the same, missing similar numbers of dangerous cases —
and that ours sits **lower** at the same horizontal position. Same cost,
fewer dangerous cases waved through.

This is called a **Pareto frontier**: the set of options where you cannot
improve one thing without sacrificing another. If our point sits below the
line the baselines form, we have expanded what is achievable rather than
just moved along it.

### The honest paragraph that has to be in there

> The four acquisition baselines optimise expected information gain and are
> competitive with, or superior to, the proposed policy on classification
> accuracy at matched annotation cost. None of them, however, models the
> clinical consequence of an error. At equal annotation cost the proposed
> policy reduces unsafe auto-accepts by X%, while accuracy remains within
> Y percentage points of the strongest baseline.

That paragraph concedes the accuracy point openly and makes the safety
argument on its own terms. A reviewer who reads it stops looking for the
thing you are hiding — because you aren't hiding it.

---

## Part 5 — One thing already in our favour

The existing ablation shows uncertainty-only sampling catching **12.58%** of
high-risk cases, against **10.21%** for picking images completely at random
at the same cost.

**Our old baseline was barely better than random.** That is a genuine
weakness in the current draft — and adding these four baselines fixes it. It
is the strongest argument for why your supervisor's request is worth 40
GPU-hours: without it, a reviewer can fairly say the comparison was against a
straw man.

---

## The short version

1. **Cost-match everything** — same labels, same rounds, so only the choice
   of images differs. Automatic in the code.
2. **Two scoreboards** — learning (they may win) and safety (they have no
   mechanism for it). Frame the paper on safety.
3. **Test on the 1,905 shared exam images**, not on 3 runs. With 3 runs, a
   p-value below 0.05 is mathematically impossible.
4. **Concede the accuracy point in writing** before a reviewer takes it.
5. The trade-off plot is the figure that carries the argument.
