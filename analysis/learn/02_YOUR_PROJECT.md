# 2 — What your project actually does

*Read `01_THE_BASICS.md` first.*

---

## 2.1 The real-world problem: labels cost money

To train a skin-cancer model you need thousands of photos **with the correct diagnosis attached**.
Attaching that diagnosis means a qualified dermatologist looking at each photo. That is slow and
expensive.

So the question becomes: **if you can only afford to have 3,000 photos labelled, which 3,000?**

Picking at random is wasteful. Most skin photos are obvious moles — the model learns nothing new
from the 500th obvious mole.

---

## 2.2 Active learning: let the AI choose

**Active learning** is the idea that the model itself picks which photos are worth paying for.

The loop:

1. Train the model on the few labelled photos you already have.
2. Let it look at all the **unlabelled** photos.
3. It picks the ones it wants labelled.
4. A doctor labels those.
5. Add them to the labelled pile.
6. **Repeat.**

Each turn through that loop is called a **round**. Your project runs **15 rounds** per experiment.

### The vocabulary of the loop

| Word | Meaning | Size in your project |
|---|---|---|
| **Pool** | All the photos the model is allowed to ask about | 8,110 |
| **Labelled set** | Ones already labelled. Grows every round. | starts at 490 |
| **Unlabelled set** | The rest of the pool. Shrinks every round. | starts at 7,620 |
| **Test set** | Locked away for grading. Never in the pool. | 1,905 |
| **Oracle** | The "doctor" who provides labels | (see below) |

### The oracle is fake, and that's fine

You obviously don't have a real dermatologist on call. So the **oracle** is simulated: since
HAM10000 already contains the true diagnosis for every photo, "asking the doctor" just means
looking up the answer that was there all along.

This is completely standard in active-learning research. It's free, repeatable, and ethical.
It **is** a limitation you must declare in the paper — a real doctor would sometimes be wrong,
tired, or disagree with a colleague. A dataset label never is.

### Two decisions the system makes per photo

For every unlabelled photo, each round, the system decides:

- **Escalate** — send it to the oracle for a real label.
- **Auto-accept** — don't bother the doctor; just accept the model's own guess.

---

## 2.3 The standard approach — and why your project exists

For 30 years the standard rule for picking photos has been:

> **Ask about the ones the model is most confused about.**

This is called **uncertainty sampling**, and the logic is decent: if the model is torn between
melanoma and mole, that photo is informative — labelling it teaches the model a lot.

**Uncertainty** just means "how confused is the model". Your project measures it four different
ways (you don't need to memorise these, just recognise them):

| Name | What it measures |
|---|---|
| **Entropy** | How spread out the seven percentages are. All on one = certain; spread evenly = confused. |
| **Least confidence** | `1 − biggest percentage`. Top guess only 40%? Then uncertainty is 0.6. |
| **Margin** | How close the top *two* guesses are. Torn between 45% and 44% = very confused. |
| **MC-dropout** | Run the same photo through 30 times, randomly disabling bits of the network each time. If the answer keeps changing, it isn't confident. |

*(MC-dropout is why some of your runs took 2.4× longer — it does 30 passes instead of 1.)*

### The flaw, and it's a serious one

Uncertainty sampling only catches **confusion**. But there's a second, far more dangerous failure:

> **The model is completely confident — and completely wrong — about a cancer.**

Uncertainty sampling will *never* flag that photo. There was no confusion to detect. The system
auto-accepts "harmless mole, 96% confident", nobody looks at it, and it was a melanoma.

**That is the exact case your project is built to catch.** In ordinary machine learning, being
confidently wrong is just an error. In medicine, it can kill someone.

---

## 2.4 Your idea: add a second signal

Your project adds a completely separate question:

> **Not "am I confused?" but "how dangerous would it be if I'm wrong?"**

Those are different questions with different answers. The model can be perfectly confident about
something extremely dangerous.

### The risk score

Three of your seven diseases are malignant or pre-malignant:

- **mel** — melanoma
- **bcc** — basal cell carcinoma
- **akiec** — actinic keratoses

The other four (moles, benign keratosis, dermatofibroma, vascular lesions) are harmless. This
split is decided in advance from clinical knowledge — it's not learned.

The **risk score** is a number from 0 to 1 = *"how likely is this lesion malignant?"*
0.9 means "90% likely dangerous".

### The two-head design

Where does the risk score come from? This is the architecture decision.

**The original version:** just add up the model's percentages for the three dangerous diseases.

**The problem with that:** it's calculated *from* the diagnosis. So if the diagnosis is confidently
wrong — "harmless mole, 96%" — then the risk score computed from it is *automatically* wrong too.
They fail together, in exactly the situation the safety net exists for. Useless.

**The fix — the two-head design:** give risk its **own mouth**.

```
                             ┌──► HEAD 1 (classification) ──► "which disease?" (7 percentages)
   photo ──► BACKBONE ───────┤
             ("eyes")        └──► HEAD 2 (risk) ──────────► "how dangerous?" (0 to 1)
```

One pair of eyes, two mouths, each with its **own separate internal numbers**. Now the risk mouth
*can* disagree with the diagnosis mouth.

> **A real example from your results.** Photo `ISIC_0027776`: the classification head said "benign
> keratosis" — wrong, it was a melanoma. But the risk head gave it **0.83**, so it was escalated to
> the doctor anyway. The safety net worked exactly as designed. We even have a picture showing the
> classification head was distracted by the dark corners of the photo while the risk head was
> looking at the actual lesion.

**Important honest caveat** (document 5 covers this properly): the two mouths still share the same
*eyes*. So when the eyes get fooled, both mouths tend to get fooled together. The design helps, but
less than you'd hope.

---

## 2.5 The escalation rule — how the decision is actually made

Each round, every unlabelled photo gets two numbers: an uncertainty score and a risk score.
The rule:

> **Escalate if the model is confused OR the case looks dangerous.**

As a grid:

|  | **Low risk** | **High risk** |
|---|---|---|
| **Low uncertainty** (confident) | auto-accept ✅ | **ESCALATE** ⚠️ ← *this cell is your contribution* |
| **High uncertainty** (confused) | escalate | escalate 🚨 |

The **top-right cell is the whole paper**: the model is *confident* but the case is *dangerous*.
The standard approach auto-accepts it. Yours escalates it.

### Two details you'll be asked about

**The threshold, recalibrated every round.** A **threshold** is the bar a score must clear to
trigger escalation. Yours isn't a fixed number — each round it's set to the **90th percentile** of
the current scores (i.e. "the top 10% most uncertain / most dangerous").

Why not just fix it? Because as the model improves, its scores shrink. A bar set against a weak
early model would soon be so high nothing ever clears it, and escalation would silently stop.
(This actually happened in your very first run — escalation collapsed from 558 photos to 1 to 0
and froze.) Re-setting the bar every round keeps it meaningful.

**Top-K budget (K = 150).** "Always send at least the 150 most-confused photos each round, even if
none clear the bar." It's a **floor, not a ceiling** — if 300 clear the bar, all 300 go.

**The risk route has no budget at all.** If 500 photos look dangerous, all 500 get escalated. The
principle: never skip a dangerous case just because you'd used up your quota. This is a deliberate
design commitment and it's why your method costs more labels.

---

## 2.6 The number that matters most: "unsafe auto-accepts"

This is your headline metric, so make sure it's clear:

> **An unsafe auto-accept** = a photo that was **genuinely cancerous**, that the system **waved
> through without a doctor ever looking at it.**

Every single one is a potential missed cancer. **Lower is better.**

Note what makes this measurement honest: it uses the true label that the system *did not see* when
it made the decision. So it measures the quality of the decision itself, not the model's opinion
of itself.

---

## 2.7 Why there are 24 experiments

You tested every combination:

**3 backbones** (ResNet-50, DenseNet-169, EfficientNet-B4)
**× 4 uncertainty measures** (entropy, MC-dropout, margin, least-confidence)
**× 2 policies** (uncertainty-only = the standard baseline; dual-metric = yours)
**= 24 experiments**, each running 15 rounds. About 94 hours of GPU time in total.

**Why bother with all 24?** Because if your method only worked with one backbone and one
uncertainty measure, that would be luck, not a finding. Testing every combination means you can
say *"it worked in all 12 matched comparisons"* — which is exactly what happened for the safety
result.

A **baseline** is the existing standard method you compare against. Yours is uncertainty-only.

---

## The five things to remember

1. **Active learning** = the AI picks which photos are worth paying a doctor to label. One cycle
   = a **round**; you run 15.
2. The standard rule ("ask when confused" = **uncertainty sampling**) has a fatal gap: it never
   catches a model that is **confident and wrong** about a cancer.
3. Your fix: a second, independent **risk score** ("how dangerous?"), from its own **head**, and
   escalate if **either** signal fires. That's the **dual-metric policy**.
4. The risk route is **uncapped** — no budget limit — which is why your method uses more labels.
5. **Unsafe auto-accepts** = genuinely cancerous photos waved through without review. This is your
   most important number, and lower is better.

---

➡️ **Next: `03_HOW_WE_MEASURE.md`** — all the measurement words.
