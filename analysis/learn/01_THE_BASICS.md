# 1 — The absolute basics

*Assumes you know nothing. Read this first.*

---

## 1.1 What is machine learning?

Normal computer programs follow rules a human wrote:

> *"If the temperature is above 30, turn on the fan."*

That works for fans. It does not work for skin cancer, because **nobody can write down the rules
for what melanoma looks like.** Even expert dermatologists can't fully explain it — they just
recognise it after seeing thousands of cases.

So instead of writing rules, we do this:

> Show the computer **thousands of photos with the correct answer attached**, and let it work out
> the rules by itself.

That is **machine learning**. The computer isn't told *what* melanoma looks like. It's shown
10,000 examples and left to figure out the pattern.

**The word "learning" is doing a lot of work here.** The computer isn't understanding anything.
It's adjusting millions of internal numbers until its guesses stop being wrong so often. That's it.

---

## 1.2 What is a neural network?

A **neural network** (or just "network", or "model") is the thing that does the guessing. Picture
it as a machine with two parts:

**The eyes** (the technical word is **backbone**)
Looks at the photo and converts it into a long list of numbers describing what it sees — how dark,
how uneven the border, how many colours, how rough the texture. Not in words, just numbers.

**The mouth** (the technical word is **head**)
Takes that list of numbers and turns it into an actual answer: *"melanoma"*.

```
   photo  ──►  [ BACKBONE / "eyes" ]  ──►  numbers  ──►  [ HEAD / "mouth" ]  ──►  "melanoma"
```

**Why the split matters:** the eyes are the expensive, general part — they learn to see skin.
The mouth is small and cheap. You can bolt on a *second* mouth that answers a different question
using the *same* eyes. **Your project does exactly that** — that's the "two-head design", and
document 2 explains it.

### The three "brands of eyes" in your project

You tested three well-known backbones: **ResNet-50**, **DenseNet-169**, **EfficientNet-B4**.
They're three different designs by three different research groups. Think of them as three brands
of camera. You test all three so nobody can say your result only works with one lucky choice.

---

## 1.3 The network never says "melanoma" — it says percentages

This is important and often confuses people.

The model does not output a word. It outputs a **probability** for every possible answer — seven
numbers that add up to 100%:

| Disease | Model's output |
|---|---|
| Melanoma | 82% |
| Ordinary mole | 11% |
| Benign keratosis | 4% |
| (four others) | 3% total |

We then just take the biggest one and call that "the prediction". So here the model *predicts*
melanoma.

Two words you'll see for this machinery:

- **Logits** — the raw, meaningless-scale numbers the mouth produces first (like `4.2, -1.3, 0.8`).
- **Softmax** — the small piece of maths that squashes logits into percentages adding to 100%.

That's all softmax is: a converter. Don't overthink it.

**The 82% is called the model's confidence.** Remember that word — a whole section of your
supervisor's request (calibration) is about whether that 82% can be trusted.

---

## 1.4 What is "training"?

**Training** is the process where the model learns. It goes like this, over and over:

1. Show it a photo.
2. It guesses — say, "ordinary mole, 70%".
3. Check the true answer — actually melanoma. **Wrong.**
4. Nudge all its internal numbers slightly, so that next time it leans a bit more toward melanoma.
5. Repeat, thousands of times.

The "nudging" is automatic maths. You don't do it by hand.

**Epoch** — one complete pass through all your training photos. "10 epochs" means the model
studied the whole set 10 times over. Your project uses 10 epochs.

**Loss** — a number measuring how wrong the model currently is. Training is just "make the loss
go down". If someone asks "did the loss go down?", they're asking "did it learn anything?"

---

## 1.5 Why we split the data (the single most important idea here)

Imagine a student who memorises the answers to last year's exam paper. Give them that exact paper
and they score 100%. Give them a *new* paper and they fail — they never learned the subject, they
memorised the answers.

Models do this constantly. It's called **overfitting**: memorising the training photos instead of
learning the disease.

So we **split the data**:

| Part | Size | Used for |
|---|---|---|
| **Training set** | most of the photos | teaching the model |
| **Test set** | 1,905 photos in your project | **grading it — never trained on, never touched** |

The test set is locked in a vault. The model never sees the answers. When we say "accuracy 89%",
we always mean **on the test set** — on photos it has never encountered.

This is what stops you fooling yourself. If someone reports a score on data the model trained on,
that number is worthless.

> **In your project, all 24 experiments used the exact same 1,905 test photos.** This was verified
> with a checksum (a digital fingerprint of the file). It matters because it lets us compare two
> methods fairly — same exam paper for both students.

---

## 1.6 Accuracy — and why it lies

**Accuracy** = what fraction did it get right. 89% accuracy = right on 89 out of 100 photos.

Simple. And **dangerously misleading**, for one reason:

### Your dataset is 67% ordinary moles.

So imagine a completely useless model that ignores the photo entirely and always answers
"ordinary mole". Its accuracy? **67%.** It looks like it works. It has learned nothing, and it
would miss *every single cancer*.

This is called **class imbalance** — some categories are far more common than others. Your seven
diseases in the test set:

| Disease | Count | Share |
|---|---|---|
| Ordinary moles (nv) | 1,327 | 70% |
| Melanoma (mel) | 209 | 11% |
| Benign keratosis (bkl) | 206 | 11% |
| Basal cell carcinoma (bcc) | 89 | 5% |
| Actinic keratoses (akiec) | 51 | 3% |
| Vascular lesions (vasc) | 14 | 0.7% |
| Dermatofibroma (df) | 9 | 0.5% |

**This is why your supervisor asked for "AUC per lesion class".** He wants a separate score for
each disease, because one overall number would hide total failure on melanoma. He is right to ask.
Document 3 explains what AUC is.

Your project fights imbalance using **class weights** — during training, mistakes on rare diseases
are made to "hurt" more than mistakes on common ones, so the model can't just ignore them.

---

## 1.7 Randomness and "seeds"

Training involves randomness — which photos come in which order, how the internal numbers start
out. Run the same training twice and you get *slightly* different models.

A **seed** is a number that locks the randomness so a run repeats identically. Your project uses
seed 42 everywhere.

**Why this matters, and it matters a lot:** if you run something once and it looks 2% better, that
could just be luck. The proper way to prove it isn't luck is to run it **five times with five
different seeds** and check it holds every time.

**You ran one seed.** That's the biggest honest limitation in your work right now, and document 4
explains what we did about it.

---

## The five things to remember

1. Machine learning = show the computer thousands of labelled examples, let it find the pattern.
2. A model has **eyes (backbone)** that look at the photo and a **mouth (head)** that gives the
   answer. You can bolt on a second mouth — that's your two-head design.
3. The model outputs **percentages**, not words. The biggest one is "the prediction", and its size
   is the model's **confidence**.
4. We always grade on a locked-away **test set** the model never trained on. Otherwise it's just
   memorising.
5. **Accuracy lies when classes are imbalanced.** 67% of your data is ordinary moles, so a useless
   model scores 67%. This is why per-disease scores matter.

---

➡️ **Next: `02_YOUR_PROJECT.md`** — what your project actually does.
