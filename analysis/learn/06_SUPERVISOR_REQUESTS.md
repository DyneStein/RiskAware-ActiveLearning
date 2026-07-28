# 6 — Your supervisor's requests, decoded

*Read documents 1–5 first.*

He sent two messages. Here is every item, what it means, why he asked, and what we did.

---

## Message 1

> **"Lack of Calibration Analysis: 1. Expected Calibration Error (ECE) 2. Brier Score
> 3. Reliability diagrams"**

**What he means:** *"Prove your model's confidence numbers can be trusted."*

**Why:** your whole safety mechanism thresholds the risk score. If "0.8" doesn't mean 80% danger,
you're thresholding a meaningless number. Accuracy can't detect this. *(Document 3.4.)*

**Status: ✅ done.** Overconfident by ~7 points (ECE 0.073). The risk head is better calibrated than
the classifier (0.056). Temperature scaling T ≈ 2.16 cuts held-out ECE from 0.073 to 0.023.
📁 `rigor/figures/17, 18, 19, 20`, `rigor/tables/calibration_metrics.csv`

---

> **"Active Learning Efficiency: Plot Accuracy vs Number of labeled samples"**

**What he means:** *"Doctor time is the expensive thing. Show accuracy against labels spent, not
against rounds."*

**Why:** he spotted that your two methods don't spend the same amount. Plotting against rounds
would flatter you for simply asking more questions.

**Status: ✅ done — and he was right to ask.** At matched budget your method is **0.35 points
behind**, not ahead. It buys safety, not efficiency. *(Document 5.4, finding 1.)*
📁 `rigor/figures/10, 11, 12, 13`

---

> **"Need AUC per lesion class especially melanoma."**

**What he means:** *"One overall score hides failure on the rare, dangerous classes."*

**Why:** 70% of your test set is ordinary moles, so a model that gave up on melanoma would still
look fine overall. *(Document 3.3.)*

**Status: ✅ done.** All 7 classes with 95% confidence intervals. Melanoma is the hardest at
**0.945 [0.929, 0.960]**, PR-AUC 0.781.
📁 `rigor/figures/21, 22, 23`, `rigor/tables/per_class_auc.csv`

---

> **"Statistical Tests: Include p-values and confidence intervals."**

**What he means:** *"Prove your improvement isn't luck."* *(Document 4.)*

**Status: ✅ done.** Wilcoxon across the 12 configurations, McNemar photo-by-photo, bootstrap CIs,
Holm correction. Safety result and label cost both significant (p = 0.003, 12/12 configurations);
F1, missed-cancer rate and melanoma recall **not** significant.
📁 `rigor/figures/24, 25`, `rigor/tables/significance_*.csv`

---

> **"Runtime: Provide 1. training time 2. inference time 3. query time"**

**What he means:** *"Is this practical to actually run?"*
- **Training time** — teaching the model
- **Inference time** — how long one prediction takes (matters for real clinic use)
- **Query time** — scoring the pool and deciding who to escalate — the overhead *your method adds*

**Status: ✅ done.** 94.1 GPU-hours total; MC-dropout 2.42× the others. Training is ~94% of each
round for three of four uncertainty measures, but querying rises to ~40% for MC-dropout. **The
escalation rule itself costs about 8 milliseconds per round** — your uncapped risk route is
essentially free in compute; its cost is annotation.
📁 `rigor/figures/26, 27`

*One thing worth telling him, because it shows real care:* the obvious way to split training vs
query time is a regression on the logged round times. **It's mathematically impossible here** —
your pool is closed, so labelled + unlabelled = 8,110 exactly every round, making the two
quantities perfectly locked together. It produced *negative* query times, which is how we caught
it. We measured each component directly instead.

---

## Message 2 — "overall need these changes"

> **"1. external validation on ISIC2019/2020"**

**What he means:** *"Does it work on somebody else's photos?"* Your model saw one dataset, from a
limited set of hospitals and cameras. This is the strongest evidence a medical AI generalises.

**Status: 🔧 code ready, needs a ~3 GB download** (best done in Colab).

### ⚠️ Tell him this — it's important

**ISIC 2019 literally contains HAM10000 inside it.** ISIC 2019 was built by combining three
sources: BCN20000 + **HAM10000** + MSK, and the HAM10000 photos keep their original `ISIC_xxxxxxx`
filenames.

**So testing on ISIC 2019 as downloaded means testing on your own training data.** The scores would
look wonderful and be worthless — and a reviewer who knows these datasets would spot it instantly.

Our script always removes the overlapping photos by filename, reports how many it removed, and
refuses to run unless the overlap is zero.

**Recommend ISIC 2020 instead** as the primary external test: different year, different patients,
no overlap — and its labels are simply benign/malignant, which is **exactly** what your risk head
predicts.

**Expect the numbers to drop.** Different cameras and hospitals always cost something. That's the
normal, reportable result.

---

> **"2. statistical significance analysis"** — ✅ done, see above.

---

> **"3. ablation studies"**

**What he means:** *"Remove one ingredient and show the recipe stops working."* You claim two
signals beat one — prove each is pulling its weight. *(Document 5.3.)*

**Status: ✅ done.** Uncertainty-only catches 12.6% of dangerous cases vs 10.2% for *random*;
risk-only 17.2% at a third of the cost; both together 29.3% — near-additive, so the two signals
flag different photos. Plus a threshold sweep tracing the safety/cost dial.
📁 `rigor/figures/14, 15, 16`

---

> **"4. calibration metrics"** — ✅ done, see above.

---

> **"5. formal mathematical definitions of the dual-metric policy"**

**What he means:** *"Right now your method is described in words and code. For a journal it must be
written in maths, precisely enough that a stranger could rebuild it exactly."*

**Status: ✅ done** — `RiskAware-ActiveLearning/METHODS.md`, written by reading your actual code
(the written docs had drifted out of date).

It also proves two small things that turned out to explain your results:

- **Proposition 1:** your policy escalates a *superset* of what the baseline escalates — it does
  everything the baseline does, plus more. So it **mathematically cannot** be less safe at fixed
  scores. That's arithmetic, not luck, and it's why the improvement appeared in 24 out of 24
  experiments with zero exceptions.
- **Proposition 2:** the extra cost is exactly the photos the risk route flags that uncertainty
  missed. Never more.

Together: your method is a **controlled trade** — strictly more safety for strictly more labels,
with a dial to set the rate. Much stronger than "our thing is better".

---

> **"6. robustness experiments"**

**What he means:** *"Real clinics don't have perfect photos. Does it survive blur, noise, bad
lighting?"*

**Status: ✅ done (4 of 5 corruptions; JPEG still to run).**
- **Good:** the risk head degrades *more slowly* than the classifier (90.3% vs 85.2% retained) —
  the system escalates rather than confidently mis-diagnosing.
- **Bad:** EfficientNet-B4 collapses under mild noise (0.863 → 0.008, below random), while the
  other two only drop to ~0.70.
- **Bad:** melanoma recall falls 0.70 → 0.31 (blur) → 0.08 (noise).
📁 `rigor/figures/29, 30, 33`

---

> **"7. explainability analysis"**

**What he means:** *"Show where the model is looking."* Skin datasets are full of shortcuts —
rulers, hair, dark lens corners. A model that learned "rulers = cancer" fails in a real clinic.

**Status: ✅ done** for all three backbones, with heat-maps for **both heads** — something only
possible because of your two-head design. Includes the case where the classifier was misled by a
camera artefact while the risk head correctly flagged the lesion at 0.83, *plus* an honest failure
case. *(Document 5.6.)*
📁 `rigor/figures/28_gradcam_panel_*.png`

---

> **"as maximum possible"**

**12 of his 13 items are done.** Only external validation is outstanding, plus one corruption type.

---

## What to send him

There is a **ready-made draft reply** at the end of `analysis/PLAIN_ENGLISH_GUIDE.md`. You can copy
it as-is or trim it.

**Two things that will impress him, and you should make sure to mention:**

1. **You caught that ISIC 2019 overlaps HAM10000.** Plenty of published papers have missed this.
2. **You're reporting results that don't favour your own method** — not label-efficient,
   missed-cancer rate not significant, the two-head redesign a tie overall, EfficientNet collapsing
   under noise. Presented alongside the strong safety result and the shared-backbone explanation
   that ties them together, that's what separates a **study** from a **sales pitch**.

**And if he asks "what's next?"** — multi-seed replication is the biggest remaining gap. Suggest
**3–4 configurations × 5 seeds (~60 GPU-hours)** targeting the safety result specifically, rather
than the full matrix (~470 hours).

---

## The five things to remember

1. **12 of his 13 requests are done.** Only external validation genuinely needs new work.
2. **He was right about the efficiency plot** — it changed a claim you would otherwise have made
   wrongly.
3. **Calibration was his top item, and yours is imperfect but fixable** — that's a good story, not
   a bad one.
4. **Warn him about ISIC 2019 containing HAM10000** before anyone runs it.
5. **Lead with the honest limitations.** Your safety result is strong enough to survive them, and
   volunteering them is what makes it credible.
