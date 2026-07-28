# 5 — What we actually found

*Read documents 1–4 first. This one uses all of that vocabulary.*

---

## 5.1 The headline: yes, the risk score works

Comparing your dual-metric policy against the standard uncertainty-only baseline, across all 12
matched configurations:

> **Unsafe auto-accepts fell by 43%.**
> **In 12 out of 12 configurations. Corrected p = 0.003.**

Remember what an unsafe auto-accept is: a **genuinely cancerous photo that the system waved through
without any doctor looking at it.** Cutting those by nearly half is the core claim of your project,
and the evidence is solid.

Supporting this: the risk score's own **AUC is 0.96** at separating dangerous from safe lesions.
That's measured independently of whether any policy uses it, so it shows the risk head really did
learn something real — it isn't noise that happened to help.

---

## 5.2 The price: more doctor labels

> **+382 more oracle labels on average (about +9%).**
> **In 12 out of 12 configurations. Corrected p = 0.003.**

Equally certain, equally consistent. Your method **costs more**. That's not a flaw to hide — it's
the deliberate consequence of the uncapped risk route (document 2.5). You chose never to skip a
dangerous case, and this is the bill.

**The honest one-sentence framing of your entire project:**

> *Dual-metric escalation substantially reduces missed dangerous cases, at a cost of ~9% more
> annotation and no meaningful change in classification quality.*

---

## 5.3 The best single result: the ablation

**Ablation** = remove one ingredient and see if the recipe still works. You claim two signals beat
one; an ablation is how you prove it.

We replayed every round under four different rules, using the exact same models and scores:

| Rule | Dangerous cases caught | Doctor labels spent |
|---|---|---|
| Random picking | 10.2% | 6,201 |
| **Uncertainty only (the standard approach)** | **12.6%** | 4,339 |
| Risk only | 17.2% | **1,917** |
| **Both together (yours)** | **29.3%** | 6,201 |

Three things fall out of this, and they're the most useful findings in your work:

**1. The standard approach is barely better than random.** 12.6% vs 10.2% — despite spending 4,339
labels. Uncertainty sampling was never designed to be a safety mechanism, and here's the proof it
isn't one. **This is your paper's motivation in a single row.**

**2. Risk-only is remarkably cheap.** 17.2% of dangerous cases caught for only 1,917 labels — less
than half the baseline's spend, and it still beats it. Worth mentioning as a budget option for
anyone with limited annotation money.

**3. The two signals are complementary, not redundant.** 12.6 + 17.2 = 29.8, and together they
achieve 29.3. **Almost perfectly additive** — which means the two routes are flagging *largely
different photos*. That is precisely what an ablation is supposed to demonstrate, and it justifies
the whole dual-metric design.

---

## 5.4 Four findings that don't flatter your method

These are uncomfortable. They also make your paper considerably stronger — and it is far better to
find them yourself than to have a reviewer find them for you.

### Finding 1: it is NOT more label-efficient

At round 15 your method shows +0.60 accuracy points. But it has spent **382 more labels** to get
there. That's not a fair comparison.

Compared at a **matched budget** — same number of labels for both — your method is **0.35 points
behind**, and needs about **300 more labels** to reach any given accuracy target.

**Why?** Your method deliberately spends labels on photos that are *dangerous*, not photos that are
*informative*. Those aren't the same photos. An informative-but-harmless photo teaches the
classifier more per label.

**What to do:** frame it as a **safety intervention with a known price**, never as an efficiency
gain. This is the first thing a reviewer would check, and the efficiency claim would not survive.

### Finding 2: the missed-cancer rate didn't significantly improve

Test-set missed-cancer rate went down 1.19 points, but **p = 0.15 — not significant**, and only
3 of 12 configurations improved. Melanoma recall (+2.43 points) is borderline at best.

**Don't claim these yet.** Say "suggestive, pending multi-seed replication."

### Finding 3: the two-head redesign is roughly a tie overall

Remember from document 2 that you replaced "add up the malignant percentages" with a separate risk
head. Testing both on the same data:

| Scoring method | AUC |
|---|---|
| Risk head (your redesign) | 0.9520 |
| Summed percentages (the original) | 0.9524 |

**A tie.** On the full test set, the redesign bought nothing.

**But that was never the claim.** The argument was about what happens when the classifier is
*wrong*. On the missed cancers specifically:

| | Still flagged as risky |
|---|---|
| Risk head | **5.6%** |
| Summed percentages | 0.6% |

So it rescues about **9× more** missed cancers — it does work where it was designed to. But 5.6% is
still a small fraction.

### Finding 4: EfficientNet-B4 collapses under noise

We re-tested every model on deliberately degraded photos (mild sensor noise, blur, dim light, low
contrast) without retraining — simulating a real clinic.

| Accuracy | Clean photos | With mild noise |
|---|---|---|
| DenseNet-169 | 0.889 | 0.692 |
| ResNet-50 | 0.896 | 0.710 |
| **EfficientNet-B4** | **0.863** | **0.008** |

**0.008 is far below random guessing** (which would be about 0.14). It collapses into predicting
essentially one wrong class for every single photo.

This isn't a bug in our test — the identical noise applied to the other two models gives the
orderly ~20-point drop you'd expect. It's specific to that architecture (likely worsened by running
B4 at 224 pixels instead of its native 380).

**Report this per-model.** The average (52.7%) hides a total failure.

Also worth flagging: **melanoma recall falls from 0.70 clean → 0.31 with blur → 0.08 with noise.**
Even on the robust models, a slightly out-of-focus camera destroys most of your melanoma detection.
That belongs in Limitations.

---

## 5.5 The finding that ties it all together

Findings 2 and 3 aren't two separate problems. **They're one mechanism**, and understanding it is
the most valuable thing in your analysis.

Go back to the picture from document 2:

```
                             ┌──► HEAD 1 (classification)
   photo ──► BACKBONE ───────┤
             ("eyes")        └──► HEAD 2 (risk)
```

The two mouths have their own separate parameters — **but they share the same eyes.**

So when the eyes are fooled (bad lighting, an odd angle, an unusual-looking lesion, a camera
artefact), **both mouths get fooled together.** They're decoupled in *parameters* but not in
*features*.

**This explains everything:**

- ✅ **Unsafe auto-accepts improved hugely** — the risk route catches many dangerous photos that
  uncertainty ignores, because "dangerous" and "confusing" are genuinely different questions.
- ❌ **Missed-cancer rate barely moved** — the specific photos the classifier gets wrong are
  largely the *same* photos the risk head gets wrong, because the shared eyes failed on both.

> **The safety net is correlated with the thing it's supposed to catch.**

That's not a hole in your paper. It's a **mechanism** — it explains your own results, it's backed
by evidence, and it points directly at the obvious next step: give the risk head its own backbone
(its own eyes), so the two can fail independently.

**One place the shared backbone did NOT hurt:** under image degradation, the risk head degraded
*more slowly* than the classifier (retaining 90.3% of its performance vs 85.2%). So the system
loses its grip on *which* disease it is faster than it loses the sense that it's *dangerous* —
meaning it escalates rather than confidently mis-diagnosing. That's genuinely good safety behaviour,
and it's a win for the two-head design that the plain AUC comparison denied it.

---

## 5.6 And one beautiful picture

`analysis/rigor/figures/28_gradcam_panel_resnet50_entropy_dual_metric.png`

**Grad-CAM** produces a heat-map showing *where the model was looking* — red means "this is what
convinced me". Skin-lesion datasets are notorious for **shortcuts**: rulers, pen marks, hair, dark
corners from the camera lens. A model can score well by learning "photos with rulers are cancer",
then fail completely in a real clinic.

Because you have two heads, we made heat-maps for **both** — "why did you say melanoma?" beside
"why did you say dangerous?" — and found this:

> Photo `ISIC_0027776` is a melanoma. The classification head got it **wrong** (said benign
> keratosis) — and the heat-map shows it was looking at the **dark corners of the image**, a camera
> artefact. Meanwhile the risk head was looking at the **actual lesion**, and scored it **0.83** —
> high enough to escalate it to a doctor.

**That single image is your entire paper's argument, in a picture.**

And it isn't a fluke: DenseNet-169 makes the *identical* mistake on the same photo, and *its* risk
head also rescues it (0.60). The failure and the rescue both reproduce across architectures.

We also included an honest failure case: a melanoma where *both* heads stared at the corners and
both missed it.

---

## The five things to remember

1. **Your core claim holds: 43% fewer dangerous cases waved through, in 12 of 12 configurations,
   p = 0.003.** The risk score's own AUC is 0.96.
2. **It costs ~9% more doctor labels**, equally consistently. Safety is bought, not free.
3. **The ablation is your best evidence:** the standard approach catches 12.6% of dangerous cases
   vs 10.2% for *random*, while yours catches 29.3%. The two signals are near-additive, so they
   flag different photos.
4. **Four results don't favour you:** not label-efficient, missed-cancer rate not significant, the
   two-head redesign a tie overall, and EfficientNet collapses under noise. Report all four.
5. **The unifying explanation: the two heads share a backbone, so they fail on the same photos.**
   This explains findings 2 and 3, and points at the next design step.

---

➡️ **Next: `06_SUPERVISOR_REQUESTS.md`** — his list, decoded, and what to reply.
