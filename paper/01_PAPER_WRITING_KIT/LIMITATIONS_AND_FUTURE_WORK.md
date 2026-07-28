# Limitations and future work

Draft text for the Limitations section, plus the future-work programme.

**Guiding principle:** every limitation here is one a competent reviewer would find. Stating them
first costs a paragraph; having them found costs the paper's credibility. The safety result is
strong enough to survive all of them.

---

## Limitations — draft text

### L1. Single random seed

Every configuration was run once, with seed 42. Standard practice is 5 seeds per configuration with
mean ± standard deviation reported, which directly measures run-to-run variance.

**Why this is not fatal:** the primary result improved in **24 of 24** experiments (rank-biserial
effect size −1.0) and in **12 of 12** matched pairs. A consistency that complete is difficult to
produce by training randomness alone. Moreover, Proposition 1 shows the direction of the effect is
structurally guaranteed at fixed scores, so only its *magnitude* is subject to seed variation.

**Why it still matters:** the weaker results — F1-macro, missed-cancer rate, melanoma recall — are
exactly the ones where seed variance could plausibly account for the observed differences, and none
of them reached significance.

**Write it as:** *"All results are from a single seed per configuration. The primary endpoint
improved in 24 of 24 experiments, and Proposition 1 guarantees its direction, but the secondary
endpoints should be regarded as suggestive pending multi-seed replication."*

### L2. Simulated oracle

The oracle is a lookup of the ground-truth label in the dataset metadata: instant, free, and always
correct. A real dermatologist is none of those. Inter-rater disagreement on dermoscopic images is
substantial, and annotation latency changes the economics of an active-learning loop.

This is standard practice in active-learning research and does not invalidate the comparison — both
policies use the identical oracle — but it means the reported annotation costs are a lower bound on
real-world cost.

### L3. No external validation yet

All results are within-dataset, on HAM10000. Cross-dataset evaluation is the strongest available
evidence that a medical model generalises, and it has not yet been run.

**Important detail to state:** ISIC 2019 is **not** a valid external test set for this work. It was
assembled from BCN20000 + **HAM10000** + MSK, and the HAM10000 images retain their original
`ISIC_xxxxxxx` identifiers. Evaluating on it as distributed would be evaluating on training data.
ISIC 2020 is the appropriate choice — independent patients, and its benign/malignant labels map
directly onto the risk head's target. The evaluation script enforces filename-level exclusion and
refuses to run unless the measured overlap is zero.

### L4. Dataset demographic skew

HAM10000 is drawn predominantly from European populations and skews towards lighter skin tones.
Fitzpatrick skin type is not recorded, so subgroup performance by skin tone **cannot be measured**
from this data at all — this is an absence of evidence, not evidence of fairness. Dermoscopic
appearance and lesion prevalence both vary with skin tone, so generalisation to darker skin is
unestablished.

*(Age, sex and anatomical site are available in the metadata and could support a subgroup analysis
of the missed-cancer rate — a straightforward addition if a reviewer requests fairness analysis.)*

### L5. EfficientNet-B4 trained below native resolution

B4 was trained at 224 × 224 for compute parity with ResNet-50 and DenseNet-169, rather than its
native 380 × 380. This likely explains both its lower clean accuracy (0.8625 vs 0.8961 and 0.8887)
and its complete collapse under Gaussian noise (accuracy 0.0079, below the ~0.14 of random
guessing). Conclusions about that backbone specifically should be treated with caution.

### L6. The method is not label-efficient

At a matched annotation budget, dual-metric escalation is **0.35 percentage points behind** on
accuracy and requires approximately **300 additional labels** to reach any given accuracy target.
The mechanism is straightforward: the policy deliberately spends labels on *dangerous* cases rather
than *informative* ones, and those are different cases. The apparent +0.60 pp gain at round 15 is
purchased with 382 extra labels.

**This belongs in the Results section, not only here.**

### L7. The ablation is decision-level, not retraining-level

The four-rule ablation replays each logged round under an alternative escalation rule **with the
model held fixed** — a one-step counterfactual on the recorded scores. It isolates the escalation
decision cleanly, which is precisely what is being compared, but it does not capture how a
different acquisition trajectory would have changed subsequent training. A full retraining ablation
would require roughly 24 additional runs.

### L8. Two safety metrics that are not interchangeable

`unsafe_auto_accepts` is measured on the unlabelled pool every round and reflects the escalation
decision directly. `fn_rate_malignant` is measured on the held-out test set at the end and reflects
what the final model learned. The first improved substantially; the second did not reach
significance. Both are reported, and the distinction is stated wherever either appears.

### L9. Shared backbone limits head independence

The classification and risk heads have independent parameters but a shared feature extractor. They
are decoupled in parameters, not in features, so a failure of the shared representation propagates
to both. This is the most likely explanation for L8's asymmetry, and it is discussed as a mechanism
rather than merely a caveat — see the Discussion.

---

## Future work

Ordered by value per unit of effort.

### F1. Multi-seed replication — highest priority
**What:** 3–4 representative configurations × 5 seeds, targeting the primary endpoint specifically.
**Cost:** ≈ 60 GPU-hours (the full 24 × 5 matrix would be ≈ 470 hours and is not necessary).
**Why first:** it converts the secondary results from "suggestive" to reportable, and it is the
first thing a reviewer will request.

### F2. External validation on ISIC 2020
**What:** evaluate the trained models on ISIC 2020, whose binary benign/malignant labels align with
the risk head. **Cost:** ≈ 3 GB download plus one inference pass; code is written and ready.
**Expect the numbers to drop** — different cameras, sites and populations always cost something.
That is the normal, reportable outcome, and reporting it honestly is worth more than omitting the
experiment.

### F3. Separate backbones for the two heads — the paper's own prescription
**What:** give the risk head its own feature extractor so the two can fail independently.
**Why:** directly targets L9, the identified mechanism limiting further gains. This is the natural
follow-up paper and should be named explicitly in the Conclusion.

### F4. Retraining-level ablation
Re-run the acquisition loop under each escalation rule rather than replaying logged scores,
removing the L7 caveat.

### F5. Calibrated thresholds
Temperature scaling reduced held-out ECE by 68%. Applying it **before** threshold calibration would
make the escalation thresholds correspond to genuine probabilities rather than raw scores — a
small change with a clean theoretical motivation.

### F6. Subgroup analysis
Missed-cancer rate broken down by age, sex and anatomical site — computable today from the existing
metadata, and a ready answer if fairness analysis is requested.

### F7. Clinician-in-the-loop study
Replace the simulated oracle with real annotators to measure genuine cost, latency and disagreement.
Out of scope here; the honest long-term direction.
