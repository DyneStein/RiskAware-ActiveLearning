# What is still missing

An honest list. **No GPU work remains.** One item needs a download; the rest are writing and
drawing tasks.

---

## Closed since the last revision

| Item | Outcome |
|---|---|
| **Comparison against published methods** | **Done.** CoreSet, BADGE, CLUE, VAAL × 3 backbones = 12 runs, 15/15 rounds each, cost-matched exactly. Unsafe auto-accepts fell in 15/15 comparisons by 25.6–60.8%. See `../COMPARISON/`. |
| **JPEG corruption** | **Done.** 90.5% of accuracy and 94.3% of risk-head AUROC retained; melanoma recall 0.702 → 0.472. |
| **EfficientNet-B4 noise collapse** | **Explained.** A progressive collapse specific to additive noise; the majority prediction flips from `nv` to `df` and 99.1% of the test set is labelled `df` by σ=0.05. Reportable as a finding. |
| **Rare-class AUC intervals** | **Diagnosed.** The narrow CIs on `df` (n=9) and `vasc` (n=14) are a ceiling artifact — 31% / 17% of bootstrap replicates return AUC exactly 1.000. Do not quote them. |
| **Pool-vs-test framing** | **Fixed** across the abstract, key numbers, claims map, outline, limitations and findings. See `POOL_VS_TEST_FRAMING.md`. |

---

## Needs a download

### 1. Multi-seed replication — **deliberately not being run**

**Decision:** seed 42 for everything, on supervisor direction. This is a recorded decision, not an
oversight, and should be presented as such.

**What substitutes for it:** `SPLIT_SEED` is frozen separately from `RANDOM_SEED`, so all 36 runs
share one identical 1,905-image test split (verified by hashing every run's split file: exactly one
distinct value). Statistical power therefore comes from **image-level** paired testing at
n = 1,905, not from replicate runs. The result also holds on all three backbones, which is
replication across architectures rather than across initialisations.

**Still worth noting as a limitation**, because it is the most likely reviewer request and it is the
only thing that would move the *secondary* endpoints from "suggestive" to reportable. It would not
change the primary endpoint's direction, which Proposition 1 fixes structurally.

**If revisited:** the capability is implemented; a replicate writes to its own `_s<seed>` folder and
cannot overwrite existing results. ~60 GPU-hours for 3–4 configurations × 5 seeds.

**Blocked on:** nothing — a decision, not a blocker.

---

### 2. External validation on ISIC 2020

**What:** evaluate the trained models on a genuinely independent dataset.

**Why it matters:** the strongest available evidence that a medical model generalises. Currently
every result is within-dataset.

**Status:** code written, tested and ready. Needs the dataset.

**Cost:** ~3 GB download plus one inference pass. Best done in Colab.

**Expect the numbers to drop.** Different cameras, hospitals and populations always cost
something. That is the normal, publishable outcome.

> ⚠️ **Use ISIC 2020, not 2019.** ISIC 2019 contains HAM10000 inside it — testing on it would mean
> testing on training data. Full explanation in `06_STATUS_AND_OPEN_ITEMS/STATUS_CHECKLIST.md`.

---

### 3. ~~JPEG corruption~~ — **done**

Run. All five robustness conditions are now complete. Across all models: **90.5%** of clean accuracy
retained and **94.3%** of risk-head AUROC — but **melanoma recall falls from 0.702 to 0.472**. Worth
quoting as a clinical caution, since re-compression happens to every image that moves between
hospital systems. Table `robustness_summary.csv`, corruption `jpeg_q30`.

---

## Writing and drawing tasks

### 4. Two figures that need to be drawn by hand

No script produces these, because they are diagrams rather than plots:

- **The two-head architecture schematic** — image → shared backbone → two heads. Reference form in
  `02_METHODS_AND_MATH/EXPERIMENTAL_SETUP.md` §3.
- **The uncertainty × risk 2×2 quadrant diagram** — showing that the *confident but dangerous* cell
  is the one the two policies disagree about. The clearest single-panel explanation of the whole
  idea, and it belongs early in the paper.

### 5. The literature review

Related Work needs a fresh search. Nothing in this package substitutes for it. The four
subsections to cover are listed in `01_PAPER_WRITING_KIT/PAPER_OUTLINE.md` §2.

### 6. Repository documentation

`RESEARCHER.md` and `README.md` in the code repository still describe the **pre-redesign**
mechanism — they predate the two-head architecture. `METHODS.md` is correct, because it was written
from the source code.

**Low priority, but fix before any public code release**, since the repository is cited in the
paper.

---

## Analyses that could be added if asked

Neither is currently needed, but both are cheap and would answer a likely reviewer request.

| If a reviewer asks for… | The answer |
|---|---|
| **Fairness / subgroup analysis** | Missed-cancer rate broken down by age, sex and anatomical site — all three are already in the metadata and in the prediction dumps. A short script. **Skin tone cannot be analysed** — HAM10000 does not record Fitzpatrick type |
| **A retraining-level ablation** | The current ablation replays logged scores with the model held fixed, which isolates the escalation decision cleanly but does not capture how a different acquisition trajectory would change later training. A full version needs ~24 more runs |

---

## What is definitively finished

So the list above is not mistaken for a longer one than it is:

- ✅ All 24 experiments, all 15 rounds
- ✅ Calibration — ECE, MCE, Brier, NLL, reliability diagrams, temperature scaling
- ✅ Per-class AUC and PR-AUC with bootstrap confidence intervals
- ✅ Statistical significance at three levels, Holm-corrected throughout
- ✅ Active-learning efficiency, including the budget-matched comparison
- ✅ Decision-level ablation and the full risk-threshold sweep
- ✅ Runtime — training, inference and query, by direct measurement
- ✅ Robustness — 4 of 5 corruptions, broken down per model
- ✅ Explainability — Grad-CAM for all 3 backbones, both heads
- ✅ Formal mathematical specification with two proved propositions
- ✅ 34 figures, 21 tables, all regenerable by one command
