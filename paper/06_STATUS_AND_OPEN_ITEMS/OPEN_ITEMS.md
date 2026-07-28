# What is still missing

An honest list. Three items need compute or a download; the rest are writing and drawing tasks.

---

## Needs compute or a download

### 1. Multi-seed replication — **the biggest gap**

**What:** every configuration was run once, with seed 42. Standard practice is 5 seeds per
configuration.

**Why it matters:** it is the first thing a reviewer will ask for, and it is what would turn the
weaker results (F1-macro, missed-cancer rate, melanoma recall) from "suggestive" into reportable.

**Why it is not fatal:** the primary result improved in **24 of 24** experiments, and Proposition 1
guarantees its *direction* mathematically. Only its magnitude is subject to seed variation.

**Cost:** ~60 GPU-hours for 3–4 representative configurations × 5 seeds, targeting the safety
result specifically. The full 24 × 5 matrix would be ~470 hours and is not necessary.

**Blocked on:** GPU access.

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

### 3. JPEG corruption — the fifth robustness condition

**What:** the robustness analysis covers blur, brightness, contrast and Gaussian noise. JPEG
re-compression is defined in the pipeline but has not been run.

**Why it matters:** mildly. Four conditions already support both robustness findings. JPEG would
make the set complete.

**Cost:** ~40 minutes on CPU. **Nothing is blocking it** — the command is in
`HOW_TO_REGENERATE.md`.

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
