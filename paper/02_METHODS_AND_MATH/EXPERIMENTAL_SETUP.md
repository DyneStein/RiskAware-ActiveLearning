# Experimental setup

Everything needed to write §4 of the paper, and everything a reader needs to reproduce the study.

---

## 1. Dataset

**HAM10000** ("Human Against Machine with 10,000 training images") — 10,015 dermoscopic images of
pigmented skin lesions across 7 diagnostic categories, with patient metadata (age, sex, anatomical
site, diagnosis confirmation method, lesion ID).

### Class distribution — full dataset and test split

| Code | Full name | Clinical risk | Test-set count | Test-set share |
|---|---|---|---|---|
| `akiec` | Actinic keratoses / intraepithelial carcinoma | **High** | 51 | 2.7% |
| `bcc` | Basal cell carcinoma | **High** | 89 | 4.7% |
| `bkl` | Benign keratosis-like lesions | Low | 206 | 10.8% |
| `df` | Dermatofibroma | Low | 9 | 0.5% |
| `mel` | Melanoma | **High** | 209 | 11.0% |
| `nv` | Melanocytic nevi | Low | 1,327 | 69.7% |
| `vasc` | Vascular lesions | Low | 14 | 0.7% |
| | **Total** | | **1,905** | 100% |

**Malignant prevalence in the test set: 349 / 1,905 = 18.3%.**

**The imbalance is severe and must be stated.** `nv` alone is nearly 70% of the data; a
degenerate model predicting `nv` for every image scores ~70% accuracy while catching no cancers at
all. This drives three design decisions reported in the paper: inverse-frequency class weighting in
the loss, **F1-macro** rather than accuracy as the quality metric, and per-class AUC rather than a
single pooled AUC.

### Data splitting

$$\mathcal{D} \;=\; \mathcal{D}_{\text{test}} \;\sqcup\; \mathcal{D}_{\text{pool}},
\qquad 1{,}905 + 8{,}110 = 10{,}015$$

- **80 / 20 split**, stratified, fixed seed.
- The test set is **never queried and never trained on**.
- The **initial labelled seed set** is 490 images (70 per class), drawn **from within the 8,110-image
  pool** — not in addition to it.
- Because the pool is closed, $|\mathcal{L}_t| + |\mathcal{U}_t| = 8{,}110$ in every round.

> **State this in the paper:** the test split is **byte-identical across all 24 experiments**,
> verified by checksum (md5 prefix `feff50b2fec0`). This is what licenses the paired image-level
> statistical tests — each pair of policies is graded on exactly the same 1,905 items, so they can
> be compared image by image with McNemar's test.

---

## 2. The experiment matrix

**24 experiments = 3 backbones × 4 uncertainty measures × 2 escalation policies.**

| Factor | Levels |
|---|---|
| **Backbone** | ResNet-50, DenseNet-169, EfficientNet-B4 (all ImageNet-pretrained) |
| **Uncertainty measure** | Shannon entropy, margin, least confidence, MC-dropout |
| **Escalation policy** | uncertainty-only (baseline), dual-metric (proposed) |

This yields **12 matched pairs** — each (backbone × uncertainty) configuration run under both
policies, everything else held identical. Those 12 pairs are the unit of the primary statistical
analysis.

---

## 3. Model architecture

A shared ImageNet-pretrained backbone $g_\phi$ feeds **two independent heads**:

```
                              ┌──►  h_cls   ──►  softmax  ──►  p ∈ Δ⁶   (7-class diagnosis)
   image ──►  BACKBONE g_φ  ──┤
              (shared)        └──►  h_risk  ──►  softmax  ──►  r ∈ [0,1] (P(malignant))
```

- The heads have **independent parameters** but a **shared feature extractor**. This distinction is
  central to the paper's Discussion: they are decoupled in parameters, not in features, so a
  failure of the shared representation propagates to both.
- The risk head is trained on the binary malignancy indicator $m(y) = \mathbb{1}[y \in
  \{\texttt{mel}, \texttt{bcc}, \texttt{akiec}\}]$.
- **The high-risk partition is fixed a priori from clinical criteria.** It is never learned and
  never changes.
- MC-dropout perturbs **only the classification head**; the risk head is always a single
  deterministic pass, so a dangerous case cannot escape review through sampling noise.

---

## 4. Hyperparameters (the reproducibility table)

Reproduce this table verbatim in the paper.

| Parameter | Symbol | Value |
|---|---|---|
| Classes | $K$ | 7 |
| High-risk classes | $\mathcal{C}_{\text{high}}$ | mel, bcc, akiec |
| Test split | | 20% (1,905 images), fixed seed, shared by all runs |
| Pool size | $\lvert\mathcal{D}_{\text{pool}}\rvert$ | 8,110 |
| Initial labelled seed set | $\lvert\mathcal{L}_1\rvert$ | 490 (70 per class) |
| Active-learning rounds | $T$ | 15 |
| Query budget (floor, top-K) | $K_{\text{budget}}$ | 150 per round |
| Epochs per round | $E$ | 10 |
| Batch size | | 32 |
| Optimiser | | Adam, cosine annealing |
| Learning rate | $\eta$ | $1 \times 10^{-4}$ |
| Weight decay | | $1 \times 10^{-5}$ |
| Input resolution | | $224 \times 224$ |
| Dropout rate | $p_{\text{drop}}$ | 0.3 |
| MC-dropout passes | $S$ | 30 |
| Threshold percentile | | 90th, **recalibrated every round** |
| Class weighting | | inverse-frequency ("balanced"), recomputed every round, applied to **both** heads |
| Random seed | | 42 |
| Backbones | | ResNet-50, DenseNet-169, EfficientNet-B4 (ImageNet-pretrained) |

**Declare as a limitation:** EfficientNet-B4 is trained at $224 \times 224$ rather than its native
$380 \times 380$, for compute parity with the other two backbones. This likely contributes to both
its lower clean accuracy and its collapse under Gaussian noise.

---

## 5. Two design choices that need justifying in the text

### Why thresholds are recalibrated every round

Thresholds are the **90th percentile of the current model's scores on its own labelled set**,
recomputed after each round's training — not fixed constants.

A threshold fixed against round 1's weak model goes stale within a few rounds: as the model
improves its scores concentrate, almost nothing clears the old bar, and escalation silently
collapses to zero. Re-deriving the bar from the current model's own behaviour keeps the escalation
*rate* roughly stable while the absolute scores drift. This is visible in the logs: $\tau^u_t$
falls from 1.88 to about 0.05 across the 15 rounds, while $\tau^r_t$ rises towards 1.

### Why the risk route has no budget

The uncertainty route has a top-K **floor** of 150 (a floor, not a ceiling — the union with the
threshold rule lets a round exceed K whenever more images genuinely qualify). The risk route has
**no budget at all**.

This is the paper's central design commitment: a case the risk head flags as dangerous must never
be skipped merely because that round's uncertainty budget was already spent. It is also exactly
what makes the method cost more — Proposition 2 shows the extra annotation equals precisely the
risk-route-only set.

---

## 6. Compute

| Quantity | Value |
|---|---|
| Total compute, all 24 experiments | **94.1 GPU-hours** |
| Mean per experiment — entropy | 2.83 h |
| Mean per experiment — least confidence | 2.88 h |
| Mean per experiment — margin | 2.96 h |
| Mean per experiment — MC-dropout | **7.00 h** (≈ 2.4× the others, from 30 forward passes) |
| Training hardware | NVIDIA T4 (Google Colab), checkpoint/resume across sessions |
| Post-hoc analysis hardware | CPU, 8 threads |

**The escalation rule itself costs ≈ 8 ms per round.** The method's cost is annotation, not compute
— worth stating explicitly, because a reader may assume a two-signal policy is expensive to run.

> **Note on absolute timings:** the component benchmarks in `04_TABLES/runtime_components_measured.csv`
> were measured on CPU and are **thread-count dependent** (the thread count is recorded in
> `runtime_benchmark.json`). The **ratios** between models and between operations are stable; the
> absolute milliseconds are not. Report ratios.

---

## 7. Reproducibility statement — draft

> All experiments use a fixed random seed (42). The train/test split is generated once, saved to
> disk, and shared byte-identically across all 24 experiments (verified by checksum). Every round
> writes its configuration, per-image scores and decisions, per-round metrics, and a model
> checkpoint. All figures and tables in this paper are regenerated from those saved artefacts by a
> single command, with no manual steps. Code and analysis scripts are released at
> `github.com/DyneStein/RiskAware-ActiveLearning`.

Regeneration instructions: `06_STATUS_AND_OPEN_ITEMS/HOW_TO_REGENERATE.md`.
