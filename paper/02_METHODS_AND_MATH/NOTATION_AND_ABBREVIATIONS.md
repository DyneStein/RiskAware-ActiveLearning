# Notation and abbreviations — complete reference

Every symbol used in `METHODS.md`, every abbreviation used anywhere in this package, and every
label that appears on a figure axis or a table column.

Use this as the paper's notation table (§3.1) as well — reviewers expect one.

---

## 1. Mathematical symbols

### Sets and data

| Symbol | Read aloud as | Meaning |
|---|---|---|
| $\mathcal{C}$ | "the class set" | The 7 lesion classes |
| $K$ | | Number of classes = **7** |
| $c_k$ | | The $k$-th class label |
| $\mathcal{C}_{\text{high}}$ | "high-risk classes" | {mel, bcc, akiec} — the malignant / pre-malignant classes |
| $\mathcal{C}_{\text{low}}$ | "low-risk classes" | The remaining four |
| $m(y)$ | "malignancy indicator" | 1 if the true class is high-risk, else 0 |
| $\mathcal{D}$ | "the dataset" | All 10,015 images with labels |
| $N$ | | 10,015 |
| $\mathcal{D}_{\text{test}}$ | "the test set" | 1,905 images. Never queried, never trained on |
| $\mathcal{D}_{\text{pool}}$ | "the pool" | 8,110 images available for labelling |
| $\mathcal{L}_t$ | "L sub t" | Images **already labelled** at round $t$ |
| $\mathcal{U}_t$ | "U sub t" | Images **still unlabelled** at round $t$ |
| $t$ | | Round index, 1 to 15 |
| $T$ | | Total rounds = **15** |
| $x_i, y_i$ | | Image $i$ and its true label |

> **The closed-pool constraint:** $|\mathcal{L}_t| + |\mathcal{U}_t| = 8{,}110$ for every round.
> Nothing enters or leaves the pool; labelling only moves an image from $\mathcal{U}$ to
> $\mathcal{L}$. This exact constraint is what makes the runtime regression in §10 impossible —
> the two counts are perfectly collinear.

### Model

| Symbol | Meaning |
|---|---|
| $g_\phi$ | The **backbone** — the shared feature extractor, parameters $\phi$ |
| $z_i$ | The feature vector for image $i$: $z_i = g_\phi(x_i) \in \mathbb{R}^d$ |
| $h_{\text{cls}}$ | The **classification head** — outputs 7 logits |
| $h_{\text{risk}}$ | The **risk head** — outputs 2 logits (benign / malignant) |
| $\mathbf{p}_i$ | The 7-class probability vector: $\operatorname{softmax}(h_{\text{cls}}(z_i))$ |
| $p_k$ | The probability assigned to class $k$ |
| $p_{(1)}, p_{(2)}$ | The largest and second-largest entries of $\mathbf{p}$ |
| $r_i$ | The **risk score** — $P(\text{malignant})$ from the risk head, in $[0,1]$ |
| $\Delta^{K-1}$ | "the probability simplex" — the set of valid 7-class probability vectors (non-negative, summing to 1) |

### Scores and decisions

| Symbol | Meaning |
|---|---|
| $u(\cdot)$ | An **uncertainty functional** — bigger means less certain |
| $u_H$ | Shannon entropy, range $[0, \log 7] \approx [0, 1.946]$ |
| $u_{\text{LC}}$ | Least confidence, $1 - \max_k p_k$, range $[0, 6/7]$ |
| $u_M$ | Margin, $1 - (p_{(1)} - p_{(2)})$, range $[0, 1]$ |
| $u_{\text{MC}}$ | MC-dropout variance over $S = 30$ passes, range $[0, 0.25]$ |
| $\tau^u_t$ | The **uncertainty threshold** at round $t$ — the 90th percentile of $u$ over $\mathcal{L}_t$ |
| $\tau^r_t$ | The **risk threshold** at round $t$ — the 90th percentile of $r$ over $\mathcal{L}_t$ |
| $K_{\text{budget}}$ | Top-K query floor = **150** per round |
| $\mathcal{A}_t$ | The **uncertainty route**: $\operatorname{Top}_{150}(u) \cup \{u_i > \tau^u_t\}$ |
| $\mathcal{B}_t$ | The **risk route**: $\{r_i > \tau^r_t\}$ — **uncapped** |
| $\mathcal{E}^{\text{unc}}_t$ | Escalation set, baseline policy $= \mathcal{A}_t$ |
| $\mathcal{E}^{\text{dual}}_t$ | Escalation set, proposed policy $= \mathcal{A}_t \cup \mathcal{B}_t$ |
| $\mathcal{S}^{\pi}_t$ | **Auto-accept set** — everything *not* escalated: $\mathcal{U}_t \setminus \mathcal{E}^{\pi}_t$ |
| $\pi$ | A policy, either `unc` or `dual` |
| $q_i$ | Quadrant tag — which side of each threshold image $i$ falls on. **Descriptive only**; the decision comes from the set algebra |

### Symbols that trip people up

| Symbol | It means |
|---|---|
| $\sqcup$ | Disjoint union — "split into two non-overlapping parts" |
| $\cup$ | Union — "either one or the other or both" |
| $\setminus$ | Set difference — "everything in the first that is not in the second" |
| $\supseteq$ | Superset — "contains all of, and possibly more" |
| $\mathbb{1}[\cdot]$ | Indicator — 1 if the condition holds, 0 otherwise |
| $\|\cdot\|$ | Size of a set — how many items it contains |
| $Q_{90}$ | The 90th percentile |

---

## 2. The seven classes

The three-letter codes appear on every figure axis and in every table.

| Code | Full name | Risk | Test-set count | Share |
|---|---|---|---|---|
| `akiec` | Actinic keratoses / intraepithelial carcinoma | **High** | 51 | 2.7% |
| `bcc` | Basal cell carcinoma | **High** | 89 | 4.7% |
| `bkl` | Benign keratosis-like lesions | Low | 206 | 10.8% |
| `df` | Dermatofibroma | Low | 9 | 0.5% |
| `mel` | **Melanoma** | **High** | 209 | 11.0% |
| `nv` | Melanocytic nevi (ordinary moles) | Low | 1,327 | 69.7% |
| `vasc` | Vascular lesions | Low | 14 | 0.7% |

**Total 1,905. Malignant (akiec + bcc + mel): 349 = 18.3%.**
*(Counts verified from `04_TABLES/per_class_auc.csv`, column `n_positive`.)*

Two consequences worth stating in the paper:

- `nv` alone is **69.7%** of the test set, so a model that predicted "ordinary mole" for every
  image would score ~70% accuracy while catching zero cancers. This is why accuracy is a poor
  headline metric here and **F1-macro** is used instead.
- `df` (9 images) and `vasc` (14 images) are so rare in the test split that their per-class AUCs
  have wide confidence intervals and should not be over-interpreted.

---

## 3. Abbreviations

### Methods

| Short | Full | Plain meaning |
|---|---|---|
| **AL** | Active learning | Model chooses which images get labelled |
| **MC-dropout** | Monte Carlo dropout | Run the model 30 times with random neurons switched off; disagreement = uncertainty |
| **LC** | Least confidence | Uncertainty = 1 − highest probability |
| **Top-K** | | Take the K highest-scoring items |
| **dual / dual_metric** | The dual-metric policy | **This project's method** |
| **unc / uncertainty_only** | The uncertainty-only policy | **The baseline** |
| **Grad-CAM** | Gradient-weighted class activation mapping | Heat-map of where the model looked |

### Metrics

| Short | Full | One-line meaning |
|---|---|---|
| **TP / TN / FP / FN** | True/false positive/negative | The four outcomes of a binary decision |
| **FN rate** | False-negative rate | Fraction of real cancers called harmless — **the safety metric** |
| **Recall / sensitivity** | | Of all real cancers, how many were caught = 1 − FN rate |
| **Precision** | | When it says cancer, how often is it right |
| **F1** | F1 score | Balanced combination of precision and recall |
| **F1-macro** | | F1 averaged over the 7 classes, **each weighted equally** |
| **AUC / AUROC** | Area under the ROC curve | Chance a random cancer scores higher than a random non-cancer. 0.5 = coin flip |
| **PR-AUC** | Area under the precision–recall curve | Fairer than AUC when the positive class is rare |
| **ECE** | Expected calibration error | Average gap between claimed confidence and actual accuracy |
| **MCE** | Maximum calibration error | The **worst** bin's gap, not the average |
| **Brier** | Brier score | Squared error of probability predictions. Lower is better |
| **NLL** | Negative log-likelihood | Another proper scoring rule; lower is better |
| **T** | Temperature | The single scalar that divides logits to fix overconfidence |

### Statistics

| Short | Full | Meaning |
|---|---|---|
| **CI** | Confidence interval | 95% throughout, percentile bootstrap |
| **pp** | Percentage points | The *difference* between two percentages |
| **n.s.** | Not significant | |
| **Holm p** | Holm–Bonferroni corrected p | Adjusted for having tested six metrics |
| **Wilcoxon** | Wilcoxon signed-rank test | Paired, rank-based, no bell-curve assumption |
| **McNemar** | McNemar's test | For two methods graded on the identical items |
| **dz** | Cohen's dz | Standardised effect size for paired data |
| **rank-biserial** | | Effect size for Wilcoxon; −1.0 means every pair moved the same way |

### Project-specific terms

| Term | Meaning |
|---|---|
| **Oracle** | The simulated expert — a lookup of the true label in the dataset metadata |
| **Round** | One cycle of: train → score the pool → escalate → get labels → repeat. 15 in total |
| **Escalate** | Send to the oracle for a label instead of accepting the model's answer |
| **Auto-accept** | Take the model's answer with no human review |
| **`unsafe_auto_accepts`** | **Malignant pool images auto-accepted.** Measured on the pool, every round. The primary endpoint |
| **`fn_rate_malignant`** | Malignant test images predicted as benign. Measured on the test set, at the end. Secondary |
| **Configuration** | One (backbone × uncertainty measure) combination — 12 of them |
| **Experiment** | One (backbone × uncertainty measure × policy) run — 24 of them |
| **Seed set** | The 490 pre-labelled images (70/class) the first model trains on |

---

## 4. Two distinctions that must never be blurred

**1. `unsafe_auto_accepts` vs `fn_rate_malignant`.**
The first is measured on the **pool**, every round, and reflects the **escalation decision** — the
thing the risk score directly controls. The second is measured on the **test set**, at the end, and
reflects **what the model learned**. The first improved by 43% (Holm p = 0.003); the second did not
reach significance (Holm p = 0.305). They are not interchangeable, and the paper must say which one
it means every single time.

**2. Percentage vs percentage point.**
Accuracy moving from 88.5% to 89.1% is **+0.60 pp**. Writing "+0.6%" means something different
(a relative increase, which would be 89.03%). Use **pp** for all differences.
