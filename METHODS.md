# Formal Specification — Risk-Aware Active Learning with a Dual-Metric Escalation Policy

This document gives the precise mathematical definition of every component of the
framework, in a form that can be lifted directly into the Methods section of the paper.
Every definition here corresponds to a specific piece of code, named in the margin note
under each block, and was written by reading that code rather than the prose docs — where
the two disagreed, the code won.

---

## 1. Notation and problem setup

Let $\mathcal{C} = \{\texttt{akiec}, \texttt{bcc}, \texttt{bkl}, \texttt{df}, \texttt{mel}, \texttt{nv}, \texttt{vasc}\}$
be the label set, $K = |\mathcal{C}| = 7$. Fix the ordering above; $c_k$ denotes the $k$-th label.

The clinical risk partition splits $\mathcal{C}$ into disjoint high- and low-risk subsets:

$$\mathcal{C} = \mathcal{C}_{\text{high}} \;\sqcup\; \mathcal{C}_{\text{low}}, \qquad
\mathcal{C}_{\text{high}} = \{\texttt{mel}, \texttt{bcc}, \texttt{akiec}\}$$

and induces the **malignancy indicator**

$$m(y) = \mathbb{1}\!\left[\, y \in \mathcal{C}_{\text{high}} \,\right] \in \{0, 1\}.$$

This partition is fixed a priori from clinical criteria (melanoma, basal cell carcinoma
and actinic keratoses are the malignant or pre-malignant classes of HAM10000); it is not
learned and never changes.

The dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}$, $N = 10{,}015$, is partitioned once,
with a fixed seed, into a held-out test set and an active-learning pool:

$$\mathcal{D} = \mathcal{D}_{\text{test}} \;\sqcup\; \mathcal{D}_{\text{pool}},
\qquad |\mathcal{D}_{\text{test}}| = 1{,}905, \qquad |\mathcal{D}_{\text{pool}}| = 8{,}110 .$$

$\mathcal{D}_{\text{test}}$ is never queried, never trained on, and is byte-identical
across all 36 experiments (verified by checksum), which is what makes the paired
image-level tests of §9 valid.

At active-learning round $t \in \{1, \dots, T\}$, $T = 15$, the pool is split into a
labelled set and an unlabelled pool:

$$\mathcal{D}_{\text{pool}} = \mathcal{L}_t \;\sqcup\; \mathcal{U}_t .$$

$\mathcal{L}_1$ is a class-stratified seed set of 70 images per class ($|\mathcal{L}_1| = 490$
before the first round's queries). Because the pool is closed,
$|\mathcal{L}_t| + |\mathcal{U}_t| = 8{,}110$ for every $t$ — a constraint that matters in §10.

> `constants.py`, `config.py`, `data/pool_manager.py`

---

## 2. The two-head model

A single backbone $g_\phi$ (ResNet-50, DenseNet-169, or EfficientNet-B4, ImageNet-pretrained)
maps an image to a shared feature vector, which feeds **two independent heads**:

$$z_i = g_\phi(x_i) \in \mathbb{R}^d$$

$$\mathbf{p}_i = \operatorname{softmax}\!\big(h_{\text{cls}}(z_i)\big) \in \Delta^{K-1}
\qquad\text{(7-class diagnostic distribution)}$$

$$r_i = \Big[\operatorname{softmax}\!\big(h_{\text{risk}}(z_i)\big)\Big]_{2} \in [0, 1]
\qquad\text{(risk score: } \widehat{P}(\text{malignant} \mid x_i) \text{)}$$

where $h_{\text{cls}}: \mathbb{R}^d \to \mathbb{R}^{7}$ and
$h_{\text{risk}}: \mathbb{R}^d \to \mathbb{R}^{2}$ are separate MLPs with dropout
(rates $p_{\text{drop}} = 0.3$), sharing no parameters with each other.

**Design rationale, stated formally.** An earlier version defined the risk score as a
functional of the classification head, $r_i = \sum_{k : c_k \in \mathcal{C}_{\text{high}}} p_{ik}$.
That construction makes the risk score measurable with respect to $\mathbf{p}_i$, so any
confident classification error forces a correspondingly wrong risk score — precisely in the
case the system exists to catch. Giving the risk head its own parameters removes that
functional dependence: $r_i$ and $\mathbf{p}_i$ are conditionally dependent only through the
shared features $z_i$. §7 of the analysis quantifies what this bought, by scoring both
definitions against the same ground truth.

> `models/base_model.py`, `risk_score/clinical_risk.py`

---

## 3. Training objective

At each round, the model is re-initialised from ImageNet weights and trained for
$E = 10$ epochs on $\mathcal{L}_t$. Both heads are trained jointly from the same labels —
the risk head's binary target is derived from the 7-class label via $m(\cdot)$, so no
additional annotation is required:

$$\mathcal{J}(\theta) \;=\; \sum_{i \in \mathcal{L}_t}
\Big[\;
\underbrace{w_{y_i} \cdot \ell\big(h_{\text{cls}}(z_i),\, y_i\big)}_{\text{classification}}
\;+\;
\underbrace{v_{m(y_i)} \cdot \ell\big(h_{\text{risk}}(z_i),\, m(y_i)\big)}_{\text{risk}}
\;\Big],
\qquad \theta = (\phi, h_{\text{cls}}, h_{\text{risk}})$$

with $\ell$ the cross-entropy loss. The two terms are summed with equal weight and
backpropagated in one optimiser step (Adam, $\eta = 10^{-4}$, weight decay $10^{-5}$,
cosine-annealed over the round's epochs).

**Class weights** follow the inverse-frequency ("balanced") rule, recomputed from the
*current* labelled set every round — so they track the evolving composition of
$\mathcal{L}_t$ rather than the original distribution:

$$w_k \;=\; \frac{|\mathcal{L}_t|}{K \cdot n_k}, \qquad n_k = \big|\{i \in \mathcal{L}_t : y_i = c_k\}\big|,$$

$$v_j \;=\; \frac{|\mathcal{L}_t|}{2 \cdot n^{\text{risk}}_j}, \qquad
n^{\text{risk}}_j = \big|\{i \in \mathcal{L}_t : m(y_i) = j\}\big|, \quad j \in \{0,1\}.$$

Both are clipped below at a count of 1 to stay finite. HAM10000 is severely imbalanced
($\texttt{nv}$ is $\approx 67\%$ of images), and unweighted training collapses toward the
majority benign class, which inflates exactly the missed-malignancy rate the paper reports.

> `active_learning/al_loop.py::compute_class_weights`, `::compute_risk_class_weights`,
> `models/base_model.py::train_model`

---

## 4. Uncertainty functionals

An uncertainty measure is a map $u : \Delta^{K-1} \to \mathbb{R}_{\geq 0}$, larger meaning
less certain. Four are implemented and each is reported on its **own natural scale** —
deliberately not rescaled to $[0,1]$, because §6 calibrates a separate threshold per method,
which makes a shared range unnecessary and a forced rescaling misleading.

| Name | Definition | Range |
|---|---|---|
| Shannon entropy | $u_{H}(\mathbf{p}) = -\sum_{k=1}^{K} p_k \log p_k$ | $[0, \log 7] \approx [0, 1.946]$ |
| Least confidence | $u_{\text{LC}}(\mathbf{p}) = 1 - \max_k p_k$ | $[0, 1 - 1/K]$ |
| Margin | $u_{M}(\mathbf{p}) = 1 - \big(p_{(1)} - p_{(2)}\big)$ | $[0, 1]$ |
| MC-dropout | $u_{\text{MC}} = \dfrac{1}{K}\sum_{k=1}^{K} \operatorname{Var}_{s}\!\big(p^{(s)}_{k}\big)$ | $[0, 0.25]$ |

where $p_{(1)} \geq p_{(2)}$ are the two largest entries of $\mathbf{p}$, and for MC-dropout
$\{\mathbf{p}^{(s)}\}_{s=1}^{S}$ are $S = 30$ stochastic forward passes with dropout active
and batch-norm frozen in evaluation mode. Probabilities are clipped to $[10^{-10}, 1]$
before the logarithm.

Note that MC-dropout perturbs **only the classification head**: the risk head is always
evaluated by a single deterministic pass. The risk route is a safety signal, not an
uncertainty-scored one, and making it stochastic would let a dangerous case escape review
through sampling noise.

> `uncertainty/*.py`, `models/base_model.py::predict_with_mc_dropout`

---

## 5. The risk score

$$r_i \;=\; \Big[\operatorname{softmax}\big(h_{\text{risk}}(g_\phi(x_i))\big)\Big]_{2}$$

a single deterministic forward pass, interpretable as the model's estimate of
$P(y_i \in \mathcal{C}_{\text{high}} \mid x_i)$. Whether that interpretation is
*earned* — i.e. whether $r_i = 0.8$ really corresponds to an 80% malignancy rate — is a
calibration question, defined in §8 and measured in the calibration analysis.

> `models/base_model.py::predict_risk`

---

## 6. Per-round threshold calibration

Thresholds are **not** fixed constants. At every round, immediately after training, the
model scores its own labelled set and the empirical 90th percentile of each score becomes
that round's threshold:

$$\tau^{u}_{t} \;=\; Q_{90}\Big(\big\{\, u(\mathbf{p}_i) \;:\; i \in \mathcal{L}_t \,\big\}\Big),
\qquad
\tau^{r}_{t} \;=\; Q_{90}\Big(\big\{\, r_i \;:\; i \in \mathcal{L}_t \,\big\}\Big)$$

where $Q_{90}$ is the linear-interpolation 90th percentile (`numpy.percentile` default).

Recalibrating every round is load-bearing. A threshold fixed against round 1's weak model
goes stale within a few rounds: as the model improves, its scores concentrate, almost
nothing clears the old bar, and escalation silently collapses to zero. Re-deriving the bar
from the current model's own behaviour keeps the escalation *rate* roughly stable while the
absolute scores drift — visible in the logs as $\tau^{u}_t$ falling from $1.88$ to $\approx 0.05$
and $\tau^{r}_t$ rising toward $1$ over 15 rounds.

A manual override $\tau^{r} \equiv \tau_{\text{fixed}}$ is available for the
threshold-sensitivity sweep.

> `active_learning/al_loop.py::calibrate_thresholds`

---

## 7. The escalation policies

Fix a round $t$. Every image in the unlabelled pool $\mathcal{U}_t$ carries a score pair
$(u_i, r_i)$. Let $K_{\text{budget}} = 150$ and let
$\operatorname{Top}_K(u, \mathcal{U}_t)$ denote the indices of the $K$ largest $u_i$.

**Uncertainty route** — budgeted, with threshold overflow:

$$\mathcal{A}_t \;=\; \operatorname{Top}_{K_{\text{budget}}}\!\big(u, \mathcal{U}_t\big)
\;\cup\;
\big\{\, i \in \mathcal{U}_t \;:\; u_i > \tau^{u}_{t} \,\big\}$$

The top-$K$ term is a **floor**, not a ceiling: it guarantees at least $K$ queries even if
nothing clears the threshold, while the union term lets the round exceed $K$ whenever more
images genuinely qualify. Setting $K_{\text{budget}} = 0$ recovers a pure threshold rule.

**Risk route** — uncapped, no budget:

$$\mathcal{B}_t \;=\; \big\{\, i \in \mathcal{U}_t \;:\; r_i > \tau^{r}_{t} \,\big\}$$

The absence of a budget here is deliberate and is the paper's central design commitment:
a case the risk head flags as dangerous must never be skipped merely because that round's
uncertainty budget was already spent.

**The two policies** are then

$$\boxed{\;\mathcal{E}^{\text{unc}}_{t} \;=\; \mathcal{A}_t \;}
\qquad\qquad
\boxed{\;\mathcal{E}^{\text{dual}}_{t} \;=\; \mathcal{A}_t \,\cup\, \mathcal{B}_t \;}$$

with the complementary auto-accept sets
$\mathcal{S}^{\pi}_{t} = \mathcal{U}_t \setminus \mathcal{E}^{\pi}_{t}$ for $\pi \in \{\text{unc}, \text{dual}\}$.

**Quadrant labelling.** For plots and analysis only, each image is additionally tagged by
which side of each threshold it falls on:

$$q_i = \big(\mathbb{1}[u_i > \tau^{u}_t],\; \mathbb{1}[r_i > \tau^{r}_t]\big)
\in \{\text{lo-lo}, \text{lo-hi}, \text{hi-lo}, \text{hi-hi}\}$$

The cell $q_i = \text{lo-hi}$ (*confident but dangerous*) is the one the two policies
disagree about: the baseline auto-accepts it, the dual-metric policy escalates it. This
labelling is descriptive — the decision is made by the set algebra above, not by $q_i$.

> `escalation/uncertainty_only.py`, `escalation/dual_metric.py`

---

## 8. Evaluation metrics, formally

**Oracle update.** For $i \in \mathcal{E}_t$ the true label is revealed, giving
$\mathcal{L}_{t+1} = \mathcal{L}_t \cup \mathcal{E}_t$, $\mathcal{U}_{t+1} = \mathcal{U}_t \setminus \mathcal{E}_t$.
The oracle is simulated by dataset lookup.

**Unsafe auto-accepts** (the primary safety metric, measured on the pool):

$$S^{\pi}_{t} \;=\; \Big|\;\big\{\, i \in \mathcal{U}_t \setminus \mathcal{E}^{\pi}_t \;:\; m(y_i) = 1 \,\big\} \;\Big|$$

— the number of genuinely malignant images the system waved through without review this
round. It uses the ground-truth label the system did **not** see, which is what makes it a
measure of the policy rather than of the model's self-assessment.

**False-negative rate on malignant classes** (the primary safety metric, measured on the
held-out test set): with $\hat{y}_i = \arg\max_k p_{ik}$,

$$\mathrm{FNR}_{\text{mal}} \;=\;
\frac{\big|\{\, i \in \mathcal{D}_{\text{test}} : m(y_i) = 1 \;\wedge\; m(\hat{y}_i) = 0 \,\}\big|}
     {\big|\{\, i \in \mathcal{D}_{\text{test}} : m(y_i) = 1 \,\}\big|}$$

Note this treats a melanoma predicted as basal cell carcinoma as **not** a false negative:
both are malignant, so the case is still escalated to a clinician. The melanoma-specific
variant $\mathrm{FNR}_{\text{mel}}$ replaces $m(\cdot)$ with $\mathbb{1}[\,\cdot = \texttt{mel}\,]$
and is therefore the stricter quantity.

**Calibration.** Partition the $n$ predictions into $M = 15$ bins $B_1, \dots, B_M$ by
confidence. With $\operatorname{acc}(B_m)$ the empirical accuracy in bin $m$ and
$\operatorname{conf}(B_m)$ its mean predicted confidence,

$$\mathrm{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n}\,\Big|\operatorname{acc}(B_m) - \operatorname{conf}(B_m)\Big|,
\qquad
\mathrm{MCE} = \max_{m}\,\Big|\operatorname{acc}(B_m) - \operatorname{conf}(B_m)\Big|.$$

Equal-width bins give the standard ECE; equal-mass bins give adaptive ECE, reported
alongside because confidences here saturate near 1 and equal-width bins are then nearly
empty in the middle of the range. The Brier score is the proper scoring rule

$$\mathrm{BS} = \frac{1}{n}\sum_{i=1}^{n}\sum_{k=1}^{K}\big(p_{ik} - \mathbb{1}[y_i = c_k]\big)^2
\;\in [0, 2],
\qquad
\mathrm{BS}_{\text{risk}} = \frac{1}{n}\sum_{i=1}^{n}\big(r_i - m(y_i)\big)^2 \in [0,1].$$

> `evaluation/metrics.py`, `evaluation/rigor/calibration.py`

---

## 9. Two structural properties of the dual-metric policy

These are elementary but worth stating, because they predict the empirical pattern and
bound what the experiments can possibly show.

> **Proposition 1 (escalation monotonicity).**
> For any round $t$ and any *fixed* score vectors $(u, r)$,
> $$\mathcal{E}^{\text{dual}}_{t} \supseteq \mathcal{E}^{\text{unc}}_{t}
> \quad\Longrightarrow\quad
> \mathcal{S}^{\text{dual}}_{t} \subseteq \mathcal{S}^{\text{unc}}_{t}
> \quad\Longrightarrow\quad
> S^{\text{dual}}_{t} \le S^{\text{unc}}_{t}.$$
>
> *Proof.* $\mathcal{E}^{\text{dual}}_t = \mathcal{A}_t \cup \mathcal{B}_t \supseteq \mathcal{A}_t = \mathcal{E}^{\text{unc}}_t$.
> Taking complements within $\mathcal{U}_t$ reverses the inclusion, and the cardinality of
> the malignant subset of a smaller set cannot be larger. $\blacksquare$

> **Proposition 2 (the cost is exactly the risk-only remainder).**
> $$\big|\mathcal{E}^{\text{dual}}_{t}\big| - \big|\mathcal{E}^{\text{unc}}_{t}\big|
> \;=\; \big|\mathcal{B}_t \setminus \mathcal{A}_t\big| \;\ge\; 0 .$$
> The additional annotation cost is precisely the set of images the risk route flags that
> the uncertainty route would have missed — never more.

**What these do and do not establish.** Proposition 1 holds *at fixed scores*, so it
applies exactly to the decision-level ablation, where every rule is replayed against one
identical model — and indeed the safety improvement there is uniform across all 24
experiments with no exceptions, as the proposition requires. It does **not** transfer
automatically to the full experiment: once the two policies request different labels, they
train on different data from round 2 onward and their score vectors diverge, so
$S^{\text{dual}}_{t} \le S^{\text{unc}}_{t}$ becomes an empirical claim rather than a
theorem. The measured result (a reduction in every one of the 12 configurations,
Wilcoxon $p = 5\times10^{-4}$) is therefore evidence, not arithmetic.

Together the two propositions frame the contribution correctly: the dual-metric policy is
not a free improvement but a **controlled trade** — strictly more safety for strictly more
annotation, with the exchange rate set by $\tau^{r}$ and traced empirically by the
threshold sweep.

---

## 10. A note on what the logs cannot identify

Because the pool is closed, $|\mathcal{L}_t| + |\mathcal{U}_t| = 8{,}110$ exactly, for every
round of every experiment. Any attempt to decompose the logged per-round wall-clock time as

$$\text{round\_time}_t \;\approx\; \beta_{\text{train}} |\mathcal{L}_t| + \beta_{\text{query}} |\mathcal{U}_t| + \gamma$$

is therefore rank-deficient: the two regressors and the intercept are exactly collinear, and
only the combination $\beta_{\text{train}} - \beta_{\text{query}}$ is identifiable. The
timings reported in the runtime analysis come from direct component-level measurement
instead, never from this regression.

---

## 11. Hyperparameters (reproducibility table)

| Parameter | Symbol | Value |
|---|---|---|
| Classes | $K$ | 7 |
| High-risk classes | $\mathcal{C}_{\text{high}}$ | mel, bcc, akiec |
| Test split | | 20% (1,905 images), fixed seed, shared by all runs |
| Pool size | $|\mathcal{D}_{\text{pool}}|$ | 8,110 |
| Seed labelled set | $|\mathcal{L}_1|$ | 490 (70 per class) |
| AL rounds | $T$ | 15 |
| Query budget (floor) | $K_{\text{budget}}$ | 150 per round |
| Epochs per round | $E$ | 10 |
| Batch size | | 32 |
| Learning rate | $\eta$ | $10^{-4}$ (Adam, cosine annealing) |
| Weight decay | | $10^{-5}$ |
| Image size | | $224 \times 224$ |
| Dropout rate | $p_{\text{drop}}$ | 0.3 |
| MC-dropout passes | $S$ | 30 |
| Threshold percentile | | 90th, recalibrated every round |
| Class weighting | | inverse-frequency, recomputed every round |
| Random seed | | 42 |
| Backbones | | ResNet-50, DenseNet-169, EfficientNet-B4 (ImageNet-pretrained) |

**Known limitation to declare in the paper:** EfficientNet-B4 is trained at $224\times224$
rather than its native $380\times380$, for compute parity with the other two backbones.
