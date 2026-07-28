# HOW TO READ — 03_FIGURES

## The short version

- **34 figures.** The 11 in `main_paper/` are selected for the manuscript; the 23 in
  `supplementary/` are for the appendix.
- **One colour rule explains almost every figure:**
  **grey = uncertainty-only (baseline), green = dual-metric (proposed).**
- **The four figures carrying the most weight:**
  - `01` — the headline: the green bar is shorter in every single configuration
  - `14` — the ablation: evidence that both signals are needed. **The strongest evidence in the study**
  - `10` — the efficiency plot: shows the method spends *more* labels, not fewer
  - `28` — Grad-CAM: the classifier attended to a camera artefact; the risk head caught the lesion
- **Two diagrams still need to be drawn by hand** — see "Figures still to draw" at the bottom.

**Everything below covers the figures one at a time** — what each shows, how to read it, and how to
report it.

---

## The visual conventions — the same in every figure

| Colour | Hex | Always means |
|---|---|---|
| ⬛ **Grey** | `#6b7280` | **Uncertainty-only** — the baseline |
| 🟩 **Green** | `#1b7a5e` | **Dual-metric** — the proposed method |
| 🟧 **Amber** | `#b45309` | Highlight / accent — usually melanoma, or a warning |

**Grey vs green is the comparison in almost every figure.** Once that is internalised, most figures
read at a glance.

Other recurring conventions:

- **Error bars and shaded bands** = 95% confidence intervals, or min–max across the four
  uncertainty methods (stated in each caption).
- **Dashed diagonal line** = perfect calibration (reliability diagrams only).
- **Dashed horizontal line at 0.5** = random-guess level (AUC plots).
- Panel titles are model names: `resnet50`, `densenet169`, `efficientnet_b4`.
- Class codes on axes: `akiec bcc bkl df mel nv vasc` — see
  `02_METHODS_AND_MATH/NOTATION_AND_ABBREVIATIONS.md` §2.
- **"pp"** on an axis means **percentage points**, i.e. a difference between two percentages.

---

# The 11 main-paper figures

### `01_unsafe_auto_accepts_total.png` — **the headline safety result**
Bar chart, one pair of bars per configuration.
**Read it:** every green bar is shorter than its grey partner. **12 of 12, no exceptions.**
**How to report it:** the primary endpoint improved in every single configuration tested — and Proposition 1
explains why it had to.
→ Paper §5.1.

### `09_headline_summary.png` — the one-glance summary
Two panels: percentage reduction in unsafe auto-accepts, and in missed-cancer rate, per
configuration, with the mean annotated in each panel title.
**Read it:** the left panel is consistently and substantially negative; the right panel is mixed.
**How to report it:** that contrast *is* the study's honest finding — the escalation decision improved reliably,
the downstream model outcome did not.
→ Paper §5.2. Also the clearest single summary figure for a presentation.

### `10_al_efficiency_accuracy_vs_labels.png` — **the honest efficiency plot**
Three panels (one per backbone). **X-axis is labelled samples, not rounds** — this is the whole
point. Plotting against rounds would flatter the method for simply asking more questions.
**Read it:** the green curve sits to the *right* — it reaches similar accuracy but spends more
labels to get there.
**How to report it:** at matched budget the method is 0.35 pp behind. It is a safety intervention, not an
efficiency gain.
→ Paper §5.4.

### `14_ablation_both_signals_needed.png` — **the strongest evidence in the study**
Three panels: high-risk catch rate, unsafe auto-accepts, and annotation cost, each with four bars —
random / uncertainty-only / risk-only / dual.
**Read it:**
- Uncertainty-only (12.6%) barely beats random (10.2%) despite spending 4,339 labels.
- Risk-only reaches 17.2% for 1,917 labels — cheapest per unit of safety.
- Dual reaches 29.3% ≈ 12.6 + 17.2, i.e. **near-additive**.
**How to report it:** near-additivity means the two routes flag largely *different* images. That is exactly what
an ablation is meant to establish, and it justifies the dual design.
→ Paper §5.5.

### `16_safety_cost_pareto.png` — the achievable frontier
X = oracle labels requested (cost), Y = unsafe auto-accepts (harm). **Down and to the left is
better.**
**Read it:** the dual-metric points define the frontier; the baseline sits above it.
**How to report it:** the method does not occupy one lucky operating point — it dominates across the range.
→ Paper §5.6.

### `17_reliability_classification.png` — calibration of the diagnosis
X = confidence claimed, Y = frequency actually observed. **Dashed diagonal = perfect.**
**Read it:** points sag **below** the diagonal ⇒ **overconfident**.
**How to report it:** claiming 95.8%, delivering 88.6% — a ~7 pp gap (ECE 0.073). Common in modern networks, and
undetectable from accuracy alone — which is precisely why calibration must be reported separately.
→ Paper §5.8.

### `18_reliability_risk_head.png` — calibration of the escalation score
Same axes, but for the risk head's P(malignant).
**Read it:** closer to the diagonal than figure 17.
**How to report it — this is the important one.** The escalation threshold is applied to *this* score, so its
trustworthiness is what the safety mechanism rests on — and it is the better-calibrated of the two
(ECE 0.056 vs 0.073).
→ Paper §5.8.

### `22_auc_per_class_with_ci.png` — **per-class AUC**
Horizontal bars, one per lesion class, with 95% bootstrap CIs. **Melanoma highlighted in amber.**
**Read it:** melanoma is last at 0.945 [0.929, 0.960].
**How to report it:** melanoma being hardest is expected and honest — it is the class most easily confused with
an ordinary mole, which is precisely why it is dangerous. Report PR-AUC (0.781) alongside; the gap
is the normal signature of a rare, hard class.
**Caution:** `df` (9 test images) and `vasc` (14) have very wide intervals. Do not over-read them.
→ Paper §5.7.

### `25_significance_heatmap.png` — the statistics at a glance
Grid of metrics × statistics, colour-coded by significance.
**Read it:** two rows are decisively significant (unsafe auto-accepts, query cost); accuracy is
significant; F1, missed-cancer rate and melanoma recall are not.
**How to report it:** always quote the **Holm-corrected** column. Melanoma recall is the trap — raw p = 0.057
looks borderline, corrected p = 0.170 is not.
→ Paper §6.

### `28_gradcam_panel_resnet50_entropy_dual_metric.png` — **the picture that tells the story**
Three columns per case: the original image, the **classification head's** heat-map ("why this
disease?"), and the **risk head's** heat-map ("why dangerous?"). Red = most influential region.
**Read it:** for `ISIC_0027776` — a true melanoma — the classification head attended to the **dark
corners of the image** (a camera artefact) and predicted benign keratosis, while the **risk head
attended to the lesion itself** and scored 0.83, high enough to escalate.
**How to report it:** this is the paper's argument in a single image — and it is not a fluke: DenseNet-169 makes
the same mistake on the same photo and its risk head also rescues it (0.60). The panel also
includes an honest failure case where both heads attended to the corners and both missed.
→ Paper §5.10.

### `33_robustness_by_model.png` — **why the average is not enough**
Metrics under each corruption, broken down **per backbone**.
**Read it:** under Gaussian noise, ResNet-50 → 0.710 and DenseNet-169 → 0.692, while
EfficientNet-B4 → **0.008**, far below the ~0.14 of random guessing.
**How to report it:** the 52.7% mean retention conceals a total architecture-specific failure. Report per model.
The identical corruption produces an orderly drop in the other two backbones, so this is not a
flaw in the test.
→ Paper §5.9.

---

# The 23 supplementary figures

### Base results (`01`–`09`, from `analysis/build_analysis.py`)

| Figure | Shows | Use |
|---|---|---|
| `02_fn_rate_malignant_final.png` | Missed-cancer rate at the final round | Shows the mixed result honestly — a few configurations are slightly worse |
| `03_accuracy_f1_final.png` | *"Safety gain does not come at the cost of accuracy"* | §5.3 |
| `04_query_cost_total.png` | Oracle labels used — **the price** | §5.2 |
| `05_unsafe_auto_accepts_trajectory_by_model.png` | Unsafe accepts per round; shaded band = min–max across the 4 methods | Shows green below grey from early rounds |
| `06_fn_rate_trajectory_by_model.png` | Missed-cancer rate over rounds | Noisier — supports the "not significant" reading |
| `07_accuracy_trajectory_by_model.png` | *"Learning curves climb the same way regardless of policy"* | Evidence the policy does not slow learning |
| `08_risk_score_auroc_trend.png` | Risk-head AUROC every round, ±1 s.d. band | The risk signal is stable across rounds and models (0.962) |

### Efficiency (`11`–`13`)

| Figure | Shows |
|---|---|
| `11_al_efficiency_by_method.png` | Accuracy vs labels for **all 12** model × method cells — the full grid behind figure 10 |
| `12_melanoma_recall_vs_labels.png` | Melanoma sensitivity vs annotation budget — the clinically meaningful version |
| `13_annotation_efficiency.png` | **Budget-matched** comparison: identical label counts for both policies. The direct evidence for the −0.35 pp result |

### Threshold sweep (`15`)

`15_risk_threshold_sweep.png` — twin y-axes: unsafe auto-accepts (green, left) and images escalated
(right) against the risk threshold. **The rightmost point is the risk route disabled**, i.e. the
baseline. Traces the entire safety/cost dial in one figure.

### Calibration (`19`, `20`)

| Figure | Shows |
|---|---|
| `19_calibration_ece_comparison.png` | ECE per configuration, classification head beside risk head — the risk head is consistently lower |
| `20_calibration_brier_comparison.png` | Same for Brier score. **Note:** the risk head's lower Brier (0.065 vs 0.188) is partly because 2 classes is an easier task than 7 — do not present it as a like-for-like win |

### Per-class discrimination (`21`, `23`)

| Figure | Shows |
|---|---|
| `21_roc_curves_per_class.png` | One-vs-rest ROC curves, dual-metric runs pooled, one panel per backbone |
| `23_melanoma_auc_dual_vs_baseline.png` | Melanoma AUC, dual vs baseline, with bootstrap CIs — the intervals overlap, i.e. no significant difference. Include it; it is honest |

### Statistics (`24`)

`24_forest_plot_accuracy.png` — forest plot of per-configuration differences (dual − uncertainty)
with 95% paired-bootstrap CIs. **Read it:** an interval crossing zero means that configuration's
difference is not individually resolvable. Most cross zero while nearly all sit on the same side —
which is precisely the "11 of 12 point the same way" argument. Pairs naturally with figure 25.

### Runtime (`26`, `27`)

| Figure | Shows |
|---|---|
| `26_runtime_breakdown.png` | Modelled seconds per round, decomposed into train / query / evaluate, from **direct microbenchmarks** — not the discarded regression |
| `27_runtime_scaling.png` | Round time vs labelled-set size, by uncertainty method. MC-dropout's slope is visibly steeper |

### Explainability (`28` ×2 more)

`28_gradcam_panel_densenet169_...` and `28_gradcam_panel_efficientnet_b4_...` — the same analysis on
the other two backbones. **DenseNet-169 reproduces both the failure and the rescue on the same
image**, which is what elevates the ResNet case study from anecdote to reproducible finding.

### Robustness (`29`, `30`)

| Figure | Shows |
|---|---|
| `29_robustness_degradation.png` | All metrics under all corruptions, mean ± s.d. |
| `30_robustness_heads_compared.png` | *"Which head degrades faster?"* — % of clean performance retained. **The risk head retains 90.3% vs the classifier's 85.2%.** A genuine win for the two-head design: the system escalates rather than confidently mis-diagnosing |

### The two-head test (`32`)

`32_risk_head_decoupling.png` — risk head vs summed malignant probabilities on nested populations.
**Read it carefully:** on all images the two are a tie (0.9520 vs 0.9524). On the classifier's
false negatives the risk head flags 5.6% vs 0.6%.
> ⚠️ **The sub-chance bars are a selection artefact.** Conditioning on classifier error defines a
> population by the classifier being wrong, which mechanically depresses any correlated score. The
> figure carries an on-plot note saying so. **Compare the two bars to each other, never to 0.5.**

---

## Figures still to draw

Two figures the paper wants that no script produces, because they are diagrams rather than plots:

1. **The two-head architecture schematic** — image → shared backbone → two heads. A schematic in
   the ASCII form given in `02_METHODS_AND_MATH/EXPERIMENTAL_SETUP.md` §3 is the reference.
2. **The uncertainty × risk 2×2 quadrant diagram** — showing that the *low-uncertainty,
   high-risk* cell (confident but dangerous) is the one the two policies disagree about. This is
   the clearest way to convey the idea in a single panel and belongs early in the paper.

Both are drawing tasks, not analysis tasks.

---

## Regenerating any figure

Every figure is reproducible from the saved checkpoints — no retraining required. See
`06_STATUS_AND_OPEN_ITEMS/HOW_TO_REGENERATE.md` for the module that produces each one.
