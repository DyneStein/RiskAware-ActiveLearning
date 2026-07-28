# Claim → evidence map

**The rule this project follows: a claim may appear in the paper only if a specific figure, table
or statistical test backs it.** This page is that mapping. Before submitting, check every claim in
the manuscript against this list.

Columns: the claim as it would be written → the artefact that proves it → the exact number →
whether it is safe to assert.

---

## ✅ Claims that are fully supported

| Claim | Evidence | Number | Status |
|---|---|---|---|
| Dual-metric escalation reduces unsafe auto-accepts | Fig `01`; `significance_configuration_level.csv` | −4,030 per run (≈43%), 12/12 configs, Holm p = 0.003 | ✅ **Assert confidently** |
| The improvement is universal, not average | `significance_ablation_level.csv` | 24/24 experiments improve; rank-biserial effect = −1.0 | ✅ Assert |
| The improvement is structurally guaranteed | METHODS §9, Proposition 1 | superset argument — proof, not statistics | ✅ Assert |
| The method costs more annotation | Fig `09`, `04`; significance table | +382 labels (+9.1%), 12/12 configs, Holm p = 0.003 | ✅ Assert (as the price) |
| The extra cost is exactly bounded | METHODS §9, Proposition 2 | cost = \|risk-route-only set\| | ✅ Assert |
| Classification quality is not harmed | Fig `03`; significance table | accuracy +0.60 pp (Holm p = 0.014); F1 n.s. | ✅ Assert as "no meaningful change" |
| Uncertainty sampling is a weak safety mechanism | Fig `14`; `ablation_decision_level.csv` | 12.58% capture vs 10.21% random | ✅ **Assert — strongest line** |
| The two signals are complementary | Fig `14`; same table | 12.58 + 17.24 ≈ 29.32 achieved (near-additive) | ✅ Assert |
| Risk-only escalation is cheap and effective | Same table | 17.24% capture for 1,917 labels vs 4,339 | ✅ Assert |
| The safety/cost trade is a continuous dial | Figs `15`, `16`; `risk_threshold_sweep.csv` | full sweep 0.05 → 1.01 | ✅ Assert |
| The risk head learned a real signal | `risk_auroc_by_experiment.csv`; Fig `08` | AUROC 0.962 across all rounds | ✅ Assert |
| Melanoma is the hardest class | Fig `22`; `per_class_auc.csv` | AUC 0.945 [0.929, 0.960]; PR-AUC 0.781 | ✅ Assert |
| Models are overconfident | Figs `17`–`20`; `calibration_metrics.csv` | ECE 0.073; 95.8% claimed vs 88.6% actual | ✅ Assert |
| The risk head is better calibrated than the classifier | Same | ECE 0.056 vs 0.073 | ✅ Assert |
| Overconfidence is cheaply correctable | Same | T ≈ 2.16, held-out ECE 0.073 → 0.023 | ✅ Assert |
| The safety signal is more corruption-robust than the diagnosis | Fig `30`; `robustness_summary.csv` | risk AUROC retains 90.3% vs accuracy 85.2% | ✅ Assert |
| The escalation rule is computationally negligible | `runtime_components_measured.csv` | ≈ 8 ms per round | ✅ Assert |
| MC-dropout is substantially more expensive | `runtime_per_experiment.csv` | 7.00 h vs ≈2.9 h mean | ✅ Assert |
| The risk head rescues cases the classifier misses | `risk_head_decoupling.csv` | 5.57% vs 0.62% of false negatives flagged | ✅ Assert (with the caveat below) |
| The model sometimes attends to camera artefacts | Fig `28`; `gradcam_cases_*.csv` | `ISIC_0027776`, reproduced across 2 backbones | ✅ Assert as a case study |

---

## ⚠️ Claims that must be softened

| Claim as tempting to write | Actual evidence | How to write it instead |
|---|---|---|
| "Reduces missed-cancer rate" | −1.19 pp, Holm p = 0.305, only 3/12 configs improved | **"Suggestive but not statistically significant; pending multi-seed replication."** |
| "Improves melanoma recall" | +2.43 pp, Holm p = 0.170 (raw 0.057) | Same softening. Do **not** quote the raw p as if significant. |
| "Improves F1-macro" | +0.86 pp, Holm p = 0.305 | "No significant change in F1-macro." |
| "More label-efficient" | **−0.35 pp at matched budget**; ~300 more labels to any target | **Never claim this.** State the opposite explicitly. |
| "The two-head design outperforms summed probabilities" | 0.9520 vs 0.9524 — a tie on the full test set | "Equivalent overall; the benefit is confined to the classifier's error region (9× more false negatives flagged)." |
| "The method is robust to image degradation" | 52.7% accuracy retention under noise; one backbone collapses | "Robust to illumination and contrast shifts; **sensitive to blur and sensor noise**, with one architecture failing entirely." |
| "The risk head has low AUC on hard cases" | Sub-chance AUCs are a selection artefact | Compare the two scoring methods to **each other**, never to 0.5. |

---

## ❌ Claims that cannot be made at all, with the current data

| Claim | Why not | What would be needed |
|---|---|---|
| The result is robust to training randomness | Single seed (42), one run per configuration | 3–4 configs × 5 seeds ≈ 60 GPU-hours |
| The method generalises to other datasets | No external validation completed | ISIC 2020 evaluation (code ready, needs ~3 GB download) |
| The method helps real clinicians | The oracle is a metadata lookup, not a dermatologist | A reader study — out of scope; declare as a limitation |
| Results hold across skin tones | HAM10000 skews to lighter skin; Fitzpatrick type is not recorded | A more diverse dataset; declare as a limitation |
| Retraining under alternative policies gives the same ablation result | The ablation is a **one-step counterfactual** on logged scores with the model held fixed | Full retraining per rule (~24 more runs) |

---

## The pre-submission check

Go through the manuscript and, for each numbered claim:

1. Does it appear in the ✅ table above? → cite the artefact in the text.
2. Does it appear in the ⚠️ table? → rewrite using the suggested phrasing.
3. Does it appear in the ❌ table? → delete it, or move it to Future Work.
4. Does it appear nowhere here? → **either find the evidence or remove the claim.**

A claim with no artefact behind it is the single most common reason a methods-heavy paper gets sent
back.
