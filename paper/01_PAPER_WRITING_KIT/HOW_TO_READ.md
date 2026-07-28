# HOW TO READ — 01_PAPER_WRITING_KIT

## The short version

- This folder contains the working materials for writing the manuscript.
- **`PAPER_OUTLINE.md` is the spine.** Every section of the paper, with the figures, tables and
  numbers that belong in it. Work down it top to bottom.
- **`KEY_NUMBERS.md` is the authoritative number source.** Every value was recomputed from the
  source CSV named beside it. Any number entering the manuscript should come from here.
- **`CLAIMS_AND_EVIDENCE_MAP.md` is the verification step.** Each claim is mapped to the artefact
  that proves it, plus claims that must be softened and claims that cannot be made at all.
- `ABSTRACT_AND_CONTRIBUTIONS.md` contains a complete 248-word draft abstract and five contribution
  statements.
- `LIMITATIONS_AND_FUTURE_WORK.md` is the Limitations section in near-final draft form.

**Three conventions that apply throughout the paper:**
1. The method is presented as a **safety intervention with a measured price**, never as a
   label-efficiency gain. At matched budget it is 0.35 pp behind.
2. **Holm-corrected p-values are quoted**, not raw ones. Six metrics were tested as a family.
3. F1-macro, missed-cancer rate and melanoma recall are described as **not significant** —
   "suggestive, pending multi-seed replication."

---

## The files

| File | Contents |
|---|---|
| `PAPER_OUTLINE.md` | Nine sections plus appendices. For each: what to write, suggested length, which figures, which tables, which numbers. Includes the recommended writing order and six paper-specific writing rules. |
| `ABSTRACT_AND_CONTRIBUTIONS.md` | Draft abstract (248 words, with notes on why it is structured as it is), five contribution statements, the reusable framing sentence, and keywords. |
| `KEY_NUMBERS.md` | Ten sections covering scale, primary results, the ablation, efficiency, per-class discrimination, calibration, robustness, runtime, the two-head test, and a closing list of numbers **not** to quote. |
| `CLAIMS_AND_EVIDENCE_MAP.md` | Three tables: claims that are fully supported, claims that must be softened (with suggested phrasing), and claims that cannot be made with the current data. Ends with a pre-submission checklist. |
| `LIMITATIONS_AND_FUTURE_WORK.md` | Nine limitations as draft prose (L1–L9) and seven future-work items ranked by value per unit of effort. |

---

## Recommended writing order

Not the order the sections appear in — this order minimises rework.

1. **Methods** (§3) — already written formally in `02_METHODS_AND_MATH/METHODS.md`. Starting here
   fixes the terminology before anything else depends on it.
2. **Experimental Setup** (§4) — already written in `02_METHODS_AND_MATH/EXPERIMENTAL_SETUP.md`.
3. **Results** (§5) — the numbers exist; work down `PAPER_OUTLINE.md` §5.1–5.11.
4. **Discussion** (§7) — leads with the shared-backbone mechanism.
5. **Limitations** (§8) — largely a copy-edit of the limitations document.
6. **Introduction** (§1) — written after the results are settled.
7. **Related Work** (§2) — needs a literature search; can proceed in parallel.
8. **Abstract** — written last.
9. **Conclusion** (§9) — a compression of the abstract plus the one prescriptive recommendation.

---

## Abbreviations used in these documents

| Short | Full |
|---|---|
| **pp** | percentage points — the *difference* between two percentages. 88.5% → 89.1% is +0.60 **pp**, not +0.6% |
| **CI** | confidence interval (95% throughout, percentile bootstrap) |
| **Holm p** | p-value after Holm–Bonferroni correction for multiple comparisons |
| **n.s.** | not statistically significant |
| **AL** | active learning |
| **FN / FP** | false negative / false positive |
| **ECE / MCE** | expected / maximum calibration error |
| **AUC / AUROC** | area under the ROC curve |
| **PR-AUC** | area under the precision–recall curve |
| **dual** | the dual-metric policy (proposed) |
| **unc** | the uncertainty-only policy (baseline) |

Complete reference: `02_METHODS_AND_MATH/NOTATION_AND_ABBREVIATIONS.md`.

---

## Not included here

- **A draft of the manuscript prose.** These are the materials and the plan; the writing itself is
  outstanding.
- **The literature review.** Related Work requires a fresh search. The four subsections to cover
  are listed in `PAPER_OUTLINE.md` §2.
- **LaTeX templates.** Venue-specific.
