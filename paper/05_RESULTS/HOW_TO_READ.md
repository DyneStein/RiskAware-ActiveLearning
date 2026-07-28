# HOW TO READ — 05_RESULTS

## The short version

- Two documents. Both present the results as prose rather than tables, and both are suitable as
  raw material for the Results and Discussion sections.
- **`FINDINGS.md`** — the overall verdict on the 24-experiment matrix. Read this first.
- **`RESPONSE_TO_REQUESTED_CHANGES.md`** — the additional analyses (calibration, statistical
  significance, ablation, runtime, robustness, explainability, formal definitions), organised by
  the requests that prompted them.
- Both include results that do not favour the method. That is deliberate.

---

## `FINDINGS.md` (~1,500 words)

| Section | Contents |
|---|---|
| The headline answer | Every metric with its significance verdict |
| **Correction on the accuracy result** | Why the +0.60 pp gain must not be described as label efficiency. **The most important paragraph in the document** |
| Figure-by-figure | Walkthrough of base figures 01–09 |
| Tables | What each of the three summary tables contains |
| Honest caveats | Single seed; why unsafe auto-accepts improved while missed-cancer rate did not; the finding that uncertainty sampling barely beats random selection |

## `RESPONSE_TO_REQUESTED_CHANGES.md` (~4,000 words)

| Section | Contents |
|---|---|
| Status at a glance | 12 of 13 requested items complete |
| §1–3 | Calibration: ECE, Brier score, reliability diagrams, temperature scaling |
| §4 | Active-learning efficiency, including the negative result |
| §5 | Per-class AUC, plus one result that was not requested but is reported anyway |
| §6–7 | Statistical testing at three levels, with what each level can and cannot conclude |
| §8 | Runtime — including the method that was discarded and why |
| §9 | The decision-level ablation |
| §10 | The formal mathematical definitions |
| §11 | Explainability / Grad-CAM |
| §12 | Robustness — three findings, two of them unfavourable |
| Closing | What still requires compute or a download, and the ISIC contamination warning |

---

## One convention to be aware of

`RESPONSE_TO_REQUESTED_CHANGES.md` sometimes reports a quantity **split by policy** (dual-metric in
one column, baseline in the other), while `01_PAPER_WRITING_KIT/KEY_NUMBERS.md` reports the **mean
across all 24 runs**. Calibration, for example, appears as "0.0719 / 0.0736" in one and "0.073" in
the other.

**These are the same data grouped two ways, not a discrepancy.** Either is usable; the two
conventions should simply not be mixed within a single passage.

**For any number entering the manuscript, `KEY_NUMBERS.md` is the authority** — its values were
recomputed directly from the source CSVs, whereas these write-ups were produced at different points
and round to different precisions.
