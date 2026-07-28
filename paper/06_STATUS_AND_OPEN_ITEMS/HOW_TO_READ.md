# HOW TO READ — 06_STATUS_AND_OPEN_ITEMS

## The short version

- **`STATUS_CHECKLIST.md`** — all 17 requested items, what each one found, and the file that proves
  it. **12 of 13 complete.** Includes the ISIC 2019 contamination warning.
- **`OPEN_ITEMS.md`** — what remains. **Three items need compute or a download**; everything else
  is writing or drawing.
- **`HOW_TO_REGENERATE.md`** — how to rebuild every figure and table.
  **One command, no retraining, no GPU, ~30–45 minutes on CPU.**

---

## The files

| File | Use it for |
|---|---|
| `STATUS_CHECKLIST.md` | Reporting progress; locating the evidence for any requested item |
| `OPEN_ITEMS.md` | Planning remaining work; answering "what's left?" |
| `HOW_TO_REGENERATE.md` | Reproducing any number; answering reviewer reproducibility questions |

---

## What remains outstanding

| Item | Cost | Blocked on |
|---|---|---|
| **Multi-seed replication** (3–4 configurations × 5 seeds) | ~60 GPU-hours | GPU access |
| **External validation on ISIC 2020** | ~3 GB download + one inference pass | The download |
| **JPEG corruption** (5th robustness condition) | ~40 min CPU | Nothing |

Multi-seed replication is the highest-value item: it is the most likely reviewer request, and it
would move the currently non-significant secondary results (F1-macro, missed-cancer rate, melanoma
recall) from "suggestive" to reportable.

---

## ⚠️ Before anyone runs external validation

**ISIC 2019 physically contains HAM10000.** It was assembled from BCN20000 + HAM10000 + MSK, and
the HAM10000 images retain their original `ISIC_xxxxxxx` identifiers. Evaluating on ISIC 2019 as
distributed means evaluating on training data — the scores would look excellent and mean nothing.

The evaluation script excludes overlapping images by filename, reports how many were removed, and
**refuses to run unless the measured overlap is zero.**

**ISIC 2020 is the recommended external test set:** independent patients, and binary
benign/malignant labels that map directly onto the risk head's target. The numbers should be
expected to drop — different cameras, sites and populations always cost something, and that is the
normal, reportable outcome.

---

## Why regeneration matters

All analysis runs from saved model checkpoints, so nothing needs retraining and nothing can be
lost. Two practical consequences:

1. **Every number in the paper is reproducible by one command**, which is a materially stronger
   position for review than analysis that cannot be re-run.
2. **Corrections propagate consistently.** If a bug is found or an analysis is added, everything
   downstream updates together rather than leaving stale figures in place.

**Verification of the reload path:** recomputing round-15 accuracy from a checkpoint gives `0.8987`
against the logged `0.8986876640419947` — identical to 14 decimal places. This is the first check
to run if any result ever appears inconsistent.

---

## Note on the ordering in `OPEN_ITEMS.md`

That document lists what is missing before it lists what is complete. This is intentional for a
research handover: a reader should be able to determine within thirty seconds what can be relied
upon and what cannot.
