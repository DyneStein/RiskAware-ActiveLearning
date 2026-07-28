# START HERE — your reading map

You are new to research. That is completely fine. Nothing in your project is actually hard;
it is just buried under vocabulary that nobody ever stops to explain.

These documents assume **you know nothing**. Read them **in order**. Each one only uses words
that an earlier one has already explained.

---

## The reading order

| # | File | What it teaches | Time |
|---|---|---|---|
| 1 | `01_THE_BASICS.md` | What machine learning even is. Neural networks, training, why we split data, why accuracy lies. | ~15 min |
| 2 | `02_YOUR_PROJECT.md` | What your project actually does and why. Active learning, the oracle, the two heads, the escalation rule. | ~20 min |
| 3 | `03_HOW_WE_MEASURE.md` | Every measurement word: recall, F1, AUC, calibration, ECE, Brier. | ~20 min |
| 4 | `04_STATISTICS.md` | p-values, confidence intervals, and why "our number is bigger" is not proof. | ~15 min |
| 5 | `05_WHAT_WE_FOUND.md` | Your actual results, explained now that you have the vocabulary. | ~15 min |
| 6 | `06_SUPERVISOR_REQUESTS.md` | Each thing your supervisor asked for, decoded, and where our answer lives. | ~15 min |

**Total: about 1 hour 40 minutes.** You do not have to do it in one sitting. Documents 1–3 are
the foundation; if you only have time for three, read those.

---

## How to read these

- **Do not skip ahead.** Document 5 will be meaningless without 1–4.
- **Do not memorise.** You only need to *recognise* these words when your supervisor uses them.
- Every document ends with **"The five things to remember"**. If you remember only those, you're fine.
- Where a word appears for the first time it is written in **bold**. That is your signal that a
  definition is right there.

---

## What is where (for later, once you've read the above)

**Learning material** (this folder, `analysis/learn/`) — the six documents above.

**The actual results:**
- `analysis/SUPERVISOR_RESPONSE.md` — the formal, technical write-up of everything, ask by ask.
  Read this *after* the six documents. It also contains a **draft reply you can send your
  supervisor**.
- `analysis/FINDINGS.md` — the core result about whether the risk score helps.
- `analysis/rigor/figures/` — 23 charts.
- `analysis/rigor/tables/` — 16 spreadsheets of numbers.

**The code and the maths:**
- `RiskAware-ActiveLearning/METHODS.md` — the formal mathematical definitions, for the paper.
  This one is genuinely technical and you do not need to understand it to use it.
- `RiskAware-ActiveLearning/evaluation/rigor/` — the code that produced everything.

---

## One thing to hold onto before you start

You have not done anything wrong by not knowing this vocabulary. Every one of these terms is
ordinary once explained — "AUC" just means "how often does it rank the cancer above the healthy
mole", and "calibration" just means "when it says 90% sure, is it right 90% of the time".

The words are the only hard part. Start with `01_THE_BASICS.md`.
