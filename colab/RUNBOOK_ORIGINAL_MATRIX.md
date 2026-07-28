# RiskAware-ActiveLearning — Colab Session & Experiment Runbook

This file is the single reference for: (1) the one-cell setup you run every time you reconnect,
(2) the full list of 24 experiments with copy-paste commands, (3) exactly how checkpoints and
`--resume` work, and (4) what to do in every disconnect scenario — planned or unplanned.

The matching setup code also lives next to this file as `setup_cell.txt`, ready to paste into one
Colab cell.

## What changed (team decision, after the H1/H2/H3/M1 audit)

- **Two-head model**: each model now shares one backbone with two output heads — the usual
  7-class classification head (drives uncertainty) and a new, independent binary
  malignant/non-malignant risk head (drives the risk score). Risk is no longer just a sum of
  three probabilities from the 7-class softmax — it's a genuinely separate signal now.
- **Class weighting is ON by default** for both heads (was previously an off-by-default
  ablation toggle). Run `--no-dynamic-weights` to test the unweighted case deliberately.
- **Escalation is now top-K (K=150/round) plus threshold overflow**: the K most-uncertain
  images get escalated every round at minimum, plus anything above that round's recalibrated
  threshold even past K — so escalation can never collapse to zero the way it did in the first
  run of `resnet50_entropy_uncertainty_only`. Override K with `--query-budget`.
- **Thresholds are recalibrated every round**, not calibrated once in round 1. The risk
  threshold has no budget cap at all — a dangerous case is always escalated regardless of K.
- **Old checkpoints are incompatible.** Any experiment run before this change (including the
  very first `resnet50_entropy_uncertainty_only` run) saved a model without the risk head — its
  checkpoint won't load into the new two-head architecture. Before running anything new, rename
  or move the existing `results/checkpoints/` folder on Drive so `--resume` doesn't try to load
  a mismatched file and error out. The old `results/logs/` CSVs are still fine to keep as a
  record of the collapse-to-zero finding.
- **Every experiment now gets its own results folder with its own plots automatically** —
  `results/experiments/{experiment_id}/`, populated as it runs: `results.csv`, `full.json`,
  per-round pool predictions, a confusion matrix every round, and an always-current learning
  curve — no need to run `--run-all` or `--plot-only` first to get any of it. Confusion matrices
  were previously computed but silently discarded; that's fixed too. See Section 3.

---

## 1. The one-cell setup — run this every time you reconnect

Paste the contents of `setup_cell.txt` into a single Colab cell and run it. It does all four
things that used to be three separate cells, in the right order:

1. Mounts your Google Drive
2. Unzips the dataset onto Colab's local fast disk (skips this step automatically if it's
   already unpacked earlier in the same session, so re-running the cell twice by accident
   doesn't waste time)
3. Clones the code fresh from GitHub (or pulls the latest if it's already there this session)
4. Rewrites `config.py` so the code points at this session's paths

**The only manual step left:** the very first time in a fresh Colab runtime, `drive.mount()`
will pop up a Google sign-in / authorize window. Click through it and grant access — this can't
be automated away, Google requires it interactively once per fresh runtime. After that, this one
cell needs no further input.

Nothing about your experiment results depends on this cell — it only rebuilds the *environment*
(code + local dataset copy). Your actual progress (trained model checkpoints, logs, calibration
values) lives permanently on Drive under `results/`, untouched by any of this.

---

## 2. The 24 experiments — full list with commands

3 models × 4 uncertainty methods × 2 policies = 24. Listed in the order we're actually running
them: cheapest model first, cheapest uncertainty method first, baseline (`uncertainty_only`)
before the comparison (`dual_metric`) within each pair — not the code's internal default order,
which would start with the heaviest model.

Every command below already includes `--resume`, so it's always safe to paste and run — if the
experiment hasn't started yet, `--resume` just starts fresh; if it's partway done, it continues
from the last checkpoint automatically.

### The 3 dials, quickly

- **Model**: `resnet50` (fastest) → `densenet169` (medium) → `efficientnet_b4` (heaviest)
- **Uncertainty method**: `entropy`, `margin`, `least_confidence` are all cheap (one forward
  pass per image). `mc_dropout` is ~30x more expensive — it runs the model 30 times per image
  with dropout switched on and measures how much the 30 answers disagree.
- **Policy**: `uncertainty_only` = escalate the top-150-most-confused images each round, plus
  anything above that round's recalibrated confusion threshold even past 150 (the **baseline**).
  `dual_metric` = the same uncertainty rule, **plus** anything the independent risk head calls
  above-threshold for malignancy risk, uncapped (**our method**, the thing this whole project is
  testing). Risk now comes from its own dedicated head, not from re-reading the classification
  head's softmax — a confidently-wrong classification no longer automatically produces a
  falsely-safe risk score too.

### ResNet-50 block — run first

```bash
# 1 — baseline
!python main.py --model resnet50 --uncertainty entropy --policy uncertainty_only --resume

# 2 — comparison (pairs with #1)
!python main.py --model resnet50 --uncertainty entropy --policy dual_metric --resume

# 3 — baseline
!python main.py --model resnet50 --uncertainty margin --policy uncertainty_only --resume

# 4 — comparison (pairs with #3)
!python main.py --model resnet50 --uncertainty margin --policy dual_metric --resume

# 5 — baseline
!python main.py --model resnet50 --uncertainty least_confidence --policy uncertainty_only --resume

# 6 — comparison (pairs with #5)
!python main.py --model resnet50 --uncertainty least_confidence --policy dual_metric --resume

# 7 — baseline, EXPENSIVE (30x scoring cost)
!python main.py --model resnet50 --uncertainty mc_dropout --policy uncertainty_only --resume

# 8 — comparison (pairs with #7), EXPENSIVE
!python main.py --model resnet50 --uncertainty mc_dropout --policy dual_metric --resume
```

### DenseNet-169 block — run second

```bash
# 9 — baseline
!python main.py --model densenet169 --uncertainty entropy --policy uncertainty_only --resume

# 10 — comparison
!python main.py --model densenet169 --uncertainty entropy --policy dual_metric --resume

# 11 — baseline
!python main.py --model densenet169 --uncertainty margin --policy uncertainty_only --resume

# 12 — comparison
!python main.py --model densenet169 --uncertainty margin --policy dual_metric --resume

# 13 — baseline
!python main.py --model densenet169 --uncertainty least_confidence --policy uncertainty_only --resume

# 14 — comparison
!python main.py --model densenet169 --uncertainty least_confidence --policy dual_metric --resume

# 15 — baseline, EXPENSIVE
!python main.py --model densenet169 --uncertainty mc_dropout --policy uncertainty_only --resume

# 16 — comparison, EXPENSIVE
!python main.py --model densenet169 --uncertainty mc_dropout --policy dual_metric --resume
```

### EfficientNet-B4 block — run last (heaviest)

```bash
# 17 — baseline
!python main.py --model efficientnet_b4 --uncertainty entropy --policy uncertainty_only --resume

# 18 — comparison
!python main.py --model efficientnet_b4 --uncertainty entropy --policy dual_metric --resume

# 19 — baseline
!python main.py --model efficientnet_b4 --uncertainty margin --policy uncertainty_only --resume

# 20 — comparison
!python main.py --model efficientnet_b4 --uncertainty margin --policy dual_metric --resume

# 21 — baseline
!python main.py --model efficientnet_b4 --uncertainty least_confidence --policy uncertainty_only --resume

# 22 — comparison
!python main.py --model efficientnet_b4 --uncertainty least_confidence --policy dual_metric --resume

# 23 — baseline, EXPENSIVE
!python main.py --model efficientnet_b4 --uncertainty mc_dropout --policy uncertainty_only --resume

# 24 — comparison, EXPENSIVE
!python main.py --model efficientnet_b4 --uncertainty mc_dropout --policy dual_metric --resume
```

### Once everything above is done (or if you'd rather not track order by hand)

```bash
!python main.py --run-all --resume
```

This walks every model/method/policy combination automatically and **skips anything already
fully complete** — safe to run any time as a "mop up whatever's left" command once you've worked
through your priority order manually, or if you just want to leave it running unattended.

---

## 3. How checkpoints and `--resume` actually work

Understanding this precisely matters for the disconnect scenarios in Section 4.

**Where everything lives:** every completed round of every experiment writes to Drive, under:
```
My Drive/Research/Active_learning and HITL/results/
├── checkpoints/                    ← operational, transient (only the latest round is kept)
│   └── {experiment_id}/
│       └── round_N/
│           ├── model.pt            ← both heads' weights + backbone
│           └── pool_state/         ← which images are labeled vs. still unlabeled
├── experiments/                    ← everything about ONE experiment, for the paper
│   └── {experiment_id}/
│       ├── results.csv             ← metrics per round, updated live — includes
│       │                             uncertainty_threshold_used / risk_threshold_used,
│       │                             so each round's recalibrated values are right here
│       ├── full.json               ← same data as results.csv, in one JSON blob
│       ├── pool_predictions/
│       │   └── round_N_pool_predictions.csv  ← every image's scores + decision that round
│       └── plots/                  ← THIS experiment's own visuals, generated automatically
│           ├── confusion_matrix_(Round_N).png     ← every round
│           ├── learning_curve.png                  ← overwritten each round, always current
│           └── uncertainty_vs_risk_scatter(Round_N).png  ← dual_metric only, round 1 + last
├── logs/
│   └── all_experiments.json        ← cross-experiment combined file, written by --run-all
├── plots/                          ← cross-experiment COMPARISON plots (all 24 together),
│                                      written by --run-all or --plot-only, not per-experiment
└── tables/
    └── comparison_table.csv/.tex   ← the main results table, same as above
```
`{experiment_id}` is the model_method_policy string, e.g. `resnet50_entropy_uncertainty_only`. The
`experiments/{experiment_id}/` folder is complete and self-contained the moment that experiment
finishes — you don't need to run `--run-all` or `--plot-only` across all 24 to get any of it.
`model.pt` now contains both heads' weights (classification + risk) in one state dict.

**The critical detail: checkpoints save at the END of a round, not continuously.** A round is:
train for 10 epochs → score the unlabeled pool → escalate/auto-accept → evaluate on the test set
→ *then* save the checkpoint. If anything interrupts execution *before* that save line runs, that
entire round's work is gone — not partially recoverable epoch-by-epoch. Training epochs are not
individually checkpointed.

**What `--resume` does when you re-run a command:** it looks inside
`checkpoints/{experiment_id}/` on Drive, finds the highest-numbered `round_N` folder that exists,
loads `model.pt` (both heads) and `pool_state` from it, and continues training from round `N+1`
onward. Thresholds are **not** reloaded from anywhere — they get recalibrated fresh every round
regardless of whether this is a brand-new run or a resumed one, using whatever the labeled set
looks like at that point. If no checkpoint folder exists yet for that experiment, `--resume` just
starts fresh at round 1 — it's always safe to include.

**Consequence:** the worst thing that can happen from any disconnect, planned or accidental, is
losing progress within the single round that was actively running — never anything already
completed. There's no scenario where you lose the whole experiment.

---

## 4. Scenario playbook — what to do in every situation

### Scenario A — You want to stop on purpose, cleanly

**Best case, do this when you can.** Wait until you see `Checkpoint saved: .../round_N/model.pt`
print in the output — that means the round just finished and saved. Once you see that line, it's
safe to stop by any method (close the tab, let it idle out, `Runtime → Disconnect and delete
runtime`). Nothing is lost.

**To resume:** open Colab again → run the one-cell setup (`setup_cell.txt`) → paste the exact
same experiment command you were running, with `--resume`. It continues at round `N+1`.

### Scenario B — You need to stop *right now*, mid-round

Click the **■ (stop)** button on the currently-running cell. This sends an interrupt and halts
Python immediately, wherever it happens to be (mid-epoch, mid-scoring, whatever).

- Everything from previously completed rounds is safe (already on Drive).
- The round that was in progress is lost and will be fully redone from scratch on resume.

**To resume:** same as Scenario A — one-cell setup, then the same command with `--resume`. It'll
just look like that round is starting over, because it is.

### Scenario C — Your wifi drops unexpectedly (not on purpose)

Important to understand: **Colab doesn't run on your laptop.** Your code executes on a Google
cloud machine; your wifi only connects your *browser* to that machine so you can see the output.
If your wifi drops:

- The training on Google's server **does not stop** — it keeps running in the background,
  invisible to you, for a while (Colab tolerates a temporary disconnect and will let you
  reconnect to the *same* running session if you're fast enough — usually within some tens of
  minutes, though free-tier limits vary and aren't guaranteed).
- If you reconnect your wifi and reopen the same Colab tab soon after, you may find it
  reconnects to the still-running session and you can watch it continue where it visually left
  off — no action needed.
- If too much time passes, or Colab decides to reclaim the machine, the session is gone and any
  round that was in progress at the moment of the drop is lost (same as Scenario B) — but again,
  every previously completed round is already safe on Drive.

**To resume (if the old session is gone):** one-cell setup, then the same command with
`--resume`.

**Practical advice:** don't rely on wifi loss as a way to intentionally pause — it's unpredictable
whether the cloud machine keeps working or gets reclaimed, and either way you can't watch it
happen. If you want to pause on purpose, use Scenario A or B instead.

### Scenario D — Colab auto-disconnects from inactivity

Free-tier Colab disconnects idle sessions (e.g., if the browser tab sits in the background too
long with no interaction). The keep-alive console snippet from earlier (auto-clicking the
connect button every 60 seconds) helps prevent this during a long unattended run, but if it
happens anyway:

- Same as Scenario C: the in-progress round may or may not have finished depending on timing.
- **To resume:** one-cell setup, then the same command with `--resume`.

### Scenario E — You close the browser tab or your laptop sleeps

Similar to wifi dropping — the cloud machine may keep running briefly, but you should treat this
as "the session might be gone" rather than count on it surviving. Same recovery either way.

**To resume:** one-cell setup, then the same command with `--resume`.

### Scenario F — Colab fully recycles the runtime (daily free-GPU quota runs out, or you get a
"runtime disconnected, GPU not available" message when reconnecting)

This is the most complete reset: everything on `/content` (the unzipped dataset, the cloned code)
is gone, and you may need to wait for the next quota window before Colab gives you a GPU again.

- **What's still safe:** everything on Drive — the dataset zip, the cloned code (if you'd ever
  put it there, which we no longer do), and critically, **every completed checkpoint, every log,
  every plot.** None of this lives on the recycled machine.
- **To resume once you have a GPU again:** exactly the normal routine — one-cell setup rebuilds
  the local dataset copy and re-clones the code from GitHub (both fast, a few minutes total) —
  then the same experiment command with `--resume` picks up from the last completed round on
  Drive, as if nothing happened.

### Universal rule of thumb

**No matter which scenario hit you, the recovery is always the same two actions:**
1. Run the one-cell setup (`setup_cell.txt`).
2. Re-run the exact experiment command you were on, with `--resume` still in it.

You never need to figure out *which* round to resume from, edit any paths, or hunt for files —
the code finds the latest checkpoint on Drive automatically.

---

## 5. Quick cheat-sheet

| Situation | What to click / do | What you lose |
|---|---|---|
| Round just finished, want to stop | Close tab / disconnect anytime | Nothing |
| Need to stop mid-round | Click ■ stop on the cell | Current round's progress only |
| Wifi drops by accident | Reconnect wifi, reopen tab, wait and see | Maybe nothing, maybe current round |
| Colab idles you out | N/A, already happened | Maybe nothing, maybe current round |
| Browser/laptop closes | N/A, already happened | Maybe nothing, maybe current round |
| Runtime fully recycled / GPU quota exhausted | Wait for GPU availability, redo setup | Nothing on Drive; current round if mid-run |

**Every recovery, every time:** run `setup_cell.txt` → re-run the same `!python main.py ...`
command with `--resume`.

---

## 6. FAQ

**Do I need to change anything in the command when resuming?** No. The exact same command you
started with, `--resume` included, is also the correct resume command. There's no separate
"resume mode" syntax.

**What if I run the wrong experiment's command with `--resume` by mistake?** It looks for a
checkpoint folder matching *that specific* model+method+policy combination
(`checkpoints/{that_experiment_id}/`). If none exists, it just starts that experiment fresh — it
won't corrupt or overwrite a different experiment's checkpoints, since each one has its own
folder.

**How do I know how many rounds an experiment has left?** Open
`results/experiments/{experiment_id}/results.csv` on Drive — one row per completed round. 15 rows
means that experiment is fully done (`AL_ROUNDS = 15` in config).

**Where are the graphs/plots for a specific experiment?**
`results/experiments/{experiment_id}/plots/` — confusion matrix every round, a learning curve
(accuracy/F1/safety metric vs. round) always up to date, and for `dual_metric` runs, the 2×2
scatter for round 1 and the final round. These appear automatically as that experiment runs, no
extra command needed. The 6 comparison plots across all 24 experiments (only meaningful once you
have more than one experiment done) live separately in `results/plots/`, generated by `--run-all`
or `--plot-only`.

**Can two experiments run at once to save time?** No — there's one GPU per Colab session, and
each `!python main.py ...` cell blocks until that experiment finishes or you stop it. Run them
one at a time, in separate cells, in the priority order from Section 2.
