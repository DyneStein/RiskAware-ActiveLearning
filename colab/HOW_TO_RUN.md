# How to Run the Experiments — Step by Step

Written to be read start to finish by someone who does not want to guess at
any of the words. Every technical term is explained the first time it
appears. If you already know the vocabulary, skip to Part 3.

**What changed since last time:** only two things.

1. The setup cell is now **`setup_cell.py`**, not `setup_cell.txt`. The old
   one no longer works — paste it and every run fails instantly.
2. Some commands now have a `--seed` or a `--strategy` on the end.

Everything else — mount Drive, paste setup cell, paste a python command,
walk away — is exactly as before.

---

# PART 1 — The vocabulary

Read this once. It is the whole language of the project.

## The data

| Term | What it means |
|---|---|
| **Dataset** | Our collection of skin-lesion photographs. HAM10000: **10,015 images**, each labelled by a doctor with one of 7 diagnoses. |
| **Class** | One of the 7 diagnoses: `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`. `mel` is melanoma — the dangerous one. `nv` is an ordinary mole and is about two-thirds of the dataset. |
| **Labelled** | The model is allowed to see this image *and* its correct answer, and learn from it. |
| **Unlabelled** | The model may look at the image but is not told the answer. |
| **Test set** | **1,905 images locked in a vault.** The model never trains on them, ever. They are the exam. If a model has trained on its own exam, its score is meaningless — this is the single most important rule in the whole project. |
| **Pool** | The **7,620 unlabelled images** the model may ask about. |
| **Seed set** | The **490 images** (70 per class) that start out labelled, so the model is not completely blind on day one. Confusingly, this has nothing to do with a "random seed" — see below. |

Adding up: 490 labelled + 7,620 pool + 1,905 test = 10,015. ✅

## The learning

| Term | What it means |
|---|---|
| **Model** / **backbone** / **architecture** | The neural network itself. We use three: **ResNet-50**, **DenseNet-169**, **EfficientNet-B4**. Think of them as three different brands of engine — same job, different internal design. We use three so nobody can say our result only works on one lucky architecture. |
| **Epoch** | One complete pass through all the labelled images. We do **10 epochs** each round — the model studies the whole set 10 times before moving on. |
| **Batch** | The model looks at **32 images at a time**, not one by one. Faster, and the maths is more stable. |
| **Round** | One full cycle: train → look at the pool → pick images to ask about → get answers → test. We do **15 rounds** per experiment. |
| **Active learning** | The whole idea: instead of labelling all 10,015 images (expensive, a doctor's time), the model *chooses* which few to ask about. Fewer questions, same accuracy. |
| **Oracle** | The stand-in "doctor" who answers. We already know every label, so answering is just a lookup. Standard practice — nobody expects a real clinician in a methods paper. |
| **Query** | One question to the oracle: "what is this image?" The cost we are trying to keep low. |

## The decision — this is your actual contribution

Every round, for every image in the pool, the model must choose:

| Term | What it means |
|---|---|
| **Auto-accept** | "I'm sure enough. Nobody needs to look at this." |
| **Escalate** | "Send this to a human." |
| **Uncertainty** | *How confused is the model?* High = it cannot decide between diagnoses. |
| **Risk** | *How dangerous is this case if I'm wrong?* Comes from a **separate part of the network** that only ever answers one question: is this malignant? |
| **Threshold** | The cut-off line. Score above it → escalate. Recalculated every round, because as the model improves, an old cut-off goes stale. |
| **Unsafe auto-accept** | **The number that matters most.** A genuinely dangerous image that the system waved through without human review. In the real world, this is a missed cancer. |

**The two policies being compared:**

| Policy | Rule |
|---|---|
| `uncertainty_only` | **The baseline.** Escalate the most confused cases. This is what everyone else does. |
| `dual_metric` | **Ours.** Escalate the most confused cases **OR** anything the risk head calls dangerous — no cap on the second part. |

The whole paper lives in one gap between those two. A normal system, faced with an image it is **confident** about but which is **dangerous**, waves it through — because confidence is all it looks at. Ours catches it. That is the entire idea.

## The randomness — this is the bit you asked about

There are **two** seeds and they do completely different jobs. Mixing them up would quietly ruin the results, which is why they are now separate.

| Seed | Value | Does it change? | Controls |
|---|---|---|---|
| **`SPLIT_SEED`** | **42 — frozen forever** | **Never** | Which 1,905 images are the exam |
| **`RANDOM_SEED`** | 42, then 43, 44 | Yes, per run | Everything else random inside training |

**What a "random seed" actually is.** Computers cannot produce true randomness — they follow a recipe that *looks* random. The seed is the starting point of that recipe. Same seed → identical run, every single time. Different seed → same method, same data, **different luck**.

`RANDOM_SEED` controls four things:
- how the network's starting values are scrambled before any learning
- the order images arrive in
- the random flips, rotations and colour shifts applied to each image
- which internal connections get switched off during training (this is deliberate — it stops memorisation)

**Why `SPLIT_SEED` must never move.** If the exam changed every time you changed the seed, you would be comparing scores from different exams. Your new runs would not be comparable to the 24 you already have, and one test set might happen to contain 12 of the rare `df` images while another has 6 — so per-class numbers would swing for reasons that have nothing to do with your method. Freezing it means **every run in the entire paper sits the identical exam.**

**Why you need more than one `RANDOM_SEED`.** Right now every number you have comes from a single run at seed 42. So when you report "+0.60% accuracy", you cannot say whether re-running the *same thing* with different luck would move it by 0.05% or by 2%. If it's 2%, your result is noise. Running seeds 43 and 44 measures that. This is the single biggest hole in the paper.

## The words on the new commands

| Term | What it means |
|---|---|
| **Baseline (SOTA)** | A well-known published method we compare against. Your supervisor asked for four: **CoreSet, BADGE, CLUE, VAAL**. |
| **Strategy** | What we call a baseline in the code — `--strategy badge`. It **replaces** the decision step entirely. |
| **Cost-matched** | The baselines are given **exactly** the number of questions your method asked in that same round. Same budget, same schedule — so the only difference is *which* images were picked, not how many. Without this the comparison is meaningless. |
| **Checkpoint** | A save file. Written at the **end of every round**. |
| **`--resume`** | "Continue from the last save." Always safe to include: if nothing is saved it just starts fresh. |

---

# PART 2 — What the experiments actually are

## Already finished — do not re-run

**3 models × 4 uncertainty methods × 2 policies = 24 experiments**, all at seed 42.
Measured cost: **94.1 GPU-hours**. These are done and on Drive. Leave them alone.

## Job A — the four baselines (your supervisor's request)

> ### 3 models × 4 baselines = **12 runs** ≈ **40 GPU-hours**

Everything fixed at **seed 42**.

| | ResNet-50 | DenseNet-169 | EfficientNet-B4 |
|---|---|---|---|
| **CoreSet** | run | run | run |
| **BADGE** | run | run | run |
| **CLUE** | run | run | run |
| **VAAL** | run | run | run |

**This is the "3 × 4" you were thinking of, and 12 is correct.** It is *not* 24 × 4 = 96. Your 24 experiments multiply models × uncertainty × policies — but a baseline has neither an uncertainty dial nor a policy dial, because **it replaces the decision step entirely**. CoreSet never even calculates uncertainty. VAAL never looks at the classifier at all.

Getting this right saves you about **240 GPU-hours.**

## Job B — the seed replication (the bigger problem)

> ### 3 models × 3 uncertainty methods × 2 policies = 18 runs, × 2 new seeds = **36 runs** ≈ **104 GPU-hours**

| | Value |
|---|---|
| Models | ResNet-50, DenseNet-169, EfficientNet-B4 |
| Uncertainty | entropy, margin, least_confidence |
| Policies | uncertainty_only, dual_metric |
| Seeds | **43 and 44** (42 is already done — it *is* the original 24) |

**Why MC-dropout is dropped:** it costs **7.0 hours** per run against 2.9 for the others — 2.4× the price — and it is the least standard of the four. Its existing seed-42 results stay in the paper, clearly labelled as single-seed.

## Every setting, in one table

| Setting | Value | Changes? |
|---|---|---|
| Rounds per experiment | 15 | No |
| Epochs per round | 10 | No |
| Batch size | 32 | No |
| Image size | 224 × 224 | No |
| Learning rate | 0.0001 | No |
| Starting labelled images | 490 | No |
| Test set | 1,905 | **Never** |
| Query budget floor | 150/round | No |
| `SPLIT_SEED` | 42 | **Never** |
| `RANDOM_SEED` | 42 → 43 → 44 | **Yes, Job B** |
| Model | 3 options | Yes |
| Uncertainty | 4 options | Yes |
| Policy / strategy | 2 policies + 4 baselines | Yes |

## The grand total

| Job | Runs | GPU-hours |
|---|---|---|
| Already done | 24 | 94.1 ✅ |
| A — baselines | 12 | ~40 |
| B — seeds | 36 | ~104 |
| **Still to run** | **48** | **~145** |

On free Colab that is realistically **5–7 weeks** of babysitting. Renting an RTX 4090 (RunPod / Vast.ai, ~$0.35/hour, about 2.5× a Colab T4) turns it into ~58 hours ≈ **$20–25 total**. That is the single best value purchase available to this project.

---

# PART 3 — Running it, step by step

## Step 1 — Open Colab and turn the GPU on

1. Go to **colab.research.google.com**
2. **File → New notebook**
3. **Runtime → Change runtime type**
4. Under *Hardware accelerator* pick **T4 GPU**. Press **Save**.

> ⚠️ Miss this step and everything runs on the processor instead of the graphics card — roughly **20× slower**. A 3-hour run becomes days. Always check.

## Step 2 — Get the setup cell text

Open this link and copy **everything** on the page:

**https://raw.githubusercontent.com/DyneStein/RiskAware-ActiveLearning/main/colab/setup_cell.py**

`Ctrl+A` then `Ctrl+C`.

> This is the same habit as before, just a different file. The `.py` on the end is only a name — it is meant to be **pasted into a Colab cell**, not run as a file, because it contains Colab-only commands like `!git clone`.

## Step 3 — Paste it and set your seed

Paste into the first cell. Near the top you will see:

```python
TRAINING_SEED = 42
```

| Doing | Set it to |
|---|---|
| **Job A** — the 12 baselines | `42` |
| **Job B** — seed run 43 | `43` |
| **Job B** — seed run 44 | `44` |

**This one line is the entire difference between a seed run and a normal one.** Leave it at 42 for now.

## Step 4 — Run the setup cell

Press **Shift + Enter**.

A Google sign-in window pops up asking permission for Drive. **Click through and allow it.** This cannot be automated — Google requires it once per session.

Then wait 2–4 minutes while it unzips the dataset. You should end with:

```
==============================================================
Tesla T4, 15360 MiB, 550.54.15
torch 2.x.x | CUDA 12.x | available=True
Training seed: 42   Split seed: 42 (frozen)
Results -> /content/drive/MyDrive/Research/Active_learning and HITL/results
==============================================================
SETUP COMPLETE — ready to run an experiment command.
==============================================================
```

**Check three things:**

| Line | Should say | If not |
|---|---|---|
| `Tesla T4` (or better) | a GPU name | Go back to Step 1 |
| `available=True` | True | Go back to Step 1 |
| `Training seed:` | the number you wanted | Fix the line, re-run the cell |

## Step 5 — Start an experiment

New cell (**+ Code**), paste **one** command, **Shift + Enter**:

```bash
!python main.py --model resnet50 --strategy coreset --resume
```

Reading that command:

| Piece | Meaning |
|---|---|
| `!` | "this is a terminal command, not Python" |
| `python main.py` | run the program |
| `--model resnet50` | use the ResNet-50 network |
| `--strategy coreset` | use the CoreSet baseline for choosing images |
| `--resume` | continue from the last save if there is one |

**One command per cell. One at a time.** There is only one GPU — a second command would just sit and wait.

## Step 6 — Check it started correctly

Within about a minute you should see:

```
======================================================================
EXPERIMENT: resnet50_baseline_coreset
  Model: resnet50
  Selection: CoreSet (Sener & Savarese, 2018)  [BASELINE]
  Budget: cost-matched per round to resnet50_entropy_dual_metric
          (total 4678 labels over 15 rounds)
  Training seed: 42 (test split is fixed at SPLIT_SEED, independent)
  Rounds: 15
======================================================================

Pool Manager initialized:
  Labeled (seed):   490 images
  Unlabeled pool:   7620 images
  Test set:         1905 images
```

**Verify:** 490 / 7620 / 1905, and the seed is what you set. If the numbers differ, stop and ask — something is wrong with the data paths.

Then each round prints:

```
--- Round 1/15 ---
  Labeled: 490 | Unlabeled: 7620 | Test: 1905
  Epoch 1/10 — Loss: 1.8234 | Acc: 31.2%
  ...
  Escalated: 312 | Auto-accepted: 7308
  Accuracy: 0.6234 | F1 Macro: 0.4102 | FN Rate (malignant): 0.5120 | Unsafe auto-accepts: 421
  Checkpoint saved: .../round_1/model.pt
```

**`Checkpoint saved` is the line that matters.** Once you see it, that round is permanently safe on Drive. You can close the laptop.

## Step 7 — Wait

About **2.9 hours** for a normal run (**4.5** for VAAL, **7.0** for MC-dropout).

Keep the browser tab open and visible if you can. Colab disconnects tabs that sit idle in the background.

## Step 8 — Next command

When it prints `EXPERIMENT COMPLETE`, paste the next command into a new cell and go again. **You do not re-run the setup cell** unless you were disconnected.

---

# PART 4 — The exact commands

## Job A — the 12 baselines

Setup cell with `TRAINING_SEED = 42`. Cheapest first, and grouped so that if you run out of time you have complete backbones rather than four half-finished ones.

```bash
# ResNet-50  (~13 hours total)
!python main.py --model resnet50 --strategy coreset --resume
!python main.py --model resnet50 --strategy clue    --resume
!python main.py --model resnet50 --strategy badge   --resume
!python main.py --model resnet50 --strategy vaal    --resume
```
```bash
# DenseNet-169  (~13 hours total)
!python main.py --model densenet169 --strategy coreset --resume
!python main.py --model densenet169 --strategy clue    --resume
!python main.py --model densenet169 --strategy badge   --resume
!python main.py --model densenet169 --strategy vaal    --resume
```
```bash
# EfficientNet-B4  (~13 hours total)
!python main.py --model efficientnet_b4 --strategy coreset --resume
!python main.py --model efficientnet_b4 --strategy clue    --resume
!python main.py --model efficientnet_b4 --strategy badge   --resume
!python main.py --model efficientnet_b4 --strategy vaal    --resume
```

**Or leave it unattended** — walks all 12, skips anything finished:

```bash
!python main.py --run-baselines --resume
```

## Job B — the seed runs

Setup cell with `TRAINING_SEED = 43`, then all 18. Then change it to `44` and repeat.

```bash
!python main.py --model resnet50 --uncertainty entropy          --policy uncertainty_only --seed 43 --resume
!python main.py --model resnet50 --uncertainty entropy          --policy dual_metric      --seed 43 --resume
!python main.py --model resnet50 --uncertainty margin           --policy uncertainty_only --seed 43 --resume
!python main.py --model resnet50 --uncertainty margin           --policy dual_metric      --seed 43 --resume
!python main.py --model resnet50 --uncertainty least_confidence --policy uncertainty_only --seed 43 --resume
!python main.py --model resnet50 --uncertainty least_confidence --policy dual_metric      --seed 43 --resume
```
```bash
!python main.py --model densenet169 --uncertainty entropy          --policy uncertainty_only --seed 43 --resume
!python main.py --model densenet169 --uncertainty entropy          --policy dual_metric      --seed 43 --resume
!python main.py --model densenet169 --uncertainty margin           --policy uncertainty_only --seed 43 --resume
!python main.py --model densenet169 --uncertainty margin           --policy dual_metric      --seed 43 --resume
!python main.py --model densenet169 --uncertainty least_confidence --policy uncertainty_only --seed 43 --resume
!python main.py --model densenet169 --uncertainty least_confidence --policy dual_metric      --seed 43 --resume
```
```bash
!python main.py --model efficientnet_b4 --uncertainty entropy          --policy uncertainty_only --seed 43 --resume
!python main.py --model efficientnet_b4 --uncertainty entropy          --policy dual_metric      --seed 43 --resume
!python main.py --model efficientnet_b4 --uncertainty margin           --policy uncertainty_only --seed 43 --resume
!python main.py --model efficientnet_b4 --uncertainty margin           --policy dual_metric      --seed 43 --resume
!python main.py --model efficientnet_b4 --uncertainty least_confidence --policy uncertainty_only --seed 43 --resume
!python main.py --model efficientnet_b4 --uncertainty least_confidence --policy dual_metric      --seed 43 --resume
```

> ⚠️ **`--seed 43` must be in the command AND `TRAINING_SEED = 43` in the setup cell.** They must agree. Results land in folders ending `_s43`, so your original 24 are never touched.

---

# PART 5 — When things go wrong

**Nothing you can do by accident will lose more than the round that was running.** Checkpoints save at the end of every round. Everything before that is on Drive.

| What happened | Do this | You lose |
|---|---|---|
| Round finished, want to stop | Close the tab whenever | Nothing |
| Must stop right now | Click ■ on the cell | The current round only |
| Wi-Fi dropped | Reconnect, reopen the tab. It may still be running | Maybe nothing |
| Colab idled you out | Setup cell → same command | Maybe the current round |
| Daily GPU quota gone | Switch account, or wait | Nothing on Drive |

**Recovery is always the same two steps, whatever happened:**
1. Run the setup cell.
2. Paste the **exact same command**, `--resume` still on it.

You never work out which round to resume from — the code finds it.

**Your account-switching trick still works.** When one account hits its limit, switch to another that has the shared folder shortcut, run the setup cell there, paste the same command. It picks up from the last saved round.

## Error messages

| Message | Meaning | Fix |
|---|---|---|
| `cannot import name 'SPLIT_SEED'` | You used the **old** `setup_cell.txt` | Use `setup_cell.py` (Step 2) |
| `Cost-matching needs the reference run ...` | That model's `entropy_dual_metric` is missing from Drive | Check Drive; that run must exist |
| `CUDA out of memory` | GPU full from a previous run | **Runtime → Restart runtime**, setup cell, same command |
| `available=False` | No GPU | Step 1 |
| `No such file or directory: archive.zip` | Drive path wrong or not mounted | Re-run the setup cell, allow Drive access |

---

# PART 6 — What order to do it in

1. **Job A, ResNet-50 block** (4 runs, ~13 h). Fastest model. Gets one complete backbone of baseline results, which is enough to start writing the comparison section.
2. **Job A, remaining 8 runs** (~26 h).
3. **Job B, seed 43** (18 runs, ~52 h).
4. **Job B, seed 44** (18 runs, ~52 h).

If time runs short, **drop VAAL first** — it is the most expensive and the least commonly demanded. CoreSet + BADGE + CLUE across all three models is 9 runs and covers what reviewers expect. **Never drop BADGE** — it is *the* reference point in this literature and its absence would be noticed immediately.

## How to tell what is finished

Open `results/experiments/<name>/results.csv` on Drive. **15 rows = complete.** Fewer means it stopped early and `--resume` will continue it.

| Job | Folder names |
|---|---|
| Original 24 | `resnet50_entropy_dual_metric` |
| Baselines | `resnet50_baseline_badge` |
| Seed runs | `resnet50_entropy_dual_metric_s43` |

---

## The one-paragraph version

Open Colab, turn on the T4 GPU, paste `setup_cell.py` into the first cell with `TRAINING_SEED` set to whichever seed you want, run it and allow Drive access, then paste one `!python main.py ...` command per cell and wait about three hours each. There are **48 runs left**: 12 baselines at seed 42, and 36 replication runs at seeds 43 and 44. If anything disconnects, re-run the setup cell and paste the same command again — you never lose more than the round in progress.
