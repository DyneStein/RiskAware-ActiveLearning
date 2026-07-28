# Runbook — Baseline Comparison

*(The multi-seed replication in section 4 is not being run — seed 42 throughout.)*

Everything needed to run the next phase of experiments on Colab. The
original 24-experiment matrix is in `RUNBOOK_ORIGINAL_MATRIX.md`; this file
covers only the new work.

**Before anything else, read this:** the setup cell changed. `setup_cell.py`
replaces the old `setup_cell.txt`, which no longer works — it wrote a
`config.py` without `SPLIT_SEED`, and the code now requires it. If you paste
the old one, every run fails immediately with `ImportError: cannot import
name 'SPLIT_SEED'`. Use `setup_cell.py`.

---

## 1. First, the thing that is not 96 runs

The natural assumption is that each baseline needs its own copy of the
24-experiment matrix, giving 24 × 4 = 96 runs. It does not, and the
difference is about 240 GPU-hours.

The 24 runs are **3 backbones × 4 uncertainty methods × 2 policies**. The
uncertainty method and the policy are dials that only *our* framework has.
Each baseline **replaces the whole selection step**, so neither dial applies
to it:

| Baseline | Why there is no uncertainty dial |
|---|---|
| **CoreSet** | Selects purely by feature-space coverage. It never computes an uncertainty score at all. |
| **BADGE** | Builds its own gradient embedding, which encodes uncertainty internally as the vector's length. |
| **CLUE** | Uses entropy as a fixed internal component of its algorithm, not as a swappable choice. |
| **VAAL** | Never looks at the classifier. It is task-agnostic by design. |

And there is no policy dial because **each baseline *is* a policy**.

> ### The correct matrix is 3 backbones × 4 baselines = **12 runs**.

| | Runs | GPU-hours |
|---|---|---|
| The assumption | 96 | ~280 |
| **Correct** | **12** | **~40** |

---

## 2. How the comparison is made fair

The four baselines are **acquisition strategies** — they answer *"given a
budget of k labels, which k?"* Our dual-metric policy is an **escalation
policy** — it answers *"which images are unsafe to auto-accept?"* and picks
its own budget as a consequence, because the risk route is uncapped by
design.

So a naive comparison is unfair in whichever direction you set it up. Give
the baselines a flat 150 per round and we win partly by spending more. Cap
ourselves at 150 and we have switched off the uncapped risk route, which is
the entire contribution being tested.

**The fix, which the code does automatically:** in round *t*, each baseline
is handed exactly the number of labels dual-metric spent in round *t* on the
same backbone. Those counts are not estimated — they are read from that
finished run's own `results.csv`. Both then spend an identical budget on an
identical schedule, and the only difference is *which* images were chosen.

This means **`<model>_entropy_dual_metric` must already be complete on Drive**
before its baselines can run. All 24 are complete, so this is satisfied — but
if a baseline errors with `FileNotFoundError: Cost-matching needs the
reference run ...`, that is why, and it fails immediately rather than three
hours in.

---

## 3. The 12 baseline runs

Run the setup cell first (`setup_cell.py`, with `TRAINING_SEED = 42`).

Every command includes `--resume`, so it is always safe to paste: if the run
has not started it begins fresh, and if it is partway through it continues
from the last completed round.

Ordered cheapest first, and grouped so that if you run out of time you still
have a complete backbone rather than four half-finished ones.

### ResNet-50 — run first (~13 GPU-h)

```bash
!python main.py --model resnet50 --strategy coreset --resume
!python main.py --model resnet50 --strategy clue    --resume
!python main.py --model resnet50 --strategy badge   --resume
!python main.py --model resnet50 --strategy vaal    --resume   # slowest, ~4.5h
```

### DenseNet-169 — second (~13 GPU-h)

```bash
!python main.py --model densenet169 --strategy coreset --resume
!python main.py --model densenet169 --strategy clue    --resume
!python main.py --model densenet169 --strategy badge   --resume
!python main.py --model densenet169 --strategy vaal    --resume
```

### EfficientNet-B4 — last (~13 GPU-h)

```bash
!python main.py --model efficientnet_b4 --strategy coreset --resume
!python main.py --model efficientnet_b4 --strategy clue    --resume
!python main.py --model efficientnet_b4 --strategy badge   --resume
!python main.py --model efficientnet_b4 --strategy vaal    --resume
```

### Or leave it running unattended

```bash
!python main.py --run-baselines --resume
```

Walks all 12 and skips anything already complete.

### Expected cost

| Strategy | Per run | × 3 backbones |
|---|---|---|
| CoreSet | ~2.9 h | 8.7 h |
| CLUE | ~2.9 h | 8.7 h |
| BADGE | ~3.0 h | 9.0 h |
| VAAL | ~4.5 h | 13.5 h |
| | | **~40 h** |

VAAL costs more because it trains a VAE and a discriminator from scratch
every round, on top of the classifier. **If compute runs short, drop VAAL
first** — CoreSet, BADGE and CLUE across all three backbones is 9 runs
(~26 h) and covers what reviewers expect. BADGE is the one that must not be
dropped; it is the reference point in this literature.

### What lands on Drive

```
results/experiments/resnet50_baseline_badge/
    results.csv          per-round metrics, plus matched_budget_this_round
    full.json
    environment.json     GPU, library versions, git commit  ← new
    pool_predictions/    per-image scores and decisions, every round
    plots/
```

---

## 4. The multi-seed replication — NOT being run

**Superseded: the supervisor's decision is seed 42 for everything.** Nothing
in this section needs doing. It is kept as a record of the design, in case the
question returns during review.

What the decision means for the paper, stated once: with a single seed there
is no estimate of run-to-run noise, so we cannot separate the size of an
effect from luck. The *direction* of the headline safety result is still
guaranteed regardless — Proposition 1 proves the dual-metric escalation set
always contains the baseline's, so unsafe auto-accepts cannot increase, which
is why the result was 24/24 with no exceptions. The **magnitudes** (43%
reduction, +382 labels, +0.60 pp accuracy) are not protected. Name the
single-seed design plainly in Limitations rather than let a reviewer find it.

The `--seed` flag still works and writes to separate `_s<seed>` folders, so
turning this on later cannot disturb the existing 24 experiments.

The rest of this section is the design as it stood.

### One trap to know about first

A paired Wilcoxon signed-rank test with **n = 5**, where every pair favours
us, has a minimum possible two-sided p-value of **2/2⁵ = 0.0625**. That is
above 0.05. **With 5 seeds you cannot reach significance no matter how clean
the result is.** Almost everyone runs 5 seeds out of habit and then finds
this out.

So the design is not a seed-level test. It is:

> Keep the primary test at the configuration level (**n = 12**), and make
> each cell a **seed-average** instead of a single run.

The n=12 structure is retained, so the minimum reachable p-value stays at
0.00049 — plenty of headroom under Holm correction. Each paired difference
becomes much less noisy. And the seeds separately give the mean ± standard
deviation error bars that reviewers actually want to see.

**MC-dropout is excluded** from the replication: 7.0 h per run against 2.9 h
(2.4× the cost), and it is the least standard of the four uncertainty
measures. Its existing single-seed results stay in the paper, clearly
labelled as such.

### What varies, and what deliberately does not

| | Varies with seed? |
|---|---|
| Weight initialisation | ✅ |
| Minibatch order | ✅ |
| Augmentation draws | ✅ |
| Dropout masks | ✅ |
| **Held-out test set** | ❌ **Frozen — `SPLIT_SEED = 42`** |
| **Starting 490 labelled images** | ❌ Fixed file on disk |

The test set is frozen on purpose. If it moved with the seed, the new runs
would not be comparable to the existing 24, the image-level paired test
would lose its pairing, and test-set difficulty would be confounded with
training noise. State this in the paper: the sweep measures *training
stochasticity*, not starting-set luck. That is a clean and defensible
choice, but it has to be stated rather than left implicit.

### Running it

For each seed, edit **one line** at the top of the setup cell:

```python
TRAINING_SEED = 43        # then 44
```

Re-run the setup cell, then:

```bash
# --- seed 43 --- (seed 42 is already done: it is the original 24)
!python main.py --model resnet50        --uncertainty entropy          --policy uncertainty_only --seed 43 --resume
!python main.py --model resnet50        --uncertainty entropy          --policy dual_metric      --seed 43 --resume
!python main.py --model resnet50        --uncertainty margin           --policy uncertainty_only --seed 43 --resume
!python main.py --model resnet50        --uncertainty margin           --policy dual_metric      --seed 43 --resume
!python main.py --model resnet50        --uncertainty least_confidence --policy uncertainty_only --seed 43 --resume
!python main.py --model resnet50        --uncertainty least_confidence --policy dual_metric      --seed 43 --resume

!python main.py --model densenet169     --uncertainty entropy          --policy uncertainty_only --seed 43 --resume
!python main.py --model densenet169     --uncertainty entropy          --policy dual_metric      --seed 43 --resume
!python main.py --model densenet169     --uncertainty margin           --policy uncertainty_only --seed 43 --resume
!python main.py --model densenet169     --uncertainty margin           --policy dual_metric      --seed 43 --resume
!python main.py --model densenet169     --uncertainty least_confidence --policy uncertainty_only --seed 43 --resume
!python main.py --model densenet169     --uncertainty least_confidence --policy dual_metric      --seed 43 --resume

!python main.py --model efficientnet_b4 --uncertainty entropy          --policy uncertainty_only --seed 43 --resume
!python main.py --model efficientnet_b4 --uncertainty entropy          --policy dual_metric      --seed 43 --resume
!python main.py --model efficientnet_b4 --uncertainty margin           --policy uncertainty_only --seed 43 --resume
!python main.py --model efficientnet_b4 --uncertainty margin           --policy dual_metric      --seed 43 --resume
!python main.py --model efficientnet_b4 --uncertainty least_confidence --policy uncertainty_only --seed 43 --resume
!python main.py --model efficientnet_b4 --uncertainty least_confidence --policy dual_metric      --seed 43 --resume
```

Then repeat the whole block with `--seed 44` (and `TRAINING_SEED = 44` in the
setup cell).

**18 runs per seed × 2 extra seeds = 36 runs ≈ 104 GPU-hours.**

Results go to `<experiment_id>_s43` and `<experiment_id>_s44`. Seed 42 keeps
its original unsuffixed folder names, so **nothing that already exists is
touched** and every analysis script continues to find the original 24 exactly
where it expects them.

---

## 5. Total compute, and the blunt part

| Job | GPU-hours |
|---|---|
| Baselines (12 runs) | ~40 |
| Seeds (36 runs) | ~104 |
| **Total** | **~145** |

Measured from `runtime_per_experiment.csv`: the original 24 runs took **94.1
GPU-hours**, not the ~12 the early plan assumed. Every number here uses the
measured figure.

On free Colab — 4-hour session caps, random disconnects, daily quota — that
is realistically 20–30 usable hours a week, so **five to seven weeks** of
babysitting.

**Renting a GPU is the better answer.** An RTX 4090 on RunPod or Vast.ai is
about $0.35/hour and roughly 2.5× a Colab T4, so ~145 T4-hours becomes ~58
wall-hours ≈ **$20–25 total**. That closes the single biggest weakness in the
paper for the price of a takeaway. Colab Pro+ (~$50/month, background
execution) is the alternative, and a department that is discussing a $3,000
open-access fee can very likely supply GPU time.

---

## 6. Disconnects

Unchanged from before, and worth restating because it is the thing that
makes long runs survivable.

Checkpoints save at the **end of each round**, never mid-round. A round is:
train 10 epochs → score the pool → select → evaluate → *then* save. So the
worst any disconnect can cost is the single round in progress. Everything
already completed is on Drive.

**Recovery is always the same two steps, whatever happened:**

1. Run the setup cell (`setup_cell.py`).
2. Re-run the exact same command, `--resume` still in it.

You never need to work out which round to resume from — the code finds the
latest checkpoint on Drive by itself.

One improvement worth knowing about: runs now **re-seed at the start of every
round** from `(seed, round_number)`. Previously a resumed run produced
different augmentation and batch order from an uninterrupted one, so results
were reproducible only in distribution. Now a resumed run reproduces an
uninterrupted one exactly.

---

## 7. What to change on Google Drive

**Short answer: nothing structural.** The new setup cell writes to exactly the
same Drive paths as before, so the existing folder layout, the shared-folder
shortcuts on the other accounts, and every completed checkpoint all keep
working untouched.

```
MyDrive/Research/Active_learning and HITL/
├── Dataset/archive.zip          unchanged
└── results/
    ├── checkpoints/             unchanged — 24 final-round models
    ├── experiments/             unchanged — new runs are added alongside
    ├── logs/  plots/  tables/   unchanged
    └── analysis/                NEW — created automatically the first time
                                 a rigor script runs on Colab
```

The account-switching trick is unaffected: the other accounts keep their
shortcut to the same shared folder, and `--resume` still finds whatever the
previous account finished.

### Two things worth doing before starting

**1. Storage — not a concern.** The Drive account has **5 TB**, with about
83 GB used. The remaining runs add roughly 1.2 GB (12 baselines, one
final-round checkpoint each at 54-97 MB, plus per-round CSVs). There is no
need to move, archive or delete anything.

Only the **final** round's checkpoint survives per experiment; the loop
deletes each previous round as it goes. So the cost is one model file per
run, not fifteen.

**2. Confirm Drive and the laptop agree.** The Drive copy and the local copy
of `results/` should be identical. From the repository:

```bash
python -m tools.build_manifest --verify
```

Anything reported as changed or missing means the two copies have drifted,
and that should be resolved before new runs are layered on top.

### What you do *not* need to do

- No need to re-upload the dataset — `archive.zip` is unchanged.
- No need to move or rename any existing results folder.
- No need to copy code to Drive. The setup cell pulls it from GitHub each
  session; that is the point of `git reset --hard origin/main` in step 4.
- No need to delete the old `setup_cell.txt` from Drive if a copy is sitting
  there — but do not paste it. It writes a `config.py` without `SPLIT_SEED`
  and every run will fail on import.

---

## 8. Quick reference

| Goal | Command |
|---|---|
| One baseline | `!python main.py --model resnet50 --strategy badge --resume` |
| All 12 baselines | `!python main.py --run-baselines --resume` |
| One seeded run | `!python main.py --model resnet50 --uncertainty entropy --policy dual_metric --seed 43 --resume` |
| Check what is done | open `results/experiments/<id>/results.csv` — 15 rows means complete |
| Rebuild the artefact index | `!python -m tools.build_manifest` |
| Verify nothing changed | `!python -m tools.build_manifest --verify` |

| Symptom | Cause |
|---|---|
| `ImportError: cannot import name 'SPLIT_SEED'` | Using the old `setup_cell.txt`. Use `setup_cell.py`. |
| `FileNotFoundError: Cost-matching needs the reference run ...` | That backbone's `entropy_dual_metric` run is not on Drive. |
| Baseline results not appearing in the comparison figures | Expected — the existing figures filter to the two policies. The baseline comparison is a separate analysis. |
