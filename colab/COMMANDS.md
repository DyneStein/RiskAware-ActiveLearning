# Every Command, Ready to Copy

**Plan: seed 42 for everything.** Multi-seed replication is not being run.

Copy a block, paste into a Colab cell, press Shift+Enter. One command per
cell, one at a time — there is only one GPU, so a second command would just
queue behind the first.

Explanation of what everything means: `HOW_TO_RUN.md`.
How the comparison works: `../docs/HOW_WE_COMPARE.md`.

---

## 0. The setup cell — run this first, every session

Copy everything from this page and paste it into the first cell:

**https://raw.githubusercontent.com/DyneStein/RiskAware-ActiveLearning/main/colab/setup_cell.py**

Leave the seed line exactly as it is:

```python
TRAINING_SEED = 42
```

Confirm before continuing:

| Line printed | Must say |
|---|---|
| GPU | `Tesla T4` (or better) |
| `available=` | `True` |
| `Training seed:` | `42` |

Re-run this cell after **any** disconnect. Never run it in the middle of an
experiment.

---

## 1. GPU JOBS — the 12 baseline runs

**3 models × 4 baselines = 12 runs ≈ 40 GPU-hours.** All at seed 42.

Each is given exactly the number of labels dual-metric spent in that same
round on that same model, so the comparison is cost-matched.

### Block A — ResNet-50 (~13 h) — do this first

```bash
!python main.py --model resnet50 --strategy coreset --resume
```
```bash
!python main.py --model resnet50 --strategy clue --resume
```
```bash
!python main.py --model resnet50 --strategy badge --resume
```
```bash
!python main.py --model resnet50 --strategy vaal --resume
```

### Block B — DenseNet-169 (~13 h)

```bash
!python main.py --model densenet169 --strategy coreset --resume
```
```bash
!python main.py --model densenet169 --strategy clue --resume
```
```bash
!python main.py --model densenet169 --strategy badge --resume
```
```bash
!python main.py --model densenet169 --strategy vaal --resume
```

### Block C — EfficientNet-B4 (~13 h)

```bash
!python main.py --model efficientnet_b4 --strategy coreset --resume
```
```bash
!python main.py --model efficientnet_b4 --strategy clue --resume
```
```bash
!python main.py --model efficientnet_b4 --strategy badge --resume
```
```bash
!python main.py --model efficientnet_b4 --strategy vaal --resume
```

### Or: all 12 unattended

Walks every combination and skips anything already finished.

```bash
!python main.py --run-baselines --resume
```

### Per-run cost

| Strategy | Per run | Why |
|---|---|---|
| CoreSet | ~2.9 h | |
| CLUE | ~2.9 h | |
| BADGE | ~3.0 h | |
| VAAL | ~4.5 h | Trains a VAE and a discriminator every round, on top of the classifier |

**If time runs short, drop VAAL first.** CoreSet + BADGE + CLUE across all
three models is 9 runs and covers what reviewers expect. **Never drop
BADGE** — it is *the* reference point in this literature.

---

## 2. CPU JOBS — no GPU needed, cheap, high value

These reuse the already-trained models. They can run on Colab **or on your
laptop** — nothing here trains anything.

### 2a. The EfficientNet-B4 noise diagnostic 🔴 do this one

**Why:** EfficientNet-B4 scores **0.6–1.0% accuracy** under mild noise, while
the other two backbones lose only about 20 points. That is below random
guessing, and its risk score goes *worse than a coin flip*. We currently
cannot explain it, and Figure 33 points straight at it. A reviewer will ask
what happens at intermediate noise levels.

A sweep answers it: a **smooth decline** means genuine architecture-specific
fragility (a real, publishable finding); a **sudden cliff** means the model
collapses onto one rare class (also reportable, but described accurately).

```bash
!python -m evaluation.rigor.dump_test_predictions --corruption gaussian_noise_0.01 --only efficientnet_b4_entropy_dual_metric efficientnet_b4_entropy_uncertainty_only
```
```bash
!python -m evaluation.rigor.dump_test_predictions --corruption gaussian_noise_0.02 --only efficientnet_b4_entropy_dual_metric efficientnet_b4_entropy_uncertainty_only
```
```bash
!python -m evaluation.rigor.dump_test_predictions --corruption gaussian_noise_0.03 --only efficientnet_b4_entropy_dual_metric efficientnet_b4_entropy_uncertainty_only
```
```bash
!python -m evaluation.rigor.dump_test_predictions --corruption gaussian_noise_0.10 --only efficientnet_b4_entropy_dual_metric efficientnet_b4_entropy_uncertainty_only
```

Then rebuild the robustness figures:

```bash
!python -m evaluation.rigor.robustness
```

**Cost:** about 20 minutes, CPU only.

### 2b. The missing JPEG corruption

`jpeg_q30` is defined in the code and named in the documentation, but was
never actually run — it is absent from the results. It is arguably the *most*
realistic corruption of the set, since every image that moves between
hospital systems gets re-compressed.

```bash
!python -m evaluation.rigor.dump_test_predictions --corruption jpeg_q30 --only densenet169_entropy_dual_metric densenet169_entropy_uncertainty_only efficientnet_b4_entropy_dual_metric efficientnet_b4_entropy_uncertainty_only resnet50_entropy_dual_metric resnet50_entropy_uncertainty_only
```
```bash
!python -m evaluation.rigor.robustness
```

**Cost:** about 40 minutes, CPU only.

### 2c. External validation on ISIC 2020 🟠 important

**Why:** every result so far is on HAM10000. Medical-imaging journals
increasingly treat external validation as mandatory. Without it, the standing
criticism is that the model learned HAM10000's cameras and lighting rather
than the disease.

> ⚠️ **Not ISIC 2019.** It *contains* HAM10000, with the original
> `ISIC_xxxxxxx` filenames intact — validating on it means testing on
> training images. **ISIC 2020** is the correct set.

Download ISIC 2020 (~3 GB) to Drive first, then:

```bash
!python -m evaluation.rigor.external_validation_isic
```

**Cost:** a few hours, mostly downloading. **No training** — one inference pass.

### 2d. Regenerate every analysis after new results land

Run these once the 12 baselines are finished.

```bash
!python -m evaluation.rigor.dump_test_predictions
```
```bash
!python -m evaluation.rigor.run_all
```
```bash
!python analysis/build_analysis.py
```

Or individually:

```bash
!python -m evaluation.rigor.statistics          # significance tests
```
```bash
!python -m evaluation.rigor.ablation_posthoc    # ablation + threshold sweep
```
```bash
!python -m evaluation.rigor.calibration         # ECE, Brier, reliability
```
```bash
!python -m evaluation.rigor.per_class_auc       # per-class AUC + CIs
```
```bash
!python -m evaluation.rigor.al_efficiency       # labels vs accuracy
```
```bash
!python -m evaluation.rigor.gradcam             # heatmaps, both heads
```
```bash
!python -m evaluation.rigor.runtime             # timing breakdown
```

---

## 3. Bookkeeping

```bash
!python -m tools.build_manifest
```
Rebuilds `MANIFEST.csv` — every artefact with its checksum. Run after new
results arrive, then commit it.

```bash
!python -m tools.build_manifest --verify
```
Re-checks every checksum. Reports anything changed, missing or new.

---

## 4. Every flag, explained

| Flag | What it does | Values |
|---|---|---|
| `--model` | Which network | `resnet50`, `densenet169`, `efficientnet_b4` |
| `--strategy` | Use a **baseline** method instead of ours. Replaces the decision step; `--uncertainty` and `--policy` are then ignored | `coreset`, `badge`, `clue`, `vaal` |
| `--policy` | Our decision rule | `uncertainty_only` (old baseline), `dual_metric` (ours) |
| `--uncertainty` | How "confused" is measured | `entropy`, `margin`, `least_confidence`, `mc_dropout` |
| `--resume` | Continue from the last save. **Always include it** — safe even on a fresh start | — |
| `--rounds` | How many rounds | default 15 |
| `--seed` | Training seed. **Leave alone** — 42 is the plan | default 42 |
| `--query-budget` | Force a fixed labels-per-round. **Do not use with `--strategy`** — it switches off cost-matching | default 150 |
| `--risk-threshold` | Manual risk cut-off, for the sensitivity sweep only | default: auto each round |
| `--no-dynamic-weights` | Turn off class weighting, for the ablation only | — |
| `--run-all` | All 24 original experiments | — |
| `--run-baselines` | All 12 baseline runs | — |
| `--plot-only` | Redraw plots from saved results, no training | — |

### Flags you will not need

`--seed` (staying at 42), `--query-budget` (breaks cost-matching),
`--risk-threshold` and `--no-dynamic-weights` (ablations already done),
`--run-all` (the 24 are finished).

---

## 5. Checking progress

Open on Drive:

```
results/experiments/<name>/results.csv
```

**15 rows = finished.** Fewer means it stopped early, and `--resume` will
carry on from there.

| Type | Folder name looks like |
|---|---|
| Original 24 | `resnet50_entropy_dual_metric` |
| New baselines | `resnet50_baseline_badge` |

Each finished run also writes `environment.json` — the GPU, CUDA version,
library versions and git commit it ran under. New; the original 24 predate it.

---

## 6. If something goes wrong

**Nothing you can do by accident loses more than the round in progress.**
Checkpoints save at the end of every round, straight to Drive.

**Recovery, always the same two steps:**
1. Re-run the setup cell.
2. Paste the exact same command, `--resume` still on it.

| Error | Cause | Fix |
|---|---|---|
| `cannot import name 'SPLIT_SEED'` | Used the old `setup_cell.txt` | Use `setup_cell.py` |
| `Cost-matching needs the reference run ...` | That model's `entropy_dual_metric` is missing from Drive | Check `results/experiments/` on Drive |
| `CUDA out of memory` | Previous run still holding the GPU | Runtime → Restart runtime, setup cell, same command |
| `available=False` | No GPU attached | Runtime → Change runtime type → T4 GPU |
| `No such file: archive.zip` | Drive not mounted or path wrong | Re-run setup cell, allow Drive access |

---

## 7. Suggested order

| # | Job | Cost | Type |
|---|---|---|---|
| 1 | EfficientNet noise diagnostic (2a) | 20 min | CPU |
| 2 | JPEG corruption (2b) | 40 min | CPU |
| 3 | ResNet-50 baselines (Block A) | ~13 h | GPU |
| 4 | ISIC 2020 external validation (2c) | ~3 h | CPU + download |
| 5 | DenseNet-169 baselines (Block B) | ~13 h | GPU |
| 6 | EfficientNet-B4 baselines (Block C) | ~13 h | GPU |
| 7 | Regenerate all analyses (2d) | ~1 h | CPU |
| 8 | Rebuild manifest (3) | 5 min | CPU |

**Jobs 1 and 2 cost an hour of CPU between them and close two real gaps.** Do
them tonight, before any GPU time.
