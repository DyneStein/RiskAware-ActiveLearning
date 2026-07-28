# Data and Artefacts — where everything lives

Every artefact this project has produced is tracked. Not all of it is
*committed*, because 1.8 GB of model weights and 2.8 GB of dataset do not
belong in git history. This file explains the three storage tiers and how to
obtain or verify anything in them.

The complete per-file index — path, size and SHA-256 checksum for every
artefact in every tier — is `MANIFEST.csv` in the repository root, with a
readable overview in `MANIFEST_SUMMARY.md`. The manifest is committed, so git
versions the *index*: any change to any file anywhere shows up as a one-line
diff in a small text file.

---

## The three tiers

| Tier | What | Size | Where |
|---|---|---|---|
| **git** | Code, docs, figures, tables, per-round results, pool predictions, manifest | ~180 MB | This repository |
| **release** | 24 trained model checkpoints | 1.8 GB | GitHub Releases → Zenodo at submission |
| **external** | HAM10000 images | 2.8 GB | Downloaded from the original source |

---

## Tier 1 — in this repository

```
RiskAware-ActiveLearning/
├── active_learning/      AL loop, escalation policies, baselines
│   └── baselines/        CoreSet, BADGE, CLUE, VAAL
├── analysis/             33 figures, 18 result tables, findings
│   ├── build_analysis.py
│   └── rigor/            figures, tables, per-experiment predictions
├── colab/                setup cell and runbooks
├── data/                 dataset, transforms, pool manager
├── docs/                 this file, limitations
├── escalation/           uncertainty-only and dual-metric policies
├── evaluation/
│   └── rigor/            statistics, calibration, ablation, robustness,
│                         Grad-CAM, external validation, runtime
├── models/               ResNet-50, DenseNet-169, EfficientNet-B4
├── paper/                writing kit: methods, maths, figures, results
├── results/
│   ├── experiments/      per-round metrics, pool predictions, plots
│   ├── logs/  plots/  tables/
│   └── checkpoints/      ← tier 2, gitignored
├── risk_score/  uncertainty/  tools/
├── MANIFEST.csv          every artefact, with checksums
└── requirements.txt
```

Each completed experiment writes:

```
results/experiments/<experiment_id>/
├── results.csv           one row per round — every metric
├── full.json             the same data in one blob
├── environment.json      GPU, CUDA, library versions, git commit
├── pool_predictions/     per-image scores and decisions, every round
└── plots/                confusion matrix per round, learning curve
```

`environment.json` is new. The original 24 runs predate it, and the Colab
environments that produced them were never recorded and cannot be recovered —
that gap is stated plainly rather than papered over.

---

## Tier 2 — model checkpoints (1.8 GB)

24 final-round checkpoints, one per experiment:

| Backbone | Per file | × 8 |
|---|---|---|
| ResNet-50 | 97 MB | 776 MB |
| EfficientNet-B4 | 74 MB | 592 MB |
| DenseNet-169 | 54 MB | 432 MB |

**Why not in git.** GitHub warns above 50 MB per file and rejects above
100 MB — ResNet-50's 97 MB is uncomfortably close to the hard limit. Weights
are binary, so every version would be stored in full, and 1.8 GB in normal
history makes the repository painful to clone for anyone who only wants the
code. Git LFS does not solve it either: the free tier is 1 GB.

**Where they are.** Published as GitHub Release assets (2 GB per file, free,
no effect on clone size), and mirrored to Zenodo at submission — which is
free, allows 50 GB per record, and mints a DOI, which is what journals ask
for.

**What needs them.** Everything in `evaluation/rigor/` that touches a model:
calibration, per-class AUC, robustness, Grad-CAM, external validation. The
decision-level ablation and the statistics do not — they read the logged
per-round CSVs, which are in tier 1.

Download and place under `results/checkpoints/<experiment_id>/round_15/`,
then verify:

```bash
python -m tools.build_manifest --verify
```

---

## Tier 3 — the HAM10000 dataset (2.8 GB)

**Not redistributed here**, for two reasons. It is 2.8 GB, and it is licensed
**CC BY-NC-SA 4.0** — attribution, non-commercial, share-alike — which is
incompatible with this repository's MIT licence. Shipping the images inside an
MIT repository would put the two licences in direct conflict.

    Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large
    collection of multi-source dermatoscopic images of common pigmented skin
    lesions. Sci. Data 5, 180161 (2018).
    https://doi.org/10.7910/DVN/DBW86T

Expected layout:

```
archive/
├── HAM10000_images_part_1/     5,000 images
├── HAM10000_images_part_2/     5,015 images
└── HAM10000_metadata.csv
```

Place it as a sibling of the repository, or point `DATA_ROOT` at it:

```bash
export DATA_ROOT=/path/to/archive
```

### The 490-image seed set

The starting labelled set is fixed, and *which* images it contains is a
scientific fact that must be reproducible. It is defined by
`Seed Data/seed_metadata.csv` — a committed list of image IDs. The images
themselves are in the repository for convenience; the ID list is what
actually matters, and is enough to reconstruct the seed set from any copy of
HAM10000.

### On ISIC 2019 — a contamination trap worth knowing about

**ISIC 2019 contains HAM10000**, with the original `ISIC_xxxxxxx` identifiers
preserved. Using it for external validation means testing on training images
and reporting an inflated number. Some published work has fallen into exactly
this.

**ISIC 2020 is the correct external set** and is what
`evaluation/rigor/external_validation_isic.py` targets.

---

## Reproducing a result

```bash
git clone https://github.com/DyneStein/RiskAware-ActiveLearning.git
cd RiskAware-ActiveLearning
pip install -r requirements.txt
export DATA_ROOT=/path/to/archive          # tier 3
python -m tools.build_manifest --verify    # confirm you have what we had
```

**Analysis only** — no GPU, no training, reads the committed per-round CSVs:

```bash
python analysis/build_analysis.py
python -m evaluation.rigor.statistics
python -m evaluation.rigor.ablation_posthoc
```

**Anything touching a model** additionally needs the tier-2 checkpoints:

```bash
python -m evaluation.rigor.calibration
python -m evaluation.rigor.robustness
python -m evaluation.rigor.gradcam
```

**Retraining from scratch** — one experiment is ~2.9 GPU-hours, ~7.0 for
MC-dropout:

```bash
python main.py --model resnet50 --uncertainty entropy --policy dual_metric --resume
```

---

## Two seeds, doing different jobs

| Constant | Value | Varies? | Controls |
|---|---|---|---|
| `SPLIT_SEED` | 42, frozen | Never | Which images form the held-out test set |
| `RANDOM_SEED` | 42, 43, … | Per run (`--seed`) | Weight init, batch order, augmentation, dropout |

They are separate on purpose. If the test set moved with the training seed,
multi-seed runs would not be comparable to each other or to the original 24,
the image-level paired test would lose its pairing, and — with only ~9 `df`
images in the whole test set — per-class metrics would swing for reasons
unrelated to the method.

Runs at seed 42 keep unsuffixed folder names, so the original 24 experiments
are untouched. Any other seed writes to `<experiment_id>_s<seed>`.
