# How to regenerate everything

**Nothing here needs retraining.** Every figure and table is produced by reloading the saved model
checkpoints and running one deterministic pass. It runs on CPU.

---

## The one command

```bash
cd RiskAware-ActiveLearning
python -m evaluation.rigor.run_all
```

That regenerates all figures and tables in dependency order. Roughly 30–45 minutes on CPU.

With the robustness passes included (slower — one extra prediction pass per corruption):

```bash
python -m evaluation.rigor.run_all --with-robustness --threads 8
```

If the prediction dumps already exist and only a plot has changed:

```bash
python -m evaluation.rigor.run_all --skip-dump
```

---

## Why this works without retraining

Each experiment saved a complete checkpoint at round 15:

```
results/checkpoints/<experiment_id>/round_15/
├── model.pt                    the trained weights
├── meta.json                   config, round, metrics
└── pool_state/
    ├── labeled.csv             which images were labelled
    ├── unlabeled.csv           which were not
    ├── test.csv                that run's own test split
    └── query_history.csv       what was escalated, round by round
```

`dump_test_predictions.py` reloads the weights, loads **that run's own `test.csv`**, and runs one
deterministic forward pass, writing every per-image probability and risk score to disk. Everything
downstream reads those dumps.

**The reload path is verified.** Recomputing round-15 accuracy from the checkpoint gives `0.8987`
against the logged `0.8986876640419947` — identical to 14 decimal places. Check this first if
anything ever looks wrong.

---

## Which module makes what

Run in this order; later ones depend on earlier ones.

| Module | Produces |
|---|---|
| `dump_test_predictions` | The per-image prediction dumps. **Everything else depends on this** |
| `al_efficiency` | Figures 10–13; `al_efficiency_budget_matched.csv`, `labels_to_reach_accuracy.csv` |
| `ablation_posthoc` | Figures 14–16; `ablation_decision_level.csv`, `risk_threshold_sweep.csv` |
| `calibration` | Figures 17–20; `calibration_metrics.csv` |
| `per_class_auc` | Figures 21–23, 32; `per_class_auc.csv`, `auc_summary_by_policy.csv`, `risk_head_decoupling.csv` |
| `statistics` | Figures 24, 25; the three `significance_*.csv` files |
| `runtime` | Figures 26, 27; the three `runtime_*.csv` files |
| `gradcam` | Figure 28 ×3; `gradcam_cases_*.csv` |
| `robustness` | Figures 29, 30, 33; `robustness_summary.csv` |
| `external_validation_isic` | External-validation figure and table *(not yet run)* |

The base figures 01–09 and the three tables in `analysis/tables/` come from a separate script:

```bash
python analysis/build_analysis.py
```

---

## Running one piece at a time

```bash
# Just the calibration analysis
python -m evaluation.rigor.calibration

# Re-dump predictions for one experiment, overwriting
python -m evaluation.rigor.dump_test_predictions \
    --only resnet50_entropy_dual_metric --overwrite --threads 8

# Runtime microbenchmarks (record the thread count!)
python -m evaluation.rigor.runtime --benchmark --threads 8
```

---

## Two known pitfalls

**1. Thread count changes the timings.**
`--threads` defaults to 8. Running the benchmark at 4 threads gives roughly 2.5× slower
per-image numbers. The value used is recorded in `analysis/rigor/runtime_benchmark.json`.
**Ratios between models and operations are stable; absolute milliseconds are not.** Report ratios.

**2. Console encoding on Windows.**
The default Windows console can't print characters like Δ and crashes with a `UnicodeEncodeError`.
`paths.py` reconfigures stdout and stderr to UTF-8 at import, which fixes it for every module —
so **import `paths` first** in any new script.

---

## Configuration

Everything lives in `evaluation/rigor/paths.py`:

```python
RESEARCH_ROOT = os.environ.get("RESEARCH_ROOT", r"C:\Users\dyssa\Desktop\Research")
```

Set the `RESEARCH_ROOT` environment variable to move the whole analysis to another machine —
no code changes needed.

The plot colours are defined there too:

```python
COLOR_UNC   = "#6b7280"   # grey  — uncertainty-only (baseline)
COLOR_DUAL  = "#1b7a5e"   # green — dual-metric (proposed)
COLOR_ACCENT= "#b45309"   # amber — highlight
```

---

## The two outstanding runs

**JPEG corruption** — the fifth robustness condition, about 40 minutes on CPU:

```bash
python -m evaluation.rigor.dump_test_predictions --corruption jpeg_q30 --threads 8 \
    --only resnet50_entropy_dual_metric resnet50_entropy_uncertainty_only \
           densenet169_entropy_dual_metric densenet169_entropy_uncertainty_only \
           efficientnet_b4_entropy_dual_metric efficientnet_b4_entropy_uncertainty_only
python -m evaluation.rigor.robustness
```

**External validation on ISIC 2020** — needs the dataset downloaded first:

```bash
python -m evaluation.rigor.external_validation_isic \
    --dataset isic2020 \
    --images-dir  /path/to/isic2020/images \
    --labels-csv  /path/to/isic2020/ground_truth.csv
```

> ⚠️ For `--dataset isic2019` the script **requires** the HAM10000 metadata so it can exclude
> overlapping images, and it **asserts the overlap is zero** before evaluating. It will refuse to
> run otherwise. This is deliberate — see `06_STATUS_AND_OPEN_ITEMS/STATUS_CHECKLIST.md`.

---

## Reproducibility statement for the paper

> All experiments use a fixed random seed (42). The train/test split is generated once, saved to
> disk, and shared byte-identically across all 24 experiments (verified by checksum). Every round
> writes its configuration, per-image scores and decisions, per-round metrics, and a model
> checkpoint. All figures and tables are regenerated from those saved artefacts by a single
> command, with no manual steps.
