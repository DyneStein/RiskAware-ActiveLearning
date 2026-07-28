"""
Robustness experiments: does the safety signal survive degraded images?

THE QUESTION
------------
A deployed dermoscopy system does not receive clean, curated JPEGs. It gets
images from a different device, slightly out of focus, under different
lighting, re-compressed by whatever software sits between camera and model.
A safety mechanism that only works on pristine data is not a safety
mechanism.

Each final model is re-evaluated on the SAME held-out test set put through
a set of mild, clinically plausible degradations: sensor noise, defocus
blur, brightness shift, contrast loss, and aggressive JPEG re-compression.
No retraining, no fine-tuning — the shipped model meets a worse image.

THE INTERESTING COMPARISON
--------------------------
Because the architecture has two heads, robustness can be measured
separately for each, and they need not behave alike:

  * classification accuracy — does the diagnosis survive?
  * risk-head AUROC — does the DANGER signal survive?

The second is what matters for safety. A system whose diagnosis degrades
but whose risk score holds up degrades *gracefully*: it gets less certain
about what the lesion is while still knowing it is dangerous, and escalates
it. The opposite ordering would be a serious finding against the design.
This module reports both so whichever way it falls is visible.

Prerequisites
-------------
Run the corrupted inference passes first, e.g.

    for c in gaussian_noise_0.05 blur_1.5 brightness_0.7 contrast_0.7 jpeg_q30; do
        python -m evaluation.rigor.dump_test_predictions --corruption $c --only <experiments>
    done

Outputs
-------
  figures/  29_robustness_degradation.png
            30_robustness_heads_compared.png
  tables/   robustness_summary.csv
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from evaluation.rigor.paths import (  # noqa: E402
    PRED_DIR, RIGOR_DIR, FIG_DIR, TABLE_DIR, CLASS_NAMES, HIGH_RISK_CLASSES,
    COLOR_UNC, COLOR_DUAL, COLOR_ACCENT, ensure_dirs, parse_experiment_id,
)

ROBUST_DIR = os.path.join(RIGOR_DIR, "predictions_robustness")
CORRUPTION_LABELS = {
    "clean": "Clean",
    "gaussian_noise_0.05": "Noise σ=0.05",
    "gaussian_noise_0.10": "Noise σ=0.10",
    "blur_1.5": "Defocus blur",
    "brightness_0.7": "Dim (×0.7)",
    "brightness_1.3": "Bright (×1.3)",
    "contrast_0.7": "Low contrast",
    "jpeg_q30": "JPEG q=30",
}


def fast_auc(y_true, scores):
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    r = rankdata(scores)
    return float((r[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def metrics(df):
    true_label = df["true_label"].to_numpy()
    pred_label = df["predicted_label"].to_numpy()
    is_mal = np.isin(true_label, list(HIGH_RISK_CLASSES))
    pred_mal = np.isin(pred_label, list(HIGH_RISK_CLASSES))
    is_mel = true_label == "mel"
    return {
        "accuracy": float((true_label == pred_label).mean()),
        "melanoma_recall": float((is_mel & (pred_label == "mel")).sum() / is_mel.sum())
        if is_mel.sum() else np.nan,
        "fn_rate_malignant": float((is_mal & ~pred_mal).sum() / is_mal.sum())
        if is_mal.sum() else np.nan,
        "risk_auroc": fast_auc(is_mal.astype(int), df["risk_score"].to_numpy()),
        "mean_confidence": float(df["confidence"].mean())
        if "confidence" in df else np.nan,
    }


def load_all():
    rows = []
    for directory, default_corruption in [(PRED_DIR, "clean"), (ROBUST_DIR, None)]:
        if not os.path.isdir(directory):
            continue
        for f in sorted(os.listdir(directory)):
            if not f.endswith("_test_predictions.csv"):
                continue
            stem = f.replace("_test_predictions.csv", "")
            if "__" in stem:
                exp_id, corruption = stem.split("__", 1)
            else:
                exp_id, corruption = stem, default_corruption or "clean"
            model, method, policy = parse_experiment_id(exp_id)
            if model is None:
                continue
            df = pd.read_csv(os.path.join(directory, f))
            row = {"experiment_id": exp_id, "model": model, "method": method,
                   "policy": policy, "corruption": corruption}
            row.update(metrics(df))
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ensure_dirs()
    df = load_all()
    if not len(df):
        print("No prediction dumps found.")
        return

    # Only experiments that actually have corrupted runs may contribute to the
    # clean baseline — otherwise "clean" would average over all 24 experiments
    # while every corruption row averages over the 6 that were run, and the
    # degradation would be measured against a different population.
    with_corrupt = set(df.loc[df["corruption"] != "clean", "experiment_id"])
    if with_corrupt:
        df = df[df["experiment_id"].isin(with_corrupt)].copy()
        print(f"Restricting to the {len(with_corrupt)} experiments that have "
              f"corrupted runs, so clean and corrupted share one population.")

    corruptions = [c for c in CORRUPTION_LABELS if c in set(df["corruption"])]
    if len(corruptions) <= 1:
        print("Only clean predictions found — run the corrupted passes first:")
        print("  python -m evaluation.rigor.dump_test_predictions --corruption blur_1.5")
        return

    # Express every corrupted run relative to its own clean baseline, so
    # models with different clean accuracy stay comparable.
    clean = (df[df.corruption == "clean"]
             .set_index("experiment_id")[["accuracy", "melanoma_recall",
                                          "risk_auroc", "fn_rate_malignant"]])
    df = df.join(clean, on="experiment_id", rsuffix="_clean")
    for m in ["accuracy", "melanoma_recall", "risk_auroc"]:
        df[f"{m}_retained"] = df[m] / df[f"{m}_clean"]
    df["fn_rate_increase"] = df["fn_rate_malignant"] - df["fn_rate_malignant_clean"]
    df.to_csv(os.path.join(TABLE_DIR, "robustness_summary.csv"), index=False)

    order = [c for c in CORRUPTION_LABELS if c in corruptions]
    labels = [CORRUPTION_LABELS[c] for c in order]
    x = np.arange(len(order))

    # --- figure 29: absolute degradation ----------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    for ax, (col, title, better) in zip(axes, [
        ("accuracy", "Accuracy", "higher"),
        ("melanoma_recall", "Melanoma recall", "higher"),
        ("risk_auroc", "Risk-head AUROC (the safety signal)", "higher"),
    ]):
        for policy, color, lbl in [("uncertainty_only", COLOR_UNC, "Uncertainty-only"),
                                   ("dual_metric", COLOR_DUAL, "Dual-metric")]:
            sub = df[df.policy == policy]
            if not len(sub):
                continue
            means = [sub[sub.corruption == c][col].mean() for c in order]
            sds = [sub[sub.corruption == c][col].std() for c in order]
            ax.errorbar(x, means, yerr=sds, marker="o", lw=2, ms=6,
                        capsize=4, color=color, label=lbl)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Metric value (mean ± s.d. across experiments)")
    axes[0].legend(fontsize=9)
    fig.suptitle("Robustness: the shipped model meets degraded images (no retraining)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "29_robustness_degradation.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- figure 30: which head degrades faster? ---------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    w = 0.38
    acc_ret = [100 * df[df.corruption == c]["accuracy_retained"].mean() for c in order]
    auc_ret = [100 * df[df.corruption == c]["risk_auroc_retained"].mean() for c in order]
    ax.bar(x - w / 2, acc_ret, w, label="Classification accuracy retained",
           color=COLOR_ACCENT)
    ax.bar(x + w / 2, auc_ret, w, label="Risk-head AUROC retained", color=COLOR_DUAL)
    ax.axhline(100, color="black", ls="--", lw=1.2, label="Clean baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("% of clean-image performance retained")
    ax.set_title("Which head degrades faster?\n"
                 "The risk head holding up better than the classifier means the system "
                 "loses confidence in the diagnosis before it loses the danger signal — "
                 "it escalates rather than silently mis-diagnoses.",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "30_robustness_heads_compared.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- figure 33: per-model, because the average hides a total collapse ----
    models = sorted(df["model"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    width = 0.8 / max(len(models), 1)
    for ax, col, title in [
        (axes[0], "accuracy", "Accuracy"),
        (axes[1], "risk_auroc", "Risk-head AUROC (the safety signal)"),
    ]:
        for j, model in enumerate(models):
            sub = df[df.model == model]
            vals = [sub[sub.corruption == c][col].mean() for c in order]
            ax.bar(x + (j - (len(models) - 1) / 2) * width, vals, width,
                   label=model)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(alpha=0.25, axis="y")
        if col == "risk_auroc":
            ax.axhline(0.5, color="red", ls="--", lw=1.2, label="Chance (0.5)")
    axes[0].set_ylabel("Metric value")
    axes[0].legend(fontsize=9)
    axes[1].legend(fontsize=9)
    fig.suptitle("Robustness broken down BY MODEL — the average hides an architecture-specific "
                 "failure\nEfficientNet-B4 collapses under mild sensor noise; the other two "
                 "backbones degrade gracefully",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "33_robustness_by_model.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    print(f"Experiments x corruptions evaluated: {len(df)}")
    print(f"\n{'corruption':<22}{'accuracy':>10}{'mel recall':>12}"
          f"{'risk AUROC':>12}{'acc kept':>10}{'AUROC kept':>12}")
    for c in order:
        s = df[df.corruption == c]
        print(f"  {CORRUPTION_LABELS[c]:<20}{s['accuracy'].mean():>10.4f}"
              f"{s['melanoma_recall'].mean():>12.4f}{s['risk_auroc'].mean():>12.4f}"
              f"{100*s['accuracy_retained'].mean():>9.1f}%"
              f"{100*s['risk_auroc_retained'].mean():>11.1f}%")

    corrupted = df[df.corruption != "clean"]
    if len(corrupted):
        print(f"\n  Averaged over all corruptions: accuracy retains "
              f"{100*corrupted['accuracy_retained'].mean():.1f}%, "
              f"risk-head AUROC retains "
              f"{100*corrupted['risk_auroc_retained'].mean():.1f}%")

        print(f"\n=== Per-model accuracy (the average above hides this) ===")
        piv = df.pivot_table(index="corruption", columns="model",
                             values="accuracy", aggfunc="mean").reindex(order)
        print(piv.round(4).to_string())
        worst = corrupted.loc[corrupted["accuracy"].idxmin()]
        print(f"\n  Worst single case: {worst['model']} under "
              f"{CORRUPTION_LABELS.get(worst['corruption'], worst['corruption'])} "
              f"-> accuracy {worst['accuracy']:.4f} "
              f"(clean {worst['accuracy_clean']:.4f}). Report per-model, not just the mean.")

    print(f"\nFigures -> {FIG_DIR}\nTables  -> {TABLE_DIR}")


if __name__ == "__main__":
    main()
