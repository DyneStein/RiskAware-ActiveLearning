"""
Per-lesion-class AUC (one-vs-rest) with bootstrap confidence intervals.

WHY PER-CLASS AND NOT JUST OVERALL
----------------------------------
HAM10000 is ~67% nevi. A model that quietly gives up on the rare classes
still posts a high overall accuracy, so an aggregate number hides exactly
the failure mode that matters clinically. Melanoma is the class the entire
project exists for, and it is only ~11% of the data — it needs its own
number, with an interval around it.

WHAT IS REPORTED
----------------
  ROC-AUC (one-vs-rest) per class   how well the class's probability
                                    separates that class from all others;
                                    prevalence-independent.
  PR-AUC / average precision        the honest companion for rare classes:
                                    ROC-AUC can look flattering when
                                    negatives vastly outnumber positives,
                                    PR-AUC does not.
  Risk-head AUC                     malignant vs benign, straight from the
                                    risk head — the policy-independent test
                                    of whether the danger signal is real.
  95% CIs                           percentile bootstrap over test images
                                    (2000 resamples), so every AUC comes
                                    with an interval rather than a bare
                                    point estimate.

AUC is computed via the Mann-Whitney rank identity rather than sklearn's
curve construction, purely so 2000 bootstrap resamples stay fast.

Outputs
-------
  figures/  21_roc_curves_per_class.png
            22_auc_per_class_with_ci.png
            23_melanoma_auc_dual_vs_baseline.png
  tables/   per_class_auc.csv
            auc_summary_by_policy.csv
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_curve

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from evaluation.rigor.paths import (  # noqa: E402
    PRED_DIR, FIG_DIR, TABLE_DIR, CLASS_NAMES, HIGH_RISK_CLASSES,
    MODELS, METHODS, COLOR_UNC, COLOR_DUAL, ensure_dirs, parse_experiment_id,
)

N_BOOT = 2000
FULL_NAMES = {
    'akiec': 'Actinic Keratoses', 'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis', 'df': 'Dermatofibroma', 'mel': 'Melanoma',
    'nv': 'Melanocytic Nevi', 'vasc': 'Vascular Lesions',
}


def fast_auc(y_true, scores):
    """ROC-AUC via the Mann-Whitney U identity. y_true in {0,1}."""
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    r = rankdata(scores)
    return float((r[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_auc_ci(y_true, scores, n_boot=N_BOOT, seed=42, alpha=0.05):
    """Percentile bootstrap CI for AUC, resampling test images."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = np.empty(n_boot)
    stats[:] = np.nan
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if yt.sum() == 0 or yt.sum() == n:
            continue
        stats[b] = fast_auc(yt, scores[idx])
    stats = stats[~np.isnan(stats)]
    if len(stats) == 0:
        return np.nan, np.nan
    return (float(np.percentile(stats, 100 * alpha / 2)),
            float(np.percentile(stats, 100 * (1 - alpha / 2))))


def analyse_one(df, exp_id, model, method, policy):
    rows = []
    true_label = df["true_label"].to_numpy()

    for c in CLASS_NAMES:
        y = (true_label == c).astype(int)
        s = df[f"prob_{c}"].to_numpy()
        auc = fast_auc(y, s)
        lo, hi = bootstrap_auc_ci(y, s)
        rows.append({
            "experiment_id": exp_id, "model": model, "method": method,
            "policy": policy, "target": c, "target_name": FULL_NAMES[c],
            "kind": "class_probability", "n_positive": int(y.sum()),
            "prevalence": float(y.mean()), "auc": auc,
            "auc_ci_low": lo, "auc_ci_high": hi,
            "pr_auc": float(average_precision_score(y, s)) if y.sum() else np.nan,
        })

    # The risk head, judged on its own: malignant vs benign.
    y_mal = df["true_label"].isin(HIGH_RISK_CLASSES).astype(int).to_numpy()
    s_risk = df["risk_score"].to_numpy()
    lo, hi = bootstrap_auc_ci(y_mal, s_risk)
    rows.append({
        "experiment_id": exp_id, "model": model, "method": method,
        "policy": policy, "target": "malignant", "target_name": "Malignant (risk head)",
        "kind": "risk_head", "n_positive": int(y_mal.sum()),
        "prevalence": float(y_mal.mean()), "auc": fast_auc(y_mal, s_risk),
        "auc_ci_low": lo, "auc_ci_high": hi,
        "pr_auc": float(average_precision_score(y_mal, s_risk)),
    })

    # Same target, but scored by summing the classifier's malignant-class
    # probabilities — the ORIGINAL pre-redesign risk definition. Included so
    # the two-head redesign can be justified with a number.
    s_sum = df[[f"prob_{c}" for c in sorted(HIGH_RISK_CLASSES)]].sum(axis=1).to_numpy()
    lo, hi = bootstrap_auc_ci(y_mal, s_sum)
    rows.append({
        "experiment_id": exp_id, "model": model, "method": method,
        "policy": policy, "target": "malignant", "target_name": "Malignant (summed class probs)",
        "kind": "summed_class_probs", "n_positive": int(y_mal.sum()),
        "prevalence": float(y_mal.mean()), "auc": fast_auc(y_mal, s_sum),
        "auc_ci_low": lo, "auc_ci_high": hi,
        "pr_auc": float(average_precision_score(y_mal, s_sum)),
    })
    return rows


# ---------------------------------------------------------------------------
# Does the two-head redesign actually buy anything?
# ---------------------------------------------------------------------------
def decoupling_analysis(dumps):
    """
    Test the ACTUAL claim behind giving the risk head its own parameters.

    Over the whole test set, the risk head and the old "sum the malignant
    class probabilities" definition score almost identically — so on a
    headline AUC the redesign looks like it bought nothing.

    But that was never the claim. The argument was about DECOUPLING: when
    the classification head is confidently wrong, a risk score computed
    *from* those same probabilities is necessarily wrong too, because it is
    a deterministic function of them. A separately-parameterised head can
    disagree. Whether it does is only visible on the subset where the
    classifier fails — which is exactly the subset the safety mechanism
    exists for.

    So the comparison is repeated on three nested populations:
      all images  ->  misclassified images  ->  CONFIDENTLY misclassified
    and, most concretely, on the false negatives: true malignant cases the
    classifier called benign. Those are the missed cancers. The question
    that matters is what fraction of them each scoring rule still flags.

    IMPORTANT — how to read the failure subsets. Conditioning on "the
    classifier was wrong" selects malignant cases it called benign together
    with benign cases it called malignant. Any score correlated with the
    classifier is therefore pushed BELOW chance on that subset, by
    construction. The absolute AUC there is a selection artefact and must
    not be read as "the risk score is worse than random". What is
    meaningful is the DIFFERENCE between the two scoring rules on the same
    subset, since both are subject to identical selection — and the missed-
    cancer flag rate, which is an absolute, operational number with no such
    problem.
    """
    rows = []
    for (model, method, policy), df in sorted(dumps.items()):
        y_mal = df["true_label"].isin(HIGH_RISK_CLASSES).to_numpy().astype(int)
        risk = df["risk_score"].to_numpy()
        summed = df[[f"prob_{c}" for c in sorted(HIGH_RISK_CLASSES)]].sum(axis=1).to_numpy()
        correct = (df["true_label"] == df["predicted_label"]).to_numpy()
        conf = df["confidence"].to_numpy()
        pred_benign = ~df["predicted_label"].isin(HIGH_RISK_CLASSES).to_numpy()

        populations = {
            "all_images": np.ones(len(df), bool),
            "misclassified": ~correct,
            "confidently_misclassified": (~correct) & (conf > 0.9),
        }
        for pop_name, mask in populations.items():
            if mask.sum() < 20 or len(np.unique(y_mal[mask])) < 2:
                continue
            rows.append({
                "model": model, "method": method, "policy": policy,
                "population": pop_name, "n": int(mask.sum()),
                "n_malignant": int(y_mal[mask].sum()),
                "auc_risk_head": fast_auc(y_mal[mask], risk[mask]),
                "auc_summed_probs": fast_auc(y_mal[mask], summed[mask]),
            })

        # The missed cancers themselves.
        fn = (y_mal == 1) & pred_benign
        if fn.sum() > 0:
            rows.append({
                "model": model, "method": method, "policy": policy,
                "population": "false_negatives_only", "n": int(fn.sum()),
                "n_malignant": int(fn.sum()),
                "auc_risk_head": np.nan, "auc_summed_probs": np.nan,
                "mean_risk_head": float(risk[fn].mean()),
                "mean_summed_probs": float(summed[fn].mean()),
                "frac_flagged_risk_head_at_0.5": float((risk[fn] > 0.5).mean()),
                "frac_flagged_summed_at_0.5": float((summed[fn] > 0.5).mean()),
            })
    return pd.DataFrame(rows)


def fig_decoupling(dec):
    pops = ["all_images", "misclassified", "confidently_misclassified"]
    pretty = {"all_images": "All test images",
              "misclassified": "Classifier got it WRONG",
              "confidently_misclassified": "Classifier CONFIDENTLY wrong\n(conf > 0.9)"}
    sub = dec[dec.population.isin(pops)]
    if not len(sub):
        return
    x = np.arange(len(pops))
    w = 0.38
    risk_m = [sub[sub.population == p]["auc_risk_head"].mean() for p in pops]
    risk_s = [sub[sub.population == p]["auc_risk_head"].std() for p in pops]
    sum_m = [sub[sub.population == p]["auc_summed_probs"].mean() for p in pops]
    sum_s = [sub[sub.population == p]["auc_summed_probs"].std() for p in pops]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w / 2, sum_m, w, yerr=sum_s, capsize=4, color="#94a3b8",
           label="Summed class probabilities (original design)")
    ax.bar(x + w / 2, risk_m, w, yerr=risk_s, capsize=4, color=COLOR_DUAL,
           label="Independent risk head (two-head redesign)")
    for xi, (a, b) in enumerate(zip(sum_m, risk_m)):
        ax.text(xi - w / 2, a + 0.012, f"{a:.3f}", ha="center", fontsize=9)
        ax.text(xi + w / 2, b + 0.012, f"{b:.3f}", ha="center", fontsize=9,
                fontweight="bold")
    ax.axhline(0.5, color="red", ls="--", lw=1.4, label="Chance (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels([pretty[p] for p in pops], fontsize=10)
    ax.set_ylabel("AUROC for malignant vs benign")
    ax.set_ylim(0.0, 1.08)
    ax.set_title("Does giving the risk head its own parameters buy anything?\n"
                 "Compare the two BARS within each group, not their height: the last two "
                 "groups are\nselected on classifier error, which depresses both scores by "
                 "construction (see note).",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower left")
    ax.text(0.5, -0.16,
            "Note: conditioning on 'the classifier was wrong' selects malignant cases the model called "
            "benign and benign cases it called malignant,\nso any score correlated with the classifier "
            "is pushed below chance there. The absolute level is therefore not interpretable; the gap "
            "between\nthe two bars is. The risk head leads in both failure groups — evidence the extra "
            "parameters decouple it, but only partially.",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5,
            style="italic", color="#374151")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "32_risk_head_decoupling.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig_roc_curves(dumps):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8))
    cmap = plt.get_cmap("tab10")
    for ax, model in zip(axes, MODELS):
        dfs = [d for (m, _, p), d in dumps.items()
               if m == model and p == "dual_metric"]
        if not dfs:
            continue
        df = pd.concat(dfs, ignore_index=True)
        tl = df["true_label"].to_numpy()
        for i, c in enumerate(CLASS_NAMES):
            y = (tl == c).astype(int)
            s = df[f"prob_{c}"].to_numpy()
            if y.sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y, s)
            lw = 3.0 if c == "mel" else 1.5
            ax.plot(fpr, tpr, lw=lw, color=cmap(i),
                    label=f"{FULL_NAMES[c]} ({fast_auc(y, s):.3f})")
        y_mal = df["true_label"].isin(HIGH_RISK_CLASSES).astype(int).to_numpy()
        fpr, tpr, _ = roc_curve(y_mal, df["risk_score"].to_numpy())
        ax.plot(fpr, tpr, lw=3.0, color="black", ls="--",
                label=f"Risk head: malignant ({fast_auc(y_mal, df['risk_score'].to_numpy()):.3f})")
        ax.plot([0, 1], [0, 1], color="gray", lw=1, ls=":")
        ax.set_title(model, fontsize=12, fontweight="bold")
        ax.set_xlabel("False positive rate")
        ax.legend(fontsize=7.5, loc="lower right")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("True positive rate (sensitivity)")
    fig.suptitle("ROC curves per lesion class (one-vs-rest), dual-metric runs pooled\n"
                 "Melanoma and the risk head drawn thick — the two clinically decisive curves",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "21_roc_curves_per_class.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_auc_with_ci(auc_df):
    sub = auc_df[auc_df.kind == "class_probability"]
    agg = sub.groupby("target").agg(
        auc=("auc", "mean"), lo=("auc_ci_low", "mean"),
        hi=("auc_ci_high", "mean"), prev=("prevalence", "mean"),
        n=("n_positive", "mean")).reindex(CLASS_NAMES)

    fig, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(agg))
    colors = [COLOR_DUAL if c == "mel" else "#94a3b8" for c in agg.index]
    ax.barh(y, agg["auc"], color=colors,
            xerr=[agg["auc"] - agg["lo"], agg["hi"] - agg["auc"]],
            capsize=4, height=0.62)
    for i, (c, r) in enumerate(agg.iterrows()):
        ax.text(r["auc"] + 0.005, i, f"{r['auc']:.3f}  [{r['lo']:.3f}, {r['hi']:.3f}]",
                va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{FULL_NAMES[c]}\n(n={agg.loc[c, 'n']:.0f}, "
                        f"{100*agg.loc[c, 'prev']:.1f}%)" for c in agg.index],
                       fontsize=9)
    ax.axvline(0.5, color="red", ls="--", lw=1.2, label="Random (0.5)")
    ax.set_xlim(0.4, 1.06)
    ax.set_xlabel("One-vs-rest ROC-AUC (mean over 24 experiments, 95% bootstrap CI)")
    ax.set_title("Per-lesion-class AUC — melanoma highlighted",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "22_auc_per_class_with_ci.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_melanoma_comparison(auc_df):
    mel = auc_df[(auc_df.kind == "class_probability") & (auc_df.target == "mel")]
    pairs = sorted({(r.model, r.method) for r in mel.itertuples()})
    x = np.arange(len(pairs))
    w = 0.38
    fig, ax = plt.subplots(figsize=(13, 6))
    for off, pol, color, lbl in [(-w / 2, "uncertainty_only", COLOR_UNC, "Uncertainty-only"),
                                 (w / 2, "dual_metric", COLOR_DUAL, "Dual-metric (ours)")]:
        vals, los, his = [], [], []
        for m, me in pairs:
            r = mel[(mel.model == m) & (mel.method == me) & (mel.policy == pol)]
            vals.append(r.auc.iloc[0] if len(r) else np.nan)
            los.append(r.auc_ci_low.iloc[0] if len(r) else np.nan)
            his.append(r.auc_ci_high.iloc[0] if len(r) else np.nan)
        vals, los, his = np.array(vals), np.array(los), np.array(his)
        ax.bar(x + off, vals, w, color=color, label=lbl,
               yerr=[vals - los, his - vals], capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n{me}" for m, me in pairs], fontsize=8)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("Melanoma ROC-AUC (95% bootstrap CI)")
    ax.set_title("Melanoma detection AUC: dual-metric vs uncertainty-only\n"
                 "Overlapping intervals mean the difference is not resolvable at this sample size",
                 fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "23_melanoma_auc_dual_vs_baseline.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ensure_dirs()
    dumps = {}
    for f in sorted(os.listdir(PRED_DIR)):
        if not f.endswith("_test_predictions.csv"):
            continue
        exp_id = f.replace("_test_predictions.csv", "")
        model, method, policy = parse_experiment_id(exp_id)
        if model is None:
            continue
        dumps[(model, method, policy)] = pd.read_csv(os.path.join(PRED_DIR, f))

    print(f"Loaded {len(dumps)} dumps. Bootstrapping {N_BOOT} resamples per AUC...")
    if not dumps:
        print("No dumps found — run evaluation.rigor.dump_test_predictions first.")
        return

    rows = []
    for (model, method, policy), df in sorted(dumps.items()):
        exp_id = f"{model}_{method}_{policy}"
        rows.extend(analyse_one(df, exp_id, model, method, policy))
        print(f"  {exp_id}")

    auc_df = pd.DataFrame(rows)
    auc_df.to_csv(os.path.join(TABLE_DIR, "per_class_auc.csv"), index=False)

    summary = (auc_df.groupby(["kind", "target", "policy"])
               .agg(auc_mean=("auc", "mean"), auc_std=("auc", "std"),
                    ci_low=("auc_ci_low", "mean"), ci_high=("auc_ci_high", "mean"),
                    pr_auc=("pr_auc", "mean")).reset_index())
    summary.to_csv(os.path.join(TABLE_DIR, "auc_summary_by_policy.csv"), index=False)

    dec = decoupling_analysis(dumps)
    dec.to_csv(os.path.join(TABLE_DIR, "risk_head_decoupling.csv"), index=False)

    fig_roc_curves(dumps)
    fig_auc_with_ci(auc_df)
    fig_melanoma_comparison(auc_df)
    fig_decoupling(dec)

    print("\n=== Per-class AUC (mean over 24 experiments, mean 95% bootstrap CI) ===")
    cls = auc_df[auc_df.kind == "class_probability"]
    for c in CLASS_NAMES:
        s = cls[cls.target == c]
        star = "  <-- MELANOMA" if c == "mel" else ""
        print(f"  {FULL_NAMES[c]:<22} AUC {s.auc.mean():.4f} "
              f"[{s.auc_ci_low.mean():.3f}, {s.auc_ci_high.mean():.3f}]   "
              f"PR-AUC {s.pr_auc.mean():.4f}   n={s.n_positive.mean():.0f}{star}")

    print("\n=== Malignant vs benign, two ways of scoring it ===")
    for kind, label in [("risk_head", "Risk head (two-head redesign)"),
                        ("summed_class_probs", "Summed class probs (original)")]:
        s = auc_df[auc_df.kind == kind]
        print(f"  {label:<38} AUC {s.auc.mean():.4f} "
              f"[{s.auc_ci_low.mean():.3f}, {s.auc_ci_high.mean():.3f}]   "
              f"PR-AUC {s.pr_auc.mean():.4f}")

    print("\n=== Does the two-head redesign buy anything? (decoupling test) ===")
    print("  Overall AUC is a tie. The design claim was about what happens when the")
    print("  classifier is WRONG, so the comparison is repeated on that subset.")
    print("  NB: conditioning on classifier error depresses BOTH scores below chance")
    print("  by construction — compare the two columns, not the absolute level.\n")
    for pop, label in [("all_images", "All test images"),
                       ("misclassified", "Classifier got it wrong"),
                       ("confidently_misclassified", "Classifier confidently wrong")]:
        s = dec[dec.population == pop]
        if not len(s):
            continue
        delta = s["auc_risk_head"].mean() - s["auc_summed_probs"].mean()
        print(f"  {label:<32} n={s['n'].mean():6.0f}  "
              f"risk head {s['auc_risk_head'].mean():.4f}  vs  "
              f"summed {s['auc_summed_probs'].mean():.4f}   "
              f"(Δ {delta:+.4f})")
    fn = dec[dec.population == "false_negatives_only"]
    if len(fn):
        print(f"\n  On the MISSED CANCERS themselves "
              f"(true malignant, called benign; n={fn['n'].mean():.0f} per experiment):")
        print(f"    mean risk-head score   {fn['mean_risk_head'].mean():.4f}   "
              f"-> {100*fn['frac_flagged_risk_head_at_0.5'].mean():.1f}% still flagged at 0.5")
        print(f"    mean summed-prob score {fn['mean_summed_probs'].mean():.4f}   "
              f"-> {100*fn['frac_flagged_summed_at_0.5'].mean():.1f}% still flagged at 0.5")

    print(f"\nFigures -> {FIG_DIR}\nTables  -> {TABLE_DIR}")


if __name__ == "__main__":
    main()
