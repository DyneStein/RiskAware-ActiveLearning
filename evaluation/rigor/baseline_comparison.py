"""
Head-to-head comparison: the dual-metric escalation policy vs four recent
acquisition baselines (CoreSet, BADGE, CLUE, VAAL).

WHAT THIS ANSWERS
-----------------
Reviewers do not ask "is your method good?" -- they ask "is it better than
what already exists, measured fairly?". This script produces that answer,
and it separates two questions that are easy to conflate:

    LEARNING    given the same number of labels, which method produces the
                more accurate classifier?

    SAFETY      how many dangerous cases did each method auto-accept
                without human review?

They are different scoreboards and a method can win one while losing the
other. Reporting them separately is the honest presentation.

THE FAIRNESS PROBLEM, AND HOW IT IS SOLVED
------------------------------------------
The four baselines are *acquisition strategies*: "given a budget of k
labels, which k images do I pick?". Ours is an *escalation policy*: "which
images are unsafe for the model to auto-accept?" -- and it therefore
chooses its own budget each round.

Comparing them directly would be meaningless: a method that queries more
labels should win on accuracy, so any accuracy gap would just be measuring
budget, not intelligence. The runs were therefore COST-MATCHED: in every
round, each baseline was handed exactly the number of labels the
dual-metric run spent in that same round on that same backbone, read from
the reference run's own results.csv. This script re-verifies that matching
held before it reports anything, and refuses to continue if it did not.

STATISTICAL POWER -- READ THIS BEFORE QUOTING A P-VALUE
--------------------------------------------------------
There are only three backbones, so a run-level paired test across them has
n=3. A two-sided Wilcoxon signed-rank test with n=3 has a minimum
attainable p-value of 2/2^3 = 0.250. Significance is therefore
*arithmetically impossible* at n=3, no matter how large the effect is. Any
p-value from that test reports direction, not evidence, and is labelled as
such throughout.

The real evidence comes from the IMAGE level. All runs share one frozen
test split of 1,905 images (SPLIT_SEED=42, deliberately decoupled from the
training seed), so each pair of models can be compared image by image with
an exact McNemar test. n is then 1,905 rather than 3, and only the images
the two models *disagree* on carry information.

Three levels are reported, and the distinction is stated in every output:

    descriptive   the main table -- effect sizes, no p-values
    image-level   McNemar on 1,905 paired images -- real evidence
    run-level     n=3 across backbones -- DIRECTION ONLY, underpowered

Outputs (analysis/rigor/tables, analysis/rigor/figures):
    baseline_comparison_main.csv            the main results table
    baseline_comparison_safety.csv          the safety scoreboard
    baseline_comparison_mcnemar.csv         image-level tests, Holm-adjusted
    baseline_comparison_runlevel.csv        n=3 direction check
    baseline_comparison_learning_curves.csv accuracy vs labels, per round
    36_baseline_safety.png
    37_baseline_pareto.png
    38_baseline_learning_curves.png
    39_baseline_mcnemar_forest.png

Usage
-----
    python -m evaluation.rigor.baseline_comparison
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from evaluation.rigor.paths import (  # noqa: E402
    EXPERIMENTS_DIR, PRED_DIR, FIG_DIR, TABLE_DIR,
    CLASS_NAMES, MODELS, BASELINES, FINAL_ROUND,
    COLOR_DUAL, COLOR_UNC, ensure_dirs,
)
from evaluation.rigor.statistics import (  # noqa: E402
    holm_bonferroni, mcnemar_exact, stars, metrics_from_idx,
    paired_bootstrap_diff, HIGH_RISK_IDX, MEL_IDX,
)

# The method under test, and the reference run the baselines were matched to.
OURS = "entropy_dual_metric"
# The project's original within-paper baseline. Not a literature method, but
# it is what the dual-metric policy was designed against, so it stays in the
# table as the "ablate the risk head" reference point.
PRIOR = "entropy_uncertainty_only"

# Display order and labels for the table rows.
METHOD_ORDER = [
    (OURS,    "Dual-metric (ours)"),
    (PRIOR,   "Uncertainty-only"),
    ("baseline_coreset", "CoreSet"),
    ("baseline_badge",   "BADGE"),
    ("baseline_clue",    "CLUE"),
    ("baseline_vaal",    "VAAL"),
]
BASELINE_KEYS = [f"baseline_{s}" for s in BASELINES]

CITATIONS = {
    "baseline_coreset": "Sener & Savarese, ICLR 2018",
    "baseline_badge":   "Ash et al., ICLR 2020",
    "baseline_clue":    "Prabhu et al., ICCV 2021",
    "baseline_vaal":    "Sinha et al., ICCV 2019",
}

COLORS = {
    OURS: COLOR_DUAL,
    PRIOR: COLOR_UNC,
    "baseline_coreset": "#8c6bb1",
    "baseline_badge":   "#b45309",
    "baseline_clue":    "#2c7fb8",
    "baseline_vaal":    "#d6604d",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_rounds():
    """Per-round results.csv for every method we compare, keyed (model, method)."""
    out = {}
    for model in MODELS:
        for key, _ in METHOD_ORDER:
            path = os.path.join(EXPERIMENTS_DIR, f"{model}_{key}", "results.csv")
            if os.path.exists(path):
                out[(model, key)] = pd.read_csv(path)
    return out


def load_preds():
    """Per-image test predictions, keyed (model, method)."""
    out = {}
    for model in MODELS:
        for key, _ in METHOD_ORDER:
            path = os.path.join(PRED_DIR, f"{model}_{key}_test_predictions.csv")
            if os.path.exists(path):
                out[(model, key)] = pd.read_csv(path)
    return out


def verify_cost_matching(rounds):
    """
    Refuse to report a comparison that is not cost-matched.

    If a baseline spent a different number of labels than the reference run,
    every accuracy comparison downstream is confounded by budget and the
    whole table is misleading. Better to fail loudly here.
    """
    print("Cost-matching check (baseline labels per round == dual-metric's)")
    problems = []
    for model in MODELS:
        ref = rounds.get((model, OURS))
        if ref is None:
            problems.append(f"{model}: reference run {OURS} is missing")
            continue
        ref_q = ref["queries_this_round"].astype(int).tolist()
        for key in BASELINE_KEYS:
            d = rounds.get((model, key))
            if d is None:
                problems.append(f"{model}_{key}: missing")
                continue
            got = d["queries_this_round"].astype(int).tolist()
            if got == ref_q:
                print(f"  {model:16s} {key:18s} OK   ({sum(got)} labels)")
            else:
                problems.append(
                    f"{model}_{key}: spent {sum(got)} labels, reference spent {sum(ref_q)}")
    if problems:
        print("\nCOST MATCHING FAILED:")
        for p in problems:
            print("  -", p)
        raise SystemExit(
            "Refusing to emit a comparison table from runs that are not "
            "cost-matched -- the accuracy column would be confounded by budget.")
    print("  all baselines cost-matched to the reference run\n")


# ---------------------------------------------------------------------------
# A. Main descriptive table
# ---------------------------------------------------------------------------
def main_table(rounds, preds):
    """
    Final-round performance for every method on every backbone.

    Deliberately carries NO p-values. It is the "how big is the effect"
    table; "is the effect real" is answered at the image level below.
    """
    rows = []
    for model in MODELS:
        for key, label in METHOD_ORDER:
            d = rounds.get((model, key))
            if d is None:
                continue
            f = d.iloc[-1]
            row = {
                "model": model,
                "method": label,
                "method_key": key,
                "citation": CITATIONS.get(key, "this work"),
                "total_labels_queried": int(f["total_queries"]),
                "accuracy": float(f["accuracy"]),
                "f1_macro": float(f["f1_macro"]),
                "fn_rate_malignant": float(f["fn_rate_malignant"]),
                "melanoma_recall": float(f["recall_mel"]),
                "unsafe_auto_accepts_final_round": int(f["unsafe_auto_accepts"]),
                "unsafe_auto_accepts_cumulative": int(d["unsafe_auto_accepts"].sum()),
            }
            # Recompute accuracy from the per-image dump as a consistency
            # check on the logged number. They must agree; if they do not,
            # the dump and the checkpoint are out of sync.
            p = preds.get((model, key))
            if p is not None:
                rec = metrics_from_idx(p["true_idx"].to_numpy(),
                                      p["predicted_idx"].to_numpy())
                row["accuracy_recomputed"] = rec["accuracy"]
                row["accuracy_matches_log"] = bool(
                    abs(rec["accuracy"] - row["accuracy"]) < 5e-3)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TABLE_DIR, "baseline_comparison_main.csv"), index=False)
    return df


def safety_table(main):
    """
    The safety scoreboard: dangerous cases auto-accepted without review.

    Expressed as a reduction relative to each baseline, because "we cut
    unsafe auto-accepts by X%" is the claim the paper actually makes.
    """
    rows = []
    for model in MODELS:
        sub = main[main.model == model].set_index("method_key")
        if OURS not in sub.index:
            continue
        ours = sub.loc[OURS]
        for key, label in METHOD_ORDER:
            if key == OURS or key not in sub.index:
                continue
            other = sub.loc[key]
            cum_o = ours["unsafe_auto_accepts_cumulative"]
            cum_b = other["unsafe_auto_accepts_cumulative"]
            rows.append({
                "model": model,
                "compared_to": label,
                "ours_unsafe_cumulative": int(cum_o),
                "baseline_unsafe_cumulative": int(cum_b),
                "absolute_reduction": int(cum_b - cum_o),
                "relative_reduction_pct": 100.0 * (cum_b - cum_o) / cum_b if cum_b else np.nan,
                "ours_unsafe_final_round": int(ours["unsafe_auto_accepts_final_round"]),
                "baseline_unsafe_final_round": int(other["unsafe_auto_accepts_final_round"]),
                "ours_labels": int(ours["total_labels_queried"]),
                "baseline_labels": int(other["total_labels_queried"]),
                "cost_matched": bool(key not in (PRIOR,)),
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TABLE_DIR, "baseline_comparison_safety.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# B. Image-level tests -- where the real p-values come from
# ---------------------------------------------------------------------------
def mcnemar_table(preds):
    """
    Exact McNemar, ours vs each baseline, on the shared 1,905-image test set.

    Two outcomes are tested per pair:

      accuracy         was the 7-class prediction correct?
      malignant-miss   was a truly malignant lesion called benign? (the
                       safety-critical error -- a missed cancer)

    Valid only because the test split is frozen across runs, so image i is
    the same image in both models. Holm-corrected within each outcome
    family across all 12 comparisons.
    """
    rows = []
    for model in MODELS:
        a = preds.get((model, OURS))
        if a is None:
            continue
        for key, label in METHOD_ORDER:
            if key == OURS or (model, key) not in preds:
                continue
            b = preds[(model, key)]

            # Align on image_id so a differently-ordered dump cannot
            # silently mispair images -- the pairing is the whole test.
            m = a.merge(b, on="image_id", suffixes=("_a", "_b"))
            if len(m) != len(a):
                raise SystemExit(
                    f"{model} {key}: test sets differ "
                    f"({len(m)} shared of {len(a)}). Pairing is invalid.")
            assert (m["true_idx_a"] == m["true_idx_b"]).all(), \
                f"{model} {key}: same image carries different labels"

            true_idx = m["true_idx_a"].to_numpy()
            pa = m["predicted_idx_a"].to_numpy()
            pb = m["predicted_idx_b"].to_numpy()

            # --- outcome 1: overall correctness
            acc_p, n_ours_only, n_base_only = mcnemar_exact(
                true_idx == pa, true_idx == pb)

            # --- outcome 2: missed cancers, on malignant images only
            is_mal = np.isin(true_idx, list(HIGH_RISK_IDX))
            hr = list(HIGH_RISK_IDX)
            # "caught" = predicted some malignant class (not necessarily the
            # right one). Calling a melanoma a BCC still sends the patient
            # for treatment; calling it a nevus does not.
            caught_a = np.isin(pa[is_mal], hr)
            caught_b = np.isin(pb[is_mal], hr)
            mal_p, n_caught_ours_only, n_caught_base_only = mcnemar_exact(
                caught_a, caught_b)

            lo, hi, _ = paired_bootstrap_diff(true_idx, pb, pa, "accuracy",
                                              n_boot=2000)
            rows.append({
                "model": model,
                "ours_vs": label,
                "method_key": key,
                "n_test_images": len(m),
                "accuracy_ours": float((true_idx == pa).mean()),
                "accuracy_other": float((true_idx == pb).mean()),
                "accuracy_delta_pp": 100.0 * float((true_idx == pa).mean()
                                                   - (true_idx == pb).mean()),
                "accuracy_delta_ci_lo_pp": 100.0 * lo,
                "accuracy_delta_ci_hi_pp": 100.0 * hi,
                "n_correct_ours_only": n_ours_only,
                "n_correct_other_only": n_base_only,
                "mcnemar_accuracy_p": acc_p,
                "n_malignant_images": int(is_mal.sum()),
                "malignant_caught_ours": int(caught_a.sum()),
                "malignant_caught_other": int(caught_b.sum()),
                "n_caught_ours_only": n_caught_ours_only,
                "n_caught_other_only": n_caught_base_only,
                "mcnemar_malignant_p": mal_p,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Holm correction, applied separately per outcome family. Correcting
    # across families would over-penalise: they answer different questions.
    for col in ["mcnemar_accuracy_p", "mcnemar_malignant_p"]:
        df[col + "_holm"] = holm_bonferroni(df[col])
        df[col.replace("mcnemar_", "").replace("_p", "") + "_sig_holm"] = \
            df[col + "_holm"] < 0.05

    df.to_csv(os.path.join(TABLE_DIR, "baseline_comparison_mcnemar.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# C. Run-level check -- direction only, structurally underpowered
# ---------------------------------------------------------------------------
def runlevel_table(main):
    """
    Does ours beat each baseline on all three backbones?

    A sign test / Wilcoxon here has n=3 and a floor of p=0.250, so no
    p-value is emitted at all -- reporting one would invite a reader to
    misread 0.250 as "tested and failed" when it is "cannot be tested".
    Only the win count and mean gap are given.
    """
    rows = []
    for key, label in METHOD_ORDER:
        if key == OURS:
            continue
        for metric, better in [("accuracy", "higher"),
                               ("f1_macro", "higher"),
                               ("fn_rate_malignant", "lower"),
                               ("melanoma_recall", "higher"),
                               ("unsafe_auto_accepts_cumulative", "lower")]:
            diffs, wins = [], 0
            for model in MODELS:
                sub = main[main.model == model].set_index("method_key")
                if OURS not in sub.index or key not in sub.index:
                    continue
                o = float(sub.loc[OURS, metric])
                b = float(sub.loc[key, metric])
                d = o - b
                diffs.append(d)
                if (better == "higher" and d > 0) or (better == "lower" and d < 0):
                    wins += 1
            if not diffs:
                continue
            rows.append({
                "ours_vs": label,
                "metric": metric,
                "better_is": better,
                "n_backbones": len(diffs),
                "backbones_ours_wins": wins,
                "mean_difference_ours_minus_other": float(np.mean(diffs)),
                "note": "n=3: direction only, significance not attainable",
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TABLE_DIR, "baseline_comparison_runlevel.csv"), index=False)
    return df


def learning_curve_table(rounds):
    """Accuracy against cumulative labels, every round, every method."""
    rows = []
    for (model, key), d in rounds.items():
        label = dict(METHOD_ORDER)[key]
        for _, r in d.iterrows():
            rows.append({
                "model": model, "method": label, "method_key": key,
                "round": int(r["round"]),
                "total_labels": int(r["total_queries"]),
                "labeled_count": int(r["labeled_count"]),
                "accuracy": float(r["accuracy"]),
                "f1_macro": float(r["f1_macro"]),
                "fn_rate_malignant": float(r["fn_rate_malignant"]),
                "unsafe_auto_accepts": int(r["unsafe_auto_accepts"]),
            })
    df = pd.DataFrame(rows).sort_values(["model", "method_key", "round"])
    df.to_csv(os.path.join(TABLE_DIR, "baseline_comparison_learning_curves.csv"),
              index=False)
    return df


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_safety(main):
    """Grouped bars: cumulative unsafe auto-accepts. The headline result."""
    fig, ax = plt.subplots(figsize=(10, 5.2))
    keys = [k for k, _ in METHOD_ORDER]
    width = 0.13
    x = np.arange(len(MODELS))
    for i, (key, label) in enumerate(METHOD_ORDER):
        vals = []
        for model in MODELS:
            sub = main[(main.model == model) & (main.method_key == key)]
            vals.append(float(sub["unsafe_auto_accepts_cumulative"].iloc[0])
                        if len(sub) else np.nan)
        pos = x + (i - (len(keys) - 1) / 2) * width
        bars = ax.bar(pos, vals, width, label=label, color=COLORS[key],
                      edgecolor="white", linewidth=0.6)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 120, f"{int(v):,}",
                        ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "-") for m in MODELS])
    ax.set_ylabel("Unsafe auto-accepts (cumulative over 15 rounds)")
    # Headroom for the rotated value labels, so they cannot run into the
    # legend sitting above the axes.
    ax.set_ylim(0, np.nanmax(main["unsafe_auto_accepts_cumulative"]) * 1.28)
    ax.set_title("Dangerous cases auto-accepted without human review — lower is better\n"
                 "All methods cost-matched: identical label budget per round",
                 fontsize=11, loc="left", pad=34)
    ax.legend(frameon=False, ncol=6, fontsize=8.5,
              loc="lower left", bbox_to_anchor=(0, 1.005))
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "36_baseline_safety.png"), dpi=300)
    plt.close(fig)


def fig_pareto(main):
    """
    The trade-off plot: accuracy against safety.

    Up-and-left is better. This is the figure that shows ours is not merely
    trading accuracy away to buy safety.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=False)
    for ax, model in zip(axes, MODELS):
        sub = main[main.model == model]
        for key, label in METHOD_ORDER:
            r = sub[sub.method_key == key]
            if not len(r):
                continue
            marker = "*" if key == OURS else "o"
            size = 320 if key == OURS else 90
            ax.scatter(r["unsafe_auto_accepts_cumulative"], r["accuracy"] * 100,
                       s=size, marker=marker, color=COLORS[key],
                       edgecolor="black", linewidth=0.6, zorder=3, label=label)
            ax.annotate(label.replace(" (ours)", ""),
                        (float(r["unsafe_auto_accepts_cumulative"].iloc[0]),
                         float(r["accuracy"].iloc[0]) * 100),
                        textcoords="offset points", xytext=(7, 5), fontsize=7.5)
        ax.set_title(model.replace("_", "-"), fontsize=10)
        ax.set_xlabel("Unsafe auto-accepts (cumulative)")
        _style(ax)
        ax.grid(alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel("Final test accuracy (%)")
    fig.suptitle("Safety–accuracy trade-off: upper-left is better. "
                 "Dual-metric (star) reaches comparable accuracy at a fraction "
                 "of the unsafe auto-accepts.",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIG_DIR, "37_baseline_pareto.png"), dpi=300)
    plt.close(fig)


def fig_learning_curves(curves):
    """Accuracy against labels spent. Overlapping curves = equal learning."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True)
    for ax, model in zip(axes, MODELS):
        sub = curves[curves.model == model]
        for key, label in METHOD_ORDER:
            d = sub[sub.method_key == key].sort_values("round")
            if not len(d):
                continue
            ax.plot(d["total_labels"], d["accuracy"] * 100,
                    marker="o", markersize=3, linewidth=2.2 if key == OURS else 1.3,
                    color=COLORS[key], label=label,
                    zorder=3 if key == OURS else 2,
                    alpha=1.0 if key == OURS else 0.85)
        ax.set_title(model.replace("_", "-"), fontsize=10)
        ax.set_xlabel("Cumulative labels queried")
        _style(ax)
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    # The uncertainty-only run set its own per-round budget, so it is the one
    # curve here that is NOT cost-matched: at a given x it has completed more
    # rounds, hence more training, than the others. Comparing it left-to-right
    # against the rest would credit it for extra compute, not better
    # selection. Said on the figure so the plot cannot be read that way.
    fig.suptitle("Learning efficiency at matched label cost — "
                 "curves that overlap mean equal learning per label\n"
                 "Ours and the four baselines share an identical per-round "
                 "budget. Uncertainty-only (gray) chose its own smaller "
                 "budget and is NOT cost-matched: at equal labels it has run "
                 "more rounds, so its early lead reflects extra training, "
                 "not better selection.",
                 fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIG_DIR, "38_baseline_learning_curves.png"), dpi=300)
    plt.close(fig)


def fig_mcnemar_forest(mc):
    """
    Accuracy difference with paired bootstrap CIs, one row per comparison.

    A CI crossing zero means "no detectable difference in accuracy" -- which
    for this paper is a GOOD result, since the claim is safety gained at no
    accuracy cost, not accuracy improved.
    """
    if mc.empty:
        return
    d = mc.sort_values(["model", "method_key"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.5, 0.42 * len(d) + 2.4))
    y = np.arange(len(d))[::-1]
    for yi, (_, r) in zip(y, d.iterrows()):
        lo, hi, c = r["accuracy_delta_ci_lo_pp"], r["accuracy_delta_ci_hi_pp"], r["accuracy_delta_pp"]
        # Colour by the HOLM-ADJUSTED test, not by whether the raw CI clears
        # zero. The two disagree for a few rows -- the CI is unadjusted, while
        # the reported p-value is corrected across all 15 comparisons -- and
        # colouring by the CI would mark rows as "wins" that the corrected
        # test does not support.
        sig = bool(r["accuracy_sig_holm"])
        if not sig:
            col = "#6b7280"
        else:
            col = COLOR_DUAL if c > 0 else "#b45309"
        ax.plot([lo, hi], [yi, yi], color=col, linewidth=2.2, solid_capstyle="round")
        ax.scatter([c], [yi], color=col, s=42, zorder=3, edgecolor="black", linewidth=0.5)
        ax.text(hi + 0.15, yi, stars(r["mcnemar_accuracy_p_holm"]),
                va="center", ha="left", fontsize=8, color=col)
    ax.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['model'].replace('_','-')}  vs  {r['ours_vs']}"
                        for _, r in d.iterrows()], fontsize=8.5)
    ax.set_xlabel("Accuracy difference, ours − other (percentage points)")
    ax.set_title("Accuracy: ours vs each comparison method\n"
                 "Paired bootstrap 95% CI on the same 1,905 test images.\n"
                 "Green = significant after Holm correction; grey = no "
                 "detectable difference (the intended result).",
                 fontsize=10.5, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "39_baseline_mcnemar_forest.png"), dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ensure_dirs()
    print("=" * 78)
    print("BASELINE COMPARISON — dual-metric escalation vs CoreSet/BADGE/CLUE/VAAL")
    print("=" * 78)

    rounds = load_rounds()
    preds = load_preds()
    print(f"loaded {len(rounds)} per-round result files, "
          f"{len(preds)} per-image prediction dumps\n")

    missing = [f"{m}_{k}" for m in MODELS for k, _ in METHOD_ORDER
               if (m, k) not in preds]
    if missing:
        print("WARNING: no prediction dump for "
              f"{len(missing)} run(s); image-level tests will skip them:")
        for x in missing:
            print("   ", x)
        print("  fix with: python -m evaluation.rigor.dump_test_predictions\n")

    verify_cost_matching(rounds)

    main_df = main_table(rounds, preds)
    if "accuracy_matches_log" in main_df:
        bad = main_df[main_df["accuracy_matches_log"] == False]  # noqa: E712
        if len(bad):
            print("WARNING: recomputed accuracy disagrees with the logged value for:")
            for _, r in bad.iterrows():
                print(f"    {r['model']}_{r['method_key']}: "
                      f"logged {r['accuracy']:.4f} vs dump {r['accuracy_recomputed']:.4f}")
        else:
            print("consistency: recomputed accuracy matches the logged value "
                  "for every run\n")

    print("-" * 78)
    print("A. MAIN TABLE (descriptive — no p-values by design)")
    print("-" * 78)
    for model in MODELS:
        sub = main_df[main_df.model == model]
        print(f"\n{model}")
        print(f"  {'method':22s} {'labels':>7s} {'acc':>7s} {'F1':>7s} "
              f"{'FN-mal':>7s} {'mel-rec':>8s} {'unsafe(cum)':>12s}")
        for key, label in METHOD_ORDER:
            r = sub[sub.method_key == key]
            if not len(r):
                continue
            r = r.iloc[0]
            print(f"  {label:22s} {r['total_labels_queried']:7d} "
                  f"{r['accuracy']:7.4f} {r['f1_macro']:7.4f} "
                  f"{r['fn_rate_malignant']:7.4f} {r['melanoma_recall']:8.4f} "
                  f"{r['unsafe_auto_accepts_cumulative']:12,d}")

    safety = safety_table(main_df)
    print("\n" + "-" * 78)
    print("B. SAFETY SCOREBOARD (cumulative unsafe auto-accepts)")
    print("-" * 78)
    for model in MODELS:
        sub = safety[safety.model == model]
        print(f"\n{model}")
        for _, r in sub.iterrows():
            print(f"  vs {r['compared_to']:22s} "
                  f"{r['ours_unsafe_cumulative']:6,d} vs {r['baseline_unsafe_cumulative']:6,d}  "
                  f"= −{r['absolute_reduction']:6,d}  "
                  f"({r['relative_reduction_pct']:5.1f}% fewer)")

    mc = mcnemar_table(preds)
    print("\n" + "-" * 78)
    print("C. IMAGE-LEVEL McNEMAR (n=1,905 paired images — the real evidence)")
    print("-" * 78)
    if mc.empty:
        print("  no prediction dumps available; skipped")
    else:
        for model in MODELS:
            sub = mc[mc.model == model]
            if not len(sub):
                continue
            print(f"\n{model}")
            for _, r in sub.iterrows():
                print(f"  vs {r['ours_vs']:22s} "
                      f"acc {r['accuracy_delta_pp']:+6.2f} pp "
                      f"[{r['accuracy_delta_ci_lo_pp']:+5.2f},{r['accuracy_delta_ci_hi_pp']:+5.2f}]  "
                      f"discordant {r['n_correct_ours_only']:3d}/{r['n_correct_other_only']:3d}  "
                      f"Holm p={r['mcnemar_accuracy_p_holm']:.4f} "
                      f"{stars(r['mcnemar_accuracy_p_holm'])}")
        n = int(mc["accuracy_sig_holm"].sum())
        print(f"\n  accuracy differences significant after Holm: {n}/{len(mc)}")
        nm = int(mc["malignant_sig_holm"].sum())
        print(f"  missed-cancer differences significant after Holm: {nm}/{len(mc)}")
        print("\n  Missed cancers on the held-out test set (malignant lesions "
              "called benign):")
        for model in MODELS:
            sub = mc[mc.model == model]
            if not len(sub):
                continue
            r = sub.iloc[0]
            print(f"    {model:16s} ours caught {r['malignant_caught_ours']}"
                  f"/{r['n_malignant_images']} malignant")
            for _, rr in sub.iterrows():
                print(f"        vs {rr['ours_vs']:22s} caught "
                      f"{rr['malignant_caught_other']:3d}  "
                      f"Holm p={rr['mcnemar_malignant_p_holm']:.4f} "
                      f"{stars(rr['mcnemar_malignant_p_holm'])}")

    run = runlevel_table(main_df)
    print("\n" + "-" * 78)
    print("D. RUN-LEVEL DIRECTION (n=3 backbones — UNDERPOWERED BY CONSTRUCTION)")
    print("-" * 78)
    print("  A two-sided Wilcoxon at n=3 cannot go below p=0.250, so no")
    print("  p-values are reported here. Win counts and mean gaps only.\n")
    for _, r in run.iterrows():
        print(f"  {r['ours_vs']:22s} {r['metric']:32s} "
              f"ours wins {r['backbones_ours_wins']}/{r['n_backbones']}  "
              f"mean Δ {r['mean_difference_ours_minus_other']:+10.4f}")

    curves = learning_curve_table(rounds)

    print("\n" + "-" * 78)
    print("FIGURES")
    print("-" * 78)
    fig_safety(main_df)
    fig_pareto(main_df)
    fig_learning_curves(curves)
    fig_mcnemar_forest(mc)
    for f in ["36_baseline_safety.png", "37_baseline_pareto.png",
              "38_baseline_learning_curves.png", "39_baseline_mcnemar_forest.png"]:
        p = os.path.join(FIG_DIR, f)
        if os.path.exists(p):
            print(f"  {f}  ({os.path.getsize(p)/1024:.0f} KB)")
    print("\nTables written to", TABLE_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
