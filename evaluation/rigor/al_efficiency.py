"""
Active-learning efficiency: accuracy vs NUMBER OF LABELLED SAMPLES.

WHY THIS IS THE RIGHT X-AXIS
----------------------------
Plotting accuracy against *round number* is misleading here, because the
two policies do not buy the same number of labels per round: dual-metric
escalates on an extra (uncapped) risk route, so by round 15 it has spent
more of the oracle's time. Judged per round it would look "better" partly
just for having asked more questions.

Plotting against *labels consumed* removes that advantage entirely and asks
the primary research question: for the same annotation
budget, which policy gives the better model? Any gap left on this plot is a
real efficiency difference, not a spending difference.

Outputs
-------
  figures/  10_al_efficiency_accuracy_vs_labels.png
            11_al_efficiency_by_method.png
            12_melanoma_recall_vs_labels.png
            13_annotation_efficiency.png
  tables/   al_efficiency_budget_matched.csv
            labels_to_reach_accuracy.csv
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
    EXPERIMENTS_DIR, FIG_DIR, TABLE_DIR, MODELS, METHODS,
    COLOR_UNC, COLOR_DUAL, ensure_dirs, parse_experiment_id,
)

ACCURACY_TARGETS = [0.75, 0.80, 0.85, 0.88]


def load_all():
    """{(model, method, policy): per-round dataframe}"""
    traj = {}
    for name in sorted(os.listdir(EXPERIMENTS_DIR)):
        csv = os.path.join(EXPERIMENTS_DIR, name, "results.csv")
        if not os.path.isfile(csv):
            continue
        model, method, policy = parse_experiment_id(name)
        if model is None:
            continue
        traj[(model, method, policy)] = pd.read_csv(csv)
    return traj


def interp_at_budget(df, budget, col="accuracy"):
    """Metric value at a given labelled-set size, linearly interpolated."""
    x = df["labeled_count"].values
    y = df[col].values
    if budget < x.min() or budget > x.max():
        return np.nan
    return float(np.interp(budget, x, y))


def labels_to_reach(df, target, col="accuracy"):
    """Smallest labelled-set size at which the metric first reaches target."""
    hit = df[df[col] >= target]
    return int(hit["labeled_count"].iloc[0]) if len(hit) else np.nan


# ---------------------------------------------------------------------------
def fig_accuracy_vs_labels(traj):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, model in zip(axes, MODELS):
        for method in METHODS:
            for policy, color in [("uncertainty_only", COLOR_UNC),
                                  ("dual_metric", COLOR_DUAL)]:
                df = traj.get((model, method, policy))
                if df is None:
                    continue
                ax.plot(df["labeled_count"], df["accuracy"], color=color,
                        alpha=0.55, lw=1.4,
                        marker="o", ms=2.5)
        ax.set_title(model, fontsize=12, fontweight="bold")
        ax.set_xlabel("Labelled samples (oracle annotations used)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Test accuracy")
    handles = [plt.Line2D([], [], color=COLOR_DUAL, lw=2.5, label="Dual-metric (ours)"),
               plt.Line2D([], [], color=COLOR_UNC, lw=2.5, label="Uncertainty-only (baseline)")]
    axes[0].legend(handles=handles, loc="lower right", fontsize=9)
    fig.suptitle("Active-learning efficiency: accuracy vs annotation budget\n"
                 "(each line = one uncertainty method; overlapping curves = equal efficiency)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "10_al_efficiency_accuracy_vs_labels.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_by_method(traj):
    fig, axes = plt.subplots(3, 4, figsize=(19, 12), sharex=False, sharey=True)
    for r, model in enumerate(MODELS):
        for c, method in enumerate(METHODS):
            ax = axes[r, c]
            for policy, color, label in [
                ("uncertainty_only", COLOR_UNC, "Uncertainty-only"),
                ("dual_metric", COLOR_DUAL, "Dual-metric"),
            ]:
                df = traj.get((model, method, policy))
                if df is None:
                    continue
                ax.plot(df["labeled_count"], df["accuracy"], color=color,
                        lw=2, marker="o", ms=3, label=label)
            ax.grid(alpha=0.25)
            if r == 0:
                ax.set_title(method, fontsize=11, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{model}\nTest accuracy", fontsize=10)
            if r == 2:
                ax.set_xlabel("Labelled samples")
            if r == 0 and c == 0:
                ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Accuracy vs labelled samples — every model × uncertainty method",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "11_al_efficiency_by_method.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_melanoma_recall(traj):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, model in zip(axes, MODELS):
        for method in METHODS:
            for policy, color in [("uncertainty_only", COLOR_UNC),
                                  ("dual_metric", COLOR_DUAL)]:
                df = traj.get((model, method, policy))
                if df is None or "recall_mel" not in df:
                    continue
                ax.plot(df["labeled_count"], df["recall_mel"], color=color,
                        alpha=0.55, lw=1.4, marker="o", ms=2.5)
        ax.set_title(model, fontsize=12, fontweight="bold")
        ax.set_xlabel("Labelled samples")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Melanoma recall (sensitivity)")
    handles = [plt.Line2D([], [], color=COLOR_DUAL, lw=2.5, label="Dual-metric (ours)"),
               plt.Line2D([], [], color=COLOR_UNC, lw=2.5, label="Uncertainty-only (baseline)")]
    axes[0].legend(handles=handles, loc="lower right", fontsize=9)
    fig.suptitle("Clinical efficiency: melanoma sensitivity vs annotation budget",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "12_melanoma_recall_vs_labels.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_annotation_efficiency(budget_df):
    """Accuracy at matched budgets — the fair head-to-head."""
    piv = budget_df.dropna(subset=["acc_dual", "acc_unc"])
    labels = [f"{m}\n{me}" for m, me in zip(piv["model"], piv["method"])]
    x = np.arange(len(piv))
    w = 0.38
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - w / 2, piv["acc_unc"], w, label="Uncertainty-only", color=COLOR_UNC)
    ax.bar(x + w / 2, piv["acc_dual"], w, label="Dual-metric (ours)", color=COLOR_DUAL)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Test accuracy at matched annotation budget")
    ax.set_ylim(min(piv[["acc_dual", "acc_unc"]].min()) - 0.03, None)
    ax.grid(alpha=0.25, axis="y")
    ax.legend()
    ax.set_title("Budget-matched comparison: same number of labels for both policies\n"
                 "(budget = the smaller of the two runs' final labelled-set size)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "13_annotation_efficiency.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ensure_dirs()
    traj = load_all()
    print(f"Loaded {len(traj)} experiment trajectories.")

    rows, reach_rows = [], []
    for model in MODELS:
        for method in METHODS:
            d = traj.get((model, method, "dual_metric"))
            u = traj.get((model, method, "uncertainty_only"))
            if d is None or u is None:
                continue

            # Budget matching: compare both at the SAME number of labels —
            # the largest budget both runs actually reached.
            budget = int(min(d["labeled_count"].max(), u["labeled_count"].max()))
            row = {
                "model": model, "method": method, "matched_budget": budget,
                "acc_dual": interp_at_budget(d, budget, "accuracy"),
                "acc_unc": interp_at_budget(u, budget, "accuracy"),
                "f1_dual": interp_at_budget(d, budget, "f1_macro"),
                "f1_unc": interp_at_budget(u, budget, "f1_macro"),
                "mel_recall_dual": interp_at_budget(d, budget, "recall_mel"),
                "mel_recall_unc": interp_at_budget(u, budget, "recall_mel"),
                "fn_malignant_dual": interp_at_budget(d, budget, "fn_rate_malignant"),
                "fn_malignant_unc": interp_at_budget(u, budget, "fn_rate_malignant"),
                "final_labels_dual": int(d["labeled_count"].max()),
                "final_labels_unc": int(u["labeled_count"].max()),
            }
            row["acc_delta_pp"] = 100 * (row["acc_dual"] - row["acc_unc"])
            row["f1_delta_pp"] = 100 * (row["f1_dual"] - row["f1_unc"])
            row["mel_recall_delta_pp"] = 100 * (row["mel_recall_dual"] - row["mel_recall_unc"])
            rows.append(row)

            for target in ACCURACY_TARGETS:
                reach_rows.append({
                    "model": model, "method": method, "target_accuracy": target,
                    "labels_dual": labels_to_reach(d, target),
                    "labels_unc": labels_to_reach(u, target),
                })

    budget_df = pd.DataFrame(rows)
    reach_df = pd.DataFrame(reach_rows)
    reach_df["labels_saved_by_dual"] = reach_df["labels_unc"] - reach_df["labels_dual"]

    budget_df.to_csv(os.path.join(TABLE_DIR, "al_efficiency_budget_matched.csv"),
                     index=False)
    reach_df.to_csv(os.path.join(TABLE_DIR, "labels_to_reach_accuracy.csv"),
                    index=False)

    fig_accuracy_vs_labels(traj)
    fig_by_method(traj)
    fig_melanoma_recall(traj)
    fig_annotation_efficiency(budget_df)

    print("\n=== Budget-matched (same labels for both policies) ===")
    print(f"Pairs compared: {len(budget_df)}")
    print(f"Mean accuracy delta (dual - unc):      {budget_df['acc_delta_pp'].mean():+.2f} pp")
    print(f"Mean F1-macro delta:                   {budget_df['f1_delta_pp'].mean():+.2f} pp")
    print(f"Mean melanoma-recall delta:            {budget_df['mel_recall_delta_pp'].mean():+.2f} pp")
    print(f"Pairs where dual wins on accuracy:     "
          f"{(budget_df['acc_delta_pp'] > 0).sum()}/{len(budget_df)}")
    print(f"Pairs where dual wins on mel. recall:  "
          f"{(budget_df['mel_recall_delta_pp'] > 0).sum()}/{len(budget_df)}")

    print("\n=== Labels needed to reach an accuracy target (negative = dual needs fewer) ===")
    for target in ACCURACY_TARGETS:
        sub = reach_df[reach_df["target_accuracy"] == target].dropna(
            subset=["labels_dual", "labels_unc"])
        if len(sub):
            print(f"  acc>={target:.2f}: dual reached it in {len(sub)} pairs, "
                  f"mean labels saved = {sub['labels_saved_by_dual'].mean():+.0f}")

    print(f"\nFigures -> {FIG_DIR}\nTables  -> {TABLE_DIR}")


if __name__ == "__main__":
    main()
