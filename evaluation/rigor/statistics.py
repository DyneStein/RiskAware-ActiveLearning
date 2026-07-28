"""
Statistical significance analysis: p-values, confidence intervals, effect
sizes, and multiple-comparison correction.

THE HONEST STARTING POINT
-------------------------
The matrix was run at ONE seed per configuration. That rules out the
textbook design (repeat each configuration over k seeds, test across
seeds). Rather than fake that, this module runs the two tests the existing
data genuinely supports, and labels exactly what each one can and cannot
conclude:

  A. CONFIGURATION-LEVEL (n = 12 paired configurations)
     Every (model, uncertainty-method) combination was run under both
     policies. Pairing on configuration and testing across the 12 pairs
     answers: "across the space of architectures and uncertainty measures,
     does the policy systematically change the outcome?" Wilcoxon
     signed-rank (non-parametric, no normality assumption, appropriate at
     n=12), corroborated by an exact sign test and an exact paired
     permutation test.
     CANNOT conclude: that the effect survives seed-to-seed noise. Nothing
     here measures run-to-run variance.

  B. IMAGE-LEVEL (n = 1905 paired test images)
     All 24 experiments were evaluated on a byte-identical test split
     (verified by checksum), so within a pair the two policies label the
     SAME images and predictions can be paired per image. McNemar's exact
     test is the correct test for paired binary outcomes, and a paired
     bootstrap over images gives CIs on the metric differences.
     CANNOT conclude: anything about training variance either — it
     quantifies uncertainty from the finite test set, given these two
     trained models.

  Together: (A) says the direction is consistent across configurations;
  (B) says whether a specific pair's gap is bigger than test-set noise.
  Neither substitutes for multi-seed replication, which stays the honest
  recommendation and is stated as such in every output.

MULTIPLE COMPARISONS
--------------------
Twelve pairs are tested at once, so ~1 in 20 would clear p<0.05 by chance.
Holm-Bonferroni adjusted p-values are reported alongside the raw ones; the
adjusted column is the one to quote.

Outputs
-------
  figures/  24_forest_plot_accuracy.png
            25_significance_heatmap.png
  tables/   significance_configuration_level.csv
            significance_image_level.csv
            significance_ablation_level.csv
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from evaluation.rigor.paths import (  # noqa: E402
    EXPERIMENTS_DIR, PRED_DIR, FIG_DIR, TABLE_DIR, CLASS_NAMES,
    HIGH_RISK_CLASSES, MODELS, METHODS, COLOR_UNC, COLOR_DUAL,
    ensure_dirs, parse_experiment_id,
)

N_BOOT = 5000
HIGH_RISK_IDX = {CLASS_NAMES.index(c) for c in HIGH_RISK_CLASSES}
MEL_IDX = CLASS_NAMES.index("mel")


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------
def holm_bonferroni(pvals):
    """Holm step-down adjusted p-values (monotone, same order as input)."""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    adj = np.full_like(p, np.nan)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return adj
    order = idx[np.argsort(p[idx])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def rank_biserial(diffs):
    """Paired rank-biserial correlation: Wilcoxon's effect size, in [-1, 1]."""
    d = np.asarray(diffs, dtype=float)
    d = d[d != 0]
    if len(d) == 0:
        return np.nan
    r = stats.rankdata(np.abs(d))
    total = r.sum()
    return float((r[d > 0].sum() - r[d < 0].sum()) / total)


def paired_permutation_p(diffs, n_perm=20000, seed=42):
    """Exact-ish paired permutation test: randomly flip the sign of each pair."""
    d = np.asarray(diffs, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return np.nan
    obs = abs(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    return float((np.sum(null >= obs - 1e-15) + 1) / (n_perm + 1))


def bootstrap_ci_mean(values, n_boot=N_BOOT, seed=42, alpha=0.05):
    """Percentile bootstrap CI for a mean."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def mcnemar_exact(a_correct, b_correct):
    """
    Exact McNemar test on paired binary outcomes.

    Only the DISCORDANT pairs carry information: images both models got
    right (or both got wrong) say nothing about which is better.
    """
    a = np.asarray(a_correct).astype(bool)
    b = np.asarray(b_correct).astype(bool)
    n01 = int((~a & b).sum())   # a wrong, b right
    n10 = int((a & ~b).sum())   # a right, b wrong
    n = n01 + n10
    if n == 0:
        return 1.0, n10, n01
    p = float(stats.binomtest(min(n01, n10), n, 0.5,
                              alternative="two-sided").pvalue)
    return p, n10, n01


def stars(p):
    if np.isnan(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


# ---------------------------------------------------------------------------
# Metrics recomputed from a prediction dump (definitions match
# evaluation/metrics.py exactly)
# ---------------------------------------------------------------------------
def metrics_from_idx(true_idx, pred_idx):
    from sklearn.metrics import f1_score
    is_mal = np.isin(true_idx, list(HIGH_RISK_IDX))
    pred_benign = ~np.isin(pred_idx, list(HIGH_RISK_IDX))
    is_mel = true_idx == MEL_IDX
    return {
        "accuracy": float((true_idx == pred_idx).mean()),
        "f1_macro": float(f1_score(true_idx, pred_idx, average="macro",
                                   labels=list(range(len(CLASS_NAMES))),
                                   zero_division=0)),
        "fn_rate_malignant": float((is_mal & pred_benign).sum() / is_mal.sum())
        if is_mal.sum() else np.nan,
        "melanoma_recall": float((is_mel & (pred_idx == MEL_IDX)).sum() / is_mel.sum())
        if is_mel.sum() else np.nan,
    }


def paired_bootstrap_diff(true_idx, pred_a, pred_b, metric, n_boot=1000, seed=42):
    """
    CI for metric(B) - metric(A), resampling the SAME images for both.

    Pairing matters: the two models' errors are correlated (same images,
    same architecture family), so an unpaired interval would be far too
    wide.
    """
    rng = np.random.default_rng(seed)
    n = len(true_idx)
    diffs = np.empty(n_boot)
    diffs[:] = np.nan
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        ma = metrics_from_idx(true_idx[idx], pred_a[idx])[metric]
        mb = metrics_from_idx(true_idx[idx], pred_b[idx])[metric]
        diffs[i] = mb - ma
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        return np.nan, np.nan, np.nan
    return (float(np.percentile(diffs, 2.5)),
            float(np.percentile(diffs, 97.5)),
            float((diffs <= 0).mean()))


# ---------------------------------------------------------------------------
# A. Configuration-level tests
# ---------------------------------------------------------------------------
def configuration_level():
    traj = {}
    for name in sorted(os.listdir(EXPERIMENTS_DIR)):
        csv = os.path.join(EXPERIMENTS_DIR, name, "results.csv")
        if not os.path.isfile(csv):
            continue
        m, me, p = parse_experiment_id(name)
        if m:
            traj[(m, me, p)] = pd.read_csv(csv)

    metrics = {
        "unsafe_auto_accepts_total": lambda d: d["unsafe_auto_accepts"].sum(),
        "final_accuracy": lambda d: d[d["round"] == d["round"].max()]["accuracy"].iloc[0],
        "final_f1_macro": lambda d: d[d["round"] == d["round"].max()]["f1_macro"].iloc[0],
        "final_fn_rate_malignant": lambda d: d[d["round"] == d["round"].max()]["fn_rate_malignant"].iloc[0],
        "final_recall_mel": lambda d: d[d["round"] == d["round"].max()]["recall_mel"].iloc[0],
        "total_queries": lambda d: d[d["round"] == d["round"].max()]["total_queries"].iloc[0],
    }

    rows = []
    for mname, fn in metrics.items():
        dual, unc, pairs = [], [], []
        for model in MODELS:
            for method in METHODS:
                d = traj.get((model, method, "dual_metric"))
                u = traj.get((model, method, "uncertainty_only"))
                if d is None or u is None:
                    continue
                dual.append(fn(d))
                unc.append(fn(u))
                pairs.append((model, method))
        dual, unc = np.array(dual, float), np.array(unc, float)
        diff = dual - unc
        if len(diff) < 3:
            continue

        try:
            w_stat, w_p = stats.wilcoxon(dual, unc)
        except ValueError:
            w_stat, w_p = np.nan, np.nan
        n_pos = int((diff > 0).sum())
        n_eff = int((diff != 0).sum())
        sign_p = (float(stats.binomtest(n_pos, n_eff, 0.5).pvalue)
                  if n_eff else np.nan)
        lo, hi = bootstrap_ci_mean(diff)

        rows.append({
            "metric": mname, "n_pairs": len(diff),
            "mean_dual": dual.mean(), "mean_uncertainty_only": unc.mean(),
            "mean_difference": diff.mean(),
            "diff_ci_low": lo, "diff_ci_high": hi,
            "pairs_dual_higher": n_pos,
            "wilcoxon_stat": w_stat, "wilcoxon_p": w_p,
            "sign_test_p": sign_p,
            "permutation_p": paired_permutation_p(diff),
            "rank_biserial_effect": rank_biserial(diff),
            "cohens_dz": float(diff.mean() / diff.std(ddof=1))
            if diff.std(ddof=1) > 0 else np.nan,
        })

    df = pd.DataFrame(rows)
    df["wilcoxon_p_holm"] = holm_bonferroni(df["wilcoxon_p"])
    df["significant_holm_0.05"] = df["wilcoxon_p_holm"] < 0.05
    return df


# ---------------------------------------------------------------------------
# B. Image-level tests
# ---------------------------------------------------------------------------
def image_level():
    dumps = {}
    if not os.path.isdir(PRED_DIR):
        return pd.DataFrame()
    for f in sorted(os.listdir(PRED_DIR)):
        if not f.endswith("_test_predictions.csv"):
            continue
        m, me, p = parse_experiment_id(f.replace("_test_predictions.csv", ""))
        if m:
            dumps[(m, me, p)] = pd.read_csv(os.path.join(PRED_DIR, f))

    rows = []
    for model in MODELS:
        for method in METHODS:
            d = dumps.get((model, method, "dual_metric"))
            u = dumps.get((model, method, "uncertainty_only"))
            if d is None or u is None:
                continue
            # Align on image_id so the pairing is exact, not positional.
            merged = u.merge(d, on="image_id", suffixes=("_unc", "_dual"))
            if len(merged) == 0:
                continue
            true_idx = merged["true_idx_unc"].to_numpy()
            pred_u = merged["predicted_idx_unc"].to_numpy()
            pred_d = merged["predicted_idx_dual"].to_numpy()

            p_acc, n_u_only, n_d_only = mcnemar_exact(pred_u == true_idx,
                                                      pred_d == true_idx)
            is_mal = np.isin(true_idx, list(HIGH_RISK_IDX))
            mal_ok_u = np.isin(pred_u[is_mal], list(HIGH_RISK_IDX))
            mal_ok_d = np.isin(pred_d[is_mal], list(HIGH_RISK_IDX))
            p_mal, n_mu, n_md = mcnemar_exact(mal_ok_u, mal_ok_d)

            m_u = metrics_from_idx(true_idx, pred_u)
            m_d = metrics_from_idx(true_idx, pred_d)

            row = {"model": model, "method": method, "n_test_images": len(merged)}
            for k in ["accuracy", "f1_macro", "fn_rate_malignant", "melanoma_recall"]:
                row[f"{k}_unc"] = m_u[k]
                row[f"{k}_dual"] = m_d[k]
                row[f"{k}_diff"] = m_d[k] - m_u[k]
                lo, hi, _ = paired_bootstrap_diff(true_idx, pred_u, pred_d, k)
                row[f"{k}_diff_ci_low"] = lo
                row[f"{k}_diff_ci_high"] = hi
                row[f"{k}_ci_excludes_zero"] = bool(
                    (not np.isnan(lo)) and (lo > 0 or hi < 0))
            row.update({
                "mcnemar_accuracy_p": p_acc,
                "mcnemar_disc_unc_only_correct": n_u_only,
                "mcnemar_disc_dual_only_correct": n_d_only,
                "mcnemar_malignant_p": p_mal,
                "mcnemar_mal_disc_unc_only_correct": n_mu,
                "mcnemar_mal_disc_dual_only_correct": n_md,
            })
            rows.append(row)

    df = pd.DataFrame(rows)
    if len(df):
        df["mcnemar_accuracy_p_holm"] = holm_bonferroni(df["mcnemar_accuracy_p"])
        df["mcnemar_malignant_p_holm"] = holm_bonferroni(df["mcnemar_malignant_p"])
        df["accuracy_sig_holm"] = df["mcnemar_accuracy_p_holm"] < 0.05
        df["malignant_sig_holm"] = df["mcnemar_malignant_p_holm"] < 0.05
    return df


# ---------------------------------------------------------------------------
# C. Ablation-level tests
# ---------------------------------------------------------------------------
def ablation_level():
    path = os.path.join(TABLE_DIR, "ablation_decision_level.csv")
    if not os.path.isfile(path):
        return pd.DataFrame()
    ab = pd.read_csv(path)
    piv = ab.pivot(index="experiment_id", columns="rule")

    rows = []
    for metric, better in [("unsafe_auto_accepts", "lower"),
                           ("high_risk_catch_rate", "higher"),
                           ("n_escalated", "lower")]:
        for rule in ["uncertainty_only", "risk_only", "random_matched"]:
            a = piv[(metric, rule)].to_numpy(float)
            b = piv[(metric, "dual_metric")].to_numpy(float)
            diff = b - a
            # random_matched is cost-matched to dual_metric by construction,
            # so their escalation counts are identical and the test is
            # degenerate (and scipy warns). Skip rather than print a NaN.
            if np.allclose(diff, 0):
                continue
            try:
                _, w_p = stats.wilcoxon(b, a)
            except ValueError:
                w_p = np.nan
            lo, hi = bootstrap_ci_mean(diff)
            rows.append({
                "metric": metric, "better_is": better,
                "comparison": f"dual_metric vs {rule}",
                "n_experiments": len(diff),
                "mean_dual": b.mean(), "mean_other": a.mean(),
                "mean_difference": diff.mean(),
                "diff_ci_low": lo, "diff_ci_high": hi,
                "wilcoxon_p": w_p,
                "rank_biserial_effect": rank_biserial(diff),
            })
    df = pd.DataFrame(rows)
    df["wilcoxon_p_holm"] = holm_bonferroni(df["wilcoxon_p"])
    return df


# ---------------------------------------------------------------------------
def fig_forest(img_df):
    if not len(img_df):
        return
    specs = [("accuracy", "Accuracy"), ("melanoma_recall", "Melanoma recall"),
             ("fn_rate_malignant", "Missed-malignant rate")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.5), sharey=True)
    labels = [f"{r.model}\n{r.method}" for r in img_df.itertuples()]
    y = np.arange(len(img_df))

    for ax, (col, title) in zip(axes, specs):
        est = img_df[f"{col}_diff"].to_numpy() * 100
        lo = img_df[f"{col}_diff_ci_low"].to_numpy() * 100
        hi = img_df[f"{col}_diff_ci_high"].to_numpy() * 100
        good_is_positive = col != "fn_rate_malignant"
        for i in range(len(est)):
            favours_dual = (est[i] > 0) == good_is_positive
            color = COLOR_DUAL if favours_dual else COLOR_UNC
            ax.plot([lo[i], hi[i]], [i, i], color=color, lw=2.2)
            ax.plot(est[i], i, "o", color=color, ms=7)
        ax.axvline(0, color="black", lw=1.2, ls="--")
        ax.set_title(f"Δ {title}\n(dual − uncertainty-only, pp)",
                     fontsize=11, fontweight="bold")
        ax.grid(alpha=0.25, axis="x")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8)
    fig.suptitle("Forest plot: per-configuration differences with 95% paired-bootstrap CIs\n"
                 "An interval crossing 0 means the difference is not distinguishable from "
                 "test-set noise. Green = favours dual-metric.",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "24_forest_plot_accuracy.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_significance_heatmap(cfg_df):
    if not len(cfg_df):
        return
    fig, ax = plt.subplots(figsize=(11, 0.75 * len(cfg_df) + 2.5))
    ax.axis("off")
    header = ["Metric", "Mean Δ (dual−baseline)", "95% CI", "Wilcoxon p",
              "Holm-adj. p", "Effect size", ""]
    cells = []
    for r in cfg_df.itertuples():
        cells.append([
            r.metric,
            f"{r.mean_difference:+,.4g}",
            f"[{r.diff_ci_low:+,.4g}, {r.diff_ci_high:+,.4g}]",
            f"{r.wilcoxon_p:.4f}",
            f"{getattr(r, 'wilcoxon_p_holm'):.4f}",
            f"{r.rank_biserial_effect:+.2f}",
            stars(getattr(r, "wilcoxon_p_holm")),
        ])
    table = ax.table(cellText=cells, colLabels=header, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.7)
    for j in range(len(header)):
        table[0, j].set_facecolor("#1f2937")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i, r in enumerate(cfg_df.itertuples(), start=1):
        sig = getattr(r, "wilcoxon_p_holm") < 0.05
        for j in range(len(header)):
            table[i, j].set_facecolor("#dcfce7" if sig else "#f3f4f6")
    ax.set_title("Configuration-level significance (n = 12 paired configurations)\n"
                 "Wilcoxon signed-rank, Holm-corrected across metrics. "
                 "*** p<0.001  ** p<0.01  * p<0.05  ns = not significant",
                 fontsize=12, fontweight="bold", pad=18)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "25_significance_heatmap.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ensure_dirs()

    print("=== A. Configuration-level (n=12 paired configurations) ===")
    cfg = configuration_level()
    cfg.to_csv(os.path.join(TABLE_DIR, "significance_configuration_level.csv"),
               index=False)
    for r in cfg.itertuples():
        print(f"  {r.metric:<28} Δ={r.mean_difference:+10.4f}  "
              f"CI[{r.diff_ci_low:+.4f}, {r.diff_ci_high:+.4f}]  "
              f"W p={r.wilcoxon_p:.4f}  Holm p={getattr(r,'wilcoxon_p_holm'):.4f} "
              f"{stars(getattr(r,'wilcoxon_p_holm'))}  "
              f"({r.pairs_dual_higher}/{r.n_pairs} pairs higher)")

    print("\n=== B. Image-level (n=1905 paired test images per configuration) ===")
    img = image_level()
    if len(img):
        img.to_csv(os.path.join(TABLE_DIR, "significance_image_level.csv"),
                   index=False)
        for r in img.itertuples():
            print(f"  {r.model:<16} {r.method:<17} "
                  f"Δacc={100*r.accuracy_diff:+6.2f}pp "
                  f"CI[{100*r.accuracy_diff_ci_low:+.2f},{100*r.accuracy_diff_ci_high:+.2f}]  "
                  f"McNemar p={r.mcnemar_accuracy_p:.4f} "
                  f"(Holm {getattr(r,'mcnemar_accuracy_p_holm'):.4f} "
                  f"{stars(getattr(r,'mcnemar_accuracy_p_holm'))})")
        n_sig = int(img["accuracy_sig_holm"].sum())
        print(f"\n  Configurations with a significant accuracy difference "
              f"after Holm correction: {n_sig}/{len(img)}")
        n_sig_mal = int(img["malignant_sig_holm"].sum())
        print(f"  Significant malignant-detection difference: {n_sig_mal}/{len(img)}")
    else:
        print("  (no prediction dumps yet — run dump_test_predictions first)")

    print("\n=== C. Ablation-level (n=24 experiments, decision-level replay) ===")
    abl = ablation_level()
    if len(abl):
        abl.to_csv(os.path.join(TABLE_DIR, "significance_ablation_level.csv"),
                   index=False)
        for r in abl.itertuples():
            print(f"  {r.metric:<22} {r.comparison:<34} "
                  f"Δ={r.mean_difference:+10.4f}  "
                  f"p={r.wilcoxon_p:.2e}  Holm={getattr(r,'wilcoxon_p_holm'):.2e} "
                  f"{stars(getattr(r,'wilcoxon_p_holm'))}")
    else:
        print("  (run ablation_posthoc first)")

    fig_forest(img)
    fig_significance_heatmap(cfg)
    print(f"\nFigures -> {FIG_DIR}\nTables  -> {TABLE_DIR}")


if __name__ == "__main__":
    main()
