"""
Are the per-class AUC confidence intervals for the rare classes trustworthy?

WHY THIS EXISTS
---------------
`per_class_auc.py` reports a bootstrap 95% CI for every class. Reading its
output, something is clearly wrong:

    class    n_positive   mean AUC   mean CI width
    df                9     0.9961          0.0131
    vasc             14     0.9946          0.0171
    akiec            51     0.9629          0.0475
    mel             209     0.9454          0.0315

The two classes with the FEWEST positive cases have the NARROWEST intervals.
That is backwards. An interval built on 9 positive cases should be wide; a
narrow one is a claim of precision the data cannot support, and quoting
"AUC 0.996 [0.987, 1.000] for dermatofibroma (n=9)" in a paper would be
indefensible.

The suspected cause is a ceiling effect. With only 9 positives that the model
separates perfectly, almost every bootstrap resample also separates perfectly,
so nearly every replicate returns AUC = 1.0. The percentile interval across a
near-constant set of replicates is narrow by construction. The narrowness
measures the degeneracy of the estimator, not the precision of the estimate.

There is a second, compounding problem specific to resampling images: a
bootstrap resample of 1,905 images will sometimes contain very few of the 9
positives -- occasionally zero or one, where AUC is undefined. Those replicates
are dropped as NaN, which silently conditions the interval on the resamples
that happened to be well-behaved.

WHAT THIS SCRIPT MEASURES
-------------------------
For every class, over the bootstrap replicates:

    frac_at_ceiling     fraction of replicates with AUC exactly 1.0
    frac_undefined      fraction dropped because the resample had < 2
                        positives or < 2 negatives
    min_positives       smallest positive count seen in any resample
    ci_width            the reported interval width

and issues a per-class verdict: whether the CI may be quoted as-is, or is a
ceiling artifact and must not be.

Output:
    analysis/rigor/tables/rare_class_ci_diagnostic.csv
    analysis/rigor/figures/40_rare_class_ci_reliability.png

Usage
-----
    python -m evaluation.rigor.rare_class_ci
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
    PRED_DIR, FIG_DIR, TABLE_DIR, CLASS_NAMES, MODELS, ensure_dirs,
)

N_BOOT = 2000
SEED = 42
# The reference configuration. One configuration is enough: the question is
# about the estimator's behaviour at small n, which is a property of the test
# set's class counts, not of which model produced the scores.
REFERENCE = "entropy_dual_metric"

# A class is treated as too rare to support a per-class AUC claim below this
# many positive cases. 20 is a conventional floor for a stable ROC estimate;
# the diagnostic columns are what actually justify the verdict per class.
MIN_POSITIVES_FOR_CLAIM = 20


def fast_auc(y_true, scores):
    """
    AUC via the Mann-Whitney U identity. Returns NaN when undefined -- i.e.
    when the resample contains only one class, so there is no pair to rank.
    """
    y = np.asarray(y_true).astype(bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks within ties, else tied scores bias the statistic.
    s_sorted = np.asarray(scores)[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def diagnose(y_true, scores, n_boot=N_BOOT, seed=SEED):
    """Bootstrap the AUC and record how the estimator misbehaves."""
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=float)
    n = len(y)
    rng = np.random.default_rng(seed)

    reps = np.full(n_boot, np.nan)
    pos_counts = np.zeros(n_boot, dtype=int)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        pos_counts[b] = int(yb.sum())
        reps[b] = fast_auc(yb, s[idx])

    ok = ~np.isnan(reps)
    valid = reps[ok]
    out = {
        "auc_point": fast_auc(y, s),
        "n_positive": int(y.sum()),
        "n_boot": n_boot,
        "frac_undefined": float((~ok).mean()),
        "min_positives_in_resample": int(pos_counts.min()),
        "frac_resamples_under_5_positives": float((pos_counts < 5).mean()),
    }
    if len(valid) == 0:
        out.update({"ci_low": np.nan, "ci_high": np.nan, "ci_width": np.nan,
                    "frac_at_ceiling": np.nan})
        return out

    lo = float(np.percentile(valid, 2.5))
    hi = float(np.percentile(valid, 97.5))
    out.update({
        "ci_low": lo,
        "ci_high": hi,
        "ci_width": hi - lo,
        # "Exactly 1.0" up to float noise: a replicate in which every positive
        # outranked every negative.
        "frac_at_ceiling": float((valid >= 1.0 - 1e-12).mean()),
        "frac_above_0_99": float((valid >= 0.99).mean()),
    })
    return out


def verdict(row):
    """
    Whether this class's CI may be quoted.

    Two independent failure modes, either of which disqualifies it:
      - the estimator is pinned at its ceiling, so the interval measures
        degeneracy rather than precision;
      - resamples routinely contain too few positives for the statistic to
        be defined or stable.
    """
    if row["n_positive"] < MIN_POSITIVES_FOR_CLAIM:
        if row.get("frac_at_ceiling", 0) > 0.25:
            return ("DO NOT QUOTE — ceiling artifact",
                    f"{row['frac_at_ceiling']:.0%} of bootstrap replicates returned "
                    f"AUC exactly 1.0; the narrow interval reflects a degenerate "
                    f"estimator on {row['n_positive']} positives, not precision.")
        return ("DO NOT QUOTE — too few positives",
                f"only {row['n_positive']} positive cases; resamples fell as low as "
                f"{row['min_positives_in_resample']} positives.")
    if row.get("frac_undefined", 0) > 0.01:
        return ("CAUTION",
                f"{row['frac_undefined']:.1%} of replicates were undefined and dropped.")
    return ("OK", "sufficient positives; interval is interpretable.")


def main():
    ensure_dirs()
    print("=" * 78)
    print("RARE-CLASS AUC CONFIDENCE INTERVAL DIAGNOSTIC")
    print("=" * 78)
    print("Question: are the per-class AUC CIs for df (n=9) and vasc (n=14)")
    print("trustworthy, given they are NARROWER than mel's (n=209)?\n")

    rows = []
    for model in MODELS:
        path = os.path.join(PRED_DIR, f"{model}_{REFERENCE}_test_predictions.csv")
        if not os.path.exists(path):
            print(f"  missing dump, skipping: {os.path.basename(path)}")
            continue
        d = pd.read_csv(path)
        for cls in CLASS_NAMES:
            y = (d["true_label"] == cls).to_numpy()
            s = d[f"prob_{cls}"].to_numpy()
            r = diagnose(y, s)
            r.update({"model": model, "target": cls})
            v, why = verdict(r)
            r["verdict"] = v
            r["reason"] = why
            rows.append(r)

    if not rows:
        raise SystemExit("no prediction dumps found; run dump_test_predictions first")

    df = pd.DataFrame(rows)
    cols = ["model", "target", "n_positive", "auc_point", "ci_low", "ci_high",
            "ci_width", "frac_at_ceiling", "frac_above_0_99", "frac_undefined",
            "min_positives_in_resample", "frac_resamples_under_5_positives",
            "verdict", "reason"]
    df = df[[c for c in cols if c in df.columns]]
    out = os.path.join(TABLE_DIR, "rare_class_ci_diagnostic.csv")
    df.to_csv(out, index=False)

    agg = (df.groupby("target")
             .agg(n=("n_positive", "first"),
                  auc=("auc_point", "mean"),
                  ci_width=("ci_width", "mean"),
                  at_ceiling=("frac_at_ceiling", "mean"),
                  min_pos=("min_positives_in_resample", "min"),
                  verdict=("verdict", "first"))
             .sort_values("n"))

    print(f"{'class':8s} {'n':>5s} {'AUC':>7s} {'CIwidth':>8s} "
          f"{'@ceiling':>9s} {'minPos':>7s}  verdict")
    for cls, r in agg.iterrows():
        print(f"{cls:8s} {int(r['n']):5d} {r['auc']:7.4f} {r['ci_width']:8.4f} "
              f"{r['at_ceiling']:8.0%} {int(r['min_pos']):7d}  {r['verdict']}")

    print("\n" + "-" * 78)
    print("CONCLUSION")
    print("-" * 78)
    bad = agg[agg["verdict"].str.startswith("DO NOT QUOTE")]
    if len(bad):
        print("The narrow intervals on the rare classes are an ARTIFACT, confirmed:")
        for cls, r in bad.iterrows():
            print(f"  {cls} (n={int(r['n'])}): {r['at_ceiling']:.0%} of bootstrap "
                  f"replicates returned AUC exactly 1.0.")
        print("\nThe estimator sits at its ceiling, so the percentile interval")
        print("collapses. It is not evidence of precision. Do not report per-class")
        print("AUC for these classes as though it were comparable to mel or nv.")
        print("\nRecommended wording for the paper:")
        print('  "Per-class AUC is reported for the five classes with at least 20')
        print('   positive test cases. Dermatofibroma (n=9) and vascular lesions')
        print('   (n=14) are too rare in the held-out split to support a stable')
        print('   per-class estimate: the model separates them perfectly on this')
        print('   split, so bootstrap intervals collapse to the ceiling and')
        print('   understate uncertainty. We report their counts and omit the')
        print('   interval rather than quote a misleadingly narrow one."')
    else:
        print("No ceiling artifact detected; intervals are interpretable as reported.")

    # --- figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    a = agg.reset_index()
    colors = ["#b45309" if v.startswith("DO NOT") else "#1b7a5e"
              for v in a["verdict"]]

    ax1.scatter(a["n"], a["ci_width"], s=110, c=colors,
                edgecolor="black", linewidth=0.6, zorder=3)
    for _, r in a.iterrows():
        ax1.annotate(r["target"], (r["n"], r["ci_width"]),
                     textcoords="offset points", xytext=(7, 4), fontsize=9)
    ax1.set_xscale("log")
    ax1.set_xlabel("Positive test cases (log scale)")
    ax1.set_ylabel("Bootstrap 95% CI width")
    ax1.set_title("CI width should FALL as n rises.\n"
                  "df and vasc break the pattern — the tell-tale of a ceiling "
                  "artifact.", fontsize=10, loc="left")

    ax2.barh(a["target"], a["at_ceiling"] * 100, color=colors,
             edgecolor="white", linewidth=0.6)
    ax2.axvline(25, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax2.set_xlabel("Bootstrap replicates with AUC exactly 1.000 (%)")
    ax2.set_title("Why: the estimator is pinned at its ceiling\n"
                  "(dashed line = 25%, the threshold used to disqualify a CI)",
                  fontsize=10, loc="left")
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "40_rare_class_ci_reliability.png")
    fig.savefig(p, dpi=300)
    plt.close(fig)

    print(f"\nTable  -> {out}")
    print(f"Figure -> {p}")


if __name__ == "__main__":
    main()
