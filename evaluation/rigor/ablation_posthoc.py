"""
Decision-level ablation + risk-threshold sensitivity sweep.

WHAT THIS ANSWERS
-----------------
"Are both signals actually needed, or is one of them carrying the result?"
and "how does the safety/cost trade-off move as the risk threshold moves?"

HOW IT WORKS (and what it does NOT claim)
-----------------------------------------
Every round, each experiment logged the per-image uncertainty score, risk
score and true label for the entire unlabelled pool
(results/experiments/<exp>/pool_predictions/round_N.csv), plus that round's
calibrated thresholds. The escalation rule is a pure function of those
numbers — so we can replay the exact same round under a DIFFERENT rule and
count precisely which images each rule would have caught or missed.

This is an ablation of the DECISION RULE, holding the model fixed. It is
not a full retraining ablation: it does not capture how a different
labelling choice would have changed the next round's model. That makes it
cheap (seconds, no GPU) and exactly controlled (all rules see identical
scores from an identical model), but it is a one-step counterfactual and is
reported as such. A full retraining ablation is a separate, GPU-cost item.

Rules compared
--------------
  uncertainty_only   top-K most uncertain, plus anything over the
                     calibrated uncertainty threshold  (the baseline)
  risk_only          anything over the calibrated risk threshold, uncapped
  dual_metric        the union of the two  (ours)
  random_matched     a random set of the same size as dual_metric's
                     (cost-matched sanity floor: does dual_metric beat
                     simply escalating that many images at random?)

Outputs
-------
  figures/  14_ablation_both_signals_needed.png
            15_risk_threshold_sweep.png
            16_safety_cost_pareto.png
  tables/   ablation_decision_level.csv
            risk_threshold_sweep.csv
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
    EXPERIMENTS_DIR, FIG_DIR, TABLE_DIR, HIGH_RISK_CLASSES,
    COLOR_UNC, COLOR_DUAL, COLOR_ACCENT, ensure_dirs, parse_experiment_id,
)

RISK_SWEEP = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
              0.95, 0.99, 0.999, 1.01]   # 1.01 == risk route disabled
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# The escalation rules, reimplemented exactly as escalation/*.py define them
# ---------------------------------------------------------------------------
def _uncertainty_route(unc, tau_u, k):
    over = np.where(unc > tau_u)[0]
    if k > 0 and len(unc) > 0:
        top_k = np.argsort(unc)[::-1][:min(k, len(unc))]
    else:
        top_k = np.array([], dtype=int)
    return np.union1d(top_k, over).astype(int)


def _risk_route(risk, tau_r):
    return np.where(risk > tau_r)[0].astype(int)


def evaluate_rule(escalate_idx, is_high_risk, n):
    """Cost (how many labels spent) and safety (how many dangerous missed)."""
    escalated = np.zeros(n, dtype=bool)
    escalated[escalate_idx] = True
    auto = ~escalated
    return {
        "n_escalated": int(escalated.sum()),
        "unsafe_auto_accepts": int((auto & is_high_risk).sum()),
        "high_risk_total": int(is_high_risk.sum()),
        "high_risk_caught": int((escalated & is_high_risk).sum()),
    }


# ---------------------------------------------------------------------------
def analyse_experiment(exp_dir, exp_id):
    """Replay every round of one experiment under each rule."""
    res_csv = os.path.join(exp_dir, "results.csv")
    pool_dir = os.path.join(exp_dir, "pool_predictions")
    if not (os.path.isfile(res_csv) and os.path.isdir(pool_dir)):
        return [], []

    rounds = pd.read_csv(res_csv).set_index("round")
    ablation_rows, sweep_rows = [], []

    for rnd in sorted(rounds.index):
        pool_csv = os.path.join(pool_dir, f"round_{rnd}_pool_predictions.csv")
        if not os.path.isfile(pool_csv):
            continue
        pool = pd.read_csv(pool_csv)
        unc = pool["uncertainty_score"].to_numpy()
        risk = pool["risk_score"].to_numpy()
        is_hr = pool["true_label"].isin(HIGH_RISK_CLASSES).to_numpy()
        n = len(pool)
        if n == 0:
            continue

        tau_u = float(rounds.loc[rnd, "uncertainty_threshold_used"])
        tau_r = float(rounds.loc[rnd, "risk_threshold_used"])
        k = int(rounds.loc[rnd, "query_budget_used"])

        unc_idx = _uncertainty_route(unc, tau_u, k)
        risk_idx = _risk_route(risk, tau_r)
        dual_idx = np.union1d(unc_idx, risk_idx).astype(int)
        rand_idx = RNG.choice(n, size=min(len(dual_idx), n), replace=False)

        for rule, idx in [("uncertainty_only", unc_idx),
                          ("risk_only", risk_idx),
                          ("dual_metric", dual_idx),
                          ("random_matched", rand_idx)]:
            row = {"experiment_id": exp_id, "round": rnd, "rule": rule,
                   "pool_size": n}
            row.update(evaluate_rule(idx, is_hr, n))
            ablation_rows.append(row)

        # Risk-threshold sensitivity: uncertainty route fixed, risk route swept.
        for t in RISK_SWEEP:
            idx = np.union1d(unc_idx, _risk_route(risk, t)).astype(int)
            row = {"experiment_id": exp_id, "round": rnd, "risk_threshold": t,
                   "pool_size": n}
            row.update(evaluate_rule(idx, is_hr, n))
            sweep_rows.append(row)

    return ablation_rows, sweep_rows


# ---------------------------------------------------------------------------
def fig_ablation(totals):
    order = ["random_matched", "uncertainty_only", "risk_only", "dual_metric"]
    pretty = {"random_matched": "Random\n(cost-matched)",
              "uncertainty_only": "Uncertainty\nonly",
              "risk_only": "Risk\nonly",
              "dual_metric": "Dual-metric\n(ours)"}
    colors = {"random_matched": "#9ca3af", "uncertainty_only": COLOR_UNC,
              "risk_only": COLOR_ACCENT, "dual_metric": COLOR_DUAL}

    g = totals.groupby("rule")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    panels = [
        (axes[0], "high_risk_catch_rate", 100.0,
         "Safety: dangerous cases sent for review\n(higher is better)",
         "% of high-risk pool images escalated", "{:.1f}%"),
        (axes[1], "unsafe_auto_accepts", 1.0,
         "Dangerous cases auto-accepted\n(lower is better)",
         "Unsafe auto-accepts, summed over 15 rounds", "{:,.0f}"),
        (axes[2], "n_escalated", 1.0,
         "Cost: oracle labels requested\n(lower is cheaper)",
         "Images escalated, summed over 15 rounds", "{:,.0f}"),
    ]
    for ax, col, scale, title, ylab, fmt in panels:
        mean = g[col].mean() * scale
        sd = g[col].std() * scale
        vals = [mean[r] for r in order]
        errs = [sd[r] for r in order]
        bars = ax.bar([pretty[r] for r in order], vals,
                      yerr=errs, capsize=4,
                      color=[colors[r] for r in order])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    fmt.format(v), ha="center", va="bottom", fontsize=10,
                    fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylab, fontsize=9)
        ax.grid(alpha=0.25, axis="y")

    fig.suptitle("Ablation: are BOTH signals needed?  "
                 "(mean ± s.d. over 24 experiments, decision-level replay)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "14_ablation_both_signals_needed.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_sweep(sweep_totals):
    g = sweep_totals.groupby("risk_threshold")
    thr = sorted(sweep_totals["risk_threshold"].unique())
    unsafe = [g["unsafe_auto_accepts"].mean()[t] for t in thr]
    cost = [g["n_escalated"].mean()[t] for t in thr]
    x = [t if t <= 1 else 1.05 for t in thr]

    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.plot(x, unsafe, "o-", color=COLOR_DUAL, lw=2.5, ms=7,
             label="Unsafe auto-accepts (safety)")
    ax1.set_xlabel("Risk threshold  (rightmost point = risk route disabled "
                   "→ pure uncertainty baseline)")
    ax1.set_ylabel("Unsafe auto-accepts (lower = safer)", color=COLOR_DUAL)
    ax1.tick_params(axis="y", labelcolor=COLOR_DUAL)
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, cost, "s--", color=COLOR_ACCENT, lw=2.5, ms=6,
             label="Oracle labels requested (cost)")
    ax2.set_ylabel("Images escalated (higher = more expensive)",
                   color=COLOR_ACCENT)
    ax2.tick_params(axis="y", labelcolor=COLOR_ACCENT)

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="center left",
               fontsize=10)
    ax1.set_title("Risk-threshold sensitivity: the safety/cost dial\n"
                  "(mean over 24 experiments, all 15 rounds)",
                  fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "15_risk_threshold_sweep.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_pareto(sweep_totals, totals):
    g = sweep_totals.groupby("risk_threshold")
    thr = sorted(sweep_totals["risk_threshold"].unique())
    unsafe = np.array([g["unsafe_auto_accepts"].mean()[t] for t in thr])
    cost = np.array([g["n_escalated"].mean()[t] for t in thr])

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(cost, unsafe, "-", color="#94a3b8", lw=1.8, zorder=1)
    sc = ax.scatter(cost, unsafe, c=[min(t, 1.0) for t in thr], cmap="viridis",
                    s=110, zorder=3, edgecolor="white", linewidth=1.2)
    plt.colorbar(sc, ax=ax, label="Risk threshold")

    for rule, color, marker, label in [
        ("uncertainty_only", COLOR_UNC, "s", "Uncertainty-only (baseline)"),
        ("risk_only", COLOR_ACCENT, "^", "Risk-only"),
        ("dual_metric", COLOR_DUAL, "*", "Dual-metric as actually run"),
        ("random_matched", "#9ca3af", "X", "Random (cost-matched)"),
    ]:
        sub = totals[totals["rule"] == rule]
        ax.scatter(sub["n_escalated"].mean(), sub["unsafe_auto_accepts"].mean(),
                   color=color, marker=marker,
                   s=340 if rule == "dual_metric" else 160,
                   zorder=4, edgecolor="black", linewidth=1.1, label=label)

    ax.set_xlabel("Cost — oracle labels requested (summed over 15 rounds)")
    ax.set_ylabel("Unsafe auto-accepts (summed over 15 rounds)")
    ax.set_title("Safety vs cost frontier\n"
                 "Down-and-left is better. The curve is the risk threshold "
                 "swept from strict to permissive.",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "16_safety_cost_pareto.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ensure_dirs()
    all_ab, all_sw = [], []

    for name in sorted(os.listdir(EXPERIMENTS_DIR)):
        exp_dir = os.path.join(EXPERIMENTS_DIR, name)
        if not os.path.isdir(exp_dir):
            continue
        model, method, policy = parse_experiment_id(name)
        if model is None:
            continue
        ab, sw = analyse_experiment(exp_dir, name)
        all_ab.extend(ab)
        all_sw.extend(sw)
        print(f"  replayed {name}: {len(ab)//4} rounds")

    ab_df = pd.DataFrame(all_ab)
    sw_df = pd.DataFrame(all_sw)

    # Per-experiment totals across all 15 rounds.
    totals = (ab_df.groupby(["experiment_id", "rule"])
              [["n_escalated", "unsafe_auto_accepts", "high_risk_caught",
                "high_risk_total"]].sum().reset_index())
    totals["high_risk_catch_rate"] = (totals["high_risk_caught"]
                                      / totals["high_risk_total"])
    sweep_totals = (sw_df.groupby(["experiment_id", "risk_threshold"])
                    [["n_escalated", "unsafe_auto_accepts", "high_risk_caught",
                      "high_risk_total"]].sum().reset_index())
    sweep_totals["high_risk_catch_rate"] = (sweep_totals["high_risk_caught"]
                                            / sweep_totals["high_risk_total"])

    totals.to_csv(os.path.join(TABLE_DIR, "ablation_decision_level.csv"),
                  index=False)
    sweep_totals.to_csv(os.path.join(TABLE_DIR, "risk_threshold_sweep.csv"),
                        index=False)

    fig_ablation(totals)
    fig_sweep(sweep_totals)
    fig_pareto(sweep_totals, totals)

    print("\n=== Decision-level ablation (mean over 24 experiments, 15 rounds) ===")
    summary = totals.groupby("rule").agg(
        unsafe=("unsafe_auto_accepts", "mean"),
        escalated=("n_escalated", "mean"),
        catch_rate=("high_risk_catch_rate", "mean"),
    ).reindex(["random_matched", "uncertainty_only", "risk_only", "dual_metric"])
    for rule, r in summary.iterrows():
        print(f"  {rule:<18} unsafe={r['unsafe']:8,.0f}   "
              f"labels={r['escalated']:8,.0f}   "
              f"high-risk caught={100*r['catch_rate']:5.1f}%")

    base = summary.loc["uncertainty_only", "unsafe_auto_accepts"] \
        if "unsafe_auto_accepts" in summary else summary.loc["uncertainty_only", "unsafe"]
    dual = summary.loc["dual_metric", "unsafe"]
    print(f"\n  dual vs uncertainty-only: {100*(base-dual)/base:.1f}% fewer unsafe auto-accepts")

    print(f"\nFigures -> {FIG_DIR}\nTables  -> {TABLE_DIR}")


if __name__ == "__main__":
    main()
