"""
Calibration analysis: ECE, Brier score, reliability diagrams.

METHODOLOGICAL JUSTIFICATION
----------------------------
The whole safety argument rests on the model's scores meaning something.
If the risk head says 0.8, roughly 80% of those cases had better actually
be malignant — otherwise "escalate everything above threshold T" is
thresholding a number that doesn't correspond to real-world probability,
and the safety story is decorative. Accuracy cannot detect this: a model
can be 90% accurate and still wildly overconfident.

WHAT IS MEASURED
----------------
Two separate probability outputs are assessed, because the system uses
both and they can fail independently:

  1. the 7-class classification head's confidence (max softmax), and
  2. the risk head's P(malignant) — the one that actually drives escalation.

  ECE   Expected Calibration Error. Bin predictions by confidence, and in
        each bin compare average confidence to actual accuracy. ECE is the
        weighted mean gap. 0 = perfect, and ~0.05 means "on average the
        stated confidence is off by 5 percentage points". Reported with
        equal-width bins (standard) and equal-mass bins (adaptive ECE,
        which is more robust when confidences pile up near 1.0 — which
        they do here).
  MCE   The worst single bin's gap: the worst-case failure, not the average.
  Brier Mean squared error between the predicted probability vector and
        the one-hot truth. Unlike ECE it is a proper scoring rule — it
        penalises being wrong AND being badly calibrated, so it cannot be
        gamed by a model that just predicts the base rate.
  NLL   Negative log-likelihood; punishes confident mistakes hardest.

TEMPERATURE SCALING
-------------------
Also fits a single temperature on one half of the test set and reports ECE
on the other half. This does not change any decision the system already
made — it is a diagnostic showing how much of the miscalibration is the
cheap, fixable kind (a single scalar) versus something structural.

Outputs
-------
  figures/  17_reliability_classification.png
            18_reliability_risk_head.png
            19_calibration_ece_comparison.png
            20_calibration_brier_comparison.png
  tables/   calibration_metrics.csv
"""

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from evaluation.rigor.paths import (  # noqa: E402
    PRED_DIR, FIG_DIR, TABLE_DIR, CLASS_NAMES, HIGH_RISK_CLASSES,
    MODELS, METHODS, COLOR_UNC, COLOR_DUAL, ensure_dirs, parse_experiment_id,
)

N_BINS = 15


# ---------------------------------------------------------------------------
# Core calibration maths
# ---------------------------------------------------------------------------
def ece_equal_width(confidences, correct, n_bins=N_BINS):
    """Standard ECE: equal-width confidence bins."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(confidences)
    ece, mce = 0.0, 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if not mask.any():
            continue
        gap = abs(correct[mask].mean() - confidences[mask].mean())
        ece += (mask.sum() / n) * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def ece_equal_mass(confidences, correct, n_bins=N_BINS):
    """Adaptive ECE: equal-count bins. Robust when confidence saturates."""
    n = len(confidences)
    order = np.argsort(confidences)
    conf, corr = confidences[order], correct[order]
    ece = 0.0
    for chunk_conf, chunk_corr in zip(np.array_split(conf, n_bins),
                                      np.array_split(corr, n_bins)):
        if len(chunk_conf) == 0:
            continue
        ece += (len(chunk_conf) / n) * abs(chunk_corr.mean() - chunk_conf.mean())
    return float(ece)


def reliability_bins(confidences, correct, n_bins=N_BINS):
    """Per-bin (centre, accuracy, mean confidence, count) for the diagram."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        out.append({
            "center": (lo + hi) / 2,
            "accuracy": float(correct[mask].mean()) if mask.any() else np.nan,
            "confidence": float(confidences[mask].mean()) if mask.any() else np.nan,
            "count": int(mask.sum()),
        })
    return pd.DataFrame(out)


def multiclass_brier(probs, true_idx):
    """Mean squared error vs one-hot truth. Range [0, 2]."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(true_idx)), true_idx] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def binary_brier(p, y):
    return float(((p - y) ** 2).mean())


def nll(probs, true_idx, eps=1e-12):
    return float(-np.log(np.clip(probs[np.arange(len(true_idx)), true_idx],
                                 eps, 1.0)).mean())


def fit_temperature(probs, true_idx, eps=1e-12):
    """
    Fit one scalar T minimising NLL, by grid + local refine on log-probs.

    Working from stored probabilities rather than logits: log(p)/T then
    renormalised is equivalent to scaling the logits, since softmax is
    shift-invariant.
    """
    logp = np.log(np.clip(probs, eps, 1.0))

    def loss(T):
        z = logp / T
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        return -np.log(np.clip(p[np.arange(len(true_idx)), true_idx], eps, 1.0)).mean()

    grid = np.linspace(0.25, 6.0, 116)
    best = min(grid, key=loss)
    fine = np.linspace(max(0.05, best - 0.1), best + 0.1, 81)
    return float(min(fine, key=loss))


def apply_temperature(probs, T, eps=1e-12):
    z = np.log(np.clip(probs, eps, 1.0)) / T
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
def analyse_one(df):
    """All calibration metrics for one experiment's test-set dump."""
    prob_cols = [f"prob_{c}" for c in CLASS_NAMES]
    probs = df[prob_cols].to_numpy()
    true_idx = df["true_idx"].to_numpy()
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == true_idx).astype(float)

    risk = df["risk_score"].to_numpy()
    is_malignant = df["true_label"].isin(HIGH_RISK_CLASSES).to_numpy().astype(float)

    ece, mce = ece_equal_width(conf, correct)
    risk_ece, risk_mce = ece_equal_width(risk, is_malignant)

    # Temperature scaling: fit on half, evaluate on the other half.
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(df))
    a, b = idx[: len(idx) // 2], idx[len(idx) // 2:]
    T = fit_temperature(probs[a], true_idx[a])
    probs_b_scaled = apply_temperature(probs[b], T)
    ece_b_before, _ = ece_equal_width(probs[b].max(axis=1),
                                      (probs[b].argmax(1) == true_idx[b]).astype(float))
    ece_b_after, _ = ece_equal_width(probs_b_scaled.max(axis=1),
                                     (probs_b_scaled.argmax(1) == true_idx[b]).astype(float))

    return {
        "accuracy": float(correct.mean()),
        "mean_confidence": float(conf.mean()),
        "overconfidence_gap": float(conf.mean() - correct.mean()),
        "ece": ece,
        "ece_adaptive": ece_equal_mass(conf, correct),
        "mce": mce,
        "brier_multiclass": multiclass_brier(probs, true_idx),
        "nll": nll(probs, true_idx),
        "risk_ece": risk_ece,
        "risk_ece_adaptive": ece_equal_mass(risk, is_malignant),
        "risk_mce": risk_mce,
        "risk_brier": binary_brier(risk, is_malignant),
        "risk_mean_score": float(risk.mean()),
        "risk_base_rate": float(is_malignant.mean()),
        "temperature": T,
        "ece_heldout_before_T": ece_b_before,
        "ece_heldout_after_T": ece_b_after,
    }


# ---------------------------------------------------------------------------
def fig_reliability(dumps, which, fname, title, score_fn, label_fn, xlabel):
    """Reliability diagram grid: one panel per model, both policies overlaid."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, model in zip(axes, MODELS):
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6,
                label="Perfect calibration")
        for policy, color, lbl in [("uncertainty_only", COLOR_UNC, "Uncertainty-only"),
                                   ("dual_metric", COLOR_DUAL, "Dual-metric")]:
            curves = []
            for (m, meth, pol), df in dumps.items():
                if m != model or pol != policy:
                    continue
                bins = reliability_bins(score_fn(df), label_fn(df))
                curves.append(bins["accuracy"].values)
            if not curves:
                continue
            arr = np.vstack(curves)
            # Bins can be empty (confidences saturate near 1.0, so the low
            # bins hold nothing) -> all-NaN columns. That is expected, not an
            # error; suppress the warning rather than the NaN.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mean = np.nanmean(arr, axis=0)
                sd = np.nanstd(arr, axis=0)
            centers = reliability_bins(score_fn(list(dumps.values())[0]),
                                       label_fn(list(dumps.values())[0]))["center"].values
            ax.plot(centers, mean, "o-", color=color, lw=2.2, ms=5, label=lbl)
            ax.fill_between(centers, mean - sd, mean + sd, color=color, alpha=0.18)
        ax.set_title(model, fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Observed frequency")
    axes[0].legend(fontsize=9, loc="upper left")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_metric_comparison(cal, cols, titles, fname, suptitle):
    fig, axes = plt.subplots(1, len(cols), figsize=(6 * len(cols), 5.5))
    if len(cols) == 1:
        axes = [axes]
    pairs = sorted({(r["model"], r["method"]) for _, r in cal.iterrows()})
    labels = [f"{m}\n{me}" for m, me in pairs]
    x = np.arange(len(pairs))
    w = 0.38
    for ax, col, title in zip(axes, cols, titles):
        unc, dual = [], []
        for m, me in pairs:
            u = cal[(cal.model == m) & (cal.method == me) &
                    (cal.policy == "uncertainty_only")][col]
            d = cal[(cal.model == m) & (cal.method == me) &
                    (cal.policy == "dual_metric")][col]
            unc.append(u.iloc[0] if len(u) else np.nan)
            dual.append(d.iloc[0] if len(d) else np.nan)
        ax.bar(x - w / 2, unc, w, label="Uncertainty-only", color=COLOR_UNC)
        ax.bar(x + w / 2, dual, w, label="Dual-metric (ours)", color=COLOR_DUAL)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5, rotation=45, ha="right")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(alpha=0.25, axis="y")
    axes[0].legend(fontsize=9)
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def load_dumps():
    dumps = {}
    for f in sorted(os.listdir(PRED_DIR)):
        if not f.endswith("_test_predictions.csv"):
            continue
        exp_id = f.replace("_test_predictions.csv", "")
        model, method, policy = parse_experiment_id(exp_id)
        if model is None:
            continue
        dumps[(model, method, policy)] = pd.read_csv(os.path.join(PRED_DIR, f))
    return dumps


def main():
    ensure_dirs()
    dumps = load_dumps()
    print(f"Loaded {len(dumps)} test-set prediction dumps.")
    if not dumps:
        print("No dumps found — run evaluation.rigor.dump_test_predictions first.")
        return

    rows = []
    for (model, method, policy), df in sorted(dumps.items()):
        r = {"model": model, "method": method, "policy": policy}
        r.update(analyse_one(df))
        rows.append(r)
    cal = pd.DataFrame(rows)
    cal.to_csv(os.path.join(TABLE_DIR, "calibration_metrics.csv"), index=False)

    prob_cols = [f"prob_{c}" for c in CLASS_NAMES]
    fig_reliability(
        dumps, "classification", "17_reliability_classification.png",
        "Reliability diagram — 7-class classification confidence\n"
        "(mean ± s.d. across uncertainty methods; below the diagonal = overconfident)",
        score_fn=lambda d: d[prob_cols].to_numpy().max(axis=1),
        label_fn=lambda d: (d[prob_cols].to_numpy().argmax(axis=1)
                            == d["true_idx"].to_numpy()).astype(float),
        xlabel="Predicted confidence",
    )
    fig_reliability(
        dumps, "risk", "18_reliability_risk_head.png",
        "Reliability diagram — risk head P(malignant)\n"
        "This is the score that drives escalation: does 0.8 really mean 80% malignant?",
        score_fn=lambda d: d["risk_score"].to_numpy(),
        label_fn=lambda d: d["true_label"].isin(HIGH_RISK_CLASSES).to_numpy().astype(float),
        xlabel="Predicted P(malignant)",
    )
    fig_metric_comparison(
        cal, ["ece", "risk_ece"],
        ["Classification ECE (lower = better)", "Risk-head ECE (lower = better)"],
        "19_calibration_ece_comparison.png",
        "Expected Calibration Error by configuration",
    )
    fig_metric_comparison(
        cal, ["brier_multiclass", "risk_brier"],
        ["Multi-class Brier score (lower = better)",
         "Risk-head Brier score (lower = better)"],
        "20_calibration_brier_comparison.png",
        "Brier score by configuration (proper scoring rule)",
    )

    print("\n=== Calibration summary (mean over experiments) ===")
    for pol in ["uncertainty_only", "dual_metric"]:
        s = cal[cal.policy == pol]
        if not len(s):
            continue
        print(f"\n  {pol}  (n={len(s)})")
        print(f"    accuracy {s.accuracy.mean():.4f} | mean confidence "
              f"{s.mean_confidence.mean():.4f} | over-confidence gap "
              f"{s.overconfidence_gap.mean():+.4f}")
        print(f"    classification: ECE {s.ece.mean():.4f}  "
              f"adaptive-ECE {s.ece_adaptive.mean():.4f}  MCE {s.mce.mean():.4f}  "
              f"Brier {s.brier_multiclass.mean():.4f}  NLL {s.nll.mean():.4f}")
        print(f"    risk head:      ECE {s.risk_ece.mean():.4f}  "
              f"MCE {s.risk_mce.mean():.4f}  Brier {s.risk_brier.mean():.4f}  "
              f"(base rate malignant = {s.risk_base_rate.mean():.3f})")
        print(f"    temperature scaling: T={s.temperature.mean():.3f}, "
              f"held-out ECE {s.ece_heldout_before_T.mean():.4f} -> "
              f"{s.ece_heldout_after_T.mean():.4f}")

    print(f"\nFigures -> {FIG_DIR}\nTables  -> {TABLE_DIR}")


if __name__ == "__main__":
    main()
