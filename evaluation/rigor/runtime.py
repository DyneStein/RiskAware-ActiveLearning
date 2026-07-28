"""
Runtime analysis: training time, inference time, query time.

THE MEASUREMENT PROBLEM, AND A DEAD END WORTH RECORDING
-------------------------------------------------------
The experiments logged one number per round, `round_time_seconds`, which
bundles three costs:

    round_time = training (10 epochs over the labelled set)
               + querying (one scoring pass over the unlabelled pool, then
                 the escalation rule)
               + fixed overhead (test-set evaluation, checkpoint, plots)

The obvious move is to regress round_time on |labelled| and |pool| across
the 360 logged rounds and read off the coefficients. THAT DOES NOT WORK
HERE, and the reason is worth stating rather than hiding: every round,

    |labelled| + |pool| = 8110  (exactly, all 24 experiments)

because the pool is a closed set — every image the oracle labels leaves the
pool and joins the labelled set. The two predictors are therefore perfectly
collinear with each other and with the intercept, the design matrix is
rank-deficient, and the fit returns coefficients that look plausible but
are arbitrary (it produced *negative* query times, which is how the problem
announced itself). Only the combined slope is identifiable from these logs,
never the split.

WHAT IS DONE INSTEAD
--------------------
  1. DIRECT MICROBENCHMARK. Each component is timed in isolation on real
     model objects at the real input size: one training step (forward +
     backward + optimiser), one inference pass (forward only), a 30-pass
     MC-dropout inference, and the escalation rule itself on a
     realistically sized pool array. These are honest per-image costs.

  2. COMPOSITION. Those per-image costs are multiplied by the actual
     per-round set sizes to give the train / query / overhead split.

  3. LOGGED WALL-CLOCK. Total GPU-hours per experiment, straight from the
     logs, with no modelling at all — including the MC-dropout overhead
     factor, which is measured, not inferred.

The benchmark runs on whatever hardware invokes it (CPU on the laptop, T4
on Colab); the logged wall-clock is from the Colab T4 the experiments
actually ran on. Both are reported with the device named, and the ratio
between them is given as a clearly-labelled scaling estimate rather than
being silently mixed.

Usage
-----
    python -m evaluation.rigor.runtime               # report from logs + cached benchmark
    python -m evaluation.rigor.runtime --benchmark   # (re)run the microbenchmark first

Outputs
-------
  figures/  26_runtime_breakdown.png
            27_runtime_scaling.png
  tables/   runtime_per_experiment.csv
            runtime_components_measured.csv
            runtime_round_composition.csv
"""

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from evaluation.rigor.paths import (  # noqa: E402
    EXPERIMENTS_DIR, PRED_DIR, RIGOR_DIR, FIG_DIR, TABLE_DIR, MODELS, METHODS,
    COLOR_UNC, COLOR_DUAL, COLOR_ACCENT, ensure_dirs, parse_experiment_id,
)

EPOCHS_PER_ROUND = 10
TEST_SET_SIZE = 1905
POOL_TOTAL = 8110
MC_PASSES = 30
BENCH_PATH = os.path.join(RIGOR_DIR, "runtime_benchmark.json")


# ---------------------------------------------------------------------------
# 1. Direct microbenchmark
# ---------------------------------------------------------------------------
def benchmark_components(n_batches=6, batch_size=32, image_size=224, threads=0):
    from models.model_factory import create_model
    from escalation import dual_metric, uncertainty_only

    # Thread count materially changes CPU timings, so it is set explicitly
    # and recorded — otherwise these numbers cannot be compared with the
    # inference timings captured during the prediction dump.
    if threads:
        torch.set_num_threads(threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {"device": device.type, "batch_size": batch_size,
           "n_batches": n_batches, "torch_threads": torch.get_num_threads(),
           "models": {}}

    for name in MODELS:
        model = create_model(name, num_classes=7, pretrained=False).to(device)
        x = torch.randn(batch_size, 3, image_size, image_size, device=device)
        y = torch.randint(0, 7, (batch_size,), device=device)
        y_risk = torch.randint(0, 2, (batch_size,), device=device)

        def sync():
            if device.type == "cuda":
                torch.cuda.synchronize()

        # --- inference: forward only, eval mode -----------------------------
        model.eval()
        with torch.no_grad():
            model(x)
            sync()
            t0 = time.perf_counter()
            for _ in range(n_batches):
                model(x)
            sync()
            infer_s = (time.perf_counter() - t0) / (n_batches * batch_size)

        # --- MC-dropout inference: 30 stochastic passes ---------------------
        model.enable_dropout()
        with torch.no_grad():
            sync()
            t0 = time.perf_counter()
            for _ in range(MC_PASSES):
                model(x)
            sync()
            mc_s = (time.perf_counter() - t0) / batch_size
        model.eval()

        # --- training step: forward + backward + optimiser ------------------
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        crit = nn.CrossEntropyLoss()
        for _ in range(2):  # warm-up
            opt.zero_grad()
            cl, rl = model(x)
            (crit(cl, y) + crit(rl, y_risk)).backward()
            opt.step()
        sync()
        t0 = time.perf_counter()
        for _ in range(n_batches):
            opt.zero_grad()
            cl, rl = model(x)
            (crit(cl, y) + crit(rl, y_risk)).backward()
            opt.step()
        sync()
        train_s = (time.perf_counter() - t0) / (n_batches * batch_size)

        out["models"][name] = {
            "ms_per_image_inference": 1000 * infer_s,
            "ms_per_image_mc_dropout": 1000 * mc_s,
            "ms_per_image_train_step": 1000 * train_s,
            "train_to_inference_ratio": train_s / infer_s if infer_s else None,
        }
        print(f"  {name:<16} infer {1000*infer_s:7.2f} ms/img | "
              f"mc-dropout {1000*mc_s:8.2f} ms/img | "
              f"train step {1000*train_s:7.2f} ms/img")
        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # --- the escalation rule itself, on a realistic pool -------------------
    rng = np.random.default_rng(0)
    n = 6000
    unc = rng.random(n)
    risk = rng.random(n)
    reps = 200
    t0 = time.perf_counter()
    for _ in range(reps):
        dual_metric.decide(unc, risk, 0.5, 0.9, 150)
    dual_ms = 1000 * (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for _ in range(reps):
        uncertainty_only.decide(unc, 0.5, 150)
    unc_ms = 1000 * (time.perf_counter() - t0) / reps
    out["policy"] = {"pool_size": n,
                     "dual_metric_ms_per_call": dual_ms,
                     "uncertainty_only_ms_per_call": unc_ms}
    print(f"  escalation rule on a {n}-image pool: "
          f"dual-metric {dual_ms:.2f} ms, uncertainty-only {unc_ms:.2f} ms per round")
    return out


# ---------------------------------------------------------------------------
def load_rounds():
    rows = []
    for name in sorted(os.listdir(EXPERIMENTS_DIR)):
        csv = os.path.join(EXPERIMENTS_DIR, name, "results.csv")
        if not os.path.isfile(csv):
            continue
        model, method, policy = parse_experiment_id(name)
        if model is None:
            continue
        df = pd.read_csv(csv)
        df["experiment_id"], df["model"] = name, model
        df["method"], df["policy"] = method, policy
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def compose_round_costs(rounds, bench):
    """Per-round train/query/eval split from measured per-image costs."""
    rows = []
    pol = bench.get("policy", {})
    for r in rounds.itertuples():
        b = bench["models"].get(r.model)
        if b is None:
            continue
        infer = b["ms_per_image_inference"] / 1000
        mc = b["ms_per_image_mc_dropout"] / 1000
        train = b["ms_per_image_train_step"] / 1000
        per_pool_image = mc if r.method == "mc_dropout" else infer
        policy_s = (pol.get("dual_metric_ms_per_call", 0)
                    if r.policy == "dual_metric"
                    else pol.get("uncertainty_only_ms_per_call", 0)) / 1000
        train_s = EPOCHS_PER_ROUND * r.labeled_count * train
        query_s = r.unlabeled_count * per_pool_image + policy_s
        eval_s = TEST_SET_SIZE * infer
        total = train_s + query_s + eval_s
        rows.append({
            "experiment_id": r.experiment_id, "model": r.model,
            "method": r.method, "policy": r.policy, "round": r.round,
            "labeled_count": r.labeled_count, "unlabeled_count": r.unlabeled_count,
            "train_seconds": train_s, "query_seconds": query_s,
            "test_eval_seconds": eval_s, "modelled_total_seconds": total,
            "logged_round_seconds": r.round_time_seconds,
            "train_share": train_s / total, "query_share": query_s / total,
            "eval_share": eval_s / total,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", action="store_true",
                    help="(Re)run the microbenchmark before reporting.")
    ap.add_argument("--threads", type=int, default=8,
                    help="torch CPU threads for the benchmark (0 = library default). "
                         "Match this to the value used by dump_test_predictions "
                         "so the two sets of timings are comparable.")
    args = ap.parse_args()

    ensure_dirs()
    rounds = load_rounds()
    if not len(rounds):
        print("No results found.")
        return
    print(f"Loaded {len(rounds)} round records from "
          f"{rounds['experiment_id'].nunique()} experiments.\n")

    if args.benchmark or not os.path.isfile(BENCH_PATH):
        print("=== Microbenchmark (isolating each component) ===")
        bench = benchmark_components(threads=args.threads)
        json.dump(bench, open(BENCH_PATH, "w"), indent=2)
        print()
    else:
        bench = json.load(open(BENCH_PATH))
        print(f"Using cached benchmark ({bench['device']}) from {BENCH_PATH}\n")

    # --- logged wall clock -------------------------------------------------
    per_exp = (rounds.groupby(["experiment_id", "model", "method", "policy"])
               .agg(total_seconds=("round_time_seconds", "sum"),
                    mean_round_seconds=("round_time_seconds", "mean"),
                    final_labeled=("labeled_count", "max"),
                    total_queries=("total_queries", "max")).reset_index())
    per_exp["total_hours"] = per_exp["total_seconds"] / 3600
    per_exp.to_csv(os.path.join(TABLE_DIR, "runtime_per_experiment.csv"), index=False)

    comp = compose_round_costs(rounds, bench)
    comp.to_csv(os.path.join(TABLE_DIR, "runtime_round_composition.csv"), index=False)

    mrows = []
    for name, b in bench["models"].items():
        mrows.append({"model": name, "device": bench["device"], **b})
    pd.DataFrame(mrows).to_csv(
        os.path.join(TABLE_DIR, "runtime_components_measured.csv"), index=False)

    # ------------------------------------------------------------------ plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    agg = (comp.groupby(["model", "method"])
           [["train_seconds", "query_seconds", "test_eval_seconds"]].mean())
    labels = [f"{m}\n{me}" for m, me in agg.index]
    x = np.arange(len(agg))
    tr = agg["train_seconds"].to_numpy()
    qu = agg["query_seconds"].to_numpy()
    ev = agg["test_eval_seconds"].to_numpy()
    ax.bar(x, tr, label="Training (10 epochs)", color=COLOR_DUAL)
    ax.bar(x, qu, bottom=tr, label="Querying (scoring the pool)", color=COLOR_ACCENT)
    ax.bar(x, ev, bottom=tr + qu, label="Test-set evaluation", color=COLOR_UNC)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5, rotation=45, ha="right")
    ax.set_ylabel(f"Modelled seconds per round ({bench['device'].upper()})")
    ax.set_title("Where the time goes, per round\n"
                 "measured per-image costs × actual set sizes",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    for model, marker in zip(MODELS, ["o", "s", "^"]):
        g = rounds[rounds.model == model].groupby("round")["round_time_seconds"].mean()
        ax.plot(g.index, g.values, marker=marker, lw=2, ms=5, label=model)
    ax.set_xlabel("Active-learning round")
    ax.set_ylabel("Mean logged round time (s)")
    ax.set_title("Logged round time grows with the labelled set\n(Colab T4, as run)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "26_runtime_breakdown.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    for method, color in zip(METHODS, ["#1b7a5e", "#b45309", "#6b7280", "#7c3aed"]):
        sub = rounds[rounds.method == method]
        ax.scatter(sub["labeled_count"], sub["round_time_seconds"], s=16,
                   alpha=0.55, color=color, label=method)
    ax.set_xlabel("Labelled-set size")
    ax.set_ylabel("Logged round time (s)")
    ax.set_title("Round time vs labelled-set size, by uncertainty method\n"
                 "MC-dropout sits far above the rest: 30 forward passes per pool image",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "27_runtime_scaling.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    # ----------------------------------------------------------------- report
    print("=== Logged wall-clock (Colab T4, exactly as run) ===")
    print(f"  All {len(per_exp)} experiments: {per_exp['total_hours'].sum():.1f} GPU-hours")
    print(f"  Mean per experiment: {per_exp['total_hours'].mean():.2f} h "
          f"(min {per_exp['total_hours'].min():.2f}, max {per_exp['total_hours'].max():.2f})")
    by_method = per_exp.groupby("method")["total_hours"].mean().sort_values()
    for m, v in by_method.items():
        print(f"    {m:<18} {v:.2f} h/experiment")
    mc_h = by_method.get("mc_dropout", np.nan)
    others = by_method.drop("mc_dropout", errors="ignore").mean()
    if others > 0:
        print(f"  MC-dropout overhead factor: {mc_h/others:.2f}x "
              f"(measured from wall-clock, not modelled)")

    dev = bench["device"].upper()
    print(f"\n=== Measured per-image component costs "
          f"({dev}, {bench.get('torch_threads', '?')} threads) ===")
    for name, b in bench["models"].items():
        print(f"  {name:<16} inference {b['ms_per_image_inference']:7.2f} ms | "
              f"MC-dropout {b['ms_per_image_mc_dropout']:8.2f} ms | "
              f"training step {b['ms_per_image_train_step']:7.2f} ms "
              f"({b['train_to_inference_ratio']:.1f}x inference)")
    pol = bench.get("policy", {})
    if pol:
        print(f"  Escalation rule itself, per round on a {pol['pool_size']}-image pool: "
              f"dual-metric {pol['dual_metric_ms_per_call']:.2f} ms, "
              f"uncertainty-only {pol['uncertainty_only_ms_per_call']:.2f} ms "
              f"(negligible — the cost of querying is the scoring pass, not the rule)")

    print(f"\n=== Modelled per-round split ({dev} costs x real set sizes) ===")
    share = comp.groupby("method")[["train_share", "query_share", "eval_share"]].mean()
    for m, r in share.iterrows():
        print(f"  {m:<18} training {100*r['train_share']:5.1f}% | "
              f"querying {100*r['query_share']:5.1f}% | "
              f"test eval {100*r['eval_share']:5.1f}%")

    ratio = (comp.groupby("model")
             .apply(lambda g: g["modelled_total_seconds"].sum()
                    / g["logged_round_seconds"].sum(), include_groups=False))
    print(f"\n  Cross-check — modelled {dev} total / logged T4 total:")
    for m, v in ratio.items():
        print(f"    {m:<16} {v:5.2f}x  "
              f"(>1 means this {dev} is slower than the T4 the runs used)")

    print(f"\nFigures -> {FIG_DIR}\nTables  -> {TABLE_DIR}")


if __name__ == "__main__":
    main()
