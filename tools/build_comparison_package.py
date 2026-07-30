"""
Assemble the self-contained comparison package.

WHY A SCRIPT AND NOT A HAND-COPIED FOLDER
-----------------------------------------
The package is a derived artefact. If it is assembled by hand it drifts:
someone regenerates a figure, forgets to re-copy it, and the folder handed
to a supervisor or uploaded as supplementary material no longer matches the
numbers in the repository. Building it with a script means it can always be
rebuilt from scratch, and the LaTeX table is generated from the same CSV
the figures came from rather than retyped.

WHAT GOES IN
------------
Only the head-to-head comparison evidence -- the material answering "is
this better than existing methods, measured fairly?". Not the full paper
package: calibration, Grad-CAM, robustness, runtime and the ablations live
in analysis/rigor and are referenced, not duplicated.

Usage
-----
    python -m tools.build_comparison_package
"""

import os
import shutil
import sys

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluation.rigor.paths import TABLE_DIR, FIG_DIR, MODELS  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, "paper", "COMPARISON")

# (source table, destination name) -- numbered so the reading order is obvious
TABLES = [
    ("baseline_comparison_main.csv",            "01_main_comparison.csv"),
    ("baseline_comparison_safety.csv",          "02_safety_scoreboard.csv"),
    ("baseline_comparison_mcnemar.csv",         "03_significance_image_level.csv"),
    ("baseline_comparison_runlevel.csv",        "04_direction_across_backbones.csv"),
    ("baseline_comparison_learning_curves.csv", "05_learning_curves_per_round.csv"),
]

FIGURES = [
    ("36_baseline_safety.png",          "fig1_safety_headline.png"),
    ("37_baseline_pareto.png",          "fig2_safety_accuracy_tradeoff.png"),
    ("38_baseline_learning_curves.png", "fig3_learning_curves.png"),
    ("39_baseline_mcnemar_forest.png",  "fig4_accuracy_significance.png"),
]

MODEL_LABEL = {
    "resnet50": "ResNet-50",
    "densenet169": "DenseNet-169",
    "efficientnet_b4": "EfficientNet-B4",
}

METHOD_LABEL_ORDER = [
    "Dual-metric (ours)", "Uncertainty-only",
    "CoreSet", "BADGE", "CLUE", "VAAL",
]


def copy_artifacts():
    tdir = os.path.join(OUT_DIR, "tables")
    fdir = os.path.join(OUT_DIR, "figures")
    os.makedirs(tdir, exist_ok=True)
    os.makedirs(fdir, exist_ok=True)

    missing = []
    for src, dst in TABLES:
        s = os.path.join(TABLE_DIR, src)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(tdir, dst))
        else:
            missing.append(src)
    for src, dst in FIGURES:
        s = os.path.join(FIG_DIR, src)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(fdir, dst))
        else:
            missing.append(src)
    return missing


def provenance_table():
    """
    One row per run: the hardware, library versions, seeds and git commit it
    actually executed under, read from each run's own environment.json.

    This is what turns "we used seed 42" into a checkable claim. Reviewers
    asking "what did this run on?" get an answer per run, not a general
    statement in a methods section.
    """
    from evaluation.rigor.paths import EXPERIMENTS_DIR
    import json

    rows = []
    for name in sorted(os.listdir(EXPERIMENTS_DIR)):
        p = os.path.join(EXPERIMENTS_DIR, name, "environment.json")
        if not os.path.exists(p):
            # The original 24 predate the provenance writer. Recorded as
            # blank rather than omitted, so the gap is visible.
            rows.append({"experiment_id": name, "environment_json": "absent"})
            continue
        e = json.load(open(p, encoding="utf-8"))
        rows.append({
            "experiment_id": name,
            "environment_json": "present",
            "captured_at_utc": e.get("captured_at_utc"),
            "seed": e.get("run", {}).get("seed"),
            "split_seed": e.get("run", {}).get("split_seed"),
            "strategy": e.get("run", {}).get("strategy"),
            "git_commit": e.get("git", {}).get("commit"),
            "gpu": e.get("hardware", {}).get("gpu_name"),
            "cuda": e.get("hardware", {}).get("cuda_version"),
            "python": (e.get("hardware", {}).get("python") or "").split()[0],
            "torch": e.get("packages", {}).get("torch"),
            "numpy": e.get("packages", {}).get("numpy"),
            "sklearn": e.get("packages", {}).get("sklearn"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "tables", "06_run_provenance.csv"),
              index=False)
    return df


def latex_main_table(main):
    """
    The paper's main comparison table, generated from the CSV.

    booktabs, siunitx-free (plain numbers so it compiles anywhere). Ours is
    bolded per column where it is best within its backbone block.
    """
    cols = [
        ("total_labels_queried", "Labels", 0, "higher_not_better"),
        ("accuracy", "Acc.", 4, "higher"),
        ("f1_macro", "F1", 4, "higher"),
        ("fn_rate_malignant", "FN-mal", 4, "lower"),
        ("melanoma_recall", "Mel-rec", 4, "higher"),
        ("unsafe_auto_accepts_cumulative", "Unsafe", 0, "lower"),
    ]
    lines = [
        r"% Generated by tools/build_comparison_package.py -- do not edit by hand.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Dual-metric escalation against four recent acquisition "
        r"baselines, all cost-matched to an identical per-round label budget. "
        r"\textbf{Unsafe} is the cumulative count of high-risk cases "
        r"auto-accepted without human review over 15 rounds (lower is better) "
        r"-- the safety-critical quantity. Best value per backbone in bold.}",
        r"\label{tab:baseline-comparison}",
        r"\begin{tabular}{ll" + "r" * len(cols) + "}",
        r"\toprule",
        "Backbone & Method & " + " & ".join(c[1] for c in cols) + r" \\",
        r"\midrule",
    ]
    for bi, model in enumerate(MODELS):
        sub = main[main.model == model]
        best = {}
        for key, _, _, direction in cols:
            if direction == "higher":
                best[key] = sub[key].max()
            elif direction == "lower":
                best[key] = sub[key].min()
        first = True
        for label in METHOD_LABEL_ORDER:
            r = sub[sub.method == label]
            if not len(r):
                continue
            r = r.iloc[0]
            cells = []
            for key, _, dp, direction in cols:
                v = r[key]
                txt = f"{int(v):,}" if dp == 0 else f"{v:.{dp}f}"
                if direction != "higher_not_better" and key in best and v == best[key]:
                    txt = r"\textbf{" + txt + "}"
                cells.append(txt)
            name = MODEL_LABEL.get(model, model) if first else ""
            first = False
            lines.append(f"{name} & {label} & " + " & ".join(cells) + r" \\")
        if bi < len(MODELS) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def latex_safety_table(safety):
    """Reduction in unsafe auto-accepts, ours vs each method."""
    lines = [
        r"% Generated by tools/build_comparison_package.py -- do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Reduction in unsafe auto-accepts achieved by dual-metric "
        r"escalation, relative to each comparison method, at an identical "
        r"label budget.}",
        r"\label{tab:safety-reduction}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Backbone & vs. & Ours & Other & Reduction \\",
        r"\midrule",
    ]
    for bi, model in enumerate(MODELS):
        sub = safety[safety.model == model]
        first = True
        for _, r in sub.iterrows():
            name = MODEL_LABEL.get(model, model) if first else ""
            first = False
            lines.append(
                f"{name} & {r['compared_to']} & "
                f"{int(r['ours_unsafe_cumulative']):,} & "
                f"{int(r['baseline_unsafe_cumulative']):,} & "
                f"{r['relative_reduction_pct']:.1f}\\% \\\\")
        if bi < len(MODELS) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def write_latex(main, safety):
    tdir = os.path.join(OUT_DIR, "tables")
    with open(os.path.join(tdir, "main_comparison.tex"), "w", encoding="utf-8") as f:
        f.write(latex_main_table(main))
    with open(os.path.join(tdir, "safety_reduction.tex"), "w", encoding="utf-8") as f:
        f.write(latex_safety_table(safety))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Building comparison package ->", OUT_DIR)

    missing = copy_artifacts()
    if missing:
        print("\nWARNING: these artefacts do not exist yet and were not copied:")
        for m in missing:
            print("   ", m)
        print("  regenerate with: python -m evaluation.rigor.baseline_comparison\n")

    main_df = pd.read_csv(os.path.join(TABLE_DIR, "baseline_comparison_main.csv"))
    safety_df = pd.read_csv(os.path.join(TABLE_DIR, "baseline_comparison_safety.csv"))
    write_latex(main_df, safety_df)

    prov = provenance_table()
    have = int((prov["environment_json"] == "present").sum())
    print(f"provenance: {have}/{len(prov)} runs carry an environment.json "
          f"(the original 24 predate the provenance writer)")

    n_files = sum(len(fs) for _, _, fs in os.walk(OUT_DIR))
    print(f"\n{n_files} files in the package:")
    for root, _, files in sorted(os.walk(OUT_DIR)):
        rel = os.path.relpath(root, OUT_DIR)
        for fn in sorted(files):
            p = os.path.join(root, fn)
            shown = fn if rel == "." else os.path.join(rel, fn)
            print(f"  {shown:52s} {os.path.getsize(p)/1024:8.1f} KB")
    print("\nDone.")


if __name__ == "__main__":
    main()
