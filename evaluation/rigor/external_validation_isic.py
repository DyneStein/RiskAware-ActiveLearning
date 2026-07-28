"""
External validation on ISIC 2019 / ISIC 2020.

READ THIS BEFORE RUNNING — THE CONTAMINATION TRAP
--------------------------------------------------
**ISIC 2019 CONTAINS HAM10000.** The ISIC 2019 training set is the union of
BCN20000, HAM10000 and MSK, and the HAM10000 images keep their original
ISIC_xxxxxxx identifiers. Every model here was trained on HAM10000. So
evaluating on ISIC 2019 as-downloaded is not external validation at all —
a large share of the "unseen" images were in the training pool, and the
resulting numbers would be inflated and indefensible.

This script therefore ALWAYS excludes, by image id, every image that
appears in HAM10000_metadata.csv, and reports how many it removed so the
exclusion is auditable rather than assumed. Overlap of zero after
filtering is asserted, not hoped for.

**ISIC 2020 is genuinely independent** — a different challenge year, ~33k
images, different patients and sites, no HAM10000 overlap. Its labels are
binary (benign / malignant), which maps exactly onto the risk head's
target. That makes ISIC 2020 the cleaner external test of the paper's
central claim, and it is the recommended primary choice; ISIC 2019 (after
filtering) additionally supports the 7-class comparison.

WHAT IS EVALUATED
-----------------
The risk head transfers directly (malignant vs benign is the same task on
both datasets), so its AUROC / PR-AUC / calibration are the headline
external numbers. The 7-class head transfers only to ISIC 2019, and only
over the label subset both datasets share — ISIC 2019's SCC class has no
HAM10000 equivalent, so SCC images are excluded from the 7-class
comparison but retained as positives for the binary malignant evaluation
(squamous cell carcinoma is malignant).

Label mapping (ISIC 2019 -> HAM10000):
    MEL->mel  NV->nv  BCC->bcc  AK->akiec  BKL->bkl  DF->df  VASC->vasc
    SCC-> (no equivalent; binary-malignant only)

EXPECTED OUTCOME, STATED IN ADVANCE
-----------------------------------
Performance will drop. Different scanners, sites and populations always
cost something, and a drop is the normal, reportable result — the question
a reviewer asks is how much, and whether the SAFETY signal degrades faster
than the diagnosis. Writing that expectation down before running is the
point: it stops a mediocre transfer number from being quietly reframed
afterwards.

HOW TO GET THE DATA (run on Colab — it is several GB)
-----------------------------------------------------
ISIC 2020 (recommended, ~3 GB resized):
    from a Kaggle/ISIC mirror, you need
      - the JPEG images directory
      - ISIC_2020_Training_GroundTruth.csv  (columns: image_name, target)

ISIC 2019 (~9 GB):
      - ISIC_2019_Training_Input/           (images)
      - ISIC_2019_Training_GroundTruth.csv  (one-hot columns MEL, NV, ...)

Usage
-----
    python -m evaluation.rigor.external_validation_isic \
        --dataset isic2020 \
        --images-dir /content/isic2020/train \
        --labels-csv /content/isic2020/ISIC_2020_Training_GroundTruth.csv

Outputs
-------
  tables/   external_validation_<dataset>.csv
            external_validation_<dataset>_summary.csv
  figures/  31_external_validation_<dataset>.png
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from data.dataset import HAM10000Dataset          # noqa: E402
from data.transforms import get_eval_transforms   # noqa: E402
from models.model_factory import create_model     # noqa: E402
from evaluation.rigor.paths import (              # noqa: E402
    RESEARCH_ROOT, CHECKPOINTS_DIR, FIG_DIR, TABLE_DIR, CLASS_NAMES,
    HIGH_RISK_CLASSES, MODELS, FINAL_ROUND, COLOR_UNC, COLOR_DUAL,
    ensure_dirs, parse_experiment_id,
)

HAM_METADATA = os.path.join(RESEARCH_ROOT, "archive", "HAM10000_metadata.csv")

ISIC2019_TO_HAM = {
    "MEL": "mel", "NV": "nv", "BCC": "bcc", "AK": "akiec",
    "BKL": "bkl", "DF": "df", "VASC": "vasc",
}
ISIC2019_MALIGNANT = {"MEL", "BCC", "AK", "SCC"}  # SCC has no HAM equivalent


def fast_auc(y_true, scores):
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    r = rankdata(scores)
    return float((r[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_ci(y, s, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    out = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if 0 < y[i].sum() < n:
            out.append(fast_auc(y[i], s[i]))
    if not out:
        return np.nan, np.nan
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# ---------------------------------------------------------------------------
def load_isic2020(labels_csv):
    df = pd.read_csv(labels_csv)
    col = "image_name" if "image_name" in df.columns else "image_id"
    out = pd.DataFrame({"image_id": df[col].astype(str)})
    out["is_malignant"] = df["target"].astype(int)
    # ISIC 2020 has no 7-class label, so `dx` is a placeholder that keeps the
    # existing Dataset class usable; it is never used as a target here.
    out["dx"] = np.where(out["is_malignant"] == 1, "mel", "nv")
    return out, "binary"


def load_isic2019(labels_csv):
    df = pd.read_csv(labels_csv)
    col = "image" if "image" in df.columns else df.columns[0]
    onehot = [c for c in df.columns if c.upper() in
              set(ISIC2019_TO_HAM) | {"SCC", "UNK"}]
    labels = df[onehot].to_numpy().argmax(axis=1)
    names = [onehot[i].upper() for i in labels]
    out = pd.DataFrame({"image_id": df[col].astype(str), "isic_label": names})
    out = out[out["isic_label"] != "UNK"].copy()
    out["is_malignant"] = out["isic_label"].isin(ISIC2019_MALIGNANT).astype(int)
    out["dx"] = out["isic_label"].map(ISIC2019_TO_HAM)
    return out, "multiclass"


def exclude_ham_overlap(df):
    """Remove every image that is also in HAM10000. Non-negotiable."""
    if not os.path.isfile(HAM_METADATA):
        raise FileNotFoundError(
            f"HAM10000 metadata not found at {HAM_METADATA}. The overlap "
            f"exclusion cannot be verified, so the run is aborted rather "
            f"than producing contaminated external-validation numbers."
        )
    ham_ids = set(pd.read_csv(HAM_METADATA)["image_id"].astype(str))
    before = len(df)
    overlap = df["image_id"].isin(ham_ids)
    df = df[~overlap].copy()
    n_removed = int(overlap.sum())
    print(f"  HAM10000 overlap check: {before} images -> removed {n_removed} "
          f"({100*n_removed/max(before,1):.1f}%) -> {len(df)} remain")
    assert not df["image_id"].isin(ham_ids).any(), "overlap exclusion failed"
    return df, n_removed


# ---------------------------------------------------------------------------
def run_model(exp_id, ckpt_dir, df, images_dir, device, batch_size=32):
    meta_model = parse_experiment_id(exp_id)[0]
    model = create_model(meta_model, num_classes=len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "model.pt"),
                                     map_location=device))
    model.device = device
    model.to(device).eval()

    dataset = HAM10000Dataset(df, [images_dir], transform=get_eval_transforms(224))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    probs, risks = [], []
    with torch.no_grad():
        for images, _, _ in loader:
            cl, rl = model(images.to(device))
            probs.append(torch.softmax(cl, 1).cpu().numpy())
            risks.append(torch.softmax(rl, 1)[:, 1].cpu().numpy())
    del model
    return np.concatenate(probs), np.concatenate(risks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["isic2019", "isic2020"], required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--labels-csv", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--max-images", type=int, default=0,
                    help="Subsample for a quick check (0 = use all).")
    args = ap.parse_args()

    ensure_dirs()
    print(f"=== External validation: {args.dataset} ===")
    df, mode = (load_isic2020(args.labels_csv) if args.dataset == "isic2020"
                else load_isic2019(args.labels_csv))
    print(f"  loaded {len(df)} labelled images")

    df, n_removed = exclude_ham_overlap(df)
    if args.max_images and len(df) > args.max_images:
        df = df.sample(args.max_images, random_state=42).reset_index(drop=True)
        print(f"  subsampled to {len(df)} images")

    prevalence = df["is_malignant"].mean()
    print(f"  malignant prevalence: {100*prevalence:.2f}%  "
          f"(HAM10000 test set, for comparison: 18.3%)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiments = []
    for name in sorted(os.listdir(CHECKPOINTS_DIR)):
        ckpt = os.path.join(CHECKPOINTS_DIR, name, f"round_{FINAL_ROUND}")
        if os.path.isfile(os.path.join(ckpt, "model.pt")):
            if args.only is None or name in args.only:
                experiments.append((name, ckpt))
    print(f"  evaluating {len(experiments)} checkpoints on {device.type}\n")

    y_mal = df["is_malignant"].to_numpy()
    rows = []
    for i, (exp_id, ckpt) in enumerate(experiments, 1):
        model, method, policy = parse_experiment_id(exp_id)
        probs, risks = run_model(exp_id, ckpt, df, args.images_dir, device)

        summed = probs[:, [CLASS_NAMES.index(c) for c in sorted(HIGH_RISK_CLASSES)]].sum(1)
        lo, hi = bootstrap_ci(y_mal, risks)
        row = {
            "experiment_id": exp_id, "model": model, "method": method,
            "policy": policy, "dataset": args.dataset, "n_images": len(df),
            "malignant_prevalence": prevalence,
            "risk_auroc": fast_auc(y_mal, risks),
            "risk_auroc_ci_low": lo, "risk_auroc_ci_high": hi,
            "risk_pr_auc": float(average_precision_score(y_mal, risks)),
            "summed_probs_auroc": fast_auc(y_mal, summed),
            "risk_brier": float(((risks - y_mal) ** 2).mean()),
            "mean_risk_score": float(risks.mean()),
        }
        if mode == "multiclass":
            valid = df["dx"].notna().to_numpy()
            pred = probs.argmax(1)
            true = np.array([CLASS_NAMES.index(d) if isinstance(d, str) else -1
                             for d in df["dx"]])
            row["accuracy_7class"] = float((pred[valid] == true[valid]).mean())
            is_mel = (true == CLASS_NAMES.index("mel")) & valid
            row["melanoma_recall"] = (
                float((pred[is_mel] == CLASS_NAMES.index("mel")).mean())
                if is_mel.sum() else np.nan)
        rows.append(row)
        print(f"[{i}/{len(experiments)}] {exp_id}: risk AUROC "
              f"{row['risk_auroc']:.4f} [{lo:.3f}, {hi:.3f}]")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(TABLE_DIR,
                            f"external_validation_{args.dataset}.csv"), index=False)
    summary = res.groupby("policy").agg(
        risk_auroc=("risk_auroc", "mean"), risk_auroc_sd=("risk_auroc", "std"),
        risk_pr_auc=("risk_pr_auc", "mean"), risk_brier=("risk_brier", "mean"),
        n=("experiment_id", "count")).reset_index()
    summary.to_csv(os.path.join(
        TABLE_DIR, f"external_validation_{args.dataset}_summary.csv"), index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(res))
    colors = [COLOR_DUAL if p == "dual_metric" else COLOR_UNC for p in res["policy"]]
    ax.bar(x, res["risk_auroc"], color=colors,
           yerr=[res["risk_auroc"] - res["risk_auroc_ci_low"],
                 res["risk_auroc_ci_high"] - res["risk_auroc"]], capsize=3)
    ax.axhline(0.5, color="red", ls="--", lw=1.2, label="Random (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(res["experiment_id"], rotation=75, ha="right", fontsize=6.5)
    ax.set_ylabel("Risk-head AUROC on external data (95% CI)")
    ax.set_title(f"External validation on {args.dataset.upper()} "
                 f"({len(df):,} images, {n_removed:,} HAM10000 overlaps excluded)\n"
                 f"Green = dual-metric, grey = uncertainty-only",
                 fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR,
                             f"31_external_validation_{args.dataset}.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\n=== Summary ===")
    print(summary.to_string(index=False))
    print(f"\nFigures -> {FIG_DIR}\nTables  -> {TABLE_DIR}")


if __name__ == "__main__":
    main()
