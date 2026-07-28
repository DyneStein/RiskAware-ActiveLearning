"""
Dump full test-set predictions for every finished experiment.

WHY THIS EXISTS
---------------
The 24 experiments logged *summary* metrics per round (accuracy, F1,
FN-rate). They never logged the raw probability vector behind each test
prediction — and calibration (ECE, Brier, reliability diagrams) and
per-class AUC are all impossible without those probabilities.

Every experiment saved its final model weights AND the exact test split it
used (results/checkpoints/<exp>/round_15/). So instead of re-training
anything, we reload each final model and run ONE forward pass over the test
set, saving the full 7-class probability vector, the risk head's
P(malignant), and the patient metadata for each image.

This is the same code path the AL loop used for its own test evaluation
(model.predict + model.predict_risk, eval transforms, no augmentation), so
the accuracy recomputed from this dump reproduces the logged round-15
accuracy. That equality is asserted downstream as a correctness check.

Runs on CPU in ~2 min/model (~50 min for all 24). Resumable: already-dumped
experiments are skipped unless --overwrite is passed.

Usage
-----
    python -m evaluation.rigor.dump_test_predictions              # all 24, clean
    python -m evaluation.rigor.dump_test_predictions --corruption gaussian_noise_0.05
    python -m evaluation.rigor.dump_test_predictions --only resnet50_entropy_dual_metric
"""

import argparse
import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader
from torchvision import transforms

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.dataset import HAM10000Dataset            # noqa: E402
from data.transforms import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from models.model_factory import create_model        # noqa: E402
from evaluation.rigor.paths import (                 # noqa: E402
    CHECKPOINTS_DIR, IMAGE_DIRS, PRED_DIR, RIGOR_DIR,
    CLASS_NAMES, FINAL_ROUND, ensure_dirs,
)

IMAGE_SIZE = 224
BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# Corruptions — for the robustness experiments (supervisor ask #6).
# Each one is a mild, clinically plausible degradation: sensor noise, an
# out-of-focus dermatoscope, bad lighting, or a re-compressed image.
# ---------------------------------------------------------------------------
class _JPEG:
    def __init__(self, quality):
        self.quality = quality

    def __call__(self, img):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class _GaussianNoise:
    """Applied on the tensor, after normalization-free ToTensor()."""

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, t):
        return torch.clamp(t + torch.randn_like(t) * self.sigma, 0.0, 1.0)


def _enhance(kind, factor):
    cls = {"brightness": ImageEnhance.Brightness,
           "contrast": ImageEnhance.Contrast}[kind]
    return lambda img: cls(img).enhance(factor)


# name -> (pil_op or None, tensor_op or None)
CORRUPTIONS = {
    "clean":                (None, None),
    "gaussian_noise_0.05":  (None, _GaussianNoise(0.05)),
    "gaussian_noise_0.10":  (None, _GaussianNoise(0.10)),
    "blur_1.5":             (lambda im: im.filter(ImageFilter.GaussianBlur(1.5)), None),
    "brightness_0.7":       (_enhance("brightness", 0.7), None),
    "brightness_1.3":       (_enhance("brightness", 1.3), None),
    "contrast_0.7":         (_enhance("contrast", 0.7), None),
    "jpeg_q30":             (_JPEG(30), None),
}


def build_transform(corruption="clean"):
    """Eval transform, optionally with a corruption injected."""
    pil_op, tensor_op = CORRUPTIONS[corruption]
    ops = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if pil_op is not None:
        ops.append(transforms.Lambda(pil_op))
    ops.append(transforms.ToTensor())
    if tensor_op is not None:
        ops.append(transforms.Lambda(tensor_op))
    ops.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return transforms.Compose(ops)


# ---------------------------------------------------------------------------
def list_experiments():
    """Every experiment with a completed final-round checkpoint."""
    out = []
    if not os.path.isdir(CHECKPOINTS_DIR):
        return out
    for name in sorted(os.listdir(CHECKPOINTS_DIR)):
        ckpt = os.path.join(CHECKPOINTS_DIR, name, f"round_{FINAL_ROUND}")
        if os.path.isfile(os.path.join(ckpt, "model.pt")):
            out.append((name, ckpt))
    return out


def load_model(model_name, weights_path, device):
    model = create_model(model_name, num_classes=len(CLASS_NAMES),
                         pretrained=False)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.device = device
    model.to(device)
    model.eval()
    return model


def run_inference(model, test_df, transform, device):
    """
    One deterministic pass over the test set.

    Returns (dataframe, timing dict). Timing separates pure model time from
    data-loading time so "inference time per image" is an honest number and
    not a measure of the laptop's disk.
    """
    dataset = HAM10000Dataset(test_df, IMAGE_DIRS, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0)

    ids, labels, cls_probs, risk_scores = [], [], [], []
    model_seconds = 0.0
    n_images = 0

    wall_start = time.perf_counter()
    with torch.no_grad():
        for images, batch_labels, batch_ids in loader:
            images = images.to(device)

            t0 = time.perf_counter()
            class_logits, risk_logits = model(images)
            probs = torch.softmax(class_logits, dim=1)
            risk = torch.softmax(risk_logits, dim=1)[:, 1]
            if device.type == "cuda":
                torch.cuda.synchronize()
            model_seconds += time.perf_counter() - t0

            ids.extend(list(batch_ids))
            labels.extend(batch_labels.numpy().tolist())
            cls_probs.append(probs.cpu().numpy())
            risk_scores.append(risk.cpu().numpy())
            n_images += images.size(0)
    wall_seconds = time.perf_counter() - wall_start

    cls_probs = np.concatenate(cls_probs, axis=0)
    risk_scores = np.concatenate(risk_scores, axis=0)

    df = pd.DataFrame({"image_id": ids})
    df["true_label"] = [CLASS_NAMES[i] for i in labels]
    df["true_idx"] = labels
    df["predicted_idx"] = cls_probs.argmax(axis=1)
    df["predicted_label"] = [CLASS_NAMES[i] for i in df["predicted_idx"]]
    df["confidence"] = cls_probs.max(axis=1)
    for i, c in enumerate(CLASS_NAMES):
        df[f"prob_{c}"] = cls_probs[:, i]
    df["risk_score"] = risk_scores

    # Carry patient metadata through for the fairness / subgroup breakdown.
    meta_cols = [c for c in ["age", "sex", "localization", "dx_type", "lesion_id"]
                 if c in test_df.columns]
    if meta_cols:
        df = df.merge(test_df[["image_id"] + meta_cols], on="image_id", how="left")

    timing = {
        "n_images": int(n_images),
        "model_seconds": model_seconds,
        "wall_seconds": wall_seconds,
        "ms_per_image_model_only": 1000.0 * model_seconds / max(n_images, 1),
        "ms_per_image_end_to_end": 1000.0 * wall_seconds / max(n_images, 1),
        "images_per_second_model_only": n_images / model_seconds if model_seconds else None,
        "batch_size": BATCH_SIZE,
        "device": device.type,
    }
    return df, timing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corruption", default="clean", choices=list(CORRUPTIONS))
    ap.add_argument("--only", nargs="*", default=None,
                    help="Limit to these experiment ids.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--threads", type=int, default=0,
                    help="torch CPU threads (0 = leave default).")
    args = ap.parse_args()

    ensure_dirs()
    if args.threads:
        torch.set_num_threads(args.threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suffix = "" if args.corruption == "clean" else f"__{args.corruption}"
    out_dir = PRED_DIR if args.corruption == "clean" else os.path.join(
        RIGOR_DIR, "predictions_robustness")
    os.makedirs(out_dir, exist_ok=True)

    experiments = list_experiments()
    if args.only:
        experiments = [(n, p) for n, p in experiments if n in args.only]

    print(f"device={device.type} threads={torch.get_num_threads()} "
          f"corruption={args.corruption}")
    print(f"experiments to dump: {len(experiments)}")

    timings = {}
    timing_path = os.path.join(out_dir, f"_inference_timing{suffix}.json")
    if os.path.isfile(timing_path):
        timings = json.load(open(timing_path))

    for i, (exp_id, ckpt_dir) in enumerate(experiments, 1):
        out_csv = os.path.join(out_dir, f"{exp_id}{suffix}_test_predictions.csv")
        if os.path.isfile(out_csv) and not args.overwrite:
            print(f"[{i}/{len(experiments)}] {exp_id}: already dumped, skipping")
            continue

        meta = json.load(open(os.path.join(ckpt_dir, "meta.json")))
        test_df = pd.read_csv(os.path.join(ckpt_dir, "pool_state", "test.csv"))

        t0 = time.perf_counter()
        model = load_model(meta["model_name"],
                           os.path.join(ckpt_dir, "model.pt"), device)
        df, timing = run_inference(model, test_df,
                                   build_transform(args.corruption), device)
        del model

        acc = float((df["true_idx"] == df["predicted_idx"]).mean())
        timing.update({"experiment_id": exp_id, "model": meta["model_name"],
                       "corruption": args.corruption, "accuracy": acc})
        timings[exp_id] = timing

        df.to_csv(out_csv, index=False)
        json.dump(timings, open(timing_path, "w"), indent=2)

        print(f"[{i}/{len(experiments)}] {exp_id}: n={timing['n_images']} "
              f"acc={acc:.4f} "
              f"{timing['ms_per_image_model_only']:.1f} ms/img "
              f"({time.perf_counter() - t0:.0f}s total)")

    print(f"\nDone. Predictions -> {out_dir}")


if __name__ == "__main__":
    main()
