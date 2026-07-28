"""
Explainability: Grad-CAM for BOTH heads.

WHY THIS MATTERS HERE SPECIFICALLY
----------------------------------
Dermoscopy datasets are notorious for shortcuts — rulers, ink marks, hair,
vignetting and colour-calibration stickers correlate with diagnosis because
of how the images were collected, not because of the pathology. A model can
score well while attending to a ruler. Grad-CAM is the standard check that
the evidence lies on the lesion.

The two-head architecture makes this more interesting than usual: the risk
head has its own parameters, so it can be interrogated separately. Asking
"what makes this image look DANGEROUS?" is a different question from "what
makes this image look like melanoma?", and this module renders both, side by
side, from the same forward pass. If the risk head attends to the lesion —
and especially if it attends to the lesion on cases the classification head
gets wrong — that is direct visual evidence for the design decision to give
it separate parameters.

Cases are chosen to be informative rather than flattering: correctly caught
melanomas, MISSED melanomas (the failure the paper is about), and cases
where the two heads disagree.

Usage
-----
    python -m evaluation.rigor.gradcam
    python -m evaluation.rigor.gradcam --experiment densenet169_entropy_dual_metric

Outputs
-------
  figures/  28_gradcam_panel_<experiment>.png
  tables/   gradcam_cases_<experiment>.csv
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from data.transforms import get_eval_transforms  # noqa: E402
from models.model_factory import create_model     # noqa: E402
from evaluation.rigor.paths import (              # noqa: E402
    CHECKPOINTS_DIR, IMAGE_DIRS, PRED_DIR, FIG_DIR, TABLE_DIR,
    CLASS_NAMES, HIGH_RISK_CLASSES, FINAL_ROUND, ensure_dirs,
    parse_experiment_id,
)

DEFAULT_EXPERIMENT = "resnet50_entropy_dual_metric"

# Last spatial (pre-pooling) block of each backbone — the layer Grad-CAM
# needs. See models/{resnet,densenet,efficientnet}.py for how each
# `backbone` Sequential is assembled.
TARGET_LAYER = {
    "resnet50": lambda m: m.backbone[7],        # layer4
    "densenet169": lambda m: m.backbone[0],     # features
    "efficientnet_b4": lambda m: m.backbone[0],  # features
}


class GradCAM:
    """Minimal Grad-CAM: one forward hook for activations, one for gradients."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out
        out.register_hook(self._save_grad)

    def _save_grad(self, grad):
        self.gradients = grad

    def __call__(self, x, score_fn):
        """
        score_fn maps (class_logits, risk_logits) -> a scalar to differentiate.
        Returns a (H, W) heatmap normalised to [0, 1].
        """
        self.model.zero_grad(set_to_none=True)
        class_logits, risk_logits = self.model(x)
        score = score_fn(class_logits, risk_logits)
        score.backward(retain_graph=True)

        acts = self.activations.detach()          # (1, C, h, w)
        grads = self.gradients.detach()           # (1, C, h, w)
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear",
                            align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam


def find_image(image_id):
    for d in IMAGE_DIRS:
        p = os.path.join(d, f"{image_id}.jpg")
        if os.path.exists(p):
            return p
    return None


def select_cases(pred_df, n_each=2):
    """Pick informative, not flattering, cases."""
    mel = pred_df[pred_df.true_label == "mel"]
    caught = mel[mel.predicted_label == "mel"].nlargest(n_each, "risk_score")
    missed = mel[~mel.predicted_label.isin(HIGH_RISK_CLASSES)]
    missed_high_risk = missed.nlargest(n_each, "risk_score")   # heads disagree
    missed_low_risk = missed.nsmallest(1, "risk_score")        # both wrong
    benign = pred_df[(pred_df.true_label == "nv") &
                     (pred_df.predicted_label == "nv")].nsmallest(1, "risk_score")

    cases = []
    for df, tag in [(caught, "CAUGHT\nboth heads agree"),
                    (missed_high_risk, "CLASSIFIER MISSED IT\nrisk head flagged it"),
                    (missed_low_risk, "BOTH HEADS\nMISSED IT"),
                    (benign, "BENIGN\ncorrectly low risk")]:
        for _, r in df.iterrows():
            cases.append({**r.to_dict(), "case_type": tag})
    return pd.DataFrame(cases)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    ap.add_argument("--n-each", type=int, default=2)
    args = ap.parse_args()

    ensure_dirs()
    exp_id = args.experiment
    model_name, method, policy = parse_experiment_id(exp_id)
    if model_name is None:
        print(f"Cannot parse experiment id: {exp_id}")
        return

    pred_path = os.path.join(PRED_DIR, f"{exp_id}_test_predictions.csv")
    if not os.path.isfile(pred_path):
        print(f"No prediction dump for {exp_id}. Run dump_test_predictions first.")
        return
    pred_df = pd.read_csv(pred_path)

    ckpt = os.path.join(CHECKPOINTS_DIR, exp_id, f"round_{FINAL_ROUND}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(model_name, num_classes=len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(torch.load(os.path.join(ckpt, "model.pt"),
                                     map_location=device))
    model.device = device
    model.to(device).eval()

    cam = GradCAM(model, TARGET_LAYER[model_name](model))
    tf = get_eval_transforms(224)

    cases = select_cases(pred_df, args.n_each)
    cases.to_csv(os.path.join(TABLE_DIR, f"gradcam_cases_{exp_id}.csv"), index=False)
    print(f"Selected {len(cases)} cases from {exp_id}")

    mel_idx = CLASS_NAMES.index("mel")
    n = len(cases)
    fig, axes = plt.subplots(n, 3, figsize=(11, 3.5 * n))
    if n == 1:
        axes = axes[None, :]

    for i, (_, row) in enumerate(cases.iterrows()):
        path = find_image(row["image_id"])
        if path is None:
            continue
        pil = Image.open(path).convert("RGB").resize((224, 224))
        x = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        x.requires_grad_(True)

        pred_i = CLASS_NAMES.index(row["predicted_label"])
        cam_cls = cam(x, lambda cl, rl: cl[0, pred_i])
        cam_risk = cam(x, lambda cl, rl: rl[0, 1])   # malignant logit

        axes[i, 0].imshow(pil)
        axes[i, 0].set_title(
            f"{row['image_id']}\ntrue={row['true_label']}  pred={row['predicted_label']}  "
            f"risk={row['risk_score']:.3f}", fontsize=8)
        axes[i, 1].imshow(pil)
        axes[i, 1].imshow(cam_cls, cmap="jet", alpha=0.45)
        axes[i, 1].set_title(f"Classification head\n(why '{row['predicted_label']}'?)",
                             fontsize=9)
        axes[i, 2].imshow(pil)
        axes[i, 2].imshow(cam_risk, cmap="jet", alpha=0.45)
        axes[i, 2].set_title("Risk head\n(why dangerous?)", fontsize=9)

        for j in range(3):
            axes[i, j].axis("off")
        axes[i, 0].text(-0.10, 0.5, row["case_type"], transform=axes[i, 0].transAxes,
                        rotation=90, va="center", ha="center", fontsize=7.5,
                        fontweight="bold", linespacing=1.6)

    fig.suptitle(f"Grad-CAM — {exp_id}\n"
                 "Left: image. Middle: what drove the diagnosis. "
                 "Right: what drove the risk score.\n"
                 "The two heads have separate parameters, so these can differ — "
                 "and on missed melanomas, that difference is the safety margin.",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0.02, 0, 1, 0.97])
    out = os.path.join(FIG_DIR, f"28_gradcam_panel_{exp_id}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure -> {out}")
    print("\nCases rendered:")
    for _, r in cases.iterrows():
        print(f"  {r['image_id']}  true={r['true_label']:<6} pred={r['predicted_label']:<6} "
              f"risk={r['risk_score']:.4f}  [{r['case_type']}]")


if __name__ == "__main__":
    main()
