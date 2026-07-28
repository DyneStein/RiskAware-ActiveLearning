"""
Feature extraction for the acquisition baselines.

WHY A SEPARATE MODULE
---------------------
CoreSet, BADGE and CLUE all select images by comparing them in some
representation space, but they do not agree on which space:

  * CoreSet and CLUE want the network's *representation* of the image —
    the shared backbone output that both heads see. Two images that the
    backbone maps near each other are, as far as the model is concerned,
    the same kind of lesion.
  * BADGE wants the *gradient* the image would produce if the model's own
    guess were the true label. That needs the activation entering the
    final linear layer, not the backbone output, because the gradient of
    the last layer's weights is (predicted probabilities − one-hot guess)
    outer-producted with exactly that activation.

Extracting both in one pass costs one forward pass over the pool instead
of two, which matters: the pool is ~7,600 images and this runs every round
of every baseline experiment.

WHAT COMES BACK
---------------
    backbone_features : (N, D)  D = 2048 resnet50 / 1664 densenet169 /
                                    1792 efficientnet_b4
    penultimate       : (N, 256) activation entering head[-1]
    probs             : (N, 7)   classification softmax
    image_ids         : (N,)

The penultimate width is 256 because the classification head is
Linear(D,512) → ReLU → Dropout → Linear(512,256) → ReLU → Dropout →
Linear(256,7); see models/base_model.py _build_head().
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def extract_features(model, dataset, batch_size=32, num_workers=2,
                     desc="Extracting features"):
    """
    One forward pass over `dataset`, capturing everything the baselines
    need.

    Runs under no_grad and in eval mode — dropout off, BatchNorm frozen —
    so the representation is the deterministic one the model would use at
    inference, not a stochastic training-time view.

    Parameters
    ----------
    model : BaseModel
        Already trained and moved to its device.
    dataset : HAM10000Dataset
        Yields (image, label, image_id).

    Returns
    -------
    dict with keys 'backbone_features', 'penultimate', 'probs', 'image_ids'
    """
    device = model.device
    model.eval()
    model.to(device)

    # The activation entering the final linear layer is not returned by
    # forward(), so it is captured with a hook on that layer's input.
    captured = {}

    def _hook(module, inputs, output):
        captured["penultimate"] = inputs[0].detach()

    handle = model.head[-1].register_forward_hook(_hook)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    all_backbone, all_penult, all_probs, all_ids = [], [], [], []
    try:
        for images, _labels, image_ids in tqdm(loader, desc=desc):
            images = images.to(device)

            features = model.backbone(images)
            if features.dim() > 2:
                features = features.view(features.size(0), -1)

            # Calling the head (rather than model.forward) is what fires
            # the hook; the risk head is irrelevant to every baseline, so
            # it is deliberately not evaluated here.
            class_logits = model.head(features)
            probs = torch.softmax(class_logits, dim=1)

            all_backbone.append(features.cpu().numpy())
            all_penult.append(captured["penultimate"].cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_ids.extend(list(image_ids))
    finally:
        handle.remove()

    return {
        "backbone_features": np.concatenate(all_backbone, axis=0),
        "penultimate": np.concatenate(all_penult, axis=0),
        "probs": np.concatenate(all_probs, axis=0),
        "image_ids": np.array(all_ids),
    }


def l2_normalise(x, eps=1e-12):
    """
    Scale every row to unit length.

    Distances between raw backbone activations are dominated by how
    strongly a given image happens to activate the network overall, which
    is not the similarity CoreSet and CLUE are trying to measure. Unit-
    normalising first makes Euclidean distance a monotone function of
    cosine similarity, so "close" means "similar-looking" rather than
    "similarly bright".
    """
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)
