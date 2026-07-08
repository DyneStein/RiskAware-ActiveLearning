"""
DenseNet-169 model for skin lesion classification.
Pretrained on ImageNet, fine-tuned with MC Dropout head.
"""

import torchvision.models as models
import torch.nn as nn
from .base_model import BaseModel


class DenseNet169(BaseModel):
    """
    DenseNet-169 (Facebook/Cornell, 2017) — 14M parameters.
    Dense connections reuse features aggressively, good for medical imaging.
    Links to the foundational 2019 risk-aware classifier paper.
    """

    def __init__(self, num_classes=7, dropout_rate=0.3, pretrained=True):
        super().__init__(num_classes=num_classes, dropout_rate=dropout_rate)

        # Load pretrained DenseNet-169
        weights = models.DenseNet169_Weights.DEFAULT if pretrained else None
        full_model = models.densenet169(weights=weights)

        # Extract backbone (features + adaptive pool)
        self.backbone = nn.Sequential(
            full_model.features,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        # Get the number of features
        in_features = full_model.classifier.in_features  # 1664 for DenseNet-169

        # Build our custom head with Dropout for MC Dropout
        self._build_head(in_features)

        print(f"DenseNet-169 initialized (pretrained={pretrained}, "
              f"classes={num_classes}, dropout={dropout_rate})")
