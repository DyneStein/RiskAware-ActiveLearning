"""
EfficientNet-B4 model for skin lesion classification.
Pretrained on ImageNet, fine-tuned with MC Dropout head.
"""

import torchvision.models as models
import torch.nn as nn
from .base_model import BaseModel


class EfficientNetB4(BaseModel):
    """
    EfficientNet-B4 (Google, 2019) — 19M parameters.
    Best accuracy-per-parameter ratio of the three models.
    """

    def __init__(self, num_classes=7, dropout_rate=0.3, pretrained=True):
        super().__init__(num_classes=num_classes, dropout_rate=dropout_rate)

        # Load pretrained EfficientNet-B4
        weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
        full_model = models.efficientnet_b4(weights=weights)

        # Extract backbone (everything except the final classifier)
        self.backbone = nn.Sequential(
            full_model.features,
            full_model.avgpool,
            nn.Flatten(),
        )

        # Get the number of features from the backbone
        in_features = full_model.classifier[1].in_features

        # Build our custom head with Dropout for MC Dropout
        self._build_head(in_features)

        print(f"EfficientNet-B4 initialized (pretrained={pretrained}, "
              f"classes={num_classes}, dropout={dropout_rate})")
