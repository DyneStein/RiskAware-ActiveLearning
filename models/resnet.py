"""
ResNet-50 model for skin lesion classification.
Pretrained on ImageNet, fine-tuned with MC Dropout head.
"""

import torchvision.models as models
import torch.nn as nn
from .base_model import BaseModel


class ResNet50(BaseModel):
    """
    ResNet-50 (Microsoft, 2015) — 25M parameters.
    The standard workhorse baseline in computer vision.
    """

    def __init__(self, num_classes=7, dropout_rate=0.3, pretrained=True):
        super().__init__(num_classes=num_classes, dropout_rate=dropout_rate)

        # Load pretrained ResNet-50
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        full_model = models.resnet50(weights=weights)

        # Extract backbone (everything except the final fc layer)
        # We remove the last fc and replace with our head
        layers = list(full_model.children())[:-1]  # Remove final fc
        self.backbone = nn.Sequential(
            *layers,
            nn.Flatten(),
        )

        # Get the number of features
        in_features = full_model.fc.in_features  # 2048 for ResNet-50

        # Build our custom head with Dropout for MC Dropout
        self._build_head(in_features)

        print(f"ResNet-50 initialized (pretrained={pretrained}, "
              f"classes={num_classes}, dropout={dropout_rate})")
