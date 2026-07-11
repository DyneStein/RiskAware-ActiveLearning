"""
Base model class with MC Dropout support.
All model architectures (EfficientNet, ResNet, DenseNet) inherit from this.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from constants import CLASS_NAMES, HIGH_RISK_CLASSES

# Maps each 7-class label index to 0 (non-malignant) or 1 (malignant), in
# CLASS_NAMES order -- used to derive the risk head's training labels from
# the same (image, 7-class label) pairs the classification head already
# trains on, with no dataset changes needed.
_MALIGNANT_LOOKUP = [1 if name in HIGH_RISK_CLASSES else 0 for name in CLASS_NAMES]


class BaseModel(nn.Module):
    """
    Abstract base class for all classification models.

    Provides:
    - Two independent output heads sharing one backbone: a 7-class
      classification head (drives uncertainty) and a binary
      malignant/non-malignant risk head (drives the risk score). Separate
      parameters mean a case the classification head confidently gets
      wrong doesn't automatically also produce a falsely-low risk score --
      see risk_score/clinical_risk.py for why that mattered.
    - Standard joint training loop (combined cross-entropy loss, optionally
      class-weighted per head)
    - Prediction with softmax probabilities (classification and risk)
    - MC Dropout inference for the classification head only (multiple
      forward passes with dropout ON) -- the risk head stays deterministic,
      since it's a separate, always-uncapped safety signal, not an
      uncertainty-scored one
    - Checkpoint save/load
    """

    def __init__(self, num_classes=7, dropout_rate=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.backbone = None    # Set by subclass
        self.head = None        # Set by subclass, via _build_head()
        self.risk_head = None   # Set by subclass, via _build_risk_head()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _build_head(self, in_features):
        """
        Build the classification head with Dropout layers.
        Dropout layers are what MC Dropout activates during inference.
        """
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(256, self.num_classes),
        )

    def _build_risk_head(self, in_features):
        """
        Build the independent risk head: malignant vs. non-malignant.

        Smaller than the classification head and trained on an easier,
        different problem (2-way, not 7-way) from the same shared features
        -- the goal is a risk signal that doesn't automatically fail
        whenever the classification head is confidently wrong.
        """
        self.risk_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(256, 2),  # [P(non-malignant), P(malignant)]
        )

    def forward(self, x):
        """
        Forward pass through the shared backbone, then both heads.

        Returns
        -------
        class_logits : torch.Tensor, shape (B, num_classes)
        risk_logits : torch.Tensor, shape (B, 2)
        """
        features = self.backbone(x)
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        class_logits = self.head(features)
        risk_logits = self.risk_head(features)
        return class_logits, risk_logits

    def predict(self, images):
        """
        Get classification softmax probabilities for a batch of images.

        Parameters
        ----------
        images : torch.Tensor
            Batch of image tensors, shape (B, 3, H, W).

        Returns
        -------
        np.ndarray
            Softmax probabilities, shape (B, num_classes).
        """
        self.eval()
        with torch.no_grad():
            images = images.to(self.device)
            class_logits, _ = self.forward(images)
            probs = torch.softmax(class_logits, dim=1)
        return probs.cpu().numpy()

    def predict_risk(self, images):
        """
        Get the independent risk head's malignant probability for a batch.

        This is the risk score used throughout the framework (see
        risk_score/clinical_risk.py) -- a single deterministic forward
        pass through the risk head, no MC Dropout involved.

        Parameters
        ----------
        images : torch.Tensor
            Batch of image tensors, shape (B, 3, H, W).

        Returns
        -------
        np.ndarray, shape (B,)
            P(malignant), in [0, 1].
        """
        self.eval()
        with torch.no_grad():
            images = images.to(self.device)
            _, risk_logits = self.forward(images)
            risk_probs = torch.softmax(risk_logits, dim=1)
        return risk_probs[:, 1].cpu().numpy()

    def predict_with_mc_dropout(self, images, n_passes=30):
        """
        MC Dropout inference on the classification head: run N forward
        passes with dropout enabled. The risk head is not involved here --
        it's scored separately and deterministically via predict_risk().

        Returns the mean prediction and the variance (uncertainty).

        Parameters
        ----------
        images : torch.Tensor
            Batch of image tensors, shape (B, 3, H, W).
        n_passes : int
            Number of stochastic forward passes.

        Returns
        -------
        mean_probs : np.ndarray
            Mean softmax probabilities, shape (B, num_classes).
        variance : np.ndarray
            Variance across passes, shape (B, num_classes).
        all_probs : np.ndarray
            All predictions, shape (n_passes, B, num_classes).
        """
        self.enable_dropout()  # Keep dropout ON during inference
        images = images.to(self.device)

        all_probs = []
        with torch.no_grad():
            for _ in range(n_passes):
                class_logits, _ = self.forward(images)
                probs = torch.softmax(class_logits, dim=1)
                all_probs.append(probs.cpu().numpy())

        all_probs = np.array(all_probs)       # (n_passes, B, num_classes)
        mean_probs = all_probs.mean(axis=0)   # (B, num_classes)
        variance = all_probs.var(axis=0)      # (B, num_classes)

        self.eval()  # Reset to eval mode
        return mean_probs, variance, all_probs

    def enable_dropout(self):
        """Enable dropout layers during inference for MC Dropout."""
        self.train()  # Sets all modules to training mode
        # But freeze BatchNorm layers — we only want dropout active
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.eval()

    def train_model(self, train_loader, epochs, lr, weight_decay=1e-5,
                     class_weights=None, risk_class_weights=None):
        """
        Jointly train the classification head and risk head for a given
        number of epochs, sharing one backbone.

        The two heads' losses are summed with equal weight
        (classification_loss + risk_loss) and backpropagated together each
        batch -- both heads and the shared backbone update from one
        optimizer step.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader. Yields (images, labels, image_ids) where
            labels are 7-class indices -- binary risk labels are derived
            from these automatically via _MALIGNANT_LOOKUP, no separate
            risk labels needed.
        epochs : int
            Number of training epochs.
        lr : float
            Learning rate.
        weight_decay : float
            L2 regularization weight.
        class_weights : array-like, optional
            Per-class weights (length num_classes) for the classification
            loss, to counter class imbalance. If None (default), plain
            unweighted cross-entropy is used. See
            config.USE_DYNAMIC_CLASS_WEIGHTS and
            active_learning.al_loop.compute_class_weights().
        risk_class_weights : array-like, optional
            Per-class weights (length 2: [non-malignant, malignant]) for
            the risk loss, same idea as class_weights one level down --
            malignant cases are still the minority even in binary framing.
            See active_learning.al_loop.compute_risk_class_weights().

        Returns
        -------
        list of float
            Combined training loss per epoch.
        """
        self.to(self.device)
        self.train()

        if class_weights is not None:
            weight_tensor = torch.as_tensor(
                class_weights, dtype=torch.float32, device=self.device
            )
            class_criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            class_criterion = nn.CrossEntropyLoss()

        if risk_class_weights is not None:
            risk_weight_tensor = torch.as_tensor(
                risk_class_weights, dtype=torch.float32, device=self.device
            )
            risk_criterion = nn.CrossEntropyLoss(weight=risk_weight_tensor)
        else:
            risk_criterion = nn.CrossEntropyLoss()

        malignant_lookup = torch.tensor(
            _MALIGNANT_LOOKUP, dtype=torch.long, device=self.device
        )

        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )

        epoch_losses = []
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels, _ in tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False
            ):
                images = images.to(self.device)
                labels = labels.to(self.device)
                risk_labels = malignant_lookup[labels]

                optimizer.zero_grad()
                class_logits, risk_logits = self.forward(images)
                loss = (
                    class_criterion(class_logits, labels)
                    + risk_criterion(risk_logits, risk_labels)
                )
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, predicted = class_logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            scheduler.step()
            epoch_loss = running_loss / total
            epoch_acc = 100.0 * correct / total
            epoch_losses.append(epoch_loss)
            print(f"  Epoch {epoch+1}/{epochs} — "
                  f"Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.1f}%")

        return epoch_losses

    def save_checkpoint(self, path):
        """Save model weights (both heads + backbone) to disk."""
        torch.save(self.state_dict(), path)
        print(f"  Checkpoint saved: {path}")

    def load_checkpoint(self, path):
        """Load model weights (both heads + backbone) from disk."""
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.to(self.device)
        print(f"  Checkpoint loaded: {path}")
