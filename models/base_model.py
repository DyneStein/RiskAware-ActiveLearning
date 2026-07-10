"""
Base model class with MC Dropout support.
All model architectures (EfficientNet, ResNet, DenseNet) inherit from this.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm


class BaseModel(nn.Module):
    """
    Abstract base class for all classification models.

    Provides:
    - Standard training loop with cross-entropy loss
    - Prediction with softmax probabilities
    - MC Dropout inference (multiple forward passes with dropout ON)
    - Checkpoint save/load
    """

    def __init__(self, num_classes=7, dropout_rate=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.backbone = None   # Set by subclass
        self.head = None       # Set by subclass
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

    def forward(self, x):
        """Forward pass through backbone + head."""
        features = self.backbone(x)
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        logits = self.head(features)
        return logits

    def predict(self, images):
        """
        Get softmax probabilities for a batch of images.

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
            logits = self.forward(images)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def predict_with_mc_dropout(self, images, n_passes=30):
        """
        MC Dropout inference: run N forward passes with dropout enabled.

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
                logits = self.forward(images)
                probs = torch.softmax(logits, dim=1)
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
                     class_weights=None):
        """
        Train the model for a given number of epochs.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader.
        epochs : int
            Number of training epochs.
        lr : float
            Learning rate.
        weight_decay : float
            L2 regularization weight.
        class_weights : array-like, optional
            Per-class weights (length num_classes) for CrossEntropyLoss, to
            counter class imbalance. If None (default), plain unweighted
            cross-entropy is used. See config.USE_DYNAMIC_CLASS_WEIGHTS and
            active_learning.al_loop.compute_class_weights().

        Returns
        -------
        list of float
            Training loss per epoch.
        """
        self.to(self.device)
        self.train()

        if class_weights is not None:
            weight_tensor = torch.as_tensor(
                class_weights, dtype=torch.float32, device=self.device
            )
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            criterion = nn.CrossEntropyLoss()
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

                optimizer.zero_grad()
                logits = self.forward(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, predicted = logits.max(1)
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
        """Save model weights to disk."""
        torch.save(self.state_dict(), path)
        print(f"  Checkpoint saved: {path}")

    def load_checkpoint(self, path):
        """Load model weights from disk."""
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.to(self.device)
        print(f"  Checkpoint loaded: {path}")
