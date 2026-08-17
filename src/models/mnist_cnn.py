"""Small CNN classifier trained on MNIST digits.

Trained standalone as a 10-way digit classifier (src/train_classifier.py).
src/metrics/fid.py then reuses its penultimate embedding layer as an
alternative FID feature extractor: the default FID extractor
(get_inception_feature_extractor) uses ImageNet-pretrained Inception-v3,
but Inception's features were learned on natural RGB photos and are a poor
domain match for 28x28 grayscale digits (Salimans et al. 2016 and later
MNIST-focused papers instead extract features from a small classifier
trained on MNIST itself). This module is that classifier.
"""

import torch
from torch import nn


class MNISTClassifierCNN(nn.Module):
    """LeNet-style CNN. Resolution path: 28 -> (pool) 14 -> (pool) 7.
    Channel path: 1 -> 32 -> 64 -> flatten -> feature_dim -> num_classes."""

    def __init__(
        self,
        num_classes: int = 10,
        feature_dim: int = 128,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14 -> 7
        )
        # This Linear+ReLU layer is the "embedding" reused as FID features:
        # a fixed-size summary of the image that is informative enough to
        # separate the 10 digit classes downstream.
        self.embed = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, feature_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """(N,1,28,28) -> (N, feature_dim) penultimate-layer embedding."""
        return self.embed(self.conv(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N,1,28,28) -> (N, num_classes) logits."""
        return self.classifier(self.extract_features(x))


def build_default_mnist_cnn() -> MNISTClassifierCNN:
    """Single source of truth for the classifier's hyperparameters, so
    training (src/train_classifier.py) and FID feature extraction
    (src/metrics/fid.py) always instantiate an identical architecture."""
    return MNISTClassifierCNN(num_classes=10, feature_dim=128, dropout=0.5)
