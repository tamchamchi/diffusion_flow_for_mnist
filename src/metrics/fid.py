"""Frechet Inception Distance (FID; Heusel et al. 2017) for MNIST samples.

FID compares feature-space statistics of real vs. generated images. Two
feature extractors are supported:

- get_inception_feature_extractor: the standard, literature-comparable
  choice -- 2048-dim ImageNet-pretrained Inception-v3 pool features. MNIST
  is 28x28 grayscale, so every image is resized to Inception's expected
  299x299 and channel-replicated to 3 channels before extraction.
- get_mnist_cnn_feature_extractor: 128-dim features from a small CNN
  trained on MNIST itself (src/train_classifier.py). Inception's features
  were learned on natural RGB photos, a poor domain match for MNIST
  digits, so this is the more MNIST-appropriate option; it just isn't
  comparable to FID numbers reported elsewhere.

Both extractors are returned as (model, transform) so every downstream
function below (compute_activation_statistics, compute_or_load_real_statistics)
works identically regardless of which one is used.
"""

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import scipy.linalg
import torch
from torch import nn
from torchvision.models import Inception_V3_Weights, inception_v3

from src.models import build_default_mnist_cnn

logger = logging.getLogger(__name__)


def get_inception_feature_extractor(
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, Callable[[torch.Tensor], torch.Tensor]]:
    """Returns (model, transform) where model(transform(images)) yields
    (N, 2048) pooled features. Requires internet access the first time (to
    download ImageNet-pretrained Inception-v3 weights)."""
    logger.info("Loading ImageNet-pretrained Inception-v3 weights (downloads on first use)")
    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights, aux_logits=True)
    model.fc = nn.Identity()  # type: ignore # expose the 2048-dim pooled features directly
    model.eval()
    model.to(device)
    preprocess = weights.transforms()

    def transform(images: torch.Tensor) -> torch.Tensor:
        # images: (N, 1, 28, 28) in [-1, 1] -> (N, 3, 299, 299) Inception input
        images = (images.clamp(-1, 1) + 1) / 2  # [-1,1] -> [0,1]
        images = images.repeat(1, 3, 1, 1)
        return preprocess(images)

    return model, transform


class _MNISTClassifierFeatureExtractor(nn.Module):
    """Thin wrapper so model(x) yields the classifier's penultimate-layer
    embedding directly, matching the (N, feature_dim)-out interface
    compute_activation_statistics expects from any extractor."""

    def __init__(self, classifier: nn.Module) -> None:
        super().__init__()
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier.extract_features(x)


def get_mnist_cnn_feature_extractor(
    ckpt_path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, Callable[[torch.Tensor], torch.Tensor]]:
    """Returns (model, transform) where model(transform(images)) yields
    (N, 128) features from the penultimate layer of the MNIST classifier
    trained by src/train_classifier.py. Unlike Inception, no resize/channel
    replication is needed -- the classifier already consumes (N,1,28,28)
    images in the same [-1,1] range training uses."""
    logger.info("Loading MNIST classifier weights from %s", ckpt_path)
    classifier = build_default_mnist_cnn()
    classifier.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model = _MNISTClassifierFeatureExtractor(classifier)
    model.eval()
    model.to(device)

    def transform(images: torch.Tensor) -> torch.Tensor:
        return images.clamp(-1, 1)

    return model, transform


@torch.no_grad()
def compute_activation_statistics(
    images: torch.Tensor,
    extractor: nn.Module,
    transform: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device | str = "cpu",
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """(N,1,28,28) images in [-1,1] -> (mu, sigma) of their extractor features."""
    if images.shape[0] < 2:
        raise ValueError(
            f"compute_activation_statistics got {images.shape[0]} image(s); "
            "estimating a covariance matrix needs at least 2, and a "
            "meaningful FID needs far more than that (Heusel et al. 2017 "
            "use tens of thousands) -- pass a larger --num-fid-samples."
        )
    logger.info("Extracting Inception features for %d images (batch_size=%d)", images.shape[0], batch_size)
    features = []
    for i in range(0, images.shape[0], batch_size):
        batch = images[i : i + batch_size].to(device)
        feats = extractor(transform(batch))
        features.append(feats.cpu().numpy())
        logger.debug("Feature extraction progress: %d/%d", min(i + batch_size, images.shape[0]), images.shape[0])
    features_np = np.concatenate(features, axis=0)
    mu = features_np.mean(axis=0)
    sigma = np.cov(features_np, rowvar=False)
    logger.info("Feature extraction done: %d images -> %d-dim statistics", images.shape[0], mu.shape[0])
    return mu, sigma


def fid_from_statistics(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Frechet distance between N(mu1,sigma1) and N(mu2,sigma2)."""
    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean, _ = scipy.linalg.sqrtm(
            (sigma1 + offset) @ (sigma2 + offset), disp=False
        )
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def compute_or_load_real_statistics(
    loader,
    cache_path: Path,
    extractor: nn.Module,
    transform: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device | str = "cpu",
    num_samples: int = 50_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Real-image activation statistics are identical for every method being
    compared, so this caches them to disk and computes them only once."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        logger.info("Loading cached real-image activation statistics from %s", cache_path)
        data = np.load(cache_path)
        return data["mu"], data["sigma"]

    logger.info(
        "No cached real-image statistics at %s; collecting up to %d real images",
        cache_path, num_samples,
    )
    collected = []
    n_seen = 0
    for x0, _ in loader:
        if n_seen >= num_samples:
            break
        collected.append(x0)
        n_seen += x0.shape[0]
        logger.debug("Real-image collection progress: %d/%d", n_seen, num_samples)
    images = torch.cat(collected, dim=0)[:num_samples]

    mu, sigma = compute_activation_statistics(images, extractor, transform, device)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, mu=mu, sigma=sigma)
    logger.info("Cached real-image activation statistics to %s", cache_path)
    return mu, sigma
