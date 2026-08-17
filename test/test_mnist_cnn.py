# tests/test_mnist_cnn.py
import pytest
import torch

from src.metrics.fid import (
    compute_activation_statistics,
    fid_from_statistics,
    get_mnist_cnn_feature_extractor,
)
from src.models import build_default_mnist_cnn


def test_forward_returns_class_logits():
    model = build_default_mnist_cnn()
    x = torch.randn(4, 1, 28, 28)
    logits = model(x)
    assert logits.shape == (4, 10)
    assert torch.isfinite(logits).all()


def test_extract_features_returns_penultimate_embedding():
    model = build_default_mnist_cnn()
    x = torch.randn(4, 1, 28, 28)
    features = model.extract_features(x)
    assert features.shape == (4, model.feature_dim)
    assert torch.isfinite(features).all()
    # ReLU is the last op in the embedding layer.
    assert (features >= 0).all()


def test_get_mnist_cnn_feature_extractor_matches_fid_interface(tmp_path):
    """An (untrained) checkpoint should load and plug into the same
    compute_activation_statistics/fid_from_statistics pipeline the
    Inception extractor uses, with feature dim 128 instead of 2048."""
    ckpt_path = tmp_path / "mnist_cnn.pt"
    torch.save(build_default_mnist_cnn().state_dict(), ckpt_path)

    extractor, transform = get_mnist_cnn_feature_extractor(ckpt_path, device="cpu")
    # More samples than feature dims, so the covariance estimate is
    # non-singular and scipy.linalg.sqrtm behaves numerically well.
    images = torch.randn(200, 1, 28, 28).clamp(-1, 1)

    mu, sigma = compute_activation_statistics(images, extractor, transform, device="cpu")
    assert mu.shape == (128,)
    assert sigma.shape == (128, 128)

    # Identical statistics against themselves -> ~zero Frechet distance.
    assert fid_from_statistics(mu, sigma, mu, sigma) == pytest.approx(0.0, abs=1e-4)
