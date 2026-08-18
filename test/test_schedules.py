import torch

from src.schedules import T_MIN, OTPath, VPPath, beta, expand_t


def test_ot_path_boundaries():
    t = torch.tensor([0.0, 1.0])
    assert torch.allclose(OTPath.alpha(t), torch.tensor([1.0, 0.0]))
    assert torch.allclose(OTPath.sigma(t), torch.tensor([0.0, 1.0]))


def test_ot_path_derivatives_are_constant():
    t = torch.tensor([0.2, 0.7])
    assert torch.allclose(OTPath.alpha_dot(t), torch.tensor([-1.0, -1.0]))
    assert torch.allclose(OTPath.sigma_dot(t), torch.tensor([1.0, 1.0]))


def test_vp_path_is_variance_preserving():
    t = torch.linspace(T_MIN, 1.0, 10)
    variance = VPPath.alpha(t) ** 2 + VPPath.sigma(t) ** 2
    assert torch.allclose(variance, torch.ones_like(variance), atol=1e-5)


def test_vp_path_boundaries():
    t_min = torch.tensor([T_MIN])
    t_max = torch.tensor([1.0])
    assert torch.allclose(VPPath.alpha(t_min), torch.tensor([1.0]), atol=1e-3)
    assert VPPath.sigma(t_max).item() > 0.999
    assert VPPath.alpha(t_max).item() < 0.01


def test_vp_path_derivatives_match_autograd():
    t = torch.linspace(T_MIN, 1.0 - 1e-3, 8, requires_grad=True)
    alpha = VPPath.alpha(t)
    sigma = VPPath.sigma(t)
    (autograd_alpha_dot,) = torch.autograd.grad(alpha.sum(), t, retain_graph=True)
    (autograd_sigma_dot,) = torch.autograd.grad(sigma.sum(), t)
    with torch.no_grad():
        assert torch.allclose(VPPath.alpha_dot(t), autograd_alpha_dot, atol=1e-4)
        assert torch.allclose(VPPath.sigma_dot(t), autograd_sigma_dot, atol=1e-4)


def test_beta_is_linear():
    """beta(t) decreases in this module's t=0-noise/t=1-data convention:
    BETA_MAX at t=0, BETA_MIN at t=1 (see src/schedules.py's docstring)."""
    t = torch.tensor([0.0, 0.5, 1.0])
    expected = 20.0 - t * (20.0 - 0.1)
    assert torch.allclose(beta(t), expected)


def test_expand_t_broadcasts_against_image_tensor():
    t = torch.tensor([0.1, 0.2, 0.3])
    x = torch.zeros(3, 1, 28, 28)
    te = expand_t(t, x)
    assert te.shape == (3, 1, 1, 1)
    assert (te * x).shape == x.shape
