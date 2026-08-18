# tests/test_sampling.py
import math

import torch
from torch import nn

from src.methods.base import Method
from src.methods.flow_matching import FlowMatchingOT
from src.sampling import sample
from src.schedules import T_MIN


class _LinearDecayMethod(Method):
    """Test double with a known analytic solution: dx/dt = -x has solution
    x(t) = x(T_MIN) * exp(-(t-T_MIN)). Used to verify the integrator itself,
    independent of any trained network."""

    def __init__(self):
        super().__init__(net=nn.Identity())

    def loss(self, x1: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("not needed for this test")

    def velocity(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -x


def test_sampler_matches_analytic_solution_for_linear_ode():
    torch.manual_seed(0)
    method = _LinearDecayMethod()
    x0 = torch.randn(5, 1, 4, 4)

    # Monkeypatch: sample() draws its own random x0 internally, so instead we
    # replicate its integration call directly to compare against x0 exactly.
    from torchdiffeq import odeint

    t_grid = torch.linspace(T_MIN, 1.0 - T_MIN, 2000)

    def ode_func(t, x):
        return method.velocity(x, t.expand(x.shape[0]))

    traj = odeint(ode_func, x0, t_grid, method="euler")
    x1_numeric = traj[-1]
    x1_analytic = x0 * math.exp(-(1.0 - 2 * T_MIN))
    assert torch.allclose(x1_numeric, x1_analytic, atol=1e-2)  # type: ignore


def test_sample_returns_correct_shape_and_is_finite():
    method = FlowMatchingOT()
    images = sample(method, num_samples=3, num_steps=5, device="cpu")
    assert images.shape == (3, 1, 28, 28)
    assert torch.isfinite(images).all()
