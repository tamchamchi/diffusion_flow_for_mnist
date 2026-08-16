"""Shared Probability-Flow-ODE sampler used to draw samples from any of the
five methods. Every Method.velocity(x, t) already returns dx/dt in the same
(t≈0 data, t=1 noise) convention, so this integrator is method-agnostic.
"""

import torch
from torchdiffeq import odeint

from src.methods.base import Method
from src.schedules import T_MIN


@torch.no_grad()
def sample(
    method: Method,
    num_samples: int,
    solver: str = "euler",
    shape: tuple[int, int, int] = (1, 28, 28),
    num_steps: int = 50,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Integrate dx/dt = method.velocity(x, t) from t=1 (noise) to t=T_MIN
    (data) with a fixed-step Euler solver, identical for all 5 methods."""
    method.eval()
    x1 = torch.randn(num_samples, *shape, device=device)
    t_grid = torch.linspace(1.0, T_MIN, num_steps, device=device)

    def ode_func(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t_batch = t.expand(x.shape[0])
        return method.velocity(x, t_batch)

    trajectory = odeint(ode_func, x1, t_grid, method=solver)
    return trajectory[-1]  # type: ignore
