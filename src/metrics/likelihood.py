"""NLL / bits-per-dimension via the instantaneous change-of-variables
formula for continuous normalizing flows (Grathwohl et al. 2018,
"FFJORD"), integrated along the shared probability-flow ODE
(Method.velocity, src/methods/base.py) from t=T_MIN (data) to t=1 (noise),
using a Hutchinson trace estimator for the divergence.
"""

import math

import torch
from torchdiffeq import odeint

from src.methods.base import Method
from src.schedules import T_MIN

# src.data._TRANSFORM maps pixel/255 (not the textbook pixel/256) to
# [-1, 1]: x = 2 * (pixel/255) - 1, so one raw pixel step is this big in
# the model's x-space.
PIXEL_SCALE = 2.0 / 255.0
BPD_CONSTANT = math.log2(1.0 / PIXEL_SCALE)  # = log2(255/2) ~= 6.994353


def standard_normal_log_prob(x: torch.Tensor) -> torch.Tensor:
    """log N(x; 0, I), per sample (summed over all non-batch dims), nats."""
    dim = x[0].numel()
    flat = x.flatten(1)
    return -0.5 * (flat**2).sum(dim=1) - 0.5 * dim * math.log(2 * math.pi)


def dequantize(x0: torch.Tensor) -> torch.Tensor:
    """Uniform dequantization directly in x-space (standard variational
    bound for evaluating a continuous density model on discrete pixels;
    Theis et al. 2016)."""
    return x0 + torch.rand_like(x0) * PIXEL_SCALE


def log_prob_to_bpd(log_p: torch.Tensor, dim: int) -> torch.Tensor:
    """log_p: (B,) nats, per-sample log-density from compute_log_prob ->
    (B,) bits/dim, converting both the nats->bits base and the
    continuous-density->discrete-pixel change of variables (BPD_CONSTANT)."""
    return -log_p / (dim * math.log(2)) + BPD_CONSTANT


def compute_log_prob(
    method: Method,
    x0: torch.Tensor,
    solver: str = "dopri5",
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> torch.Tensor:
    """x0: (B, 1, 28, 28) -> (B,) log p_model(x0) in nats, per sample."""
    # eps: (B, 1, 28, 28) Rademacher vector, fixed for the whole ODE solve
    # (one Hutchinson trace estimator per sample, not resampled per step).
    eps = torch.randint(0, 2, x0.shape, device=x0.device, dtype=x0.dtype) * 2 - 1

    def augmented_ode(t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        # state = (x, logdet), x: (B, 1, 28, 28), logdet: (B,) -> matching
        # shapes (dx/dt, d(logdet)/dt) for torchdiffeq to integrate jointly.
        x, _ = state
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            t_batch = t.expand(x_req.shape[0])
            v = method.velocity(x_req, t_batch)
            # Vector-Jacobian product (v^T eps) w.r.t. x_req, cheaper than
            # the full Jacobian: eps^T (dv/dx) eps is an unbiased estimator
            # of tr(dv/dx), the instantaneous change-of-variables term.
            (vjp,) = torch.autograd.grad((v * eps).sum(), x_req)
        trace = (vjp * eps).flatten(1).sum(dim=1)  # (B,)
        return v.detach(), trace

    logdet0 = torch.zeros(x0.shape[0], device=x0.device)
    t_grid = torch.tensor([T_MIN, 1.0], device=x0.device)
    x1, logdet = odeint(
        augmented_ode, (x0, logdet0), t_grid, method=solver, rtol=rtol, atol=atol
    )
    x1_final, logdet_final = x1[-1], logdet[-1]
    return standard_normal_log_prob(x1_final) + logdet_final


def compute_bpd(
    method: Method,
    x0: torch.Tensor,
    num_mc_samples: int = 1,
    solver: str = "dopri5",
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> torch.Tensor:
    """Per-sample bits/dim, averaged over num_mc_samples independent
    dequantization + Hutchinson-trace noise draws."""
    dim = x0[0].numel()
    estimates = []
    for _ in range(num_mc_samples):
        x_deq = dequantize(x0)
        log_p = compute_log_prob(method, x_deq, solver=solver, rtol=rtol, atol=atol)
        estimates.append(log_prob_to_bpd(log_p, dim))
    return torch.stack(estimates).mean(dim=0)
